"""Lightweight project assistant.

* Operational questions ("which suppliers are high risk?", "how many shipment alerts?") are answered
  from live database queries.
* Conceptual questions ("how is supplier risk calculated?") are answered by retrieving the most relevant
  documentation section with TF-IDF cosine similarity (scikit-learn). No external API or LLM is used.
"""
import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..ml.predictor import load_metrics
from ..models import Machine, PredictionHistory, Shipment, Supplier, User

KB = [
    ("System overview",
     "The Supply Chain Risk Prediction and Decision Support System combines rule-based risk scoring and machine "
     "learning. Modules: Overview dashboard, Procurement (supplier risk), Production (machine failure prediction), "
     "Inventory (stockout risk and reorder), Transportation (late delivery prediction), Demand Analysis, AI Assistant "
     "and Prediction History. Backend FastAPI with SQLite and SQLAlchemy, frontend React Vite Tailwind Recharts."),
    ("Supplier risk calculation",
     "Supplier risk is rule based, score 0 to 100 where higher means riskier. Five factors are converted to 0-100 "
     "sub-scores: delivery = (100 - on-time delivery %) x 2.5, weight 30%; quality = defect rate % x 12.5, weight 25%; "
     "lead time = variability / average lead time x 200, weight 15%; fulfillment = (100 - fulfillment %) x 4, weight "
     "15%; financial = (5 - stability rating) / 4 x 100, weight 15%. Score below 35 is Low risk, 35 to 60 Medium, "
     "60 or more High. The largest sub-score determines the recommendation."),
    ("Inventory stockout risk and reorder calculation",
     "Net stock = available stock - reserved stock. Days of cover = net stock / average daily demand. High stockout "
     "risk if net stock is at or below safety stock or days of cover is less than the lead time. Medium risk if net "
     "stock is at or below the reorder level. Otherwise Low. Recommended order quantity = reorder level + average "
     "daily demand x 30 day review period - net stock. Safety stock = 1.65 x standard deviation of daily demand x "
     "square root of lead time (95% service level). Reorder level (reorder point) = average daily demand x lead time "
     "+ safety stock. Average daily demand and its variability come from real DataCo order history."),
    ("Machine failure prediction model",
     "The machine failure model is trained on the AI4I 2020 Predictive Maintenance dataset (10,000 records, about "
     "3.4% failures). Inputs: machine type L M H, air temperature K, process temperature K, rotational speed rpm, "
     "torque Nm, tool wear min, plus engineered temperature difference and mechanical power. The failure-mode "
     "columns TWF HDF PWF OSF RNF are excluded because they describe the failure itself (target leakage); UDI and "
     "Product ID are identifiers and are excluded. A Random Forest classifier with balanced class weights is used in "
     "the application and compared with Logistic Regression on a stratified 80/20 split."),
    ("Machine risk levels and maintenance recommendations",
     "Failure probability below 0.20 is Low risk, 0.20 to 0.50 Medium, 0.50 or more High and predicted as failure. "
     "High risk: schedule immediate maintenance. Medium: plan preventive maintenance within the next shift. Low: "
     "routine monitoring. Extra checks are added when tool wear is above 200 minutes, the process-air temperature "
     "gap is below 8.6 K, power is outside 3.5-9 kW, or torque is above 60 Nm."),
    ("Late delivery prediction model",
     "The late delivery model is trained on the DataCo Smart Supply Chain dataset (180,519 order lines). Target "
     "Late_delivery_risk. Inputs known at order time: payment type, shipping mode, scheduled shipping days, market, "
     "order region, product category, department, customer segment, quantity, product price, discount rate, sales, "
     "order month, weekday and hour. Excluded to prevent target leakage: actual days for shipping, delivery status, "
     "shipping date, order status and realised profit. Personal data such as customer names, email, password, street "
     "and zipcode is removed. The split groups by Order Id so items of one order are never in both train and test. "
     "Random Forest is used and compared with Logistic Regression."),
    ("Shipment risk levels and actions",
     "Late delivery probability below 0.40 is Low risk, 0.40 to 0.60 Medium, 0.60 or more High. High risk: upgrade "
     "the shipping mode or notify the customer and prioritise the order. Medium: monitor and confirm carrier pickup. "
     "Low: no action needed."),
    ("Demand analysis",
     "Demand analysis uses real DataCo order records aggregated by month, product and category (cancelled and "
     "suspected fraud orders excluded). The trend is a least-squares linear slope over the last 12 months, expressed "
     "as percent of mean demand per month: above +2% Increasing, below -2% Decreasing, otherwise Stable. A 3-month "
     "moving average is shown. The DataCo data has far fewer order lines from October 2017, so the consistent period "
     "January 2015 to September 2017 is the default for trend analysis. No separate forecasting model is used."),
    ("Data sources and sample data",
     "ai4i2020.csv and DataCoSupplyChainDataset.csv are the real datasets. suppliers.csv and inventory.csv were not "
     "provided, so sample templates were generated: supplier performance values are simulated, inventory stock "
     "quantities are simulated but demand rates come from real DataCo history. You can replace them by uploading "
     "your own CSV files on the Procurement and Inventory pages."),
    ("CSV upload",
     "CSV uploads are available on Procurement (suppliers), Inventory, Production (machine readings, predicted in "
     "batch) and Transportation (shipments, predicted in batch). Download the template button to get the required "
     "columns. Files must be .csv, at most 5 MB and 5000 rows. Each row is validated; invalid rows are rejected with "
     "the row number and reason while valid rows are saved."),
    ("Authentication and security",
     "Users sign up with full name, a valid email and a password of at least 8 characters with a letter and a number, "
     "confirmed twice. Duplicate emails are rejected. Passwords are hashed with bcrypt and never stored in plain text. "
     "Sign in returns a JWT access token (HS256) which protects every API route. The secret key is read from the .env "
     "file. Logout removes the token from the browser."),
    ("Prediction history",
     "Every machine failure and late delivery prediction made by a user, manually or by CSV upload, is saved in the "
     "prediction history table with inputs, probability, risk level, recommendation and timestamp. Users can filter, "
     "delete single entries or clear their history."),
]

_LIVE_HINT = re.compile(r"\b(how many|which|list|show|top|current|currently|now|today|riskiest|highest|worst|alerts?|"
                        r"count|status|summary|any)\b")
_CONCEPT_HINT = re.compile(r"\b(calculat\w*|formula|how (is|are|does|do)|why|what is|what are|explain|define|meaning)\b")


@lru_cache(maxsize=1)
def _index():
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform([f"{t}. {t}. {body}" for t, body in KB])
    return vec, mat


def _retrieve(q: str):
    vec, mat = _index()
    sims = cosine_similarity(vec.transform([q]), mat)[0]
    order = sims.argsort()[::-1]
    return [(KB[i][0], KB[i][1], float(sims[i])) for i in order[:2]]


def _fmt_pct(x):
    return f"{x * 100:.1f}%"


def _live(q: str, db: Session, user: User):
    code = re.search(r"sup-\d{3}", q)
    if code:
        s = db.query(Supplier).filter(func.lower(Supplier.supplier_code) == code.group(0)).first()
        if not s:
            return f"No supplier with code {code.group(0).upper()} was found."
        return (f"{s.supplier_code} {s.name} ({s.country}) has a risk score of {s.risk_score} - {s.risk_level} risk. "
                f"On-time delivery {s.on_time_delivery_rate}%, defect rate {s.defect_rate}%, fulfilment "
                f"{s.fulfillment_rate}%, lead time {s.avg_lead_time_days} +/- {s.lead_time_variability_days} days, "
                f"financial stability {s.financial_stability}/5.")
    if not _LIVE_HINT.search(q) or _CONCEPT_HINT.search(q):
        if not re.search(r"\b(accuracy|metrics?|performance|f1|precision|recall|auc)\b", q):
            return None
    if re.search(r"\b(accuracy|metrics?|performance|f1|precision|recall|auc|compare|comparison)\b", q):
        m = load_metrics()
        if not m:
            return "The models have not been trained yet. Run python -m app.ml.train."
        lines = []
        for key, label in [("machine_failure", "Machine failure"), ("late_delivery", "Late delivery")]:
            for model, r in m[key]["models"].items():
                lines.append(f"{label} - {model.replace('_', ' ')}: accuracy {_fmt_pct(r['accuracy'])}, precision "
                             f"{_fmt_pct(r['precision'])}, recall {_fmt_pct(r['recall'])}, F1 {r['f1']:.3f}, "
                             f"ROC-AUC {r['roc_auc']:.3f}")
        return "Hold-out test results from the last training run:\n" + "\n".join(lines)
    if re.search(r"suppliers?|procurement|vendor", q):
        counts = dict(db.query(Supplier.risk_level, func.count()).group_by(Supplier.risk_level).all())
        high = db.query(Supplier).filter(Supplier.risk_level == "High").order_by(Supplier.risk_score.desc()).limit(5).all()
        txt = (f"There are {sum(counts.values())} suppliers: {counts.get('High', 0)} High, {counts.get('Medium', 0)} "
               f"Medium and {counts.get('Low', 0)} Low risk.")
        if high:
            txt += " Highest risk: " + ", ".join(f"{s.name} ({s.risk_score})" for s in high) + "."
        return txt
    if re.search(r"machines?|production|maintenance|failures?", q):
        counts = dict(db.query(Machine.risk_level, func.count()).group_by(Machine.risk_level).all())
        top = db.query(Machine).filter(Machine.risk_level == "High").order_by(Machine.failure_probability.desc()).limit(5).all()
        txt = (f"{sum(counts.values())} machines are monitored: {counts.get('High', 0)} High, {counts.get('Medium', 0)} "
               f"Medium and {counts.get('Low', 0)} Low failure risk.")
        if top:
            txt += " Machine failure alerts: " + ", ".join(f"{m.machine_code} ({_fmt_pct(m.failure_probability)})" for m in top) + "."
        return txt
    if re.search(r"shipments?|deliver\w*|late|delay\w*|transport\w*|orders?", q):
        counts = dict(db.query(Shipment.risk_level, func.count()).group_by(Shipment.risk_level).all())
        modes = db.query(Shipment.shipping_mode, func.avg(Shipment.late_probability)).group_by(Shipment.shipping_mode) \
            .order_by(func.avg(Shipment.late_probability).desc()).all()
        txt = (f"{sum(counts.values())} shipments are tracked: {counts.get('High', 0)} High, {counts.get('Medium', 0)} "
               f"Medium and {counts.get('Low', 0)} Low late-delivery risk.")
        if modes:
            txt += " Average late probability by mode: " + ", ".join(f"{m} {_fmt_pct(p)}" for m, p in modes) + "."
        return txt
    if re.search(r"inventory|stock\w*|reorder|warehouse", q):
        from ..routers.inventory import all_items
        items = all_items(db)
        alerts = [i for i in items if i["stockout_risk"] != "Low"]
        txt = (f"{len(items)} inventory records: {sum(i['stockout_risk'] == 'High' for i in items)} High and "
               f"{sum(i['stockout_risk'] == 'Medium' for i in items)} Medium stockout risk.")
        if alerts:
            txt += " Reorder now: " + ", ".join(f"{i['product_name']} ({i['recommended_order_qty']} units)" for i in alerts[:5]) + "."
        return txt
    if re.search(r"\b(my )?(predictions?|history)\b", q):
        n = db.query(PredictionHistory).filter(PredictionHistory.user_id == user.id).count()
        hi = db.query(PredictionHistory).filter(PredictionHistory.user_id == user.id,
                                                PredictionHistory.risk_level == "High").count()
        return f"You have made {n} predictions, {hi} of them classified as High risk."
    if re.search(r"summary|status|overview", q):
        return " ".join(filter(None, [_live(f"how many {w}", db, user) for w in ["suppliers", "machines", "shipments", "inventory"]]))
    return None


def answer(question: str, db: Session, user: User) -> dict:
    q = question.lower().strip()
    live = _live(q, db, user)
    if live:
        return {"answer": live, "type": "live_data", "sources": ["Live database query"]}
    hits = _retrieve(q)
    if hits[0][2] < 0.08:
        return {"answer": "I could not find that in the project documentation. Try asking about supplier risk, "
                          "machine failure prediction, late delivery prediction, inventory reorder rules, demand "
                          "trends, model accuracy, CSV uploads or security - or ask a live question such as "
                          "'Which suppliers are high risk?'.", "type": "fallback", "sources": []}
    text = hits[0][1]
    sources = [hits[0][0]]
    if hits[1][2] > 0.6 * hits[0][2] and hits[1][2] > 0.12:
        text += "\n\n" + hits[1][1]
        sources.append(hits[1][0])
    return {"answer": text, "type": "documentation", "sources": sources}


SUGGESTIONS = ["Which suppliers are high risk?", "How is supplier risk calculated?", "How many machine failure alerts are there?",
               "What features does the late delivery model use?", "Show inventory items that need reorder",
               "What is the model accuracy?", "How is the demand trend calculated?", "How are passwords stored?"]
