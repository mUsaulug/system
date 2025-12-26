import hashlib
import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

dataset_path = os.path.join("data", "triage_dataset.json")
if os.path.exists(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
else:
    records = [
        {"text": "Kartımdan bilgim dışında 500 TL çekilmiş.", "category": "FRAUD_UNAUTHORIZED_TX", "urgency": "RED"},
        {"text": "Hesabımda tanımadığım bir işlem var, iptal edin.", "category": "FRAUD_UNAUTHORIZED_TX", "urgency": "RED"},
        {"text": "Kredi kartım çalındı, hemen kapatın.", "category": "FRAUD_UNAUTHORIZED_TX", "urgency": "RED"},
        {"text": "Aynı işlemden iki kere ücret alınmış.", "category": "CHARGEBACK_DISPUTE", "urgency": "YELLOW"},
        {"text": "İade ettiğim ürünün parası hala yatmadı.", "category": "CHARGEBACK_DISPUTE", "urgency": "YELLOW"},
        {"text": "Siparişi iptal ettim ama param iade edilmedi.", "category": "CHARGEBACK_DISPUTE", "urgency": "YELLOW"},
        {"text": "Yaptığım EFT 3 saattir karşı hesaba geçmedi.", "category": "TRANSFER_DELAY", "urgency": "YELLOW"},
        {"text": "Havale işlemim hala beklemede görünüyor.", "category": "TRANSFER_DELAY", "urgency": "YELLOW"},
        {"text": "Para transferi yaptım ama ulaşmadı.", "category": "TRANSFER_DELAY", "urgency": "YELLOW"},
        {"text": "Mobil uygulamaya giriş yapamıyorum.", "category": "ACCESS_LOGIN_MOBILE", "urgency": "RED"},
        {"text": "Şifremi unuttum, yenileme linki gelmiyor.", "category": "ACCESS_LOGIN_MOBILE", "urgency": "RED"},
        {"text": "İnternet bankacılığı açılmıyor, hata veriyor.", "category": "ACCESS_LOGIN_MOBILE", "urgency": "RED"},
        {"text": "Kredi kartı limitimi nasıl arttırabilirim?", "category": "CARD_LIMIT_CREDIT", "urgency": "GREEN"},
        {"text": "Şube çalışma saatleriniz nedir?", "category": "INFORMATION_REQUEST", "urgency": "GREEN"},
        {"text": "Yeni kampanya detaylarını öğrenmek istiyorum.", "category": "CAMPAIGN_POINTS_REWARDS", "urgency": "GREEN"},
        {"text": "IBAN numaramı nereden görebilirim?", "category": "INFORMATION_REQUEST", "urgency": "GREEN"},
    ]

df = pd.DataFrame(records, columns=["text", "category", "urgency"])

def hash_dataset(frame: pd.DataFrame) -> str:
    payload = "|".join(
        f"{row.text}::{row.category}::{row.urgency}" for row in frame.itertuples(index=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
dataset_hash = hash_dataset(df)

train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df["category"],
)

# 2. Train Category Model
print("Training Category Model...")
category_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('clf', LogisticRegression(random_state=42))
])
category_pipeline.fit(train_df["text"], train_df["category"])

# 3. Train Urgency Model
print("Training Urgency Model...")
urgency_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('clf', LogisticRegression(random_state=42))
])
urgency_pipeline.fit(train_df["text"], train_df["urgency"])

# 4. Evaluate Models
category_preds = category_pipeline.predict(test_df["text"])
urgency_preds = urgency_pipeline.predict(test_df["text"])

category_f1 = f1_score(test_df["category"], category_preds, average="macro")
urgency_f1 = f1_score(test_df["urgency"], urgency_preds, average="macro")

category_confusion = confusion_matrix(test_df["category"], category_preds).tolist()
urgency_confusion = confusion_matrix(test_df["urgency"], urgency_preds).tolist()

os.makedirs("reports", exist_ok=True)
report_path = os.path.join("reports", f"triage_eval_{timestamp}.json")
with open(report_path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "timestamp": timestamp,
            "dataset_hash": dataset_hash,
            "category_f1_macro": category_f1,
            "urgency_f1_macro": urgency_f1,
            "category_confusion_matrix": category_confusion,
            "urgency_confusion_matrix": urgency_confusion,
        },
        handle,
        ensure_ascii=False,
        indent=2,
    )

# 5. Save Models
os.makedirs("models", exist_ok=True)
category_model_path = os.path.join("models", f"category_model_{timestamp}.pkl")
urgency_model_path = os.path.join("models", f"urgency_model_{timestamp}.pkl")
joblib.dump(category_pipeline, category_model_path)
joblib.dump(urgency_pipeline, urgency_model_path)

latest_metadata = {
    "timestamp": timestamp,
    "dataset_hash": dataset_hash,
    "category_model_path": category_model_path,
    "urgency_model_path": urgency_model_path,
}

with open(os.path.join("models", "latest.json"), "w", encoding="utf-8") as handle:
    json.dump(latest_metadata, handle, ensure_ascii=False, indent=2)

print("Models saved to 'models/' directory.")
