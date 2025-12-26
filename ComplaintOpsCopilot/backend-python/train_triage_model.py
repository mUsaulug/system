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

# 1. Create Synthetic Dataset
data = [
    # FRAUD_UNAUTHORIZED_TX
    ("Kartımdan bilgim dışında 500 TL çekilmiş.", "FRAUD_UNAUTHORIZED_TX", "RED"),
    ("Hesabımda tanımadığım bir işlem var, iptal edin.", "FRAUD_UNAUTHORIZED_TX", "RED"),
    ("Kredi kartım çalındı, hemen kapatın.", "FRAUD_UNAUTHORIZED_TX", "RED"),
    
    # CHARGEBACK_DISPUTE
    ("Aynı işlemden iki kere ücret alınmış.", "CHARGEBACK_DISPUTE", "YELLOW"),
    ("İade ettiğim ürünün parası hala yatmadı.", "CHARGEBACK_DISPUTE", "YELLOW"),
    ("Siparişi iptal ettim ama param iade edilmedi.", "CHARGEBACK_DISPUTE", "YELLOW"),

    # TRANSFER_DELAY
    ("Yaptığım EFT 3 saattir karşı hesaba geçmedi.", "TRANSFER_DELAY", "YELLOW"),
    ("Havale işlemim hala beklemede görünüyor.", "TRANSFER_DELAY", "YELLOW"),
    ("Para transferi yaptım ama ulaşmadı.", "TRANSFER_DELAY", "YELLOW"),

    # TECHNICAL_ISSUE
    ("Mobil uygulamaya giriş yapamıyorum.", "ACCESS_LOGIN_MOBILE", "RED"),
    ("Şifremi unuttum, yenileme linki gelmiyor.", "ACCESS_LOGIN_MOBILE", "RED"),
    ("İnternet bankacılığı açılmıyor, hata veriyor.", "ACCESS_LOGIN_MOBILE", "RED"),

    # GENERAL_INFO
    ("Kredi kartı limitimi nasıl arttırabilirim?", "CARD_LIMIT_CREDIT", "GREEN"),
    ("Şube çalışma saatleriniz nedir?", "INFORMATION_REQUEST", "GREEN"),
    ("Yeni kampanya detaylarını öğrenmek istiyorum.", "CAMPAIGN_POINTS_REWARDS", "GREEN"),
     ("IBAN numaramı nereden görebilirim?", "INFORMATION_REQUEST", "GREEN"),
]

df = pd.DataFrame(data, columns=["text", "category", "urgency"])

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
