import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import os

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

# 2. Train Category Model
print("Training Category Model...")
category_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('clf', LogisticRegression(random_state=42))
])
category_pipeline.fit(df["text"], df["category"])

# 3. Train Urgency Model
print("Training Urgency Model...")
urgency_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('clf', LogisticRegression(random_state=42))
])
urgency_pipeline.fit(df["text"], df["urgency"])

# 4. Save Models
os.makedirs("models", exist_ok=True)
joblib.dump(category_pipeline, "models/category_model.pkl")
joblib.dump(urgency_pipeline, "models/urgency_model.pkl")

print("Models saved to 'models/' directory.")
