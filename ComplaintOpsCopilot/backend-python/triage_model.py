import joblib
import os
import pandas as pd

class TriageEngine:
    def __init__(self):
        self.category_model = None
        self.urgency_model = None
        self._load_models()

    def _load_models(self):
        try:
            # Check if models exist
            if os.path.exists("models/category_model.pkl") and os.path.exists("models/urgency_model.pkl"):
                self.category_model = joblib.load("models/category_model.pkl")
                self.urgency_model = joblib.load("models/urgency_model.pkl")
            else:
                print("Models not found. Please run train_triage_model.py first.")
        except Exception as e:
            print(f"Error loading models: {e}")

    def predict(self, text: str):
        if not self.category_model or not self.urgency_model:
            return {
                "category": "UNKNOWN",
                "category_confidence": 0.0,
                "urgency": "LOW",
                "urgency_confidence": 0.0
            }

        # Predict Category
        cat_pred = self.category_model.predict([text])[0]
        cat_probs = self.category_model.predict_proba([text])[0]
        cat_conf = max(cat_probs)

        # Predict Urgency
        urg_pred = self.urgency_model.predict([text])[0]
        urg_probs = self.urgency_model.predict_proba([text])[0]
        urg_conf = max(urg_probs)

        return {
            "category": cat_pred,
            "category_confidence": float(cat_conf),
            "urgency": urg_pred,
            "urgency_confidence": float(urg_conf)
        }

triage_engine = TriageEngine()
