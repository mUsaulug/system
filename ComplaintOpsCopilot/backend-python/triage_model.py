import joblib
import logging
import os
import json

class TriageEngine:
    def __init__(self):
        self.category_model = None
        self.urgency_model = None
        self.model_loaded = False
        self.logger = logging.getLogger("complaintops.triage_model")
        self._load_models()

    def _load_models(self):
        try:
            metadata_path = os.path.join("models", "latest.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                category_path = metadata.get("category_model_path")
                urgency_path = metadata.get("urgency_model_path")
                if category_path and urgency_path:
                    self.category_model = joblib.load(category_path)
                    self.urgency_model = joblib.load(urgency_path)
            elif os.path.exists("models/category_model.pkl") and os.path.exists("models/urgency_model.pkl"):
                self.category_model = joblib.load("models/category_model.pkl")
                self.urgency_model = joblib.load("models/urgency_model.pkl")
            else:
                self.logger.warning("Models not found. Please run train_triage_model.py first.")
        except Exception as e:
            self.logger.error("Error loading models: %s", e)

        self.model_loaded = bool(self.category_model and self.urgency_model)

    def predict(self, text: str):
        if not self.model_loaded:
            return {
                "category": "UNKNOWN",
                "category_confidence": 0.0,
                "urgency": "LOW",
                "urgency_confidence": 0.0,
                "model_loaded": False,
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
            "urgency_confidence": float(urg_conf),
            "model_loaded": True,
        }

triage_engine = TriageEngine()
