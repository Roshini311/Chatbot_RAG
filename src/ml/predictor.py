import os
import pickle
import numpy as np
from config.settings import settings
from src.ml.dataset_prep import CATEGORIES

class DocumentClassifierPredictor:
    """
    Predicts document technical domain using trained ML/DL model.
    """
    def __init__(self):
        self.model = None
        self.tokenizer_data = None
        self.backend = None
        self._load_model()

    def _load_model(self):
        token_path = settings.TOKENIZER_PATH
        if not os.path.exists(token_path):
            from src.ml.train_classifier import train_and_save_classifier
            train_and_save_classifier()

        try:
            with open(token_path, "rb") as f:
                self.tokenizer_data = pickle.load(f)

            self.backend = self.tokenizer_data.get("backend", "tf")

            if self.backend == "tf":
                import tensorflow as tf
                tf_model_path = settings.MODEL_PATH
                if not tf_model_path.endswith(".h5") and not tf_model_path.endswith(".keras"):
                    tf_model_path += ".h5"
                self.model = tf.keras.models.load_model(tf_model_path)
            else:
                model_path = settings.MODEL_PATH.replace(".h5", ".pkl")
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load trained ML classifier ({e}). Defaulting to rule-based fallback.")
            self.model = None

    def predict_category(self, text: str) -> str:
        if not text or len(text.strip()) == 0:
            return "General / Uncategorized"

        sample_text = text[:1500].lower()

        # Rule-based fallback check first if model loading failed
        if self.model is None:
            if any(k in sample_text for k in ["security", "cipher", "attack", "malware", "auth", "firewall"]):
                return "Cyber Security"
            elif any(k in sample_text for k in ["neural", "learning", "model", "prompt", "transformer", "agent"]):
                return "Artificial Intelligence"
            elif any(k in sample_text for k in ["cloud", "aws", "azure", "docker", "kubernetes", "devops"]):
                return "Cloud Computing"
            elif any(k in sample_text for k in ["robot", "actuator", "ros", "motor", "kinematics", "slam"]):
                return "Robotics"
            elif any(k in sample_text for k in ["pandas", "dataframe", "eda", "statistics", "data science", "etl"]):
                return "Data Science"
            return "General Knowledge"

        try:
            if self.backend == "tf":
                preds = self.model.predict(np.array([text[:1000]]), verbose=0)
                class_idx = int(np.argmax(preds[0]))
                return CATEGORIES[class_idx]
            else:
                vectorizer = self.tokenizer_data["vectorizer"]
                X = vectorizer.transform([text[:1000]])
                class_idx = int(self.model.predict(X)[0])
                return CATEGORIES[class_idx]
        except Exception:
            return "Artificial Intelligence"
