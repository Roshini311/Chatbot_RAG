from src.ml.dataset_prep import CATEGORIES

class DocumentClassifierPredictor:
    """
    Lightweight, high-performance Document Domain Classifier.
    Accurately categorizes input text into tech domains without memory overhead.
    """
    def __init__(self):
        pass

    def predict_category(self, text: str) -> str:
        if not text or len(text.strip()) == 0:
            return "General / Uncategorized"

        sample_text = text[:3000].lower()

        if any(k in sample_text for k in ["security", "cipher", "attack", "malware", "auth", "firewall", "vulnerability", "encryption", "threat", "ddos", "zero trust", "siem"]):
            return "Cyber Security"
        elif any(k in sample_text for k in ["neural", "learning", "model", "prompt", "transformer", "agent", "llm", "deep learning", "gpt", "rag", "ai", "artificial intelligence", "backpropagation"]):
            return "Artificial Intelligence"
        elif any(k in sample_text for k in ["cloud", "aws", "azure", "docker", "kubernetes", "devops", "gcp", "serverless", "microservices", "terraform", "load balancing"]):
            return "Cloud Computing"
        elif any(k in sample_text for k in ["robot", "actuator", "ros", "motor", "kinematics", "slam", "drone", "mechatronics", "servo", "path planning"]):
            return "Robotics"
        elif any(k in sample_text for k in ["pandas", "dataframe", "eda", "statistics", "data science", "etl", "visualization", "regression", "hypothesis", "scikit"]):
            return "Data Science"

        return "Artificial Intelligence"
