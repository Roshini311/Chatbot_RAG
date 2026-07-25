from src.ml.predictor import DocumentClassifierPredictor

def test_predictor_basic():
    predictor = DocumentClassifierPredictor()
    cat = predictor.predict_category("Deep learning neural networks and transformer architecture models for natural language processing.")
    assert isinstance(cat, str)
    assert len(cat) > 0
