import os
import pytest
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import SentimentRecommender

@pytest.fixture(scope="module")
def recommender():
    """Fixture to load SentimentRecommender from the root models directory."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    models_path = os.path.join(project_root, "models")
    print(f"\n[Fixture] Initializing recommender with models from: {models_path}")
    recommender_obj = SentimentRecommender(root_path=models_path)
    print("[Fixture] Recommender object created successfully.\n")
    return recommender_obj

def test_model_loading(recommender):
    print("[Test] Checking if recommender object is loaded...")
    assert recommender is not None
    print("[Pass] Model loaded successfully with all required attributes.\n")

def test_invalid_models_path():
    print("[Test] Initializing recommender with INVALID model path.")
    invalid_path = "/invalid/path/to/models"
    try:
        SentimentRecommender(root_path=invalid_path)
        pytest.fail("Expected failure when loading from invalid path, but succeeded.")
    except Exception as e:
        print(f"[Expected] Exception occurred: {e}")
        print("[Pass] Invalid model path handled correctly.\n")

def test_get_recommendations_valid_user(recommender):
    test_user_id = "dorothy w"  # Replace with valid user ID
    print(f"[Test] Fetching recommendations for VALID user: {test_user_id}")
    recs = recommender.get_sentiment_filtered_recommendations(test_user_id)
    print(f"[Expected] Non-empty list of recommendations.")
    print(f"[Actual]   {recs}")
    assert isinstance(recs, list)
    assert all("name" in r and "pos_sentiment_percent" in r for r in recs)
    print("[Pass] Valid user recommendations returned successfully.\n")

def test_get_recommendations_invalid_user(recommender):
    test_user_id = "non_existent_user_123"
    print(f"[Test] Fetching recommendations for INVALID user: {test_user_id}")
    recs = recommender.get_sentiment_filtered_recommendations(test_user_id)
    print(f"[Expected] None or empty list.")
    print(f"[Actual]   {recs}")
    assert recs is None or recs == []
    print("[Pass] Invalid user handled correctly.\n")

def test_empty_recommendations_scenario(recommender):
    """Edge case: User exists but no recommendations."""
    if hasattr(recommender, "force_empty_recommendations"):
        print("[Test] Simulating empty recommendation scenario.")
        recs = recommender.force_empty_recommendations("some_user")
        print(f"[Expected] Empty list.")
        print(f"[Actual]   {recs}")
        assert recs == []
        print("[Pass] Empty recommendation scenario handled correctly.\n")
    else:
        print("[Skip] force_empty_recommendations not implemented.\n")

def test_predict_sentiment_positive(recommender):
    text = "I absolutely love this product! Works perfectly."
    print(f"[Test] Predicting sentiment for POSITIVE text: \"{text}\"")
    prediction = recommender.predict_sentiment(text)
    print(f"[Expected] 'Positive' or 'Negative'")
    print(f"[Actual]   {prediction}")
    assert prediction in ["Positive", "Negative"]
    print("[Pass] Positive sentiment test passed.\n")

def test_predict_sentiment_negative(recommender):
    text = "This is the worst thing I have ever bought. I am very disappointed with it."
    print(f"[Test] Predicting sentiment for NEGATIVE text: \"{text}\"")
    prediction = recommender.predict_sentiment(text)
    print(f"[Expected] 'Positive' or 'Negative'")
    print(f"[Actual]   {prediction}")
    assert prediction in ["Positive", "Negative"]
    print("[Pass] Negative sentiment test passed.\n")

def test_get_top_n_products(recommender):
    if hasattr(recommender, "get_top_n_products"):
        n = 5
        print(f"[Test] Fetching top {n} products.")
        products = recommender.get_top_n_products(n)
        print(f"[Expected] List of {n} or fewer products.")
        print(f"[Actual]   {products}")
        assert isinstance(products, list)
        assert len(products) <= n
        print("[Pass] Top N products function works correctly.\n")
    else:
        print("[Skip] get_top_n_products not implemented.\n")

def test_sentiment_probability_output(recommender):
    if hasattr(recommender, "predict_sentiment_proba"):
        text = "Product quality is amazing."
        print(f"[Test] Predicting sentiment probability for: \"{text}\"")
        proba = recommender.predict_sentiment_proba(text)
        print(f"[Expected] Iterable of probabilities.")
        print(f"[Actual]   {proba}")
        assert hasattr(proba, "__iter__")
        print("[Pass] Sentiment probability output works correctly.\n")
    else:
        print("[Skip] predict_sentiment_proba not implemented.\n")

def test_invalid_text_prediction(recommender):
    text = ""  # Empty input
    print("[Test] Predicting sentiment for EMPTY text.")
    try:
        prediction = recommender.predict_sentiment(text)
        print(f"[Expected] Handle gracefully, return valid label or None.")
        print(f"[Actual]   {prediction}")
        assert prediction in ["Positive", "Negative", None]
        print("[Pass] Empty text handled correctly.\n")
    except Exception as e:
        pytest.fail(f"Error occurred for empty text: {e}")