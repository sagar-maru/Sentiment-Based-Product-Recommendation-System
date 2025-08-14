# app.py
from flask import Flask, render_template, request
import os

from model import SentimentRecommender

app = Flask(__name__, template_folder="templates", static_folder="static")

# If your models folder is elsewhere, pass root_path below:
MODEL_ROOT = os.environ.get("MODEL_ROOT", "models")

# Create the recommender instance once at startup
try:
    recommender = SentimentRecommender(root_path=MODEL_ROOT)
except Exception as e:
    # If models fail to load, keep the app running but show error on the page
    recommender = None
    model_load_error = str(e)
else:
    model_load_error = None

@app.route("/", methods=["GET"])
def index():
    # Render empty form
    return render_template("index.html", error=model_load_error, recs=None, sentiment=None, user_id=None)

@app.route("/recommend", methods=["POST"])
def recommend():
    if model_load_error:
        return render_template("index.html", error=f"Model load error: {model_load_error}", recs=None, sentiment=None, user_id=None)

    user_id = request.form.get("user_id")
    if user_id is None:
        return render_template("index.html", error="Missing user_id in form", recs=None, sentiment=None, user_id=None)

    try:
        recs = recommender.get_sentiment_filtered_recommendations(user_id)
    except Exception as e:
        return render_template("index.html", error=f"Internal error: {e}", recs=None, sentiment=None, user_id=user_id)

    if recs is None:
        # user not found
        return render_template("index.html", error=f"User '{user_id}' not found. Please enter a valid user id.", recs=None, sentiment=None, user_id=user_id)

    if not recs:
        return render_template("index.html", error=f"No recommendations / reviews available for user '{user_id}'.", recs=None, sentiment=None, user_id=user_id)

    # recs is already a list of dicts: [{'name':..., 'brand':..., 'pos_sentiment_percent': ...}, ...]
    # But the index.html expects 'pos_sentiment_percent' as numeric and will render it with a '%' sign in template
    # ensure float formatting
    for r in recs:
        # ensure key name matches index.html: pos_sentiment_percent
        if 'pos_sentiment_percent' not in r and 'pos_sentiment_percent' in r:
            r['pos_sentiment_percent'] = r.get('pos_sentiment_percent', 0.0)

    return render_template("index.html", error=None, recs=recs, sentiment=None, user_id=user_id)


@app.route("/predict_sentiment", methods=["POST"])
def predict_sentiment():
    if model_load_error:
        return render_template("index.html", error=f"Model load error: {model_load_error}", recs=None, sentiment=None, user_id=None)

    review_text = request.form.get("review_text", "")
    if not review_text.strip():
        return render_template("index.html", error="Please provide some review text to predict.", recs=None, sentiment=None, user_id=None)

    try:
        label = recommender.predict_sentiment(review_text)
    except Exception as e:
        return render_template("index.html", error=f"Prediction error: {e}", recs=None, sentiment=None, user_id=None)

    return render_template("index.html", error=None, recs=None, sentiment=label, user_id=None)


if __name__ == "__main__":
    # debug True for development; set False in production
    app.run(host="0.0.0.0", port=5000, debug=True)