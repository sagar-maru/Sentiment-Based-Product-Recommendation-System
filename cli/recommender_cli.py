import argparse
import sys
import os

# Add root path so Python can find 'model.py'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import SentimentRecommender  # Your main recommendation class


def main():
    parser = argparse.ArgumentParser(description="🎯 Sentiment-aware Product Recommender CLI")
    parser.add_argument(
        "-m", "--models",
        default="models",
        help="Path to the models directory (default: ./models)"
    )
    parser.add_argument(
        "-u", "--user",
        default=None,
        help="User ID to get recommendations for (optional). If omitted, you'll be prompted."
    )
    parser.add_argument(
        "-s", "--sentiment",
        default=None,
        help="Enter a review text to predict sentiment (optional)."
    )
    args = parser.parse_args()

    try:
        recommender = SentimentRecommender(root_path=args.models)
    except Exception as e:
        print(f"❌ [ERROR] Failed to load models from '{args.models}': {e}")
        return

    # If sentiment prediction mode
    if args.sentiment:
        try:
            prediction = recommender.predict_sentiment(args.sentiment)
            print(f"💬 Review: {args.sentiment}")
            print(f"🔮 Predicted Sentiment: {prediction}")
        except Exception as e:
            print(f"❌ Error predicting sentiment: {e}")
        return

    # If recommendation mode
    user = args.user or input("Enter user id: ").strip()
    try:
        recs = recommender.get_sentiment_filtered_recommendations(user)
    except Exception as e:
        print(f"❌ [ERROR] Internal error while getting recommendations: {e}")
        return

    if recs is None:
        print(f"⚠️ No such user '{user}' in the recommendation model.")
    elif not recs:
        print(f"ℹ️ No recommendations available for user '{user}'.")
    else:
        print(f"\n🏆 Top recommendations for {user}:\n")
        for i, r in enumerate(recs, 1):
            print(f"{i}. {r['name']} — 👍 {r['pos_sentiment_percent']:.2f}% positive")
        print("")


if __name__ == "__main__":
    main()
