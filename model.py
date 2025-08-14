# model.py
import os
import pickle
import pandas as pd
import numpy as np

class SentimentRecommender:
    """
    Loads required models and data from disk and provides:
      - get_sentiment_filtered_recommendations(user_id)
      - predict_sentiment(review_text)
    """
    root_model_path = "models"
    sentiment_model_fname = "top_sentiment_classifier_model.pkl"
    tfidf_vectorizer_fname = "tfidf_vectorizer.pkl"
    best_recommender_fname = "final_recommendation_model.pkl"
    clean_dataframe_fname = "cleansed_data.pkl"

    def __init__(self, root_path: str = None):
        if root_path:
            self.root_model_path = root_path

        # helper to build path
        def p(fname): return os.path.join(self.root_model_path, fname)

        # Load models and data with fallbacks and clear error messages
        try:
            with open(p(self.sentiment_model_fname), "rb") as f:
                self.sentiment_model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load sentiment model: {p(self.sentiment_model_fname)}: {e}")

        try:
            # tfidf_vectorizer may have been saved via pandas or pickle; try both
            try:
                self.tfidf_vectorizer = pd.read_pickle(p(self.tfidf_vectorizer_fname))
            except Exception:
                with open(p(self.tfidf_vectorizer_fname), "rb") as f:
                    self.tfidf_vectorizer = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load tfidf vectorizer: {p(self.tfidf_vectorizer_fname)}: {e}")

        try:
            with open(p(self.best_recommender_fname), "rb") as f:
                self.user_final_rating = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load recommender matrix/model: {p(self.best_recommender_fname)}: {e}")

        try:
            with open(p(self.clean_dataframe_fname), "rb") as f:
                self.cleaned_data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load cleaned dataframe: {p(self.clean_dataframe_fname)}: {e}")

        # Ensure expected columns exist in cleaned_data
        if not isinstance(self.cleaned_data, pd.DataFrame):
            raise RuntimeError("cleansed_data.pkl did not load into a pandas DataFrame.")

        # Commonly used column names (adjust if your data uses different names)
        expected_cols = {"id", "name", "reviews_lemmatized"}
        missing = expected_cols.difference(set(self.cleaned_data.columns))
        if missing:
            # don't break here; just warn
            print(f"Warning: expected columns missing from cleansed_data: {missing}")

    def _resolve_user_index(self, user_id):
        """
        Attempt to find the user in user_final_rating index.
        Accepts integer-like or string user_id values.
        Returns the exact index key if found, else raises KeyError.
        """
        idx = self.user_final_rating.index

        # try exact match first
        if user_id in idx:
            return user_id

        # try conversions: int -> str, str -> int (if possible)
        try:
            # if user_id is str of integer numbers, try int
            if isinstance(user_id, str) and user_id.isdigit():
                candidate = int(user_id)
                if candidate in idx:
                    return candidate
        except Exception:
            pass

        try:
            # if index contains string keys but user provided int
            if isinstance(user_id, (int, np.integer)):
                candidate = str(user_id)
                if candidate in idx:
                    return candidate
        except Exception:
            pass

        # Last resort: try matching by str equivalence for all index entries
        str_map = {str(x): x for x in idx}
        if str(user_id) in str_map:
            return str_map[str(user_id)]

        raise KeyError(f"user_id '{user_id}' not found in recommender index")

    def get_sentiment_filtered_recommendations(self, user_id, top_k=5, top_n_for_sentiment=20):
        """
        Returns top_k recommended products for the user filtered/ranked by positive sentiment percent.
        Output: list of dicts: [{'name':..., 'pos_sentiment_percent': ...}, ...]
        """
        try:
            user_key = self._resolve_user_index(user_id)
        except KeyError:
            return None  # caller (app) will handle error message

        # Extract top N product ids for this user from recommender
        try:
            user_row = self.user_final_rating.loc[user_key]
        except Exception as e:
            # If the recommender is e.g. a sparse matrix or unexpected type
            raise RuntimeError(f"Unexpected format for user_final_rating: {e}")

        # Ensure it's a Series
        if isinstance(user_row, pd.DataFrame):
            # take first row if happens
            user_row = user_row.iloc[0]

        # sort and pick top_n_for_sentiment
        try:
            top_products = list(user_row.sort_values(ascending=False).head(top_n_for_sentiment).index)
        except Exception:
            # fallback: try numpy argsort
            vals = np.asarray(user_row)
            cols = list(user_row.index)
            order = np.argsort(-vals)[:top_n_for_sentiment]
            top_products = [cols[i] for i in order]

        if not top_products:
            return []

        # Filter cleaned_data rows belonging to the top_products
        # assume cleaned_data['id'] contains same ids as recommender columns
        df_top = self.cleaned_data[self.cleaned_data['id'].isin(top_products)].copy()

        if df_top.empty:
            # no reviews available for these products
            return []

        # Prepare reviews for prediction (ensure string type)
        reviews = df_top['reviews_lemmatized'].astype(str).values

        # Transform via tfidf and predict
        try:
            X = self.tfidf_vectorizer.transform(reviews)
        except Exception as e:
            raise RuntimeError(f"Failed to transform reviews with TF-IDF vectorizer: {e}")

        try:
            preds = self.sentiment_model.predict(X)
        except Exception as e:
            raise RuntimeError(f"Failed to predict sentiment: {e}")

        df_top = df_top.assign(predicted_sentiment=preds)
        df_top['positive_sentiment'] = df_top['predicted_sentiment'].apply(lambda x: 1 if str(x).lower().startswith("pos") else 0)

        # Aggregate counts by product name
        agg = df_top.groupby('name').agg(
            pos_sent_count=('positive_sentiment', 'sum'),
            total_sent_count=('predicted_sentiment', 'count')
        ).reset_index()

        # Avoid division by zero
        agg['pos_sent_count'] = agg['pos_sent_count'].fillna(0)
        agg['total_sent_count'] = agg['total_sent_count'].replace(0, np.nan)
        agg['pos_sent_percentage'] = np.round((agg['pos_sent_count'] / agg['total_sent_count']) * 100, 2)
        agg['pos_sent_percentage'] = agg['pos_sent_percentage'].fillna(0.0)

        # Sort by percentage and return top_k
        topk = agg.sort_values(by='pos_sent_percentage', ascending=False).head(top_k)

        # Build output list of dicts
        result = []
        for _, row in topk.iterrows():
            result.append({
                'name': row['name'],
                'pos_sentiment_percent': float(np.round(row['pos_sent_percentage'], 2))
            })

        return result

    def predict_sentiment(self, review_text: str):
        """
        Predict sentiment label for a single review text.
        Returns predicted label (string).
        """
        if not review_text or not str(review_text).strip():
            return None

        try:
            X = self.tfidf_vectorizer.transform([str(review_text)])
            pred = self.sentiment_model.predict(X)
            return pred[0]
        except Exception as e:
            raise RuntimeError(f"Failed to predict sentiment for provided text: {e}")
