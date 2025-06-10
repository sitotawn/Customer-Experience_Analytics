from transformers import pipeline
import pandas as pd
import os

class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initializes the sentiment analyzer using a pre-trained Hugging Face model.
        Args:
            model_name (str): The name of the pre-trained model to use.
                              Recommended: "distilbert-base-uncased-finetuned-sst-2-english"
                              Alternatively for simpler lexicon-based: "cardiffnlp/twitter-roberta-base-sentiment"
        """
        print(f"Loading sentiment analysis model: {model_name}...")
        try:
            # Initialize the Hugging Face sentiment-analysis pipeline
            self.classifier = pipeline("sentiment-analysis", model=model_name)
            print(f"Model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            print("Ensure you have an internet connection and the model name is correct.")
            print("You might also need to install PyTorch (`pip install torch`) or TensorFlow (`pip install tensorflow`).")
            self.classifier = None # Set to None to indicate failure

    def get_sentiment(self, text):
        """
        Computes the sentiment for a given text.
        Returns: A dictionary with 'label' (POSITIVE, NEGATIVE, NEUTRAL) and 'score'.
                 Returns None if the classifier is not loaded or text is invalid.
        """
        if self.classifier is None or not isinstance(text, str) or not text.strip():
            return {'label': 'NEUTRAL', 'score': 0.5} # Default for invalid/empty text
        try:
            # The classifier returns a list of dictionaries, e.g., [{'label': 'POSITIVE', 'score': 0.99}]
            result = self.classifier(text)[0]
            return result
        except Exception as e:
            print(f"Error computing sentiment for text: '{text[:50]}...': {e}")
            return {'label': 'ERROR', 'score': 0.0} # Indicate an error

    def add_sentiment_to_dataframe(self, df, text_column='review_text'):
        """
        Adds sentiment labels and scores to a DataFrame.
        Args:
            df (pd.DataFrame): The input DataFrame containing review texts.
            text_column (str): The name of the column containing the review text.
        Returns:
            pd.DataFrame: The DataFrame with 'sentiment_label' and 'sentiment_score' columns added.
        """
        if self.classifier is None:
            print("Sentiment analyzer not loaded. Cannot add sentiment to DataFrame.")
            df['sentiment_label'] = 'ERROR'
            df['sentiment_score'] = 0.0
            return df

        print(f"Adding sentiment to DataFrame using column: '{text_column}'...")

        # Ensure the text column exists and handle potential NaNs before processing
        if text_column not in df.columns:
            print(f"Error: Text column '{text_column}' not found in DataFrame.")
            df['sentiment_label'] = 'ERROR'
            df['sentiment_score'] = 0.0
            return df

        # Fill NaN values in the text column to avoid errors during sentiment analysis
        df[text_column] = df[text_column].fillna('')

        # Apply sentiment analysis using a generator for efficiency with large datasets
        # The pipeline can take a list of strings directly
        texts_to_analyze = df[text_column].tolist()

        # Handle cases where `texts_to_analyze` might be empty
        if not texts_to_analyze:
            print("No texts to analyze for sentiment.")
            df['sentiment_label'] = []
            df['sentiment_score'] = []
            return df

        # Batch processing for efficiency, if needed, can be implemented here,
        # but the pipeline often handles it internally.
        results = self.classifier(texts_to_analyze)

        sentiment_labels = [r['label'] for r in results]
        sentiment_scores = [r['score'] for r in results]

        df['sentiment_label'] = sentiment_labels
        df['sentiment_score'] = sentiment_scores
        print("Sentiment analysis completed.")
        return df

if __name__ == "__main__":
    # Example Usage:
    # Assuming you have an 'all_cleaned_reviews.csv' file in data/processed/
    PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    cleaned_csv_path = os.path.join(PROCESSED_DATA_DIR, 'all_cleaned_reviews.csv')

    if not os.path.exists(cleaned_csv_path):
        print(f"Error: Cleaned data CSV not found at {cleaned_csv_path}")
        print("Please run Task 1 (data collection and preprocessing) first.")
    else:
        print(f"Loading cleaned reviews from: {cleaned_csv_path}")
        df_reviews = pd.read_csv(cleaned_csv_path)
        print(f"Loaded {len(df_reviews)} reviews.")

        analyzer = SentimentAnalyzer()
        if analyzer.classifier: # Only proceed if model loaded successfully
            df_analyzed = analyzer.add_sentiment_to_dataframe(df_reviews, text_column='review_text')
            print("\nSample of reviews with sentiment:")
            print(df_analyzed[['review_text', 'sentiment_label', 'sentiment_score']].head())

            # Optional: Aggregate by bank and rating for quick check
            print("\nAverage sentiment score by Bank and Rating:")
            # Ensure 'rating' column is numeric if it's not already
            df_analyzed['rating'] = pd.to_numeric(df_analyzed['rating'], errors='coerce')
            df_agg = df_analyzed.groupby(['bank', 'rating'])['sentiment_score'].mean().unstack()
            print(df_agg)

            # Example of saving results (this will be done in the main pipeline script later)
            # output_path = os.path.join(PROCESSED_DATA_DIR, 'reviews_with_sentiment.csv')
            # df_analyzed.to_csv(output_path, index=False, encoding='utf-8')
            # print(f"\nSentiment analysis results saved to {output_path}")
        else:
            print("Sentiment analysis model failed to load. Skipping analysis.")