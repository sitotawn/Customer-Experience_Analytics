import pandas as pd
import os
from src.nlp_analysis.sentiment_analyzer import SentimentAnalyzer
from src.nlp_analysis.thematic_analyzer import ThematicAnalyzer
from src.data_processing.preprocess import preprocess_text # For fallback if clean_review_text is missing

def run_analysis_pipeline():
    """
    Runs the full NLP analysis pipeline: loads data, performs sentiment,
    thematic analysis, and saves the results to a new CSV.
    """
    print("--- Starting NLP Analysis Pipeline ---")

    # Define paths
    PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
    cleaned_csv_path = os.path.join(PROCESSED_DATA_DIR, 'all_cleaned_reviews.csv')
    output_analyzed_csv_path = os.path.join(PROCESSED_DATA_DIR, 'analyzed_reviews_with_themes_sentiment.csv')

    # 1. Load Cleaned Data
    if not os.path.exists(cleaned_csv_path):
        print(f"Error: Cleaned data CSV not found at {cleaned_csv_path}")
        print("Please run Task 1 (data collection and preprocessing) first.")
        return pd.DataFrame()

    print(f"Loading cleaned reviews from: {cleaned_csv_path}")
    df_reviews = pd.read_csv(cleaned_csv_path)
    print(f"Loaded {len(df_reviews)} reviews.")

    # Ensure 'clean_review_text' is available; create if not (should be from preprocessing)
    if 'clean_review_text' not in df_reviews.columns:
        print("Warning: 'clean_review_text' column not found. Running basic text cleaning on 'review_text' for analysis.")
        df_reviews['clean_review_text'] = df_reviews['review_text'].apply(preprocess_text)
    else:
        print("'clean_review_text' column found. Proceeding with analysis.")

    # 2. Perform Sentiment Analysis
    sentiment_analyzer = SentimentAnalyzer()
    if sentiment_analyzer.classifier: # Only proceed if model loaded successfully
        df_analyzed = sentiment_analyzer.add_sentiment_to_dataframe(df_reviews.copy(), text_column='clean_review_text')
        print(f"Sentiment analysis completed for {len(df_analyzed)} reviews.")
    else:
        print("Sentiment analysis skipped due to model loading error.")
        df_analyzed = df_reviews.copy() # Continue with original df if sentiment fails
        df_analyzed['sentiment_label'] = 'N/A'
        df_analyzed['sentiment_score'] = 0.0

    # 3. Perform Thematic Analysis
    thematic_analyzer = ThematicAnalyzer()
    # Define your custom theme rules here, based on your exploration from thematic_analyzer.py runs
    custom_theme_rules = {
        'Account Access Issues': ['login', 'log in', 'password', 'username', 'pin', 'otp', 'access', 'blocked', 'account blocked', 'reset password', 'verify'],
        'Transaction Performance': ['transfer', 'send money', 'payment', 'slow', 'fast', 'transaction', 'failed', 'delay', 'complete', 'speed', 'otp delay', 'hang', 'stuck'],
        'User Interface & Experience': ['ui', 'interface', 'design', 'easy', 'user friendly', 'difficult', 'navigate', 'layout', 'dashboard', 'looks', 'smooth', 'intuitive', 'clunky', 'confusing'],
        'Customer Support': ['support', 'customer service', 'help', 'response', 'contact', 'call center', 'agent', 'hotline', 'phone', 'feedback', 'representative'],
        'Feature Requests': ['fingerprint', 'face id', 'biometric', 'dark mode', 'new feature', 'add feature', 'update', 'qr code', 'payment options', 'notification', 'ussd'],
        'App Stability & Bugs': ['crash', 'bug', 'glitch', 'error', 'force close', 'freezes', 'unstable', 'fix app', 'malfunction', 'broken'],
        'Security & Authentication': ['security', 'authenticate', 'secure', 'fraud', 'data protection'] # Added based on common banking app concerns
    }
    # Add themes. Ensure 'clean_review_text' is used for theme assignment.
    df_final = thematic_analyzer.add_themes_to_dataframe(df_analyzed, text_column='clean_review_text', theme_rules=custom_theme_rules)
    print(f"Thematic analysis completed for {len(df_final)} reviews.")

    # 4. Save Results
    print(f"Saving final analyzed reviews to: {output_analyzed_csv_path}")
    # Ensure unique ID for each row if not already present from scraping (reviewId from google-play-scraper)
    # We'll use pandas default index as review_id for simplicity if original not there,
    # but ideally the 'reviewId' from google-play-scraper should be kept and used.
    if 'reviewId' not in df_final.columns:
         df_final['review_id'] = df_final.index
    else:
         df_final = df_final.rename(columns={'reviewId': 'review_id'}) # Standardize name

    # Select and reorder columns as per KPI: review_id, review_text, sentiment_label, sentiment_score, identified_theme(s)
    final_output_columns = [
        'review_id', 'review_text', 'sentiment_label', 'sentiment_score', 'identified_theme(s)',
        'rating', 'date', 'bank', 'source' # Keep other useful columns for reporting
    ]

    # Ensure all required columns are present, adding placeholders if necessary (shouldn't be needed if logic is correct)
    for col in final_output_columns:
        if col not in df_final.columns:
            print(f"Warning: Missing expected column '{col}' in final DataFrame. Adding as empty/default.")
            if col == 'sentiment_score': df_final[col] = 0.0
            elif col == 'identified_theme(s)': df_final[col] = [[]] * len(df_final)
            else: df_final[col] = 'N/A'


    df_final[final_output_columns].to_csv(output_analyzed_csv_path, index=False, encoding='utf-8')
    print("--- NLP Analysis Pipeline Completed ---")
    return df_final

if __name__ == "__main__":
    analyzed_df = run_analysis_pipeline()
    if not analyzed_df.empty:
        print("\nSample of final analyzed data:")
        print(analyzed_df[['review_id', 'review_text', 'sentiment_label', 'sentiment_score', 'identified_theme(s)']].head())

        # --- KPI Checks for Task 2 ---
        print("\n--- Task 2 KPI Checks ---")

        # KPI 1: Sentiment scores for 90%+ reviews.
        sentiment_coverage = (analyzed_df['sentiment_label'] != 'N/A').mean() * 100
        print(f"Sentiment coverage: {sentiment_coverage:.2f}% (Target: 90%+) ")
        if sentiment_coverage >= 90:
            print("  KPI ACHIEVED: High sentiment coverage.")
        else:
            print("  KPI WARNING: Sentiment coverage is below 90%. Investigate why some sentiments might be missing.")

        # KPI 2: 3+ themes per bank with examples.
        print("\nThemes identified per bank:")
        unique_themes_per_bank = {}
        for bank in analyzed_df['bank'].unique():
            bank_themes = set()
            # Flatten the list of lists in 'identified_theme(s)' column
            # Handle potential string representation of list (if loaded from CSV without proper parsing)
            themes_data = analyzed_df[analyzed_df['bank'] == bank]['identified_theme(s)'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x
            )
            for themes_list in themes_data:
                if isinstance(themes_list, list):
                    for theme in themes_list:
                        bank_themes.add(theme)
            if 'Other' in bank_themes and len(bank_themes) > 1: # 'Other' is a fallback, not a distinct theme for KPI count
                bank_themes.remove('Other') 
            unique_themes_per_bank[bank] = list(bank_themes)
            print(f"- {bank}: {len(unique_themes_per_bank[bank])} unique themes: {unique_themes_per_bank[bank]}")
            if len(unique_themes_per_bank[bank]) >= 3:
                print("    KPI ACHIEVED: 3+ themes identified.")
            else:
                print("    KPI WARNING: Fewer than 3 themes identified. Refine thematic rules.")

        # KPI 3: Modular pipeline code. (Checked by structure, not runtime)
        print("\nKPI: Modular pipeline code. (Assessed by reviewing src/nlp_analysis structure and script calls)")
        print("  This KPI is met through the organization of 'sentiment_analyzer.py' and 'thematic_analyzer.py' modules.")

    else:
        print("Pipeline did not produce any analyzed data.")