# src/nlp_analysis/thematic_analyzer.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from collections import Counter
import os

# Load spaCy model once when the module is imported
# ENSURE 'en_core_web_sm' is downloaded by running: python -m spacy download en_core_web_sm
try:
    NLP_MODEL = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' loaded for thematic analysis.")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("Please ensure you have downloaded it: 'python -m spacy download en_core_web_sm'")
    NLP_MODEL = None # Set to None to handle cases where model isn't available


class ThematicAnalyzer:
    def __init__(self):
        pass # No complex initialization needed for this class itself

    def extract_keywords_tfidf(self, df, text_column='clean_review_text', top_n=10):
        """
        Extracts top N TF-IDF keywords (single words and bigrams) for each bank.
        TF-IDF identifies words/phrases that are important to a document in a corpus.
        Args:
            df (pd.DataFrame): DataFrame with cleaned review texts.
            text_column (str): Column containing cleaned review text.
            top_n (int): Number of top keywords to extract per bank.
        Returns:
            dict: A dictionary with bank names as keys and lists of top TF-IDF keywords as values.
        """
        if df.empty or text_column not in df.columns:
            print(f"DataFrame is empty or missing '{text_column}'. Cannot extract keywords.")
            return {}

        print(f"Extracting top {top_n} keywords per bank using TF-IDF...")
        bank_keywords = {}
        for bank in df['bank'].unique():
            bank_df = df[df['bank'] == bank]
            corpus = bank_df[text_column].tolist()

            if not corpus:
                bank_keywords[bank] = []
                continue

            # TfidfVectorizer parameters:
            # min_df: ignore terms that appear in less than 5 documents (to remove very rare words)
            # max_df: ignore terms that appear in more than 80% of documents (to remove very common words)
            # stop_words: remove common English stopwords (e.g., 'the', 'is', 'a')
            # ngram_range: (1,2) includes single words (unigrams) and two-word phrases (bigrams)
            vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, stop_words='english', ngram_range=(1,2))

            try:
                tfidf_matrix = vectorizer.fit_transform(corpus)
                feature_names = vectorizer.get_feature_names_out()

                # Sum TF-IDF scores for each word across all reviews for the bank
                sum_tfidf_scores = tfidf_matrix.sum(axis=0)

                # Create a DataFrame of words and their summed TF-IDF scores for sorting
                word_scores = pd.DataFrame({'word': feature_names, 'tfidf_score': sum_tfidf_scores.flat})
                word_scores = word_scores.sort_values(by='tfidf_score', ascending=False)

                bank_keywords[bank] = word_scores['word'].head(top_n).tolist()
                print(f"  {bank}: {bank_keywords[bank]}")
            except ValueError as e:
                print(f"  Warning: Could not extract TF-IDF for {bank}. Error: {e}. Not enough data or features after filtering?")
                bank_keywords[bank] = [] # Set to empty if error

        return bank_keywords

    def _get_noun_chunks(text):
        """Helper function to extract noun chunks (phrases) using spaCy."""
        if NLP_MODEL is None or not isinstance(text, str):
            return []
        doc = NLP_MODEL(text)
        return [chunk.text.lower() for chunk in doc.noun_chunks]

    def extract_key_phrases_spacy(self, df, text_column='clean_review_text', top_n=10):
        """
        Extracts common noun phrases/keywords using spaCy's noun chunker for each bank.
        This is good for identifying multi-word concepts like "customer service" or "loading time".
        Args:
            df (pd.DataFrame): DataFrame with cleaned review texts.
            text_column (str): Column containing cleaned review text.
            top_n (int): Number of top phrases/keywords to extract per bank.
        Returns:
            dict: A dictionary with bank names as keys and lists of top phrases as values.
        """
        if NLP_MODEL is None:
            print("spaCy model not loaded. Skipping key phrase extraction.")
            return {}
        if df.empty or text_column not in df.columns:
            print(f"DataFrame is empty or missing '{text_column}'. Cannot extract key phrases.")
            return {}

        print(f"Extracting top {top_n} key phrases per bank using spaCy noun chunks...")
        bank_phrases = {}
        for bank in df['bank'].unique():
            bank_df = df[df['bank'] == bank]
            corpus = bank_df[text_column].tolist()

            all_phrases = []
            for text in corpus:
                all_phrases.extend(self._get_noun_chunks(text))

            # Count frequency of phrases
            phrase_counts = Counter(all_phrases)
            # Filter out single-word phrases that might be less informative here, if desired
            # phrase_counts = {p: c for p, c in phrase_counts.items() if ' ' in p} 
            bank_phrases[bank] = [phrase for phrase, count in phrase_counts.most_common(top_n)]
            print(f"  {bank}: {bank_phrases[bank]}")
        return bank_phrases

    def assign_theme(self, text, theme_rules):
        """
        Assigns themes to a review based on predefined keyword rules.
        A review can have multiple themes. If no theme found, it gets "Other".
        Args:
            text (str): The cleaned review text.
            theme_rules (dict): A dictionary where keys are theme names and values are
                                lists of keywords/phrases associated with that theme.
        Returns:
            list: A list of themes identified in the review.
        """
        identified_themes = []
        text_lower = text.lower() # Ensure text is lowercase for consistent matching
        for theme, keywords in theme_rules.items():
            for keyword in keywords:
                if keyword in text_lower:
                    identified_themes.append(theme)
                    break # Move to next theme once one keyword is found for current theme
        return identified_themes if identified_themes else ["Other"] # Default to "Other" if no theme found

    def add_themes_to_dataframe(self, df, text_column='clean_review_text', theme_rules=None):
        """
        Adds a list of identified themes to each review in the DataFrame.
        Args:
            df (pd.DataFrame): The input DataFrame.
            text_column (str): Column containing cleaned review text.
            theme_rules (dict): Dictionary defining themes and their keywords.
                                If None, a default set of example rules will be used.
        Returns:
            pd.DataFrame: DataFrame with 'identified_theme(s)' column added.
        """
        if theme_rules is None:
            # Default example theme rules.
            # !!! IMPORTANT: YOU WILL CUSTOMIZE THESE RULES based on TF-IDF/spaCy outputs and manual review !!!
            theme_rules = {
                'Account Access Issues': ['login', 'log in', 'password', 'username', 'pin', 'otp', 'access', 'blocked', 'account blocked', 'reset password', 'verify'],
                'Transaction Performance': ['transfer', 'send money', 'payment', 'slow', 'fast', 'transaction', 'failed', 'delay', 'complete', 'speed', 'otp delay', 'hang', 'stuck'],
                'User Interface & Experience': ['ui', 'interface', 'design', 'easy', 'user friendly', 'difficult', 'navigate', 'layout', 'dashboard', 'looks', 'smooth', 'intuitive', 'clunky', 'confusing'],
                'Customer Support': ['support', 'customer service', 'help', 'response', 'contact', 'call center', 'agent', 'hotline', 'phone', 'feedback', 'representative'],
                'Feature Requests': ['fingerprint', 'face id', 'biometric', 'dark mode', 'new feature', 'add feature', 'update', 'qr code', 'payment options', 'notification', 'ussd'],
                'App Stability & Bugs': ['crash', 'bug', 'glitch', 'error', 'force close', 'freezes', 'unstable', 'fix app', 'malfunction', 'broken'],
                'Security & Authentication': ['security', 'authenticate', 'secure', 'fraud', 'data protection', 'scam'] # Added based on common banking app concerns
            }
            print("Using default theme rules. It is HIGHLY RECOMMENDED to customize these for better results.")

        print("Assigning themes to reviews based on defined rules...")
        df['identified_theme(s)'] = df[text_column].apply(lambda x: self.assign_theme(x, theme_rules))
        print("Theme assignment completed.")
        return df

if __name__ == "__main__":
    # Example Usage when thematic_analyzer.py is run directly
    # This block helps you test and refine your thematic analysis
    PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    cleaned_csv_path = os.path.join(PROCESSED_DATA_DIR, 'all_cleaned_reviews.csv')

    if not os.path.exists(cleaned_csv_path):
        print(f"Error: Cleaned data CSV not found at {cleaned_csv_path}")
        print("Please run Task 1 (data collection and preprocessing) first.")
    else:
        print(f"Loading cleaned reviews from: {cleaned_csv_path}")
        df_reviews = pd.read_csv(cleaned_csv_path)
        print(f"Loaded {len(df_reviews)} reviews.")

        # Ensure 'clean_review_text' is available.
        # It should be created by src/data_processing/preprocess.py
        """    
        if 'clean_review_text' not in df_reviews.columns:
            print("Warning: 'clean_review_text' column not found. Running basic text cleaning for demonstration.")
            # Import preprocess_text function from the preprocessor to ensure data is clean
            from src.preprocess import preprocess_text
            df_reviews['clean_review_text'] = df_reviews['review_text'].apply(preprocess_text)
        else:
            print("'clean_review_text' column found. Proceeding with thematic analysis.")
        """
        analyzer = ThematicAnalyzer()

        # --- Step 1: Extract Keywords using TF-IDF ---
        bank_tfidf_keywords = analyzer.extract_keywords_tfidf(df_reviews, text_column='clean_review_text', top_n=20) # Increased top_n
        print("\n--- TF-IDF Keywords per Bank (Top 20) ---")
        for bank, keywords in bank_tfidf_keywords.items():
            print(f"- {bank}: {keywords}")

        # --- Step 2: Extract Key Phrases using spaCy (Noun Chunks) ---
        bank_spacy_phrases = analyzer.extract_key_phrases_spacy(df_reviews, text_column='clean_review_text', top_n=20) # Increased top_n
        print("\n--- spaCy Key Phrases per Bank (Top 20) ---")
        for bank, phrases in bank_spacy_phrases.items():
            print(f"- {bank}: {phrases}")

        # --- Step 3: Assign Themes based on rules ---
        # IMPORTANT: CUSTOMIZE 'my_custom_theme_rules' based on the keywords/phrases above,
        # and your understanding of the reviews. Aim for 3-5 overarching themes per bank.
        # This is where your data analyst judgment comes in!
        my_custom_theme_rules = {
            'Account Access/Login': ['login', 'log in', 'password', 'username', 'pin', 'otp', 'access', 'blocked', 'account blocked', 'reset password', 'verify'],
            'Transaction Issues': ['transfer', 'send money', 'payment', 'slow', 'fast', 'transaction', 'failed', 'delay', 'complete', 'speed', 'otp delay', 'hang', 'stuck', 'money transfer', 'bank transfer', 'transaction history'],
            'App Performance/Bugs': ['crash', 'bug', 'glitch', 'error', 'force close', 'freezes', 'unstable', 'fix app', 'malfunction', 'broken', 'loading', 'load', 'slow loading', 'not working', 'working'],
            'User Interface/UX': ['ui', 'interface', 'design', 'easy', 'user friendly', 'difficult', 'navigate', 'layout', 'dashboard', 'looks', 'smooth', 'intuitive', 'clunky', 'confusing', 'experience', 'simple', 'complex'],
            'Feature Requests/Availability': ['fingerprint', 'face id', 'biometric', 'dark mode', 'new feature', 'add feature', 'update', 'qr code', 'payment options', 'notification', 'ussd', 'online payment', 'bill payment', 'features'],
            'Customer Support/Service': ['support', 'customer service', 'help', 'response', 'contact', 'call center', 'agent', 'hotline', 'phone', 'feedback', 'representative', 'customer care'],
            # You can add more themes if needed, aiming for 3-5 *per bank* which implies a common set for all.
            # Example: 'Security Concerns': ['security', 'secure', 'safe', 'fraud']
        }
        # The 'add_themes_to_dataframe' function will use the defined rules to tag each review.
        df_themed = analyzer.add_themes_to_dataframe(df_reviews.copy(), text_column='clean_review_text', theme_rules=my_custom_theme_rules)

        print("\n--- Sample of reviews with assigned themes ---")
        print(df_themed[['review_text', 'identified_theme(s)']].head(10)) # Print more samples

        # --- KPI Check: Themes per Bank ---
        print("\n--- KPI Check: Unique Themes Identified per Bank ---")
        for bank in df_themed['bank'].unique():
            bank_themes_set = set()
            # Ensure the 'identified_theme(s)' column is a list of lists if loaded from CSV (it might be string representation)
            # This eval() attempts to convert string representations of lists back into actual lists
            themes_data_for_bank = df_themed[df_themed['bank'] == bank]['identified_theme(s)'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x
            )
            for themes_list in themes_data_for_bank:
                if isinstance(themes_list, list): # Ensure it's a list before iterating
                    for theme in themes_list:
                        bank_themes_set.add(theme)

            # Exclude 'Other' for KPI count, if it's merely a fallback and not a primary theme
            if 'Other' in bank_themes_set and len(bank_themes_set) > 1:
                bank_themes_set.remove('Other') 

            print(f"- {bank}: {len(bank_themes_set)} unique themes identified: {list(bank_themes_set)}")
            if len(bank_themes_set) >= 3:
                print("    KPI ACHIEVED: 3+ themes identified.")
            else:
                print("    KPI WARNING: Fewer than 3 themes identified. Refine thematic rules.")