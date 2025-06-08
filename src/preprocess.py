# src/data_processing/preprocess.py
import pandas as pd
import re
import os
from datetime import datetime # Import datetime for type checking/conversion

def load_raw_reviews(filepath):
    """Loads raw JSON review data from a specified filepath into a Pandas DataFrame."""
    try:
        df = pd.read_json(filepath)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

def preprocess_text(text):
    """Applies a series of text cleaning steps."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Lowercase text
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\S*@\S*\s?', '', text) # Remove emails
    text = re.sub(r'#\w+', '', text) # Remove hashtags
    text = re.sub(r'@\w+', '', text) # Remove mentions
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation and special characters
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

def normalize_date(date_str):
    """Normalizes date string to YYYY-MM-DD format."""
    if pd.isna(date_str):
        return None
    try:
        # The 'at' field from scraped data should now be an ISO format string (YYYY-MM-DDTHH:MM:SS)
        if isinstance(date_str, str):
            return pd.to_datetime(date_str).strftime('%Y-%m-%d')
        # Fallback if it's somehow still a datetime object (though our scraping fixed this)
        if isinstance(date_str, datetime):
            return date_str.strftime('%Y-%m-%d')
        return None # Return None if date cannot be parsed or is unexpected type
    except Exception:
        return None # Return None if date cannot be parsed

def clean_reviews_dataframe(df):
    """
    Applies preprocessing steps to a DataFrame of reviews.
    Expected columns from raw: 'content', 'score' (for rating), 'at' (for date), 'bank_name', 'source'
    """
    if df.empty:
        print("DataFrame is empty, skipping cleaning.")
        return pd.DataFrame()

    # --- CRITICAL FIX HERE: Rename 'content' to 'review_text' and use inplace=True ---
    df.rename(columns={'content': 'review_text', 'score': 'rating', 'at': 'date'}, inplace=True)

    # Handle missing review text: fill with empty string for preprocessing
    # This line will now correctly access the 'review_text' column
    df['review_text'] = df['review_text'].fillna('')

    # Remove duplicate reviews (based on review_text and bank_name)
    initial_rows = len(df)
    df.drop_duplicates(subset=['review_text', 'bank_name'], inplace=True)
    print(f"Removed {initial_rows - len(df)} duplicate reviews.")

    # Apply text cleaning to review text
    df['clean_review_text'] = df['review_text'].apply(preprocess_text)

    # Handle missing ratings: remove rows with missing critical data like rating
    df.dropna(subset=['rating'], inplace=True)
    print(f"Removed rows with missing ratings. Remaining rows: {len(df)}")

    # Normalize dates
    df['date'] = df['date'].apply(normalize_date)
    # Remove rows where date could not be normalized
    df.dropna(subset=['date'], inplace=True)
    print(f"Removed rows with unparseable dates. Remaining rows: {len(df)}")


    # Select and reorder final columns for the CSV output
    # Ensuring 'bank_name' and 'source' are present from scraping phase
    final_columns = ['review_text', 'rating', 'date', 'bank_name', 'source']
    
    # Check if all final columns exist, add if missing but essential (e.g., 'source' should be from scraping)
    for col in ['bank_name', 'source']:
        if col not in df.columns:
            print(f"Warning: '{col}' column not found in DataFrame. Adding as empty.")
            df[col] = '' # Add as empty to avoid error if column somehow missed

    df = df[final_columns] # Reorder to match requirements
    df.rename(columns={'bank_name': 'bank'}, inplace=True) # Rename `bank_name` to `bank` for final CSV

    return df

if __name__ == "__main__":
    # Define paths
    RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),  '..', 'data')
    PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True) # Ensure processed data directory exists

    print("Starting data preprocessing...")

    all_processed_reviews = pd.DataFrame() # Initialize an empty DataFrame to store all combined processed reviews

    # Loop through each JSON file in the raw data directory
    for bank_file in os.listdir(RAW_DATA_DIR):
        if bank_file.endswith(".json"):
            # Construct full path to the raw file
            raw_filepath = os.path.join(RAW_DATA_DIR, bank_file)
            print(f"Processing raw data from: {raw_filepath}")
            
            # Load the raw data into a DataFrame
            df_raw = load_raw_reviews(raw_filepath)
            
            # If data was loaded successfully, clean it
            if not df_raw.empty:
                df_processed_single_bank = clean_reviews_dataframe(df_raw)
                
                # Concatenate with the main DataFrame
                all_processed_reviews = pd.concat([all_processed_reviews, df_processed_single_bank], ignore_index=True)
                print(f"Finished processing {len(df_processed_single_bank)} reviews from {bank_file}.")
            else:
                print(f"No data to process for {bank_file}.")

    # After processing all banks, save the combined cleaned data to CSV
    if not all_processed_reviews.empty:
        output_csv_path = os.path.join(PROCESSED_DATA_DIR, "all_cleaned_reviews.csv")
        all_processed_reviews.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"\nAll processed reviews saved to: {output_csv_path}")
        print(f"Total cleaned reviews: {len(all_processed_reviews)}")
    else:
        print("\nNo reviews were processed and saved.")
"""
    # --- KPI Check for Missing Data ---
    total_raw_reviews_count = 0
    for f in os.listdir(RAW_DATA_DIR):
        if f.endswith(".json"):
            df_temp = load_raw_reviews(os.path.join(RAW_DATA_DIR, f))
            total_raw_reviews_count += len(df_temp)

    if total_raw_reviews_count > 0:
        missing_reviews_count = total_raw_reviews_count - len(all_processed_reviews)
        missing_percentage = (missing_reviews_count / total_raw_reviews_count) * 100
        print("\n--- KPI Check ---")
        print(f"Total raw reviews initially collected: {total_raw_reviews_count}")
        print(f"Total cleaned reviews (after preprocessing): {len(all_processed_reviews)}")
        print(f"Data loss (due to duplicates/missing critical fields): {missing_reviews_count} reviews")
        print(f"Missing data percentage: {missing_percentage:.2f}% (Target: <5%)")
        if missing_percentage < 5:
            print("KPI ACHIEVED: Missing data percentage is within acceptable limits.")
        else:
            print("KPI WARNING: Missing data percentage is higher than the 5% target. Review preprocessing steps.")
    else:
        print("KPI Check: No raw reviews found to calculate missing data percentage.")
"""