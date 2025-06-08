# scripts/scrape_reviews.py
from datetime import datetime
import json
import os
from google_play_scraper import Sort, reviews_all

# --- Custom JSON Encoder for datetime objects ---
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    # If it's not a datetime object and still not serializable, raise the original TypeError
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

# Define the target apps with their IDs and desired output filenames

APP_DETAILS = {
    "Commercial Bank of Ethiopia": {
        "app_id": "com.combanketh.mobilebanking",
        "output_file": "cbe_reviews.json"
    },
    "Bank of Abyssinia": {
        "app_id": "com.boa.boaMobileBanking", 
        "output_file": "boa_reviews.json"
    },
    "Dashen Bank": {
        "app_id": "com.dashen.dashensuperapp",
        "output_file": "dashen_reviews.json"
    }
}

# Define the directory for raw data
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
os.makedirs(RAW_DATA_DIR, exist_ok=True) # Ensure the directory exists

print(f"Starting review scraping process. Raw data will be saved in: {RAW_DATA_DIR}")

for bank_name, details in APP_DETAILS.items():
    app_id = details["app_id"]
    output_file = os.path.join(RAW_DATA_DIR, details["output_file"])

    print(f"\n--- Scraping reviews for {bank_name} (App ID: {app_id}) ---")

    try:
        # Scrape all reviews for the app
        # `count` parameter can be used to limit, but `reviews_all` by default aims for all
        # Set specific language and country if relevant for Ethiopia, e.g., lang='en', country='et'
        scraped_reviews = reviews_all(
            app_id,
            lang='en',          # Language of reviews
            country='et',       # Country to scrape from (Ethiopia)
            sort=Sort.NEWEST,   # Sort by newest reviews first
            # filter_score_with=5 # Optional: filter by score (e.g., only 5-star reviews)
        )

        # Add bank name and source to each review
        processed_reviews_for_json = []
        for review in scraped_reviews:
            # Add bank name and source
            review['bank_name'] = bank_name
            review['source'] = 'Google Play'
            
            
            # Convert 'at' (datetime object) to ISO format string
            if 'at' in review and isinstance(review['at'], datetime):
                review['at'] = review['at'].isoformat() # Converts to 'YYYY-MM-DDTHH:MM:SS.ffffff'
            
            processed_reviews_for_json.append(review)

        # Save raw reviews to a JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_reviews_for_json, f, ensure_ascii=False, indent=4, default=json_serial)

        print(f"Successfully scraped {len(scraped_reviews)} reviews for {bank_name}.")
        print(f"Raw data saved to: {output_file}")

    except Exception as e:
        print(f"Error scraping {bank_name} (App ID: {app_id}): {e}")
        print("Please ensure the App ID is correct and there are no network issues or rate limits.")

print("\nScraping process completed.")