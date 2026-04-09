import os
# os.environ["KAGGLE_CONFIG_DIR"] = "/home/alexey/"
import kaggle
import pandas as pd
import requests

# --- 1. DOWNLOAD DATA FROM KAGGLE ---

def get_kaggle_data(dataset_slug, extract_dir='kaggle_data'):
    """
    Downloads a specific file from a Kaggle dataset, extracts it,
    and loads it into a pandas DataFrame.

    :param dataset_slug: The slug of the dataset (e.g., 'titanic')
    :param extract_dir: Directory to extract files into
    :return: pandas DataFrame or None
    """
    print(f"--- Loading data from Kaggle: {dataset_slug} ---")
    try:
        # Ensure extraction directory exists
        os.makedirs(extract_dir, exist_ok=True)

        print(f"Downloading {dataset_slug}...")
        kaggle.api.dataset_download_files(dataset_slug, path=extract_dir, unzip=True)

        csv_file_path = [f for f in os.listdir(extract_dir) if os.path.isfile(os.path.join(extract_dir, f))][0]
        csv_file_path = os.path.join(extract_dir, csv_file_path) # make it a full path
        # Load the extracted CSV into a DataFrame
        if os.path.exists(csv_file_path):
            print(f"Loading {csv_file_path} into DataFrame...")
            df = pd.read_csv(csv_file_path)
            print("Kaggle data loaded successfully.")
            return df
        else:
            print(f"Error: Could not find extracted file {csv_file_path}")
            return None

    except Exception as e:
        print(f"Error loading data from Kaggle: {e}")
        return None


# --- 2. DOWNLOAD FILE FROM WEB ---

def get_web_csv_data(url):
    """
    Downloads a CSV file directly from a URL into a pandas DataFrame.

    :param url: The direct URL to the .csv file
    :return: pandas DataFrame or None
    """
    print(f"--- Loading data from Web URL: {url[:50]}... ---")
    try:
        # pandas can read a CSV directly from a URL
        df = pd.read_csv(url)
        print("Web CSV data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data from URL: {e}")
        return None


# --- 3. SCRAPE DATA FROM WEBPAGE (WIKIPEDIA) ---
def get_wikipedia_table_data(url, table_index=0):
    """
    Scrapes a specific table from a Wikipedia page into a pandas DataFrame.

    MODIFIED: Added a User-Agent header to prevent 403 Forbidden errors.

    :param url: The URL of the Wikipedia page
    :param table_index: The 0-based index of the table to scrape
    :return: pandas DataFrame or None
    """
    print(f"--- Scraping table from Wikipedia: {url[:50]}... ---")

    # Set a User-Agent to mimic a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Use requests to get the page with the headers
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}. Unable to fetch page.")
            return None

        # Pass the HTML content (response.text) to pandas
        # It returns a LIST of DataFrames (all tables on the page)
        tables = pd.read_html(response.text, flavor='bs4')

        if tables:
            print(f"Found {len(tables)} tables. Extracting table at index {table_index}...")
            # We select the specific table we want
            df = tables[table_index]
            print("Wikipedia table scraped successfully.")
            return df
        else:
            print("No tables found on the page.")
            return None

    except Exception as e:
        print(f"Error scraping Wikipedia table: {e}")
        return None

