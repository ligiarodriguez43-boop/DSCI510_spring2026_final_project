from load import get_wikipedia_table_data
import pandas as pd

def process_wiki_data(wiki_url:str)-> pd.DataFrame:
    # Note: The table index (e.g., 0) might change if Wikipedia updates the page
    # 0 is usually the main table.
    wiki_df = get_wikipedia_table_data(wiki_url, table_index=0)
    if wiki_df is not None:
        print("\nWikipedia (Largest Companies) Data Head:")
        print(wiki_df.head())

        # We need to clean this data before plotting, as it's messy
        # For example, revenue is a string like "$3,171 million"
        # This is a common next step in data science.
        # Let's try to clean one column for plotting
        try:
            # Let's clean the 'Employees' column
            if 'Employees' in wiki_df.columns:
                # Remove commas, then convert to numeric
                wiki_df['Employees_numeric'] = wiki_df['Employees'].astype(str).replace(',', '', regex=False)
                # Handle non-numeric entries by coercing them to NaN
                wiki_df['Employees_numeric'] = pd.to_numeric(wiki_df['Employees_numeric'], errors='coerce')
                # Create a new, cleaner DataFrame for plotting
                plot_df = pd.DataFrame({
                    'Employees': wiki_df['Employees_numeric'],
                    'Industry': wiki_df['Industry']['Industry']
                })
                return plot_df
            else:
                print("Could not find 'Employees' column for plotting.")
        except Exception as e:
            print(f"Could not clean/plot Wikipedia data: {e}")

    return pd.DataFrame()
