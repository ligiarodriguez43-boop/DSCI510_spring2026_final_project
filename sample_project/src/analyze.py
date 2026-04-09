import os
import matplotlib.pyplot as plt


# --- PLOT STATISTICS ---
def plot_statistics(df, dataset_name, result_dir="plots", notebook_plot=False):
    """
    Generates and saves basic plots for a given DataFrame.

    :param result_dir: where to place plots
    :param df: The pandas DataFrame
    :param dataset_name: A name for titling plots (e.g., 'Titanic')
    """
    print(f"--- Plotting statistics for {dataset_name} ---")

    # Ensure a directory for plots exists
    os.makedirs(result_dir, exist_ok=True)

    # Identify numerical and categorical columns for plotting
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Plot 1: Histogram (for a numerical column)
    if not numerical_cols.empty:
        col_to_plot = numerical_cols[0]
        plt.figure(figsize=(10, 6))
        df[col_to_plot].hist(bins=30, edgecolor='black')
        plt.title(f'Histogram of {col_to_plot} - {dataset_name}')
        plt.xlabel(col_to_plot)
        plt.ylabel('Frequency')
        plt.grid(axis='y')
        if not notebook_plot:
            plt.savefig(f'{result_dir}/{dataset_name}_histogram.png')
            print(f"Saved histogram for {col_to_plot}")
            plt.close()
        else:
            plt.plot()

    # Plot 2: Bar Chart (for a categorical column)
    if not categorical_cols.empty:
        # Use the first categorical column with less than 30 unique values
        for col_to_plot in categorical_cols:
            if df[col_to_plot].nunique() < 30:
                plt.figure(figsize=(10, 6))
                df[col_to_plot].value_counts().plot(kind='bar')
                plt.title(f'Bar Chart of {col_to_plot} - {dataset_name}')
                plt.xlabel(col_to_plot)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.grid(axis='y')
                if not notebook_plot:
                    plt.savefig(f'{result_dir}/{dataset_name}_barchart.png')
                    print(f"Saved bar chart for {col_to_plot}")
                    plt.close()
                else:
                    plt.plot()
                break  # Only plot the first suitable one

    # Plot 3: Scatter Plot (for two numerical columns)
    if len(numerical_cols) >= 2:
        col1 = numerical_cols[0]
        col2 = numerical_cols[1]
        plt.figure(figsize=(10, 6))
        plt.scatter(df[col1], df[col2], alpha=0.5)
        plt.title(f'Scatter Plot: {col1} vs {col2} - {dataset_name}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid(True)
        if not notebook_plot:
            plt.savefig(f'{result_dir}/{dataset_name}_scatterplot.png')
            print(f"Saved scatter plot for {col1} vs {col2}")
            plt.close()
        else:
            plt.plot()
