import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Generate example data (1 million records)
data = {'Amount': np.random.randn(1000000)}  # Example data
df = pd.DataFrame(data)

# Precompute histogram and bin edges
def precompute_histogram(df_column, num_bins=20):
    # Calculate bin edges using linspace between min and max values
    bin_edges = np.linspace(df_column.min(), df_column.max(), num_bins + 1)

    # Calculate histogram data using np.histogram
    hist, bin_edges = np.histogram(df_column, bins=bin_edges)
    
    return hist, bin_edges

# Precompute histogram
hist, bin_edges = precompute_histogram(df['Amount'], num_bins=20)

# Function to create the figure for histogram using precomputed values
def create_histogram_figure(hist, bin_edges):
    fig, ax = plt.subplots()
    # Use the precomputed hist and bin_edges directly in ax.hist
    ax.hist(bin_edges[:-1], bins=bin_edges, weights=hist, edgecolor='black')
    ax.set_title('Histogram of Amount')
    ax.set_xlabel('Amount')
    ax.set_ylabel('Frequency')
    return fig

# Generate the figure
fig = create_histogram_figure(hist, bin_edges)

# Display the histogram in Streamlit
st.pyplot(fig)



def precompute_boxplot_stats(column):
    """
    Precompute statistics for boxplot and outlier detection.
    """
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mean = column.mean()
    median = column.median()
    
    # Count outliers for memory efficiency
    outlier_count = ((column < lower_bound) | (column > upper_bound)).sum()
    
    # Store computed stats
    stats = {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Mean": mean,
        "Median": median,
        "Outlier Count": outlier_count
    }
    return stats

def create_boxplot_figure(stats, column, sample_size=1000):
    """
    Create a Matplotlib boxplot figure using precomputed statistics.
    """
    sample_data = column.sample(sample_size, random_state=42)  # Sample for visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(sample_data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'))
    
    # Add precomputed statistics
    ax.axvline(stats["Mean"], color='green', linestyle='--', label=f"Mean: {stats['Mean']:.2f}")
    ax.axvline(stats["Median"], color='orange', linestyle='--', label=f"Median: {stats['Median']:.2f}")
    ax.axvline(stats["Lower Bound"], color='red', linestyle='--', label=f"Lower Bound: {stats['Lower Bound']:.2f}")
    ax.axvline(stats["Upper Bound"], color='purple', linestyle='--', label=f"Upper Bound: {stats['Upper Bound']:.2f}")
    
    # Customize plot
    ax.set_title("Boxplot for Large Dataset (Sampled for Visualization)")
    ax.set_xlabel("Values")
    ax.legend()
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    return fig
