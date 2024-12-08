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
