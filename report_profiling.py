import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport

def create_data_profile(df):
    """
    Generate a data profile report using ydata-profiling
    """
    profile = ProfileReport(
        df, 
        title="Data Profiling Report",
        explorative=True,
        dark_mode=True
    )
    
    # Save profile to HTML
    profile_html = profile.to_html()
    return profile_html

def profiling_page():
    st.title('📈 Data Profiling Report')
    
    # Check if converted DataFrame exists in session state
    if 'converted_df' not in st.session_state:
        st.warning("No data available. Please convert a DataFrame first.")
        
        # Button to go back to converter
        if st.button('Go Back to Data Converter'):
            st.switch_page("app.py")
        return
    
    # Retrieve the converted DataFrame
    converted_df = st.session_state.converted_df
    
    # Display basic DataFrame info
    st.subheader('DataFrame Overview')
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Number of Rows", converted_df.shape[0])
        st.metric("Number of Columns", converted_df.shape[1])
    
    with col2:
        st.write("**Column Types:**")
        st.dataframe(converted_df.dtypes)
    
    # Generate profile report
    st.subheader('Comprehensive Data Profile')
    
    # Add a spinner while generating report
    with st.spinner('Generating comprehensive data profile...'):
        profile_html = create_data_profile(converted_df)
    
    # Display profile report in an iframe
    st.components.v1.html(profile_html, height=1000, scrolling=True)
    
    # Download profile report button
    st.download_button(
        label="Download Full Profile Report",
        data=profile_html,
        file_name='data_profile_report.html',
        mime='text/html'
    )
    
    # Button to go back to converter
    if st.button('Back to Data Converter'):
        st.switch_page("app.py")

def main():
    profiling_page()

if __name__ == '__main__':
    main()
