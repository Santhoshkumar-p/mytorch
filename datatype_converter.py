



import streamlit as st
import pandas as pd
import numpy as np

def detect_column_types(df):
    """
    Detect the initial data types of columns in the DataFrame with enhanced type detection.
    """
    column_types = {}
    for col in df.columns:
        # Comprehensive type detection
        if isinstance(df[col], pd.CategoricalDtype):
            column_types[col] = 'category'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = 'datetime'
        elif pd.api.types.is_integer_dtype(df[col]):
            column_types[col] = 'integer'
        elif pd.api.types.is_float_dtype(df[col]):
            column_types[col] = 'float'
        elif pd.api.types.is_bool_dtype(df[col]):
            column_types[col] = 'boolean'
        else:
            column_types[col] = 'string'
    return column_types

def convert_data_type(series, target_type, additional_handler=None):
    """
    Convert data type with comprehensive handling options.
    """
    try:
        if target_type == 'integer':
            return pd.to_numeric(series, errors='coerce')
        
        elif target_type == 'float':
            return pd.to_numeric(series, errors='coerce')
        
        elif target_type == 'string':
            return series.astype(str)
        
        elif target_type == 'boolean':
            # Handling various boolean representations
            bool_map = {
                'true': True, 'false': False,
                't': True, 'f': False,
                '1': True, '0': False,
                'yes': True, 'no': False,
                'y': True, 'n': False
            }
            series.map(lambda x: bool_map.get(str(x).lower(), pd.NA))
            return series.astype(bool)
        
        elif target_type == 'datetime':
            if additional_handler:
                return pd.to_datetime(series, format=additional_handler, errors='coerce')
            return pd.to_datetime(series, errors='coerce')
        
        elif target_type == 'category':
            # If additional_handler is provided, use it as the categories
            if additional_handler:
                categories = [cat.strip() for cat in additional_handler.split(',')]
                return pd.Categorical(series, categories=categories)
            return pd.Categorical(series)
        else:
            return series
    except Exception as e:
        st.error(f"Conversion error for {target_type}: {e}")
        return series

def main():
    st.title('📊 Advanced Data Type Converter')
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", 
                                     type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Read the file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
        
        # Detect initial column types
        initial_types = detect_column_types(df)
        
        # Predefined date formats
        date_formats = [
            '%Y-%m-%d', 
            '%d/%m/%Y', 
            '%m/%d/%Y', 
            '%Y/%m/%d', 
            '%d-%m-%Y', 
            '%m-%d-%Y',
            '%Y-%m-%d %H:%M:%S',
            'Custom'
        ]
        
        # Prepare conversion data
        convert_data = []
        for col, dtype in initial_types.items():
            convert_data.append({
                'column': col, 
                'source_type': dtype, 
                'target_type': dtype,
                'additional_handler': ''
            })
        
        # Convert data DataFrame
        convert_df = pd.DataFrame(convert_data)
        
        # Display original DataFrame head
        st.subheader('Original Data Preview')
        st.dataframe(df.head())
        
        # Conversion configuration
        st.subheader('Data Type Conversion')
        
        # Use column configs for interactive conversion
        convert_df_edited = st.data_editor(
            convert_df, 
            column_config={
                'column': st.column_config.TextColumn('Column Name', disabled=True),
                'source_type': st.column_config.TextColumn('Source Type', disabled=True),
                'target_type': st.column_config.SelectboxColumn(
                    'Target Type', 
                    options=['string', 'integer', 'float', 'boolean', 'datetime', 'category'],
                    width='medium'
                ),
                'additional_handler': st.column_config.TextColumn(
                    'Additional Handler',
                    help=(
                        'Datetime: Input date format\n'
                        'Category: Comma-separated list of categories'
                    )
                )
            },
            num_rows='fixed'
        )
        
        # Conversion button
        if st.button('Convert Data'):
            # Create a copy of the original DataFrame for conversion
            converted_df = df.copy()
            
            # Perform conversions
            for _, row in convert_df_edited.iterrows():
                col = row['column']
                target_type = row['target_type']
                handler = row['additional_handler'] if pd.notna(row['additional_handler']) else None
                
                converted_df[col] = convert_data_type(
                    converted_df[col], 
                    target_type, 
                    handler
                )
            
            # Side-by-side comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader('Original DataFrame')
                st.dataframe(df.dtypes)
            
            with col2:
                st.subheader('Converted DataFrame')
                st.dataframe(converted_df.dtypes)
            
            # Full DataFrame comparison
            st.subheader('Converted Data Preview')
            st.dataframe(converted_df.head())
            
            # Optional: Download converted DataFrame
            csv = converted_df.to_csv(index=False)
            st.download_button(
                label="Download Converted Data",
                data=csv,
                file_name='converted_data.csv',
                mime='text/csv'
            )

if __name__ == '__main__':
    main()
