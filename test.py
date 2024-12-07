import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from scipy import stats
from typing import Union, List, Dict, Any, Optional

class AdvancedDataProfiler:
    def __init__(self, 
                 data: Union[pd.DataFrame, str], 
                 config: Dict[str, Any] = None):
        """
        Advanced Data Profiler with comprehensive analysis capabilities
        
        :param data: DataFrame or path to data file
        :param config: Configuration dictionary for profiling settings
        """
        # Load data
        self.data = self._load_data(data) if isinstance(data, str) else data
        
        # Default configuration with extensive options
        self.config = config or {
            'missing_threshold': 0.05,  # 5% missing data
            'cardinality_threshold': 0.9,  # High cardinality threshold
            'correlation_threshold': 0.8,  # High correlation threshold
            'outlier_method': 'iqr',  # Outlier detection method
            'text_analysis': {
                'min_word_length': 3,
                'max_common_words': 10
            },
            'time_series': {
                'seasonality_test': True,
                'trend_analysis': True
            }
        }
        
        # Comprehensive metadata storage
        self.metadata = {}
        self._generate_comprehensive_metadata()
    
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """
        Advanced data loading with multiple format support
        
        :param file_path: Path to the data file
        :return: Pandas DataFrame
        """
        file_extensions = {
            '.csv': pd.read_csv,
            '.xlsx': pd.read_excel,
            '.json': pd.read_json,
            '.parquet': pd.read_parquet,
            '.tsv': lambda f: pd.read_csv(f, sep='\t'),
            '.xls': pd.read_excel
        }
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in file_extensions:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return file_extensions[ext](file_path)
    
    def _generate_comprehensive_metadata(self):
        """
        Generate in-depth metadata about the dataset
        """
        # Basic dataset information
        self.metadata = {
            'dataset_overview': {
                'total_rows': len(self.data),
                'total_columns': len(self.data.columns),
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1e6,
                'data_types': dict(self.data.dtypes)
            },
            'column_details': {}
        }
        
        # Detailed analysis for each column
        for column in self.data.columns:
            col_metadata = {
                'dtype': str(self.data[column].dtype),
                'non_null_count': self.data[column].count(),
                'null_count': self.data[column].isnull().sum(),
                'null_percentage': self.data[column].isnull().mean() * 100,
                'unique_count': self.data[column].nunique(),
                'unique_percentage': self.data[column].nunique() / len(self.data) * 100
            }
            
            # Add type-specific analysis
            if pd.api.types.is_numeric_dtype(self.data[column]):
                col_metadata.update(self._numeric_column_analysis(column))
            elif pd.api.types.is_datetime64_any_dtype(self.data[column]):
                col_metadata.update(self._datetime_column_analysis(column))
            elif pd.api.types.is_object_dtype(self.data[column]):
                col_metadata.update(self._object_column_analysis(column))
            
            self.metadata['column_details'][column] = col_metadata
    
    def _numeric_column_analysis(self, column: str) -> Dict[str, Any]:
        """
        Perform detailed numeric column analysis
        
        :param column: Column name
        :return: Numeric column metadata
        """
        return {
            'statistical_summary': {
                'mean': self.data[column].mean(),
                'median': self.data[column].median(),
                'std_dev': self.data[column].std(),
                'min': self.data[column].min(),
                'max': self.data[column].max(),
                'skewness': self.data[column].skew(),
                'kurtosis': self.data[column].kurtosis()
            },
            'outliers': self._detect_outliers(column)
        }
    
    def _datetime_column_analysis(self, column: str) -> Dict[str, Any]:
        """
        Perform detailed datetime column analysis
        
        :param column: Column name
        :return: Datetime column metadata
        """
        return {
            'time_range': {
                'start': self.data[column].min(),
                'end': self.data[column].max(),
                'total_duration': self.data[column].max() - self.data[column].min()
            },
            'periodicity_analysis': {
                'daily_frequency': self.data[column].dt.floor('D').value_counts(),
                'monthly_frequency': self.data[column].dt.to_period('M').value_counts(),
                'weekday_distribution': self.data[column].dt.day_name().value_counts()
            }
        }
    
    def _object_column_analysis(self, column: str) -> Dict[str, Any]:
        """
        Perform detailed object column analysis
        
        :param column: Column name
        :return: Object column metadata
        """
        return {
            'top_values': self.data[column].value_counts().head(10).to_dict(),
            'text_analysis': {
                'avg_length': self.data[column].str.len().mean(),
                'max_length': self.data[column].str.len().max(),
                'min_length': self.data[column].str.len().min()
            }
        }
    
    def _detect_outliers(self, column: str, method: str = 'iqr') -> Dict[str, Any]:
        """
        Detect outliers using multiple methods
        
        :param column: Column name
        :param method: Outlier detection method
        :return: Outlier detection results
        """
        if method == 'iqr':
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
            
            return {
                'method': 'IQR',
                'total_outliers': len(outliers),
                'outlier_percentage': len(outliers) / len(self.data) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        raise ValueError(f"Unsupported outlier detection method: {method}")
    
    def correlations(self, method: str = 'pearson') -> Dict[str, pd.DataFrame]:
        """
        Calculate multiple correlation matrices
        
        :param method: Correlation method
        :return: Correlation matrices
        """
        methods = ['pearson', 'spearman', 'kendall']
        if method not in methods:
            raise ValueError(f"Unsupported correlation method. Use {methods}")
        
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        return {
            method: self.data[numeric_columns].corr(method=method)
        }
    
    def data_quality_checks(self) -> Dict[str, List[str]]:
        """
        Comprehensive data quality checks
        
        :return: Dictionary of data quality alerts
        """
        alerts = {}
        
        for column, details in self.metadata['column_details'].items():
            column_alerts = []
            
            # Missing value check
            if details['null_percentage'] > self.config['missing_threshold'] * 100:
                column_alerts.append(f"HIGH_MISSING_VALUES ({details['null_percentage']:.2f}%)")
            
            # Cardinality check
            if details['unique_percentage'] > self.config['cardinality_threshold'] * 100:
                column_alerts.append("HIGH_CARDINALITY")
            
            # Add alerts if any
            if column_alerts:
                alerts[column] = column_alerts
        
        return alerts
        
    def detect_and_convert_types(df):
        for col in df.columns:
            if df[col].dtype == 'object':
                if df[col].apply(lambda x: isinstance(x, str) or isinstance(x, NoneType)).all():
                    # Check for boolean-like values
                    if df[col].str.lower().isin(['true', 'false', '0', '1']).all():
                        df[col] = df[col].map({'true': True, 'false': False, '1': True, '0': False})
                        print(f"Column '{col}' detected as Boolean.")
                    
                    # Check for numeric values
                    elif pd.to_numeric(df[col], errors='coerce').notnull().all():
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        print(f"Column '{col}' detected as Numeric.")
                    
                    # Check for datetime values
                    elif pd.to_datetime(df[col], errors='coerce').notnull().all():
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        print(f"Column '{col}' detected as Datetime.")
                    
                    # Detect categorical columns
                    elif df[col].nunique() / len(df) < 0.1:
                        df[col] = df[col].astype('category')
                        print(f"Column '{col}' detected as Categorical.")
                    # Otherwise, treat as text
                    else:
                        df[col] = df[col].astype("string")
                        print(f"Column '{col}' detected as Text.")
                elif df[col].apply(lambda x: isinstance(x, bool) or isinstance(x, NoneType)).all():
                        df[col] = df[col].astype(bool)
                        print(f"Column '{col}' detected as Boolean.")
                else:
                    print(f"Column '{col}' contains non-string values and cannot be processed as text.")
            else:
                if np.issubdtype(df[col].dtype, np.number):
                    print(f"Column '{col}' detected as Numeric.")
                elif np.issubdtype(df[col].dtype, np.bool_):
                    print(f"Column '{col}' detected as Boolean.")
                elif np.issubdtype(df[col].dtype, np.datetime64):
                    print(f"Column '{col}' detected as Datetime.")
                else:
                    print(f"Column '{col}' detected as Unsupported.")
                    
        return df

# Additional specialized analysis modules can be added similarly
