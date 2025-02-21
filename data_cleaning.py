# Comprehensive data analysis and cleaning script for buyer persona creation
# This script handles data loading, cleaning, and analysis with proper error handling
# and data validation at each step

import pandas as pd
import numpy as np
from scipy import stats  # type: ignore
import logging

class BuyerPersonaAnalysis:
    def __init__(self, file_path):
        """
        Initialize the analysis with the data file path
        
        Parameters:
        file_path (str): Path to the CSV file containing the data
        """
        self.file_path = file_path
        self.original_df = None  # Will store the original, unmodified data
        self.cleaned_df = None   # Will store the cleaned version of the data
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self):
        """
        Load the CSV file and perform initial data checks.
        Handles different encodings and provides error messages if loading fails.
        
        Returns:
        bool: True if data loaded successfully, False otherwise
        """
        try:
            # Try UTF-8 encoding first (most common encoding)
            self.original_df = pd.read_csv(self.file_path, encoding='utf-8')
            self.logger.info(f"Successfully loaded {len(self.original_df)} records")
            return True
        except UnicodeDecodeError:
            # If UTF-8 fails, try latin-1 encoding (more permissive)
            try:
                self.original_df = pd.read_csv(self.file_path, encoding='latin-1')
                self.logger.info(f"Successfully loaded {len(self.original_df)} records using latin-1 encoding")
                return True
            except Exception as e:
                self.logger.error(f"Error loading file with latin-1 encoding: {str(e)}")
                return False
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False
    
    def analyze_completeness(self):
        """
        Analyze the completeness of data and identify missing patterns.
        Provides detailed statistics about data completeness and missing values.
        
        Returns:
        tuple: (completion_rates, missing_by_column) if successful, None if no data loaded
        """
        if self.original_df is None:
            self.logger.warning("No data loaded. Please load data first.")
            return None, None
        
        # Calculate what percentage of fields are filled for each record
        completion_rates = self.original_df.notna().mean(axis=1)
        
        print("\nData Completeness Analysis:")
        print("-" * 50)
        print("\nCompletion Rate Statistics:")
        print(completion_rates.describe())
        
        # Analyze which columns have missing values
        missing_by_column = self.original_df.isnull().sum().sort_values(ascending=False)
        print("\nMissing Values by Column:")
        for column, missing_count in missing_by_column.items():
            if missing_count > 0:
                percentage = (missing_count/len(self.original_df)*100)
                print(f"{column}: {missing_count} missing ({percentage:.1f}%)")
                
        return completion_rates, missing_by_column
    
    def clean_income_value(self, income_str):
        """
        Convert income from string format ('$45,000.00') to float (45000.0)
        Handles missing values and invalid formats gracefully.
        
        Parameters:
        income_str: The income value to clean, can be string or NaN
        
        Returns:
        float: Cleaned income value or np.nan if conversion fails
        """
        if pd.isna(income_str):
            return np.nan
        try:
            # Remove currency symbol and thousands separator
            cleaned = str(income_str).replace('$', '').replace(',', '').strip()
            return float(cleaned)
        except (ValueError, AttributeError):
            return np.nan
    
    def predict_income(self, row):
        """
        Predict missing income based on education and job position.
        Uses a sophisticated approach to estimate income based on similar profiles.
        
        Parameters:
        row: A row from the DataFrame containing person's information
        
        Returns:
        float: Predicted or cleaned income value
        """
        # Check if we need to predict (if income is missing)
        if pd.isna(row['INGRESO MENSUAL']):
            # Ensure we have our numeric income column
            if 'INGRESO_MENSUAL_NUMERIC' not in self.original_df.columns:
                self.original_df['INGRESO_MENSUAL_NUMERIC'] = self.original_df['INGRESO MENSUAL'].apply(self.clean_income_value)
            
            # Handle missing job positions
            if pd.isna(row['PUESTO']):
                self.logger.info("Missing job position, using education-based prediction")
                education_based = self.original_df[
                    self.original_df['ESCOLARIDAD'] == row['ESCOLARIDAD']
                ]['INGRESO_MENSUAL_NUMERIC'].dropna()
                return education_based.median() if len(education_based) > 0 else np.nan
            
            try:
                # Create a mask for similar profiles
                education_mask = (self.original_df['ESCOLARIDAD'] == row['ESCOLARIDAD'])
                valid_job_mask = self.original_df['PUESTO'].notna()
                
                # Convert job titles to string and compare
                job_str = str(row['PUESTO']).lower()
                job_mask = self.original_df['PUESTO'].astype(str).str.lower() == job_str
                
                # Combine all conditions
                similar_profiles = self.original_df[
                    education_mask & valid_job_mask & job_mask
                ]['INGRESO_MENSUAL_NUMERIC'].dropna()
                
                if len(similar_profiles) >= 3:
                    return similar_profiles.median()
                else:
                    self.logger.info(f"Not enough similar profiles for job: {job_str}, using education-based fallback")
            
            except Exception as e:
                self.logger.error(f"Error processing job position: {str(e)}, using education-based fallback")
            
            # If we get here, we need to use education-based fallback
            education_based = self.original_df[
                self.original_df['ESCOLARIDAD'] == row['ESCOLARIDAD']
            ]['INGRESO_MENSUAL_NUMERIC'].dropna()
            
            return education_based.median() if len(education_based) > 0 else np.nan
        
        # If we have the income value, just clean it
        return self.clean_income_value(row['INGRESO MENSUAL'])
    
    def infer_marital_regime(self, row):
        """
        Infer marriage regime based on marital status and patterns in data.
        Uses logical rules and existing patterns to fill missing values.
        
        Parameters:
        row: A row from the DataFrame containing person's information
        
        Returns:
        str: Inferred marital regime or original value if not missing
        """
        if pd.isna(row['REGIMEN C']):
            # First check if ESTADO CIVIL exists and is not null
            if pd.isna(row['ESTADO CIVIL']):
                return 'PENDING'  # Return a default value for missing marital status
                
            try:
                # Convert to string and handle the comparison
                estado_civil = str(row['ESTADO CIVIL']).upper()
                if estado_civil in ['SOLTERO', 'SOLTERA']:
                    return 'N/A'
                elif estado_civil in ['CASADO', 'CASADA']:
                    # Find most common regime for married people
                    married_regimes = self.original_df[
                        self.original_df['ESTADO CIVIL'].str.upper().isin(['CASADO', 'CASADA'])
                    ]['REGIMEN C'].dropna()
                    
                    return married_regimes.mode()[0] if len(married_regimes) > 0 else 'PENDING'
                else:
                    return 'PENDING'  # For any other marital status
            except Exception as e:
                self.logger.error(f"Error processing marital regime for row with ESTADO CIVIL: {row['ESTADO CIVIL']}")
                return 'PENDING'
                
        return row['REGIMEN C']
    
    def clean_data(self):
        """
        Main data cleaning function that handles missing values and standardizes formats.
        Applies all cleaning rules and transformations to create a clean dataset.
        
        Returns:
        DataFrame: Cleaned dataset or None if no data loaded
        """
        if self.original_df is None:
            self.logger.warning("No data loaded. Please load data first.")
            return None
            
        self.cleaned_df = self.original_df.copy()
        
        # Clean and standardize income data
        self.logger.info("\nCleaning income data...")
        self.cleaned_df['INGRESO MENSUAL'] = self.cleaned_df.apply(self.predict_income, axis=1)
        
        # Clean marital regime data
        self.logger.info("\nCleaning marital regime data...")
        self.cleaned_df['REGIMEN C'] = self.cleaned_df.apply(self.infer_marital_regime, axis=1)
        
        # Standardize categorical variables
        self.logger.info("\nStandardizing categorical variables...")
        categorical_columns = ['ESCOLARIDAD', 'ESTADO CIVIL', 'TIPO DE RESIDENCIA']
        for col in categorical_columns:
            if col in self.cleaned_df.columns:
                self.cleaned_df[col] = self.cleaned_df[col].str.strip().str.upper()
            
        print("\nCleaning Results:")
        print("-" * 50)
        for column in ['INGRESO MENSUAL', 'REGIMEN C']:
            if column in self.cleaned_df.columns:
                before = self.original_df[column].isna().sum()
                after = self.cleaned_df[column].isna().sum()
                print(f"\n{column}:")
                print(f"Missing values before: {before}")
                print(f"Missing values after: {after}")
            
        return self.cleaned_df
    
    def validate_results(self):
        """
        Validate the cleaning results by comparing distributions and checking for anomalies.
        Provides detailed statistics about the changes made during cleaning.
        """
        if self.cleaned_df is None:
            self.logger.warning("No cleaned data available. Please clean data first.")
            return
            
        print("\nValidation Results:")
        print("-" * 50)
        
        # Compare income distributions
        print("\nIncome Distribution Comparison:")
        print("\nOriginal Data:")
        print(self.original_df['INGRESO MENSUAL'].describe())
        print("\nCleaned Data:")
        print(self.cleaned_df['INGRESO MENSUAL'].describe())
        
        # Check categorical variables consistency
        categorical_cols = ['ESTADO CIVIL', 'REGIMEN C', 'TIPO DE RESIDENCIA']
        for col in categorical_cols:
            if col in self.cleaned_df.columns:
                print(f"\n{col} Value Counts:")
                print(self.cleaned_df[col].value_counts(normalize=True) * 100)

    def get_cleaning_summary(self):
        """
        Generate a summary of the cleaning process and data quality metrics.
        This helps other components understand the state of the data.
        
        Returns:
        dict: Summary statistics and quality metrics
        """
        if self.cleaned_df is None:
            return "No cleaned data available. Run clean_data() first."

        summary = {
            'total_records': len(self.cleaned_df),
            'columns': list(self.cleaned_df.columns),
            'missing_values': self.cleaned_df.isnull().sum().to_dict(),
            'successful_sales': len(self.cleaned_df[self.cleaned_df['STATUS'] == 'ESCRITURADA']),
            'data_types': self.cleaned_df.dtypes.to_dict()
        }
        
        return summary

    def get_cleaned_data(self):
        """
        Public method to access the cleaned data.
        This is the main interface for other components to get the cleaned data.
        
        Returns:
        DataFrame: The cleaned dataset or None if cleaning hasn't been performed
        """
        if self.cleaned_df is None:
            self.logger.warning("No cleaned data available. Run clean_data() first.")
            return None
            
        return self.cleaned_df

def main():
    """
    Main execution function that orchestrates the analysis process.
    Handles the complete workflow from data loading to validation.
    """
    # Initialize analysis with file path
    file_path = r"C:\Users\L03140374\Desktop\buyer_persona_analysis\data\Viña Real - Analisis de Ventas UTF-8.csv"
    analysis = BuyerPersonaAnalysis(file_path)
    
    # Load and analyze data
    if analysis.load_data():
        print("\nAnalyzing data completeness...")
        completion_rates, missing_analysis = analysis.analyze_completeness()
        
        print("\nCleaning data...")
        cleaned_data = analysis.clean_data()
        
        print("\nValidating results...")
        analysis.validate_results()
        
        # Get cleaning summary
        summary = analysis.get_cleaning_summary()
        print("\nCleaning Summary:")
        for key, value in summary.items():
            print(f"\n{key}:")
            print(value)
        
        return cleaned_data
    
    return None

if __name__ == "__main__":
    cleaned_df = main()