import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import os

class EnhancedBuyerAnalysis:
    """
    Analyzes successful real estate sales to understand buyer patterns
    """
    def __init__(self, cleaned_data):
        """
        Initialize with cleaned DataFrame
        
        Parameters:
        cleaned_data (pd.DataFrame): Cleaned data from BuyerPersonaAnalysis
        """
        self.data = cleaned_data
        # Focus on successful sales only
        self.successful_sales = cleaned_data[cleaned_data['STATUS'] == 'ESCRITURADA'].copy()
        self.analysis_results = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Set specific path for plots
        self.plot_dir = r"C:\Users\L03140374\Desktop\buyer_persona_analysis\enhanced_analysis_plot"
        # Ensure directory exists (though it should already exist in this case)
        os.makedirs(self.plot_dir, exist_ok=True)
    
    def analyze_successful_buyers(self):
        """
        Main method to analyze successful buyers
        """
        self.logger.info("Starting successful buyers analysis...")
        
        # Analyze financial aspects
        financial_patterns = self._analyze_financial_background()
        
        # Analyze personal background
        personal_patterns = self._analyze_personal_background()
        
        # Analyze professional background
        professional_patterns = self._analyze_professional_background()
        
        # Combine all analyses
        self.analysis_results = {
            'financial': financial_patterns,
            'personal': personal_patterns,
            'professional': professional_patterns,
            'success_metrics': self._calculate_success_metrics()
        }
        
        return self.analysis_results
    
    def _analyze_financial_background(self):
        """
        Analyzes financial patterns of successful buyers
        """
        financial_data = {}
        
        # Income analysis
        income_data = self.successful_sales['INGRESO MENSUAL'].dropna()
        financial_data['income'] = {
            'average': income_data.mean(),
            'median': income_data.median(),
            'ranges': self._categorize_income_ranges(income_data)
        }
        
        # Payment methods
        if 'TIPO DE CREDITO' in self.successful_sales.columns:
            financial_data['payment_methods'] = (
                self.successful_sales['TIPO DE CREDITO']
                .value_counts(normalize=True)
                .to_dict()
            )
        
        return financial_data
    
    def _analyze_personal_background(self):
        """
        Analyzes personal characteristics of successful buyers
        """
        personal_data = {}
        
        # Age analysis
        if 'EDAD' in self.successful_sales.columns:
            age_data = self.successful_sales['EDAD'].dropna()
            personal_data['age'] = {
                'average_age': age_data.mean(),
                'age_groups': self._categorize_age_groups(age_data)
            }
        
        # Marital status
        if 'ESTADO CIVIL' in self.successful_sales.columns:
            personal_data['marital_status'] = (
                self.successful_sales['ESTADO CIVIL']
                .value_counts(normalize=True)
                .to_dict()
            )
        
        # Education
        if 'ESCOLARIDAD' in self.successful_sales.columns:
            personal_data['education'] = (
                self.successful_sales['ESCOLARIDAD']
                .value_counts(normalize=True)
                .to_dict()
            )
        
        return personal_data
    
    def _analyze_professional_background(self):
        """
        Analyzes professional characteristics of successful buyers
        """
        professional_data = {}
        
        # Job roles analysis
        if 'PUESTO' in self.successful_sales.columns:
            professional_data['job_roles'] = (
                self.successful_sales['PUESTO']
                .value_counts(normalize=True)
                .head(10)  # Top 10 most common roles
                .to_dict()
            )
        
        # Company analysis
        if 'EMPRESA' in self.successful_sales.columns:
            professional_data['company_types'] = (
                self.successful_sales['EMPRESA']
                .value_counts(normalize=True)
                .head(10)  # Top 10 most common employers
                .to_dict()
            )
        
        return professional_data
    
    def _calculate_success_metrics(self):
        """
        Calculates key success metrics
        """
        total_sales = len(self.data)
        successful_sales = len(self.successful_sales)
        
        return {
            'total_sales': total_sales,
            'successful_sales': successful_sales,
            'success_rate': (successful_sales / total_sales) * 100
        }
    
    def _categorize_income_ranges(self, income_data):
        """
        Categorizes income into meaningful ranges
        """
        # Define income brackets (in MXN)
        brackets = [
            (0, 30000, 'Entry Level'),
            (30001, 60000, 'Mid Level'),
            (60001, 100000, 'High Level'),
            (100001, float('inf'), 'Premium')
        ]
        
        income_categories = {}
        for min_val, max_val, label in brackets:
            mask = (income_data >= min_val) & (income_data <= max_val)
            count = mask.sum()
            income_categories[label] = {
                'count': count,
                'percentage': (count / len(income_data)) * 100
            }
        
        return income_categories
    
    def _categorize_age_groups(self, age_data):
        """
        Categorizes ages into meaningful groups
        """
        # Define age groups
        age_groups = [
            (0, 30, 'Young Adult'),
            (31, 45, 'Mid Adult'),
            (46, 60, 'Mature Adult'),
            (61, float('inf'), 'Senior')
        ]
        
        age_categories = {}
        for min_age, max_age, label in age_groups:
            mask = (age_data >= min_age) & (age_data <= max_age)
            count = mask.sum()
            age_categories[label] = {
                'count': count,
                'percentage': (count / len(age_data)) * 100
            }
        
        return age_categories
    
    def create_visualizations(self):
        """
        Creates and saves all visualizations
        """
        self.logger.info("Creating visualizations...")
        
        # Income distribution
        self._create_income_plot()
        
        # Age distribution
        self._create_age_plot()
        
        # Education distribution
        self._create_education_plot()
        
        # Payment methods
        self._create_payment_methods_plot()
    
    def _create_income_plot(self):
        """Creates income distribution plot"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.successful_sales, x='INGRESO MENSUAL', bins=30)
        plt.title('Income Distribution of Successful Buyers')
        plt.xlabel('Monthly Income (MXN)')
        plt.ylabel('Count')
        plt.savefig(f'{self.plot_dir}/income_distribution.png')
        plt.close()
    
    def _create_age_plot(self):
        """Creates age distribution plot"""
        if 'EDAD' in self.successful_sales.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.successful_sales, x='EDAD', bins=20)
            plt.title('Age Distribution of Successful Buyers')
            plt.xlabel('Age')
            plt.ylabel('Count')
            plt.savefig(f'{self.plot_dir}/age_distribution.png')
            plt.close()
    
    def _create_education_plot(self):
        """Creates education distribution plot"""
        if 'ESCOLARIDAD' in self.successful_sales.columns:
            plt.figure(figsize=(12, 6))
            education_counts = self.successful_sales['ESCOLARIDAD'].value_counts()
            education_counts.plot(kind='bar')
            plt.title('Education Levels of Successful Buyers')
            plt.xlabel('Education Level')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{self.plot_dir}/education_distribution.png')
            plt.close()
    
    def _create_payment_methods_plot(self):
        """Creates payment methods distribution plot"""
        if 'TIPO DE CREDITO' in self.successful_sales.columns:
            plt.figure(figsize=(10, 6))
            payment_counts = self.successful_sales['TIPO DE CREDITO'].value_counts()
            plt.pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%')
            plt.title('Payment Methods Distribution')
            plt.savefig(f'{self.plot_dir}/payment_methods.png')
            plt.close()
    
    def generate_report(self):
        """
        Generates a comprehensive analysis report
        """
        if not self.analysis_results:
            self.analyze_successful_buyers()
        
        report = {
            'summary': {
                'total_analyzed': len(self.successful_sales),
                'success_rate': self.analysis_results['success_metrics']['success_rate']
            },
            'key_findings': {
                'financial': self._summarize_financial_findings(),
                'personal': self._summarize_personal_findings(),
                'professional': self._summarize_professional_findings()
            },
            'visualization_paths': self._get_visualization_paths()
        }
        
        return report
    
    def _summarize_financial_findings(self):
        """Summarizes key financial findings"""
        financial = self.analysis_results.get('financial', {})
        return {
            'average_income': financial.get('income', {}).get('average'),
            'common_payment_method': max(
                financial.get('payment_methods', {}).items(),
                key=lambda x: x[1],
                default=('Unknown', 0)
            )[0]
        }
    
    def _summarize_personal_findings(self):
        """Summarizes key personal findings"""
        personal = self.analysis_results.get('personal', {})
        return {
            'average_age': personal.get('age', {}).get('average_age'),
            'most_common_education': max(
                personal.get('education', {}).items(),
                key=lambda x: x[1],
                default=('Unknown', 0)
            )[0]
        }
    
    def _summarize_professional_findings(self):
        """Summarizes key professional findings"""
        professional = self.analysis_results.get('professional', {})
        return {
            'top_job_role': max(
                professional.get('job_roles', {}).items(),
                key=lambda x: x[1],
                default=('Unknown', 0)
            )[0]
        }
    
    def _get_visualization_paths(self):
        """Returns paths to all generated visualizations"""
        return {
            'plots': [f for f in os.listdir(self.plot_dir) if f.endswith('.png')]
        }

def main():
    """
    Main execution function that orchestrates the entire analysis process
    """
    try:
        # First, get cleaned data from data_cleaning
        from data_cleaning import BuyerPersonaAnalysis
        
        # Use raw string for file path to handle backslashes properly
        file_path = r"C:\Users\L03140374\Desktop\buyer_persona_analysis\data\Viña Real - Analisis de Ventas UTF-8.csv"
        
        print("\nStarting Buyer Persona Analysis...")
        print("-" * 50)
        
        # Initialize and run data cleaning
        print("\n1. Loading and cleaning data...")
        cleaner = BuyerPersonaAnalysis(file_path)
        if not cleaner.load_data():
            print("Failed to load the data file")
            return None
            
        # Clean the data
        cleaned_data = cleaner.clean_data()
        
        if cleaned_data is not None:
            print(f"Successfully cleaned data. Shape: {cleaned_data.shape}")
            
            try:
                # Initialize enhanced analysis
                print("\n2. Initializing enhanced analysis...")
                analysis = EnhancedBuyerAnalysis(cleaned_data)
                
                # Perform analysis
                print("\n3. Analyzing successful buyers...")
                analysis.analyze_successful_buyers()
                
                # Create visualizations
                print("\n4. Generating visualizations...")
                analysis.create_visualizations()
                
                # Generate report
                print("\n5. Generating final report...")
                report = analysis.generate_report()
                
                print("\nAnalysis completed successfully!")
                print(f"Results saved in: {analysis.plot_dir}")
                
                # Print key findings
                print("\nKey Findings:")
                print("-" * 50)
                print(f"Total successful sales analyzed: {report['summary']['total_analyzed']}")
                print(f"Success rate: {report['summary']['success_rate']:.1f}%")
                
                return analysis, report
                
            except Exception as e:
                print(f"\nError during analysis: {str(e)}")
                import traceback
                print("\nError details:")
                print(traceback.format_exc())
                return None
        else:
            print("Could not obtain cleaned data - cleaning process returned None")
            return None
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        print("Error details:", e.__class__.__name__)
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True  # Prevents duplicate logging
    )
    
    # Run the analysis
    results = main()
    
    # Check if analysis was successful
    if results is not None:
        analysis, report = results
        print("\nAnalysis completed successfully.")
        print("Check the 'buyer_persona_plots' directory for visualizations.")
    else:
        print("\nAnalysis failed. Please check the error messages above.")