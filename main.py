# main.py
# This is our central script that coordinates all analysis steps

from data_cleaning import BuyerPersonaAnalysis
from enhanced_analysis import EnhancedBuyerAnalysis
from clustering_analysis import BuyerPersonaClustering

class AnalysisPipeline:
    """
    Coordinates the entire analysis process from data cleaning through clustering
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.cleaned_data = None
        self.enhanced_analysis = None
        self.clustering_results = None

    def run_complete_analysis(self):
        """
        Executes the full analysis pipeline in the correct order
        """
        print("Starting Complete Analysis Pipeline...")
        
        # Step 1: Data Cleaning
        print("\nStep 1: Cleaning Data...")
        if not self._clean_data():
            return False
            
        # Step 2: Enhanced Analysis
        print("\nStep 2: Performing Enhanced Analysis...")
        if not self._perform_enhanced_analysis():
            return False
            
        # Step 3: Clustering
        print("\nStep 3: Performing Clustering Analysis...")
        if not self._perform_clustering():
            return False
            
        print("\nAnalysis Pipeline Completed Successfully!")
        return True

    def _clean_data(self):
        """
        First step: Clean the raw data
        """
        try:
            cleaner = BuyerPersonaAnalysis(self.file_path)
            self.cleaned_data = cleaner.clean_data()
            if self.cleaned_data is None:
                print("Error: Data cleaning produced no results")
                return False
            print("Data cleaning completed successfully")
            return True
        except Exception as e:
            print(f"Error during data cleaning: {str(e)}")
            return False

    def _perform_enhanced_analysis(self):
        """
        Second step: Perform enhanced analysis on cleaned data
        """
        try:
            self.enhanced_analysis = EnhancedBuyerAnalysis(self.cleaned_data)
            self.enhanced_analysis.generate_comprehensive_profiles()
            self.enhanced_analysis.visualize_results()
            print("Enhanced analysis completed successfully")
            return True
        except Exception as e:
            print(f"Error during enhanced analysis: {str(e)}")
            return False

    def _perform_clustering(self):
        """
        Third step: Perform clustering analysis
        """
        try:
            clustering = BuyerPersonaClustering(self.cleaned_data)
            if clustering.prepare_features():
                self.clustering_results = clustering
                clustering.analyze_clusters()
                print("Clustering analysis completed successfully")
                return True
            return False
        except Exception as e:
            print(f"Error during clustering analysis: {str(e)}")
            return False

    def get_results(self):
        """
        Returns all analysis results in a structured format
        """
        return {
            'cleaned_data': self.cleaned_data,
            'enhanced_analysis': self.enhanced_analysis.analysis_results if self.enhanced_analysis else None,
            'clustering_results': self.clustering_results
        }

def main():
    """
    Main execution function
    """
    # File path for your data
    file_path = r"C:\Users\L03140374\Desktop\buyer_persona_analysis\data\Viña Real - Analisis de Ventas UTF-8.csv"
    
    # Initialize and run pipeline
    pipeline = AnalysisPipeline(file_path)
    
    if pipeline.run_complete_analysis():
        results = pipeline.get_results()
        # Here you can work with your results
        print("\nAnalysis completed successfully!")
    else:
        print("\nAnalysis failed. Please check the error messages above.")

if __name__ == "__main__":
    main()