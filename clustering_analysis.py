# clustering_analysis.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

class BuyerPersonaClustering:
    """
    Enhanced clustering analysis for buyer personas with focus on successful sales
    """
    def __init__(self, df):
        self.data = df
        self.successful_sales = df[df['STATUS'] == 'ESCRITURADA'].copy()
        self.features = None
        self.cluster_labels = None
        self.preprocessor = None
        
        # Define feature groups
        # Define expected feature names
        self.numerical_features = ['INGRESO MENSUAL', 'EDAD']  # Note: 'Edad' instead of 'EDAD'
        self.categorical_features = [
            'SEXO', 'ESTADO CIVIL', 'REGIMEN C', 'ESTADO', 'MUNICIPIO',
            'ESCOLARIDAD', 'PUESTO', 'MEDIO DE CONTACTO'
        ]
        
        # Verify columns exist
        self._verify_columns()
    
    def _verify_columns(self):
        """
        Verify that required columns exist in the dataset
        """
        all_features = self.numerical_features + self.categorical_features
        missing_columns = [col for col in all_features if col not in self.data.columns]
        
        if missing_columns:
            print("\nWarning: Missing columns in dataset:")
            for col in missing_columns:
                print(f"- {col}")
            print("\nAvailable columns:")
            print(self.data.columns.tolist())
            raise ValueError("Missing required columns in dataset")

    def prepare_features(self):
        """
        Prepare features for clustering with enhanced preprocessing
        """
        print("\nPreparing features for clustering analysis...")
        
        try:
            # Create preprocessing pipeline
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numerical_features),
                    ('cat', categorical_transformer, self.categorical_features)
                ])
            
            # Fit and transform the data
            self.features = self.preprocessor.fit_transform(self.successful_sales)
            print("Features prepared successfully.")
            return True
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            return False
    
    def perform_clustering(self, max_clusters=10):
        """
        Perform clustering analysis and determine optimal number of clusters
        """
        print("\nPerforming clustering analysis...")
        
        # Find optimal number of clusters
        silhouette_scores = []
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.features)
            score = silhouette_score(self.features, cluster_labels)
            silhouette_scores.append(score)
        
        # Get optimal number of clusters
        optimal_clusters = np.argmax(silhouette_scores) + 2
        print(f"\nOptimal number of clusters: {optimal_clusters}")
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        self.cluster_labels = kmeans.fit_predict(self.features)
        
        # Add cluster labels to successful sales data
        self.successful_sales['Cluster'] = self.cluster_labels
        
        return optimal_clusters
    
    def analyze_personas(self):
        """
        Generate detailed buyer personas from clusters
        """
        print("\nAnalyzing buyer personas...")
        
        personas = []
        for cluster_id in range(len(np.unique(self.cluster_labels))):
            cluster_data = self.successful_sales[self.successful_sales['Cluster'] == cluster_id]
            
            # Calculate cluster size and conversion rate
            cluster_size = len(cluster_data)
            conversion_rate = (cluster_size / len(self.successful_sales)) * 100
            
            persona = {
                'cluster_id': cluster_id,
                'size': cluster_size,
                'conversion_rate': conversion_rate,
                'demographics': self._analyze_demographics(cluster_data),
                'professional': self._analyze_professional(cluster_data),
                'financial': self._analyze_financial(cluster_data),
                'contact_preferences': self._analyze_contact_preferences(cluster_data)
            }
            
            personas.append(persona)
        
        return personas
    
    def _analyze_demographics(self, cluster_data):
        """
        Analyze demographic characteristics of a cluster
        """
        return {
            'gender_distribution': cluster_data['SEXO'].value_counts(normalize=True).to_dict(),
            'age_stats': {
                'mean': cluster_data['EDAD'].mean(),
                'range': (cluster_data['EDAD'].min(), cluster_data['EDAD'].max())
            },
            'marital_status': {
                'status': cluster_data['ESTADO CIVIL'].value_counts(normalize=True).to_dict(),
                'regime': cluster_data['REGIMEN C'].value_counts(normalize=True).to_dict()
            },
            'location': {
                'state': cluster_data['ESTADO'].value_counts(normalize=True).to_dict(),
                'municipality': cluster_data['MUNICIPIO'].value_counts(normalize=True).to_dict()
            }
        }
    
    def _analyze_professional(self, cluster_data):
        """
        Analyze professional characteristics of a cluster
        """
        return {
            'education': cluster_data['ESCOLARIDAD'].value_counts(normalize=True).to_dict(),
            'job_positions': cluster_data['PUESTO'].value_counts(normalize=True).head(5).to_dict()
        }
    
    def _analyze_financial(self, cluster_data):
        """
        Analyze financial characteristics of a cluster
        """
        return {
            'income_stats': {
                'mean': cluster_data['INGRESO MENSUAL'].mean(),
                'median': cluster_data['INGRESO MENSUAL'].median(),
                'range': (cluster_data['INGRESO MENSUAL'].min(), 
                         cluster_data['INGRESO MENSUAL'].max())
            }
        }
    
    def _analyze_contact_preferences(self, cluster_data):
        """
        Analyze contact channel preferences of a cluster
        """
        return {
            'preferred_channels': cluster_data['MEDIO DE CONTACTO']
                                .value_counts(normalize=True).to_dict()
        }
    
    def visualize_clusters(self):
        """
        Create visualizations of cluster characteristics
        """
        self._plot_cluster_sizes()
        self._plot_income_distribution()
        self._plot_age_distribution()
        self._plot_contact_channels()
    
    def _plot_cluster_sizes(self):
        """Plot cluster size distribution"""
        plt.figure(figsize=(10, 6))
        cluster_sizes = self.successful_sales['Cluster'].value_counts()
        cluster_sizes.plot(kind='bar')
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Buyers')
        plt.tight_layout()
        plt.savefig('cluster_sizes.png')
        plt.close()
    
    def _plot_income_distribution(self):
        """Plot income distribution by cluster"""
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.successful_sales, x='Cluster', y='INGRESO MENSUAL')
        plt.title('Income Distribution by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Monthly Income (MXN)')
        plt.tight_layout()
        plt.savefig('cluster_income.png')
        plt.close()
    
    def _plot_age_distribution(self):
        """Plot age distribution by cluster"""
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.successful_sales, x='Cluster', y='EDAD')
        plt.title('Age Distribution by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Age')
        plt.tight_layout()
        plt.savefig('cluster_age.png')
        plt.close()
    
    def _plot_contact_channels(self):
        """Plot contact channel preferences by cluster"""
        plt.figure(figsize=(12, 6))
        channel_data = pd.crosstab(
            self.successful_sales['Cluster'],
            self.successful_sales['MEDIO DE CONTACTO'],
            normalize='index'
        ) * 100
        channel_data.plot(kind='bar', stacked=True)
        plt.title('Contact Channel Preferences by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Percentage')
        plt.legend(title='Channel', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig('cluster_channels.png')
        plt.close()

def main():
    """
    Main execution function
    """
    try:
        # Import cleaned data
        file_path = r"C:\Users\L03140374\Desktop\buyer_persona_analysis\data\Viña Real - Analisis de Ventas UTF-8.csv"
        cleaned_df = pd.read_csv(file_path)
        
        # Initialize clustering
        clustering = BuyerPersonaClustering(cleaned_df)
        
        # Prepare features
        if clustering.prepare_features():
            # Perform clustering
            n_clusters = clustering.perform_clustering()
            
            # Generate personas
            personas = clustering.analyze_personas()
            
            # Create visualizations
            clustering.visualize_clusters()
            
            # Print persona profiles
            print("\nBuyer Persona Profiles:")
            print("-" * 50)
            for persona in personas:
                print(f"\nPersona Cluster {persona['cluster_id'] + 1}")
                print(f"Size: {persona['size']} buyers")
                print(f"Conversion Rate: {persona['conversion_rate']:.1f}%")
                print("\nKey Characteristics:")
                print(f"- Demographics: {persona['demographics']}")
                print(f"- Professional: {persona['professional']}")
                print(f"- Financial: {persona['financial']}")
                print(f"- Contact Preferences: {persona['contact_preferences']}")
                print("-" * 50)
            
            return clustering, personas
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    results = main()
    if results is not None:
        print("\nAnalysis completed successfully!")
        print("Check the generated visualization files for detailed insights.")