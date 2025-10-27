# Patient Risk Prediction for Diabetes
# Machine Learning Project

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class DiabetesRiskPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, filepath='diabetes.csv'):
        """Load the diabetes dataset"""
        try:
            self.data = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"Error: {filepath} not found. Please upload the dataset.")
            return None
            
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        # Separate features and target
        X = self.data.drop('Outcome', axis=1)
        y = self.data['Outcome']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        print("Model trained successfully!")
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance"""
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy, y_pred
    
    def save_results(self, accuracy, filepath='results.txt'):
        """Save results to file"""
        with open(filepath, 'w') as f:
            f.write(f"Diabetes Risk Prediction Model Results\n")
            f.write(f"Model Accuracy: {accuracy:.4f}\n")
        print(f"\nResults saved to {filepath}")

def main():
    print("=" * 50)
    print("Patient Risk Prediction System for Diabetes")
    print("=" * 50)
    
    # Initialize predictor
    predictor = DiabetesRiskPredictor()
    
    # Load data
    data = predictor.load_data('diabetes.csv')
    if data is None:
        return
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = predictor.preprocess_data()
    
    # Train model
    print("\nTraining model...")
    predictor.train_model(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy, predictions = predictor.evaluate_model(X_test, y_test)
    
    # Save results
    predictor.save_results(accuracy)
    
    print("\n" + "=" * 50)
    print("Process completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
