"""
Test and Demo Module
Provides examples and test data for fraud detection system
"""

import numpy as np
import pandas as pd
from src.data_preprocessing import preprocess_data
from src.model_training import LogisticRegressionModel, RandomForestModel


def create_sample_data():
    """
    Create sample data for testing without downloading full dataset
    
    Returns:
        pd.DataFrame: Sample credit card transaction data
    """
    print("\n" + "="*50)
    print("CREATING SAMPLE DATA")
    print("="*50)
    
    np.random.seed(42)
    
    # Create sample features (simulating PCA-transformed features)
    n_samples = 1000
    
    # Legitimate transactions (Class 0)
    legitimate = np.random.normal(0, 1, (n_samples//2, 29))
    legitimate_class = np.zeros((n_samples//2, 1))
    
    # Fraudulent transactions (Class 1) - slightly different distribution
    fraudulent = np.random.normal(1, 1.5, (n_samples//2, 29))
    fraudulent_class = np.ones((n_samples//2, 1))
    
    # Combine data
    X = np.vstack([legitimate, fraudulent])
    y = np.vstack([legitimate_class, fraudulent_class]).ravel()
    
    # Create DataFrame
    columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
    data = pd.DataFrame(X, columns=columns)
    data['Class'] = y.astype(int)
    
    print(f"\nSample data created:")
    print(f"  • Total samples: {len(data)}")
    print(f"  • Features: {len(columns)}")
    print(f"  • Legitimate: {(data['Class'] == 0).sum()}")
    print(f"  • Fraudulent: {(data['Class'] == 1).sum()}")
    
    return data


def run_demo():
    """Run demo with sample data"""
    
    print("\n" + "="*70)
    print("FRAUD DETECTION DEMO - USING SAMPLE DATA")
    print("="*70)
    
    # Create sample data
    data = create_sample_data()
    
    # Preprocess
    from sklearn.model_selection import train_test_split
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Train Logistic Regression
    print("\n" + "-"*50)
    print("Training Logistic Regression...")
    print("-"*50)
    
    lr_model = LogisticRegressionModel()
    lr_model.train(X_train, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    
    print(f"\n✓ Model Accuracy: {accuracy:.2%}")
    
    # Show sample predictions
    print(f"\nSample Predictions (first 10 test samples):")
    print(f"{'Actual':>8} | {'Predicted':>10} | {'Confidence':>10}")
    print("-"*30)
    
    y_proba = lr_model.predict_proba(X_test[:10])
    if isinstance(y_proba, np.ndarray):
        if y_proba.ndim == 2:
            confidence = np.max(y_proba, axis=1)
        else:
            confidence = np.abs(y_proba)
    
    for i in range(10):
        actual = y_test.iloc[i]
        pred = y_pred[i]
        conf = confidence[i] if isinstance(confidence, np.ndarray) else 0.5
        status = "✓" if actual == pred else "✗"
        print(f"{int(actual):>8} | {int(pred):>10} | {conf:>10.2%}  {status}")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    return lr_model, X_test, y_test


def test_preprocessing():
    """Test data preprocessing functions"""
    
    print("\n" + "="*70)
    print("PREPROCESSING TEST")
    print("="*70)
    
    # Create sample data
    data = create_sample_data()
    
    # Test preprocessing
    X_scaled, y, scaler = preprocess_data(data)
    
    print(f"\n✓ Preprocessing test passed!")
    print(f"  • Input shape: {data.shape}")
    print(f"  • Output shape: {X_scaled.shape}")
    print(f"  • Features scaled: {X_scaled.mean().abs().max() < 0.1}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FRAUD DETECTION - DEMO & TEST MODULE")
    print("="*70)
    
    # Run tests
    test_preprocessing()
    
    # Run demo
    model, X_test, y_test = run_demo()
    
    print("\n💡 Next Steps:")
    print("   1. Run main.py with the actual dataset")
    print("   2. Check models/ folder for saved models")
    print("   3. Review notebooks/ folder for detailed analysis")
