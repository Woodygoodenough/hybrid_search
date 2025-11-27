"""
Train and save the Logistic Regression model for PRE vs POS selection

This script trains a logistic regression model to predict which search method
(PRE or POS) will be faster, and saves it for later use during search.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data():
    """Load data and prepare features and labels"""
    print("=" * 80)
    print("LOADING AND PREPARING DATA")
    print("=" * 80)

    # Load data
    df = pd.read_csv('timed_results.csv')

    # Filter for ADAP methods
    adap_df = df[df['method'].isin(['ADAP_PRE', 'ADAP_POS'])]

    # Pivot to get PRE and POS side by side
    pivot_df = adap_df.pivot_table(
        index=['k', 'num_survivors', 'predicates'],
        columns='method',
        values=['total', 'db_search', 'histo_filter', 'faiss_search',
                'ud_params', 'intersect', 'residual', 'finalize', 'iterations']
    )

    pivot_df.columns = ['_'.join(col).lower() for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    # Create label: 1 if POS is faster, 0 if PRE is faster
    pivot_df['pos_faster'] = (pivot_df['total_adap_pos'] < pivot_df['total_adap_pre']).astype(int)

    # Calculate selectivity
    total_docs = 150000
    pivot_df['selectivity'] = pivot_df['num_survivors'] / total_docs

    # Features for modeling
    feature_cols = ['k', 'num_survivors', 'selectivity']
    X = pivot_df[feature_cols]
    y = pivot_df['pos_faster']

    print(f"Dataset size: {len(X)}")
    print(f"POS faster: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"PRE faster: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")
    print(f"Features: {feature_cols}\n")

    return X, y, feature_cols


def train_logistic_regression(X, y, feature_cols):
    """Train logistic regression model"""
    print("=" * 80)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 80)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)

    print("Model Coefficients:")
    for feat, coef in zip(feature_cols, lr_model.coef_[0]):
        print(f"  {feat:15s}: {coef:8.4f}")
    print(f"  {'Intercept':15s}: {lr_model.intercept_[0]:8.4f}")

    # Evaluate on test set
    y_pred = lr_model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print()

    return lr_model, scaler, feature_cols


def save_model(lr_model, scaler, feature_cols, filepath='lr_model.pkl'):
    """Save the trained model, scaler, and feature columns"""
    model_package = {
        'model': lr_model,
        'scaler': scaler,
        'feature_cols': feature_cols
    }

    with open(filepath, 'wb') as f:
        pickle.dump(model_package, f)

    print(f"✅ Model saved to: {filepath}")
    print(f"   - Logistic Regression model")
    print(f"   - StandardScaler")
    print(f"   - Feature columns: {feature_cols}")


def load_model(filepath='lr_model.pkl'):
    """Load the trained model (for testing purposes)"""
    with open(filepath, 'rb') as f:
        model_package = pickle.load(f)

    print(f"\n✅ Model loaded from: {filepath}")
    print(f"   Features: {model_package['feature_cols']}")

    return model_package


def main():
    """Main execution function"""
    # Load and prepare data
    X, y, feature_cols = load_and_prepare_data()

    # Train model
    lr_model, scaler, feature_cols = train_logistic_regression(X, y, feature_cols)

    # Save model
    save_model(lr_model, scaler, feature_cols, filepath='lr_model.pkl')

    # Test loading (optional verification)
    print("\n" + "=" * 80)
    print("TESTING MODEL LOADING")
    print("=" * 80)
    loaded_model_package = load_model('lr_model.pkl')

    # Test prediction with a sample
    print("\n" + "=" * 80)
    print("TESTING PREDICTION")
    print("=" * 80)
    sample_input = pd.DataFrame({
        'k': [100],
        'num_survivors': [50000],
        'selectivity': [50000 / 150000]
    })

    sample_scaled = loaded_model_package['scaler'].transform(sample_input)
    prediction = loaded_model_package['model'].predict(sample_scaled)[0]
    proba = loaded_model_package['model'].predict_proba(sample_scaled)[0]

    print(f"Sample input: k={sample_input['k'][0]}, num_survivors={sample_input['num_survivors'][0]}")
    print(f"Prediction: {'POS' if prediction == 1 else 'PRE'}")
    print(f"Probability: PRE={proba[0]:.4f}, POS={proba[1]:.4f}")

    print("\n" + "=" * 80)
    print("✅ MODEL TRAINING AND SAVING COMPLETE!")
    print("=" * 80)
    print("\nYou can now use 'lr_model.pkl' in your search.py")
    print("Use lr_based_adap_search() to automatically select PRE or POS")


if __name__ == "__main__":
    main()
