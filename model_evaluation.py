"""
Model Evaluation Script for PRE vs POS Search Method Selection

This script trains and evaluates 4 different models to predict which search method
(PRE or POS) will be faster for a given query, and times their inference speed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


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
    print(f"Features: {feature_cols}")

    return X, y, feature_cols, pivot_df


def split_data(X, y):
    """Split data into train and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print()

    return X_train, X_test, y_train, y_test


def train_simple_rule_based(X_train, y_train, X_test, y_test):
    """Model 1: Simple Rule-Based (Single threshold on num_survivors)"""
    print("=" * 80)
    print("MODEL 1: SIMPLE RULE-BASED (THRESHOLD ON NUM_SURVIVORS)")
    print("=" * 80)

    # Find optimal threshold using training data
    best_threshold = None
    best_accuracy = 0

    for threshold in range(1000, 100000, 1000):
        y_pred = (X_train['num_survivors'] <= threshold).astype(int)
        acc = accuracy_score(y_train, y_pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

    print(f"Optimal threshold: {best_threshold:,}")
    print(f"Training accuracy: {best_accuracy:.4f}")

    # Time inference on test set
    start_time = time.perf_counter()
    y_pred = (X_test['num_survivors'] <= best_threshold).astype(int)
    inference_time = (time.perf_counter() - start_time) * 1000  # ms

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nTest Set Performance:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Inference time: {inference_time:.4f} ms ({len(X_test)} samples)")
    print(f"Avg per sample: {inference_time/len(X_test):.6f} ms")
    print()

    return {
        'name': 'Simple Rule-Based',
        'model': {'threshold': best_threshold},
        'predictions': y_pred,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'inference_time': inference_time
    }


def train_advanced_rule_based(X_train, y_train, X_test, y_test, fallback_threshold):
    """Model 2: Advanced Rule-Based (K-dependent thresholds)"""
    print("=" * 80)
    print("MODEL 2: ADVANCED RULE-BASED (K-DEPENDENT THRESHOLDS)")
    print("=" * 80)

    # Find optimal threshold for each k value
    k_thresholds = {}
    train_df = X_train.copy()
    train_df['y'] = y_train.values

    for k_val in train_df['k'].unique():
        k_data = train_df[train_df['k'] == k_val]
        best_thresh = None
        best_acc = 0

        for threshold in range(1000, 100000, 1000):
            y_pred = (k_data['num_survivors'] <= threshold).astype(int)
            acc = accuracy_score(k_data['y'], y_pred)
            if acc > best_acc:
                best_acc = acc
                best_thresh = threshold

        k_thresholds[k_val] = best_thresh

    print("K-dependent thresholds:")
    for k, thresh in sorted(k_thresholds.items()):
        print(f"  k={k:6.0f} -> threshold={thresh:,}")

    # Time inference on test set
    start_time = time.perf_counter()
    y_pred = X_test.apply(
        lambda row: 1 if row['num_survivors'] <= k_thresholds.get(row['k'], fallback_threshold) else 0,
        axis=1
    )
    inference_time = (time.perf_counter() - start_time) * 1000  # ms

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nTest Set Performance:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Inference time: {inference_time:.4f} ms ({len(X_test)} samples)")
    print(f"Avg per sample: {inference_time/len(X_test):.6f} ms")
    print()

    return {
        'name': 'Advanced Rule-Based (K-dependent)',
        'model': {'k_thresholds': k_thresholds},
        'predictions': y_pred,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'inference_time': inference_time
    }


def train_logistic_regression(X_train, y_train, X_test, y_test, feature_cols):
    """Model 3: Logistic Regression"""
    print("=" * 80)
    print("MODEL 3: LOGISTIC REGRESSION")
    print("=" * 80)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)

    print(f"Logistic Regression Coefficients:")
    for feat, coef in zip(feature_cols, lr_model.coef_[0]):
        print(f"  {feat:15s}: {coef:8.4f}")

    # Time inference on test set
    start_time = time.perf_counter()
    y_pred = lr_model.predict(X_test_scaled)
    inference_time = (time.perf_counter() - start_time) * 1000  # ms

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nTest Set Performance:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Inference time: {inference_time:.4f} ms ({len(X_test)} samples)")
    print(f"Avg per sample: {inference_time/len(X_test):.6f} ms")
    print()

    return {
        'name': 'Logistic Regression',
        'model': {'lr': lr_model, 'scaler': scaler},
        'predictions': y_pred,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'inference_time': inference_time
    }


def train_decision_tree(X_train, y_train, X_test, y_test, feature_cols):
    """Model 4: Decision Tree"""
    print("=" * 80)
    print("MODEL 4: DECISION TREE")
    print("=" * 80)

    # Train model
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=20)
    dt_model.fit(X_train, y_train)

    print(f"Decision Tree Feature Importances:")
    for feat, importance in zip(feature_cols, dt_model.feature_importances_):
        print(f"  {feat:15s}: {importance:.4f}")

    # Time inference on test set
    start_time = time.perf_counter()
    y_pred = dt_model.predict(X_test)
    inference_time = (time.perf_counter() - start_time) * 1000  # ms

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nTest Set Performance:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Inference time: {inference_time:.4f} ms ({len(X_test)} samples)")
    print(f"Avg per sample: {inference_time/len(X_test):.6f} ms")
    print()

    return {
        'name': 'Decision Tree',
        'model': dt_model,
        'predictions': y_pred,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'inference_time': inference_time
    }


def compare_models(models, X_test):
    """Compare all models and select the best one"""
    print("=" * 80)
    print("MODEL COMPARISON - SORTED BY F1 SCORE (DESC) THEN INFERENCE TIME (ASC)")
    print("=" * 80)

    # Create results dataframe
    results = pd.DataFrame({
        'Model': [m['name'] for m in models],
        'Accuracy': [m['accuracy'] for m in models],
        'Precision': [m['precision'] for m in models],
        'Recall': [m['recall'] for m in models],
        'F1 Score': [m['f1'] for m in models],
        'Inference Time (ms)': [m['inference_time'] for m in models],
        'Avg Time per Sample (ms)': [m['inference_time'] / len(X_test) for m in models]
    })

    # Sort by F1 Score (descending) then by inference time (ascending)
    results = results.sort_values(['F1 Score', 'Inference Time (ms)'], ascending=[False, True])
    results = results.reset_index(drop=True)

    print(results.to_string(index=False))
    print("=" * 80)

    return results


def select_best_model(results):
    """Select the best model based on composite score"""
    print("\n" + "=" * 80)
    print("FINAL MODEL SELECTION (Composite Score: 70% F1 + 20% Accuracy + 10% Speed)")
    print("=" * 80)

    # Normalize metrics to 0-1 range
    results['Accuracy_norm'] = results['Accuracy']
    results['F1_norm'] = results['F1 Score']

    # Avoid division by zero
    time_range = results['Inference Time (ms)'].max() - results['Inference Time (ms)'].min()
    if time_range > 0:
        results['Speed_norm'] = 1 - (results['Inference Time (ms)'] - results['Inference Time (ms)'].min()) / time_range
    else:
        results['Speed_norm'] = 1.0

    # Composite score: 70% F1, 20% Accuracy, 10% Speed
    results['Composite_Score'] = (0.7 * results['F1_norm'] +
                                  0.2 * results['Accuracy_norm'] +
                                  0.1 * results['Speed_norm'])

    results_sorted = results.sort_values('Composite_Score', ascending=False)

    print(results_sorted[['Model', 'Accuracy', 'F1 Score', 'Inference Time (ms)', 'Composite_Score']].to_string(index=False))
    print("=" * 80)

    best_model = results_sorted.iloc[0]
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   Accuracy:  {best_model['Accuracy']:.4f}")
    print(f"   Precision: {best_model['Precision']:.4f}")
    print(f"   Recall:    {best_model['Recall']:.4f}")
    print(f"   F1 Score:  {best_model['F1 Score']:.4f}")
    print(f"   Inference Time: {best_model['Inference Time (ms)']:.4f} ms")
    print(f"   Avg per sample: {best_model['Avg Time per Sample (ms)']:.6f} ms")
    print(f"   Composite Score: {best_model['Composite_Score']:.4f}")
    print("=" * 80)

    return results_sorted


def plot_comparison(results):
    """Visualize model comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Accuracy vs Inference Time
    ax1 = axes[0]
    scatter = ax1.scatter(results['Inference Time (ms)'],
                          results['Accuracy'],
                          s=200,
                          alpha=0.6,
                          c=range(len(results)),
                          cmap='viridis')

    for idx, row in results.iterrows():
        ax1.annotate(row['Model'],
                     (row['Inference Time (ms)'], row['Accuracy']),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax1.set_xlabel('Inference Time (ms)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Trade-off: Accuracy vs Inference Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Metric comparison
    ax2 = axes[1]
    x = np.arange(len(results))
    width = 0.2

    bars1 = ax2.bar(x - 1.5*width, results['Accuracy'], width, label='Accuracy', alpha=0.8)
    bars2 = ax2.bar(x - 0.5*width, results['Precision'], width, label='Precision', alpha=0.8)
    bars3 = ax2.bar(x + 0.5*width, results['Recall'], width, label='Recall', alpha=0.8)
    bars4 = ax2.bar(x + 1.5*width, results['F1 Score'], width, label='F1 Score', alpha=0.8)

    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(results['Model'], rotation=45, ha='right')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: model_comparison.png")
    plt.show()


def plot_confusion_matrices(models, y_test):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, model in enumerate(models):
        ax = axes[idx // 2, idx % 2]

        cm = confusion_matrix(y_test, model['predictions'])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['PRE', 'POS'],
                    yticklabels=['PRE', 'POS'])

        ax.set_title(f"{model['name']}\nConfusion Matrix", fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)

        # Add accuracy
        acc = model['accuracy']
        ax.text(0.5, 1.08, f"Accuracy: {acc:.4f}",
                transform=ax.transAxes,
                ha='center',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Saved: confusion_matrices.png")
    plt.show()


def plot_decision_tree(dt_model, feature_cols):
    """Visualize the decision tree"""
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model['model'],
              feature_names=feature_cols,
              class_names=['PRE', 'POS'],
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title('Decision Tree for PRE vs POS Selection', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
    print("Saved: decision_tree.png")
    plt.show()


def main():
    """Main execution function"""
    # Load and prepare data
    X, y, feature_cols, pivot_df = load_and_prepare_data()

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train all models
    model1 = train_simple_rule_based(X_train, y_train, X_test, y_test)
    model2 = train_advanced_rule_based(X_train, y_train, X_test, y_test,
                                       fallback_threshold=model1['model']['threshold'])
    model3 = train_logistic_regression(X_train, y_train, X_test, y_test, feature_cols)
    model4 = train_decision_tree(X_train, y_train, X_test, y_test, feature_cols)

    # Collect all models
    models = [model1, model2, model3, model4]

    # Compare models
    results = compare_models(models, X_test)

    # Select best model
    results_sorted = select_best_model(results)

    # Plot comparisons
    plot_comparison(results_sorted)
    plot_confusion_matrices(models, y_test)
    plot_decision_tree(model4, feature_cols)

    print("\n✅ Model evaluation complete!")


if __name__ == "__main__":
    main()
