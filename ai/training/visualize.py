import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def visualize_results(results, best_model_name, X_test, y_test, best_model):
    """Create visualizations of model performance"""
    
    # 1. Model Comparison Bar Plot
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    test_acc = [results[m]['test_accuracy'] for m in models]
    cv_acc = [results[m]['cv_mean_accuracy'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, test_acc, width, label='Test Accuracy')
    plt.bar(x + width/2, cv_acc, width, label='CV Accuracy')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison: Test vs Cross-Validation Accuracy')
    plt.xticks(x, models, rotation=45)
    plt.ylim(0, 1.1)
    plt.legend()
    
    # Add values on top of bars
    for i, v in enumerate(test_acc):
        plt.text(i - width/2, v + 0.02, f"{v:.3f}", ha='center')
    for i, v in enumerate(cv_acc):
        plt.text(i + width/2, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    # 2. Confusion Matrix for Best Model
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # 3. Feature Importance for Tree-based Models
    if hasattr(best_model.named_steps[best_model.steps[-1][0]], 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        feature_importances = best_model.named_steps[best_model.steps[-1][0]].feature_importances_
        features = X_test.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values('Importance', ascending=False).head(20)
        
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Top 20 Important Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()