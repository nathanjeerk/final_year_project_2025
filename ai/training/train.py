import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle, warnings
from time import time

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 1234
np.random.seed(RANDOM_STATE)

def load_and_prepare_data(filepath):
    """Load and prepare the dataset with proper header handling"""
    try:
        # First try reading with headers
        df = pd.read_csv(filepath)
        
        # Check if the first column is the exercise name
        if df.columns[0] != 'exercise':
            # If not, try reading without headers
            df = pd.read_csv(filepath, header=None)
            df = df.rename(columns={0: 'exercise'})
            
            # The remaining columns are pose coordinates
            # Create proper column names for pose coordinates
            num_pose_cols = len(df.columns) - 1
            if num_pose_cols % 4 == 0:  # 4 values per landmark (x,y,z,visibility)
                num_landmarks = num_pose_cols // 4
                pose_cols = []
                for i in range(num_landmarks):
                    for coord in ['x', 'y', 'z', 'visibility']:
                        pose_cols.append(f'pose_{i}_{coord}')
                df.columns = ['exercise'] + pose_cols
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise

    print("=== Data Summary ===")
    print(f"Total samples: {len(df)}")
    print("Exercise distribution:")
    print(df['exercise'].value_counts())
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Separate features and target
    X = df.drop("exercise", axis=1)
    y = df["exercise"]
    
    # Convert all feature columns to float
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Drop any rows with NaN values that resulted from conversion
    X = X.dropna()
    y = y[X.index]
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    return X, y

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models"""
    
    # Define models with optimized parameters
    pipelines = {
        'Logistic Regression': make_pipeline(
            StandardScaler(), 
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        'Ridge Classifier': make_pipeline(
            StandardScaler(), 
            RidgeClassifier(random_state=RANDOM_STATE)
        ),
        'Random Forest': make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        ),
        'Gradient Boosting': make_pipeline(
            StandardScaler(),
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                random_state=RANDOM_STATE
            )
        )
    }
    
    results = {}
    fit_models = {}
    
    print("\n=== Model Training ===")
    for name, pipeline in pipelines.items():
        start_time = time()
        
        # Training with cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1)
        
        # Final training
        model = pipeline.fit(X_train, y_train)
        fit_models[name] = model
        
        # Evaluation
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        end_time = time()
        duration = end_time - start_time
        
        results[name] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'training_time': duration
        }
        
        print(f"\n{name}:")
        print(f"  Training Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"  Training Time: {duration:.2f}s")
        
        # Classification report
        print("\n  Classification Report:")
        print(classification_report(y_test, test_pred))
    
    return fit_models, results

def save_model(fit_models, results):
    """Save the best model and results"""
    # Find best model based on test accuracy
    best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
    best_model = fit_models[best_model_name]
    
    print(f"\n=== Best Model: {best_model_name} ===")
    print(f"Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
    
    # Save the best model
    with open("best_workout_pose_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    # Save all models
    with open("all_workout_pose_models.pkl", "wb") as f:
        pickle.dump(fit_models, f)
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('model_results.csv')
    
    return best_model_name, best_model