from train import load_and_prepare_data, train_and_evaluate_models, save_model
from visualize import visualize_results
from sklearn.model_selection import train_test_split

# Main Execution
if __name__ == "__main__":
    # Load and prepare data
    X, y = load_and_prepare_data("../collect/coordinate.csv")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1234
    )
    
    # Train and evaluate models
    fit_models, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Save models and get best model
    best_model_name, best_model = save_model(fit_models, results)
    
    # Visualize results
    visualize_results(results, best_model_name, X_test, y_test, best_model)