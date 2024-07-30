from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import create_dummy_vars
from src.models.train_model import linear_regression, decision_tree, random_forest_regressor
from src.models.predict_model import evaluate_model, dt_evaluate_model, rf_evaluate_model


if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/final.csv"
    df = load_and_preprocess_data(data_path)
    x, y = create_dummy_vars(df)
    
    lrmodel, y_train, x_train,x_test,y_test = linear_regression(x,y)
    
    train_mae, test_mae = evaluate_model(lrmodel, y_train, x_train,x_test,y_test)
    
    print('Train error is', train_mae)
    print('Test error is', test_mae)
    
    dtmodel, x_test, y_test,x_train, y_train = decision_tree(x,y)
    
    dt_test_mae,dt_train_mae = dt_evaluate_model (dtmodel, x_test, y_test,x_train, y_train)
    
    print('Train error is', dt_train_mae)
    print('Test error is', dt_test_mae)
    
    rfmodel, x_train, x_test,y_test = random_forest_regressor(x,y)
    
    rf_test_mae = rf_evaluate_model (rfmodel, x_train, x_test,y_test)
    
    print('Test error is', rf_test_mae)