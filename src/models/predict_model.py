from sklearn.metrics import mean_absolute_error


def evaluate_model(lrmodel, y_train, x_train,x_test,y_test):
    train_pred = lrmodel.predict(x_train)
    train_mae = mean_absolute_error(train_pred, y_train)
    ypred = lrmodel.predict(x_test)
    test_mae = mean_absolute_error(ypred, y_test)
    
    return train_mae, test_mae

def dt_evaluate_model(dtmodel, x_test, y_test,x_train, y_train):
    
     ytest_pred = dtmodel.predict(x_test)
     dt_test_mae = mean_absolute_error(ytest_pred, y_test)
     ytrain_pred = dtmodel.predict(x_train)
     dt_train_mae = mean_absolute_error(ytrain_pred, y_train)
     
     return dt_test_mae,dt_train_mae
 
def rf_evaluate_model (rfmodel, x_train, x_test,y_test):
    
    ytest_pred = rfmodel.predict(x_test)
    
    rf_test_mae = mean_absolute_error(ytest_pred, y_test)
    
    return rf_test_mae