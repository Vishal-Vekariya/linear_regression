from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle


def linear_regression(x,y):
   
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1234 )
    lrmodel = LinearRegression().fit(x_train,y_train)
    
    with open('models/linear_regression.pkl', 'wb') as f:
        pickle.dump(lrmodel, f)
    return lrmodel, y_train, x_train,x_test,y_test
    
def decision_tree(x,y):
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1234 )
    dt = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=567)
    dtmodel = dt.fit(x_train,y_train)
    
    with open('models/decision_tree.pkl', 'wb') as f:
        pickle.dump(dtmodel, f)
    
    return dtmodel, x_test, y_test,x_train,  y_train

def random_forest_regressor(x,y):
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1234 )
    rf = RandomForestRegressor(n_estimators=200, criterion='absolute_error')
    rfmodel = rf.fit(x_train,y_train)
    
    with open('models/random_forest_regressor.pkl', 'wb') as f:
        pickle.dump(rfmodel, f)
    return rfmodel, x_train, x_test,y_test