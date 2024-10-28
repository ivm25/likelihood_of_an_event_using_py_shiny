import pandas as pd
import numpy as np
import sklearn

import matplotlib.pyplot as plt
import seaborn as sns

# Standard machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Scikit-learn utilities
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import mlflow
import mlflow.sklearn
import mlflow.xgboost


# PyMC3 for Bayesian Inference
import pymc as pm

import arviz as az
import bambi as bmb

from xgboost import XGBClassifier
import joblib

# Read the data



heart = pd.read_csv('data/heart_disease.csv')

heart.info()


# Replacing any special characters
heart['ca'] = heart['ca'].replace('?', 0)

# Converting column ['ca'] to numeric column

heart['ca'] = pd.to_numeric(heart['ca'])

# Checking potential predictors
grouped_data = heart.groupby("present").agg(
            {
        "age": "mean",
        "sex": "mean",
        "cp": "mean",
        "trestbps": "mean",
        "chol": "mean",
        "fbs": "mean",
        "restecg": "mean",
        "thalach": "mean",
        "exang": "mean",
        "oldpeak": "mean",
        "slope": "mean",
        "ca": "mean"
       
    }
)
     
def process_data(df):

    data_encoded = pd.get_dummies(df, columns = ['sex',
                                                 'cp',
                                                 'fbs',
                                                 'restecg',
                                                 'exang',
                                                 'slope',
                                                 'ca',
                                                 'thal'
                                                 ],
                                                 drop_first = True)
    
    # Define the target and predictors
    target = data_encoded['present']

    predictors = data_encoded.drop('present', axis = 1)

    # standardising the numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(predictors)
    # Convert the scaled data back to a DataFrame and assign the original column names
    X_scaled_df = pd.DataFrame(X_scaled, columns=predictors.columns)
    
    return X_scaled_df, target



def split_data(X, y, test_size=0.3, random_state=42):
    """
    Split the data into training and testing sets.
    
    Parameters:
    X (DataFrame): Features.
    y (Series): Target variable.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    X_train, X_test, y_train, y_test: Split data.
    """
    return train_test_split(X,
                             y,
                               test_size=test_size,
                                 random_state=random_state)




# Creating the model

def build_and_train_model(x_train, y_train):
    """
    Build and train the logistic regression model.
    
    Parameters:
    X_train (DataFrame): Training features.
    y_train (Series): Training target variable.
    
    Returns:
    model (LogisticRegression): Trained logistic regression model.
    """
   


    model = LogisticRegression()
    model.fit(x_train, y_train)
    joblib.dump(model, "output/logistic_model.pkl") 
    return model


def cross_val_log_regression(x_train, y_train):
    """
    Build and train the logistic regression model.
    
    Parameters:
    X_train (DataFrame): Training features.
    y_train (Series): Training target variable.
    
    Returns:
    model (LogisticRegression): Trained logistic regression model.
    """
    # Define the parameter grid
    param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
                    }
    
    grid_search = GridSearchCV(LogisticRegression(),
                                param_grid, 
                                cv=5, 
                                scoring='accuracy')
    
    
    grid_search.fit(x_train, y_train)
    

    # Log best parameters and score

    best_score = grid_search.best_score_

    best_parameters = grid_search.best_params_

    # save the best Model
    best_model = grid_search.best_estimator_


    print(f"Best Parameters: {best_parameters}")
    
    print(f"Best Model: {best_model}")

    print(f"Best Score: {best_score}")

   
    return best_model


def mlflow_log_regression(x_train, y_train):
    """
    Build and train the logistic regression model.
    
    Parameters:
    X_train (DataFrame): Training features.
    y_train (Series): Training target variable.
    
    Returns:
    model (LogisticRegression): Trained logistic regression model.
    """
    # Define the parameter grid
    param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
                    }
    
    grid_search = GridSearchCV(LogisticRegression(),
                                param_grid, 
                                cv=5, 
                                scoring='accuracy')
    
    with mlflow.start_run():
        grid_search.fit(x_train, y_train)
        

        # Log best parameters and score
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_score",
                          grid_search.best_score_)
        
        # Log the best Model
        best_model = grid_search.best_estimator_
        mlflow.sklearn.log_model(best_model, "best_model")

        print(f"Best Parameters: {grid_search.best_params_}")
        

   
    return best_model



def build_and_train_xgboost_model(x_train, y_train):
    """
    Build and train the xgboost classification model.
    
    Parameters:
    X_train (DataFrame): Training features.
    y_train (Series): Training target variable.
    
    Returns:
    model (xgboost): Trained xgboost classification model.
    """

    model = XGBClassifier()
    model.fit(x_train, y_train)
    joblib.dump(model, "output/xgboost_model.pkl")
    return model


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model.
    
    Parameters:
    model : Trained model.
    X_test (DataFrame): Testing features.
    y_test (Series): Testing target variable.
    
    Returns:
    accuracy (float): Accuracy of the model.
    roc_auc (float): ROC-AUC score of the model.
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, roc_auc




def calc_roc(model, x_test, y_test):
  # Calculate the area under the roc curve

  probs = model.predict_proba(x_test)[:,1]

  auc = roc_auc_score(y_test, probs)
  # Calculate metrics for the roc curve
  fpr, tpr, thresholds = roc_curve(y_test, probs)
  
  g = plt.figure()
  plt.style.use('bmh')
  plt.figure(figsize = (8, 8))
  
  # Plot the roc curve
  plt.plot(fpr, tpr, 'b')
  plt.xlabel('False Positive Rate', size = 16)
  plt.ylabel('True Positive Rate', size = 16)
  plt.title('Receiver Operating Characteristic Curve, AUC = %0.4f' % auc, 
            size = 18)
  
  return g


def senstivity_analysis(model, y_test, x_test):

    preds = model.predict(x_test)

    tp = sum((preds == 1) & (y_test == 1))
    fp = sum((preds == 1) & (y_test == 0))
    tn = sum((preds == 0) & (y_test == 0))
    fn = sum((preds == 0) & (y_test == 1))
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)

    return sens, spec



def coefs(model):

    # coefs = m.coef_


    c = ["age", 
        "thalach",
        "restecg",
        "ca",
        #  'sex',
        # 'cp',
        # 'trestbps'
       ]
    

# Checking in terms of log-odds 
    coef_values = []
    for coef, val  in zip(c, model.coef_[0]):
        print(coef, val)
        s = coef, ":", round(val, 2)
        coef_values.append(s)
        return coef_values
    
    
def odds(model, predictors, outcome):

    all_odds =[]

    cf = model.coef_

    for z in cf[0]:
        log_odds = z

        o = np.exp(log_odds)
        all_odds.append(o)

    
    key_indicators = pd.DataFrame(predictors.columns)

    key_indicators['odds'] = all_odds

    key_indicators.rename(columns = {0:"key_indicators"},
                          inplace = True)
    

    key_indicators_sorted = key_indicators.sort_values(by = 'odds',
                                                        ascending = False)
    

    key_indicators_sorted['percentage_effect'] = (key_indicators_sorted['odds'])

    return key_indicators_sorted





# Building the bayesian model
def bayesian_model():

    heart_model = bmb.Model("present ~ age + thalach + restecg + ca",
                        heart, family="bernoulli")

    fitted_model = heart_model.fit()

    az.to_netcdf(fitted_model, "output/heart_model.nc")

    return fitted_model

