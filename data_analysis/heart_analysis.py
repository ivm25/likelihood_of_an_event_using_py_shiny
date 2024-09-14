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
     
# Some columns have a small, 
# but noticeable difference when 
# stratified by predictors.
#  Based on the differences and some 
# knowledge about heart disease,
#  these seem like good candidates for predictors:

# - age
# - thalach (maximum heart rate achieved)
# - restecg (resting ECG)
# - ca (number of vessels colored by fluoroscopy)

# predictor_columns = ['age',
#              'thalach',
#              'restecg',
#              'ca',
       
#             ]


# predictors = heart[predictor_columns]



# outcome_var = heart['present']

# Divide the data into train test datasets

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
    return train_test_split(X, y, test_size=test_size, random_state=random_state)




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

