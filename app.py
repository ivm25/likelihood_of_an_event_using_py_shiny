from shiny import App, render, ui, reactive, Session
import shinyswatch
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

from data_analysis.heart_analysis import *
import pymc as pm

import arviz as az
import bambi as bmb

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report



# Read the data
heart = pd.read_csv('data/heart_disease.csv')

# Replacing any special characters
heart['ca'] = heart['ca'].replace('?', 0)

# Converting column ['ca'] to numeric column

heart['ca'] = pd.to_numeric(heart['ca'])

heart = heart.drop(columns = ['Unnamed: 0'])

predictor_columns = ['age',
             'thalach',
             'restecg',
             'ca',
            ]


predictors = heart[predictor_columns]


y = heart['present']


x_train, x_test, y_train, y_test = split_data(predictors, 
                                              y,
                                              test_size=0.2,
                                                random_state=42)



bayesian_model = az.from_netcdf('output/heart_model.nc')


logit_model = joblib.load('output/logistic_model.pkl')

xgboost_model = joblib.load('output/xgboost_model.pkl')




# User interface (UI) definition


app_ui = ui.page_fluid(ui.page_navbar(
                                      title = "Odds of Heart Disease",
                                      ),
            
    ui.navset_card_pill(ui.nav_panel("Distributions",  
    ui.layout_sidebar(
    
    ui.sidebar( # A select input for choosing the variable to plot
    ui.input_select(
        "var", "Select variable", choices= list(heart.columns),
        selected = 'age'
    ),),
    # Add a title to the page with some top padding
    ui.panel_title(ui.h2("Distribution of key factors", class_="pt-5")),
    # A container for plot output
    ui.output_plot("hist"),
   
)
    ),ui.nav_panel("Outliers",  
    ui.layout_sidebar(
    
    ui.sidebar( # A select input for choosing the variable to plot
    ui.input_select(
        "var_2", "Select variable", choices= list(heart.columns),
        selected = 'age'
    ),),
    # Add a title to the page with some top padding
    ui.panel_title(ui.h2("Outliers", class_="pt-5")),
    # A container for plot output
    ui.output_plot("boxplots"),

    )
    ), ui.nav_panel("bayesian posterior probabilities",  
    ui.layout_sidebar(
    
    ui.sidebar( # A select input for choosing the variable to plot
    ui.input_select(
        "var_3", "Select variable", choices= list(heart.columns),
        selected = 'age'
    ),),
    # Add a title to the page with some top padding
    ui.panel_title(ui.h2("Posterior Probabilities", class_="pt-5")),
    # A container for plot output
    ui.output_plot("posteriors"),

    )
    ), ui.nav_panel("roc-auc",  
    ui.layout_sidebar(
    
    ui.sidebar( # A select input for choosing the variable to plot
    ui.input_radio_buttons(
        "model_type", "Select model", choices= [1,
                                                2],
        # selected = logit_model
    ),),
    # Add a title to the page with some top padding
    ui.panel_title(ui.h2("ROC-AUC", class_="pt-5")),
    # A container for plot output
    ui.output_plot("roc_plot"),

    )
    )   
    )
)


# Server function provides access to client-side input values
def server(input, output, session: Session):
    @render.plot
    def hist():
        # Histogram of the selected variable (input.var())
        p = sns.displot(heart, x = input.var(),
                        hue = 'present',
                        kde = True,
                          edgecolor="black")
        return p.set(xlabel=None)

    @render.plot
    def boxplots():
        # Histogram of the selected variable (input.var())
        p = sns.violinplot(heart, 
                        y = input.var_2(),
                        hue = 'present',
                         )
        return p.set(xlabel=None)
    
    @render.plot
    def posteriors():
        
        z = az.plot_forest(bayesian_model
               )
        return z
    

    @render.plot
    def roc_plot():

        probs = logit_model.predict_proba(x_test)[:,1]

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
        
        


app = App(app_ui, server)


