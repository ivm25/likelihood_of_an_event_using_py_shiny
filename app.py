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

# Styles

plt.style.use('ggplot')    

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
             'sex',
             'cp',
             'trestbps'
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


glossary = {
    "Age": "Age of the person",
    "sex": "Sex of the person (1 = male; 0 = female)",
    "cp": "chest pain type",
       "Value 1": "typical angina",
       "Value 2": "atypical angina",
       "Value 3": "non-anginal pain",
       "Value 4": "asymptomatic",
    "trestbps": " resting blood pressure (in mm Hg on admission to the hospital",
    "chol": " serum cholestoral in mg/dl",
    "fbs": "(fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)",
    "restecg": "resting electrocardiographic results",
    "thalach": "maximum heart rate achieved",
    "exang": "exercise induced angina (1 = yes; 0 = no)",
    "oldpeak": " ST depression induced by exercise relative to rest",
    "slope": "the slope of the peak exercise ST segment",
    "ca": "number of major vessels (0-3) colored by flourosopy",
    "thal": "3 = normal; 6 = fixed defect; 7 = reversable defect",
    "present": "Presence of Heart Disease"
    # Add more terms as needed
}

# User interface (UI) definition


app_ui = ui.page_fluid(
    
    ui.page_navbar(
                    title = ui.tags.div(ui.img(src = 'heart.jpeg',
                                               height = '50px',
                                               style = 'margin:5px;'),
                                        ui.h4("Understanding Heart Disease"))
                 ),
            
    ui.navset_card_pill(ui.nav_panel("Understanding the data and its terminologies",  
    ui.layout_sidebar(
    
    ui.sidebar( # A select input for choosing the variable to plot
 
    ),
    # Add a title to the page with some top padding
    ui.panel_title(ui.h2("Meaning of various columns", class_="pt-5")),
    # A container for plot output
    ui.output_ui("glossary_content"),
   
    )
    ),ui.nav_panel("Distributions of various factors",  
    ui.layout_sidebar(
    
    ui.sidebar( # A select input for choosing the variable to plot
    ui.input_select(
        "var", "Select variable", choices= list(heart.columns),
        selected = 'age'
    ),),
    # Add a title to the page with some top padding
    ui.panel_title(ui.h2("Distribution numbers of various factors", class_="pt-5")),
    # A container for plot output
    ui.output_plot("hist"),
   
    )
    ),ui.nav_panel("Presence of Heart Disease",  
    ui.layout_sidebar(
    
    ui.sidebar( # A select input for choosing the variable to plot
    ui.input_select(
        "var_2", "Select variable", choices= list(heart.columns),
        selected = 'age'
    ),),
    # Add a title to the page with some top padding
    ui.panel_title(ui.h2("Presence of Heart disease for Males and females fitted as a Logit function", class_="pt-5")),
    # A container for plot output
    ui.output_plot("logit_plot"),

    )
    ),ui.nav_panel("Odds of Heart Disease",  
    ui.layout_sidebar(
    
    ui.sidebar( # A select input for choosing the variable to plot
    # ui.input_select(
    #     "var_3", "Select variable", choices= list(heart.columns),
    #     selected = 'age'
    # )
    ),
    # Add a title to the page with some top padding
    ui.panel_title(ui.h2("Odds", class_="pt-5")),
    # A container for plot output
    ui.output_table("Odds_summary"),

    )
    )


    # , ui.nav_panel("bayesian posterior probabilities",  
    # ui.layout_sidebar(
    
    # ui.sidebar( # A select input for choosing the variable to plot
    # ui.input_select(
    #     "var_3", "Select variable", choices= list(heart.columns),
    #     selected = 'age'
    # ),),
    # # Add a title to the page with some top padding
    # ui.panel_title(ui.h2("Posterior Probabilities", class_="pt-5")),
    # # A container for plot output
    # ui.output_plot("posteriors"),

    # )
    # )
    , ui.nav_panel("roc-auc",  
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
    ), theme = shinyswatch.theme.journal(),
)


# Server function provides access to client-side input values
def server(input, output, session: Session):
    @render.plot
    def hist():
        # Histogram of the selected variable (input.var())
        p = sns.displot(heart, x = input.var(),
                        
                        col = 'present',
                        hue = 'present',
                        kde = True,
                          edgecolor="black")
        return p.set(xlabel=input.var())

    @render.plot
    def logit_plot():
        # Histogram of the selected variable (input.var())
        p = sns.lmplot(x = input.var_2(),
                       y = 'present',
                       data = heart,
                       col = 'sex',
                       hue = 'sex',
                       logistic=True
                         )
        
            
        return p.set(xlabel=input.var_2(),
                     ylabel = 'Presence of Heart Disease',
                     )
    

    @render.table
    def Odds_summary():
        # Histogram of the selected variable (input.var())
        p = pd.DataFrame(odds(logit_model,
                 predictors,
                 y))
        
            
        return p
    

    # @render.plot
    # def posteriors():
        
    #     z = az.plot_forest(bayesian_model
    #            )
    #     return z
    

    @render.plot
    def roc_plot():

        probs = logit_model.predict_proba(x_test)[:,1]

        auc = roc_auc_score(y_test, probs)
        # Calculate metrics for the roc curve
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        
        g = plt.figure()
        plt.style.use('ggplot')
        plt.figure(figsize = (8, 8))
        
        # Plot the roc curve
        plt.plot(fpr, tpr, 'b')
        plt.xlabel('False Positive Rate', size = 16)
        plt.ylabel('True Positive Rate', size = 16)
        plt.title('Receiver Operating Characteristic Curve, AUC = %0.4f' % auc, 
                    size = 18)
        
        
    @output
    @render.ui
    def glossary_content():
        items = [ui.tags.li(ui.tags.strong(term), ": ", description) for term, description in glossary.items()]
        return ui.tags.ul(*items)

www_dir = Path(__file__).parent /"www"
app = App(app_ui, server,static_assets=www_dir)

