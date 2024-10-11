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



from plotnine import *
from plotnine import aes, ggplot
from plotnine.geoms import geom_col, geom_line
from plotnine.geoms.geom_point import geom_point
from plotnine.geoms.geom_smooth import geom_smooth



# Styles

plt.style.use('ggplot')    

# Read the data
heart = pd.read_csv('data/heart_disease.csv')

# Replacing any special characters
heart['ca'] = heart['ca'].replace('?', 0)

# Converting column ['ca'] to numeric column

heart['ca'] = pd.to_numeric(heart['ca'])

heart = heart.drop(columns = ['Unnamed: 0'])



predictors = process_data(heart)[0]

y = process_data(heart)[1]

x_train, x_test, y_train, y_test = split_data(predictors, 
                                              y,
                                              test_size=0.2,
                                                random_state=42)


mfw_output = mlflow_log_regression(x_train, y_train)
# heart_model = bmb.Model("present ~ age + thalach + restecg + ca",
#                         heart, family="bernoulli")

bayesian_model = az.from_netcdf('output/heart_model.nc')


# age_grid = np.linspace(heart['age'].min(), heart['age'].max(), 100)


# # Create a DataFrame to hold the predictions
# pred_df = pd.DataFrame({
#     'age': np.tile(age_grid, len(heart['ca'].unique())),
#     'ca': np.repeat(heart['ca'].unique(), len(age_grid)),
#     'thalach': heart['thalach'].mean(),  # Assuming mean value for other predictors
#     'restecg': heart['restecg'].mode()[0] 
# })


# # Generate predictions
# predictions = heart_model.predict(bayesian_model, data = pred_df,
#                                   )

# # Add predictions to the DataFrame
# pred_df['probability'] = predictions
# presence_posterior = az.extract_dataset(bayesian_model, num_samples = 100)["p"]




# _, ax = plt.subplots(figsize=(7, 5))

# for c in heart['ca'].unique():
#     # Which rows in new_data correspond to party?
#     idx = pred_df.index[pred_df["ca"] == c].tolist()
#     print(idx)
#     ax.plot(pred_df.loc[idx, 'age'], presence_posterior[idx][0], alpha=0.5)

# ax.set_ylabel("P(disease='present' | age)")
# ax.set_xlabel("Age", fontsize=15)
# ax.set_ylim(0, 1)
# ax.set_xlim(18, 90)
# plt.show()


trace_df = az.extract(bayesian_model, 
                      var_names=['age',
                                  'thalach',
                                   'restecg',
                                    'ca', 'Intercept']).to_dataframe()

trace_long = pd.melt(trace_df, id_vars=['chain', 'draw'], 
                     var_name='variable',
                       value_name='value')

summary_df = az.summary(bayesian_model).reset_index()


build_and_train_model(x_train, y_train)

logit_model = joblib.load('output/logistic_model.pkl')

xgboost_model = joblib.load('output/xgboost_model.pkl')


glossary = {
    "Aim of the study": 
    "Analyse Heart Disease Data from UCI Machine Learning Library, using Frequentist and Bayesian Classification Methods.",


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

cols_to_use_distributions = ['age',
                             'sex',
                            'thalach',
                            'restecg',
                            'ca',
                            'sex',
                            'cp',
                            'trestbps',
                            'chol',
                            'fbs',
                            'exang',
                            'thal'
                             ]


cols_to_use_logit_fit =     ['age',
                            'thalach',
                            'restecg',
                            'ca',
                            'sex',
                            'cp',
                            'trestbps',
                            'chol',
                            'fbs',
                            'exang',
                            'thal'
                             ]

# User interface (UI) definition


app_ui = ui.page_fluid(
    
    ui.page_navbar(
                    title = ui.tags.div(ui.img(src = 'heart.jpeg',
                                               height = '50px',
                                               style = 'margin:5px;'),
                                        ui.h4("Understanding Heart Disease"))
                 ),
            
    ui.navset_card_pill(ui.nav_panel("Problem Statement",  
    ui.layout_sidebar(
    
    ui.sidebar(ui.p("""This analsis is an objective take on
                        the various factors that could potentially 
                        impact heart on the basis of all the factors
                        in this dataset.
                        
                        """)
 
    ),
    # Add a title to the page with some top padding
    ui.panel_title(ui.h2("Aim of study and meaning of various columns", class_="pt-5")),
    # A container for plot output
    ui.output_ui("glossary_content"),
   
    )
    ),ui.nav_panel("Data Distributions",  
    ui.layout_sidebar(
    
    ui.sidebar( # A select input for choosing the variable to plot
    ui.input_radio_buttons(
        "var", "Select variable", choices= list(cols_to_use_distributions),
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
    ui.input_radio_buttons(
        "var_2", "Select variable", choices= list(cols_to_use_logit_fit),
        selected = 'ca'
    ),),
    # Add a title to the page with some top padding
    ui.panel_title(ui.h2("Presence of Heart disease fitted as a Logit function", class_="pt-5")),
    # A container for plot output
    ui.output_plot("logit_plot"),
    ui.output_plot("Odds_summary")
    )
    )
 
    , ui.nav_panel("Bayesian Probabilities",  
    ui.layout_sidebar(
    
    ui.sidebar( """Bayesian Posterior Probabilities 
                   show the Log Odds of Heart Disease
                   as a function of all the factors.
                   """
    ),
    # Add a title to the page with some top padding
    ui.panel_title(ui.h2("Posterior Probabilities", class_="pt-5")),
    # A container for plot output
    ui.output_plot("posteriors"),

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
    

    @render.plot
    def Odds_summary():
        # Histogram of the selected variable (input.var())
        p = pd.DataFrame(odds(mfw_output,
                 predictors,
                 y))
        
        plt.style.use('ggplot')

        g = (ggplot(data = p, mapping = aes(x = 'reorder(key_indicators, odds)',
                                            y = 'percentage_effect')
                                            )
            + geom_point(aes(x = 'key_indicators',
                             y = 'percentage_effect'),
                             color = 'red',
                             size = 5)
            + geom_col(color = 'red',
                       fill = 'grey',
                       alpha = 0.2)

            + geom_segment(aes(y = 0,
                               x = 'key_indicators',
                               yend = 'percentage_effect',
                               xend = 'key_indicators'))     

            + theme_seaborn()

            + labs(title = 'Odds of a person having heart disease',
                   x = 'key Indicators',
                   y = 'Odds')

            + coord_flip()

             )
        
        return g
    
  
    
    @render.plot
    def posteriors():
        
      
        z = (ggplot(trace_long, aes(x='value', fill = 'variable')) +
                  geom_histogram(bins=30, alpha=0.4) 
                +  facet_wrap('~variable', scales='free')
                  )
        return z
    
    @render.table
    def posterior_summary():
        
        return summary_df
    
    

    # @render.plot
    # def roc_plot():

    #     probs = logit_model.predict_proba(x_test)[:,1]

    #     auc = roc_auc_score(y_test, probs)
    #     # Calculate metrics for the roc curve
    #     fpr, tpr, thresholds = roc_curve(y_test, probs)
        
    #     g = plt.figure()
    #     plt.style.use('ggplot')
    #     plt.figure(figsize = (8, 8))
        
    #     # Plot the roc curve
    #     plt.plot(fpr, tpr, 'b')
    #     plt.xlabel('False Positive Rate', size = 16)
    #     plt.ylabel('True Positive Rate', size = 16)
    #     plt.title('Receiver Operating Characteristic Curve, AUC = %0.4f' % auc, 
    #                 size = 18)
        
        
    @output
    @render.ui
    def glossary_content():
        items = [ui.tags.li(ui.tags.strong(term), ": ", description) for term, description in glossary.items()]
        return ui.tags.ul(*items)

www_dir = Path(__file__).parent /"www"
app = App(app_ui, server,static_assets=www_dir)


