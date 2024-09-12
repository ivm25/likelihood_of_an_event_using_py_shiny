
# Heart Disease Analysis and Classification

## Project Overview

This project aims to analyze the Heart Disease dataset from the UCI Machine Learning Repository. The primary objectives are to perform classification using logistic regression and XGBoost, compare these results with Bayesian modeling using Bernoulli distributions, and present the findings in a Python Shiny app.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Analysis](#models-and-analysis)
  - [Logistic Regression](#logistic-regression)
  - [Bayesian Modeling](#bayesian-modeling)
- [Results](#results)
- [Shiny App](#shiny-app)
- [Contributing](#contributing)


## Dataset

The dataset used in this project is the Heart Disease dataset from the UCI Machine Learning Repository. It contains various medical attributes related to heart disease diagnosis.

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

- **Attributes**: Age, Sex, Chest Pain Type, Resting Blood Pressure, Serum Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Induced Angina, ST Depression, Slope of ST, Number of Major Vessels, Thal, and Target.

## Installation

To run this project, you need to have Python installed along with the following packages:

- Use the requirements.txt file to install the required packages in your virtual environment
```

## Usage



## Models and Analysis

### Logistic Regression

Logistic regression is used to model the probability of a binary outcome based on one or more predictor variables. In this project, we use logistic regression to classify the presence of heart disease.


### Bayesian Modeling

Bayesian modeling using Bernoulli distributions is employed to compare the results with traditional classification methods. This approach provides a probabilistic framework for classification.

## Results

The results of the analysis are presented in the Python Shiny app, which includes:

- Distribution of key factors
- Logistic regression model fitting
- Odds of heart disease
- Bayesian Probabilities

## Shiny App

The Shiny app provides an interactive interface to explore the analysis results. It includes various panels for different aspects of the analysis:

- **Distributions**: Visualize the distribution of key factors.
- **Heart DIsease**: View logistic regression model fitting.
- **Odds of having Heart Disease**: Summarize the odds of having heart disease.
- **Bayesian Approach**: Demonstrate the Bayesian posterior probabilities of  factors causing Heart Disease.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



