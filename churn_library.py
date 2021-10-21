
"""
author: Alex Spiers
date:20th October 2021
This python file is an exercise in creating production ready code by practising:
- Conforming to clean code principles (specifically PEP 8 coding style)
- Testing (e.g. unit testing or in this case, using pytest)
- Logging
-
The original jupyter notebook, "churn_notebook.ipynb", was designed to predict customer
churn. This python file takes the EDA, model fitting and model evaluation from the logistic regression
and random forest classifiers that were created in that notebook and creates modular, concise and clean
code so that it can tested properly and imported in production.
"""

# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(bank_data_path):
    '''
    returns dataframe for the csv found at pth

    input:
            bank_data_path: a path to the data in csv format
    output:
            bank_data_df: pandas dataframe containing customer data
    '''
    bank_data_df = pd.read_csv(bank_data_path, index_col=1)

    # remove unnamed first column if exists
    if bank_data_df.columns[0] == 'Unnamed: 0':
        bank_data_df = bank_data_df.iloc[:, 1:]

    # Add churn column i.e. 0 or 1 based on 'Attrition_Flag'
    bank_data_df['Churn'] = bank_data_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    bank_data_df = bank_data_df.drop(['Attrition_Flag'], axis=1)
    return bank_data_df


def perform_eda(customer_df, bivariate_cols=None):
    '''
    This function peforms EDA on customer_df and saved figures to images folder
    input:
            df: pandas dataframe

    output:
            None - saves files directly to images folder
    '''
    # Split columns into numeric/quant columns and categorical columns
    category_cols = customer_df.select_dtypes(
        exclude=['int64', 'float64']).columns.tolist()
    quant_cols = customer_df.select_dtypes(
        include=['int64', 'float64']).columns.tolist()

    # Perform univariate EDA plots on categorical variables
    plt.figure(figsize=(20, 10))
    for col in category_cols:
        customer_df[col].value_counts('normalize').plot(kind='bar')
        plt.savefig('images/eda/' + col)
        plt.clf()

    # Perform univariate EDA plots on numeric variables
    for col in quant_cols:
        sns.distplot(customer_df[col])
        plt.savefig('images/eda/' + col)
        plt.clf()

    plt.close()

    # Create pairs plot (bivariate plots) between just four most important
    # variables
    bivariate_plot_cols = [
        'Total_Trans_Ct',
        'Total_Trans_Amt',
        'Total_Revolving_Bal',
        'Total_Ct_Chng_Q4_Q1']
    plt.figure(figsize=(30, 15))
    sns.pairplot(customer_df[bivariate_plot_cols])
    plt.savefig('images/eda/bivariate_pairs_plot.png')
    plt.close()


def encoder_helper(customer_df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could 
            be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for col in category_lst:
        col_groups = customer_df.groupby(col).mean()[response]
        customer_df.loc[:, col] = customer_df[col].replace(
            col_groups.to_dict())

    customer_df = customer_df.rename(
        columns=dict(
            zip(
                category_lst,
                [colname + '_Churn' for colname in category_lst]
            )
        )
    )

    return customer_df


def perform_feature_engineering(customer_df, response='Churn', test_size=0.3, seed=42):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could 
              be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = customer_df.drop(columns=[response])
    y = customer_df[response]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test




def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Train logistic regression model
    lrc = LogisticRegression()
    lrc.fit(X_train, y_train)
    # Create predictions based on LR model
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

if __name__ == "__main__":
    BANK_DATA = import_data("./data/bank_data.csv")
    # print(BANK_DATA.head())
    #print("Completed printing and importing")
    # perform_eda(BANK_DATA)
    #print("Completed EDA")
    print(
        BANK_DATA.select_dtypes(
            exclude=[
                'int64',
                'float64']).columns.tolist())
    BANK_DATA = encoder_helper(
        BANK_DATA,
        BANK_DATA.select_dtypes(
            exclude=[
                'int64',
                'float64']).columns.tolist())
    X_train, X_test, y_train, y_test = perform_feature_engineering(BANK_DATA)
    print(X_train.shape)
    print(X_test.shape)
