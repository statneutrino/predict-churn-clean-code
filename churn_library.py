
"""
author: Alex Spiers
date:20th October 2021
This python file is an exercise in creating production ready code by practising:
- Conforming to clean code principles (specifically PEP 8 coding style)
- Testing (e.g. unit testing or in this case, using pytest)
- Logging
The original jupyter notebook, "churn_notebook.ipynb", was designed to predict customer
churn. This python file takes the EDA, model fitting and model evaluation from the logistic
regression and random forest classifiers that were created in that notebook and creates
modular, concise and clean code so that it can tested properly and imported in production.
"""

# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
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

    return bank_data_df


def perform_eda(customer_df, bivariate_cols=None):
    '''
    This function peforms EDA on customer_df and saved figures to images folder
    input:
            df: pandas dataframe

    output:
            None - saves files directly to images folder
    '''
    # remove unnamed first column if exists
    if customer_df.columns[0] == 'Unnamed: 0':
        customer_df = customer_df.iloc[:, 1:]

    # Add churn column i.e. 0 or 1 based on 'Attrition_Flag' if not already added
    if 'Churn' not in customer_df.columns:
        customer_df['Churn'] = customer_df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        customer_df = customer_df.drop(['Attrition_Flag'], axis=1)

    # Split columns into numeric/quant columns and categorical columns
    category_cols = customer_df.select_dtypes(
        exclude=['int64', 'float64']).columns.tolist()
    quant_cols = customer_df.select_dtypes(
        include=['int64', 'float64']).columns.tolist()

    # Perform univariate EDA plots on categorical variables
    plt.figure(figsize=(20, 10))
    for col in category_cols:
        customer_df[col].value_counts('normalize').plot(kind='bar')
        plt.savefig('./images/eda/' + col)
        plt.clf()

    # Perform univariate EDA plots on numeric variables
    for col in quant_cols:
        sns.distplot(customer_df[col])
        plt.savefig('./images/eda/' + col)
        plt.clf()

    plt.close()

    # Create pairs plot (bivariate plots) between just four most important
    # variables unless specified otherwise
    if bivariate_cols is not None:
        bivariate_plot_cols = bivariate_cols
    else:
        bivariate_plot_cols = [
            'Total_Trans_Ct',
            'Total_Trans_Amt',
            'Total_Revolving_Bal',
            'Total_Ct_Chng_Q4_Q1']
    plt.figure(figsize=(30, 15))
    sns.pairplot(customer_df[bivariate_plot_cols])
    plt.savefig('./images/eda/bivariate_pairs_plot.png')
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


def perform_feature_engineering(
        customer_df,
        response='Churn',
        test_size=0.3,
        seed=42):
    '''
    Performs simple feature engineering (spliting into training and test sets)
    No variable scaling is done in this function (could be added at a future date)
    input:
              df: pandas dataframe with no categorical 
              response: string of response name [optional argument that could
              be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    # remove unnamed first column if exists
    if customer_df.columns[0] == 'Unnamed: 0':
        customer_df = customer_df.iloc[:, 1:]

    # Add churn column i.e. 0 or 1 based on 'Attrition_Flag'
    customer_df['Churn'] = customer_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    customer_df = customer_df.drop(['Attrition_Flag'], axis=1)

    # Encode categorical columns
    category_cols = customer_df.select_dtypes(
        exclude=['int64', 'float64']).columns.tolist()
    if len(category_cols) >= 1:
        customer_df = encoder_helper(customer_df, category_cols)

    #Split data into train and test datasets
    X = customer_df.drop(columns=[response])
    y = customer_df[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    # Scale/normalize features to have mean zero and unit variance so that comparisons can be made between features for feature importance
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = pd.DataFrame(
        data=scaler.transform(X_train), 
        index=X_train.index,
        columns=X_train.columns)
    X_test = pd.DataFrame(
        data=scaler.transform(X_test), 
        index=X_test.index,
        columns=X_test.columns)
    
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
    # Create logistic regression classification report
    plt.rc('figure', figsize=(7, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    # Save image
    plt.savefig('./images/results/classification_report_lr.png', bbox_inches='tight')
    plt.close()
    # Create random forest classification report
    plt.rc('figure', figsize=(7, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    # Save image
    plt.savefig('./images/results/classification_report_rf.png', bbox_inches='tight')
    plt.close()


def shap_explainer_plot(model, X_data, output_pth):
    '''
    creates and stores the shap_values importances to output_pth
    input:
            model: sklearn model object
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Create shap explainer plot
    plt.figure(figsize = [12,6])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    plt.rcParams.update({'font.size': 8})
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, plot_size=[12, 6])
    plt.savefig(output_pth, bbox_inches='tight')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances to output_pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Create feature importance plots
    
    # LR regression
    if isinstance(model, LogisticRegression):
        importances = abs(model.coef_[0])

    # Random forest models
    else:
        importances = model.feature_importances_
    
    importances = 100.0 * (importances / importances.max())
    indices = np.argsort(importances)[::-1] # Sort feature importances in descending order
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    plt.figure(figsize=(20,5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth, bbox_inches='tight')


def train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        seed=42,
        custom_params=None):
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
    lrc = LogisticRegression(solver='liblinear')
    lrc.fit(X_train, y_train)

    # Create predictions based on LR model
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Fit random forest classifier and tune parameters using grid search
    # during CV
    rfc = RandomForestClassifier(random_state=seed)
    if custom_params is not None:
        param_grid = custom_params
    else:
        param_grid = {
            'n_estimators': [100, 50],  # 'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # Create predictions based on best RF model
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Save results of LR model as text file
    # with open('./models/training_results.txt', "w") as train_results:
    #     train_results.write('Logistic Regression Results\n')
    #     train_results.write('training data results\n')
    #     train_results.write(
    #         classification_report(
    #             y_train,
    #             y_train_preds_lr) +
    #         '\n')
    #     train_results.write('test data results\n')
    #     train_results.write(
    #         classification_report(
    #             y_test, y_test_preds_lr) + '\n')
    #     # Save results of RF model in same text file
    #     train_results.write('Random Forest Results\n')
    #     train_results.write('training data results\n')
    #     train_results.write(
    #         classification_report(
    #             y_train,
    #             y_train_preds_rf) +
    #         '\n')
    #     train_results.write('test data results\n')
    #     train_results.write(
    #         classification_report(
    #             y_test, y_test_preds_rf) + '\n')
    # train_results.close()

    # Save classification report
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # Save ROC plot
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/ROC_curve.png')
    plt.close()

    # Save models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Save feature importances
    feature_importance_plot(cv_rfc.best_estimator_, X_test, './images/results/feature_importance_rf.png') # rf model
    feature_importance_plot(lrc, X_test, './images/results/feature_importance_lf.png') # lr model

    # Save shap explainer plots for random forest ONLY
    shap_explainer_plot(cv_rfc.best_estimator_, X_test, './images/results/shap_rf.png') # rf model
    # shap_explainer_plot(lrc, X_test, './images/results/shap_lf.png') # lr model

    # Save predictions
    train_predictions = pd.DataFrame(
        {
            'y_train_preds_lr': y_train_preds_lr,
            'y_train_preds_rf': y_train_preds_rf
        }
    ).set_index(X_train.index)
    test_predictions = pd.DataFrame(
        {
            'y_test_preds_lr': y_test_preds_lr,
            'y_test_preds_rf': y_test_preds_rf
        }
    ).set_index(X_test.index)

    train_predictions.to_csv(path_or_buf="./models/train_predictions.csv")
    test_predictions.to_csv(path_or_buf="./models/test_predictions.csv")


if __name__ == "__main__":
    BANK_DATA = import_data("./data/bank_data.csv")
    print("Imported Data")
    perform_eda(BANK_DATA)
    print("Completed EDA")
    X_train, X_test, y_train, y_test = perform_feature_engineering(BANK_DATA)
    print("Performed Feature Engineering")
    print(train_models(X_train, X_test, y_train, y_test))
    print("Trained Models")

    print(X_train.head())
    print(X_test.head())
    print(X_train.shape)
    print(X_test.shape)
