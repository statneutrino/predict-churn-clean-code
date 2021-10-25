"""
author: Alex Spiers
date:25th October 2021
This python script is a part of a project in creating production ready code by practising:
- Conforming to clean code principles (specifically PEP 8 coding style)
- Testing (e.g. unit testing or in this case, using pytest)
- Logging
The original jupyter notebook, "churn_notebook.ipynb", was designed to predict customer
churn. This python file takes the EDA, model fitting and model evaluation from the logistic
regression and random forest classifiers that were created in that notebook and creates
modular, concise and clean code so that it can tested properly and imported in production.

This module specifically tests the functions that make up the churn_library.py module.

The functions tested are:

- import_data: imports csv file
- perform_eda: creates plots using matplotlib for exploratory data analysis
- encoder_helper: helper function to encode categorical features using mean churn
- perform_feature_engineering: encodes cat features, scales features and
splits data into train and test sets
- train_models: trains logistic regression and random forest machine learning models,
creates predictions and evaluates model performance

The results of the tests are saved in './log/churn_library.log'
"""

import os
import logging
import pandas as pd
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        customer_df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert customer_df.shape[0] > 0
        assert customer_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, customer_df, path):
    '''
    test perform eda function
    '''
    # Check if data is passed to function in correct format
    bank_data_df = cl.import_data("./data/bank_data.csv")
    path = './images/eda/'
    try:
        assert 'Churn' in bank_data_df.columns or 'Attrition_Flag' in bank_data_df.columns
        logging.info(
            "Testing dataframe passed to perform_eda: Dataframe contains expected response")
    except AssertionError as err:
        logging.error(
            "Testing dataframe passed to perform_eda: Dataframe does not contain \
			expected churn response. Cannot test perform_eda without correct data")
        raise err

    try:
        perform_eda(customer_df)
        total_files_in_path = len(os.listdir('./images/eda/'))
        assert total_files_in_path >= 1
        logging.info(
            "Testing perform_eda - files saved: SUCCESS. Files found are:%s",
            str(os.listdir(path)))

        total_png_files = sum(ext[-3:] == 'png' for ext in os.listdir(path))
        assert total_png_files >= 1
        logging.info(
            "Testing perform_eda PNG files found: SUCCESS.\
                There are %d files saved as png images in path",
            total_png_files)
    except AssertionError as err:
        logging.error(
            "Testing EDA images: no files in path %s with png extension found",
            path)
        raise err


def test_encoder_helper(encoder_helper, customer_df):
    '''
    test encoder helper
    '''
    category_cols = customer_df.select_dtypes(
        exclude=['int64', 'float64']).columns.tolist()
    if len(category_cols) < 1:
        logging.warning(
            "Dataframe passed to encoder_helper contains no categorical columns; \
				functionality not tested.")
    else:
        try:
            customer_df = encoder_helper(customer_df, category_cols)
            assert len(
                customer_df.select_dtypes(
                    exclude=[
                        'int64',
                        'float64']).columns.tolist()) == 0
            logging.info(
                "Testing categorical encoding - SUCCESS: all categorical columns removed")
        except AssertionError as err:
            logging.error(
                "Testing categorical encoding: \
					some categorical columns remain in dataframe not encoded")
            raise err

        try:
            new_category_names = [
                colname + '_Churn' for colname in category_cols]
            for colname in new_category_names:
                assert colname in customer_df.columns
                logging.info(
                    "Testing categorical encoding - SUCCESS: %s exists in dataframe",
                    colname)
                assert pd.api.types.is_numeric_dtype(customer_df[colname])
                logging.info(
                    "Testing cat encoding - SUCCESS: '%s' column is now numeric",
                    colname)
        except AssertionError as err:
            logging.error(
                "Testing cat encoding: categories either not numeric or renamed to 'x_Churn'")
            raise err


def test_perform_feature_engineering(
        perform_feature_engineering, customer_df, test_size=0.3):
    '''
    test perform_feature_engineering
    '''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            customer_df, test_size=0.3)
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_test.size > 0
        assert y_train.size > 0
        logging.info("Testing perform_feature_engineering shape: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering output shape:\
                either zero length objects returned, or \
			    not enough (4) objects returned")
        raise err

    try:
        assert X_test.shape[0] - \
            1 <= round(test_size * customer_df.shape[0]) <= X_test.shape[0] + 1
        logging.info(
            "Testing X_test dataset size is %s original dataset: SUCCESS",
            test_size)
        assert y_test.size - \
            1 <= round(test_size * customer_df.shape[0]) <= y_test.size + 1
        logging.info(
            "Testing y_test dataset size is %s original y: SUCCESS",
            test_size)
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering shape: \
			test dataset size is NOT %s x train dataset",
            test_size)
        raise err


def test_train_models(train_models, data_tuple, results_path, model_path):
    '''
    test train_models
    '''
    X_train, X_test, y_train, y_test = data_tuple

    try:
        train_models(X_train, X_test, y_train, y_test)
        results_path_files = os.listdir(results_path)
        assert len(results_path_files) >= 1
        logging.info(
            "Testing train_models; files exist in %s: SUCCESS. Files found are: %s",
                results_path, results_path_files)
    except AssertionError as err:
        logging.error(
            "Testing train_models: no files in %s",
            results_path)
        raise err

    # Check classification report saved
    try:
        class_report_num = sum(
            'classification_report' in file for file in results_path_files)
        assert class_report_num >= 1
        logging.info("Testing classification reports produced: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing classifcation report;\
                no classification reports found in %s",
                results_path)
        raise err
    # Check models saved
    try:
        model_path_files = os.listdir(model_path)
        assert len(model_path_files) >= 1
        logging.info(
            "Testing train_models. Files saved in %s: SUCCESS.\
                Files found are: %s",
                model_path, model_path_files)
        total_pkl_files = sum(ext[-3:] == 'pkl' for ext in model_path_files)
        assert total_pkl_files >= 1
        logging.info("Testing files saved as PNG: SUCCESS.\
            There are %s files saved as pkl files in path",
            total_pkl_files)
    except AssertionError as err:
        logging.error(
            "Testing classifcation report;\
            no files with pkl extension found in %s", model_path)
        raise err
    # Check feature importance plots saved
    try:
        feature_imp_num = sum(
            'feature_importance' in file for file in results_path_files)
        assert feature_imp_num >= 1
        logging.info("Testing feature importance plots produced: SUCCESS. %d files found.",
            feature_imp_num)
    except AssertionError as err:
        logging.error(
            "Testing classifcation report;\
                no feature importance plots found in %s",
                format(results_path))
        raise err


if __name__ == "__main__":

    test_import(cl.import_data)

    CUSTOMER_DF = cl.import_data('./data/bank_data.csv')

    test_eda(cl.perform_eda, CUSTOMER_DF, "./data/bank_data.csv")

    # Include initial feature_processing code to allow test_encoder_helper to
    # test core functionality
    if CUSTOMER_DF.columns[0] == 'Unnamed: 0':
        CUSTOMER_DF = CUSTOMER_DF.iloc[:, 1:]

    CUSTOMER_DF['Churn'] = CUSTOMER_DF['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    CUSTOMER_DF = CUSTOMER_DF.drop(['Attrition_Flag'], axis=1)

    test_encoder_helper(cl.encoder_helper, CUSTOMER_DF)

    # Reload customer dataframe for testing
    CUSTOMER_DF = cl.import_data('./data/bank_data.csv')

    # test_perform_feature_engineering(cl.perform_feature_engineering, CUSTOMER_DF)

    PROCESSED_DATA = cl.perform_feature_engineering(CUSTOMER_DF)
    test_train_models(
        cl.train_models,
        PROCESSED_DATA,
        './images/results/',
        './models/')
