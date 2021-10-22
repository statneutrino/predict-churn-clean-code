import os
import logging
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):
	'''
	test perform eda function
	'''
	# Check if data is passed to function in correct format
	df = cl.import_data("./data/bank_data.csv")
	path = './images/eda/'
	try:
		assert 'Churn' in df.columns or 'Attrition_Flag' in df.columns
		logging.info("Testing dataframe passed to perform_eda: Dataframe contains expected response")
	except AssertionError as err:
		logging.error("Testing dataframe passed to perform_eda: Dataframe does not contain \
			expected churn response. Cannot test perform_eda without correct data")

	try:
		perform_eda(df)
		total_files_in_path = len(os.listdir('./images/eda/'))
		assert total_files_in_path >= 1
		logging.info("Testing files exist in directory: SUCCESS. Files found are:" + str(os.listdir(path)))

		total_png_files = sum(ext[-3:] == 'png' for ext in os.listdir(path))
		assert total_png_files >= 1
		logging.info("Testing files saved as PNG: SUCCESS. There are {} files saved as png images in path".format(total_png_files))
	except AssertionError as err:
		logging.error("Testing EDA images: no files in path" + path + "with png extension found.")
		raise err


def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":

	# test_import(cl.import_data)
	# test_eda(cl.perform_eda)

	CUSTOMER_DF = cl.import_data('./data/bank_data.csv')
	if CUSTOMER_DF.columns[0] == 'Unnamed: 0':
		CUSTOMER_DF = CUSTOMER_DF.iloc[:, 1:]
	
	CUSTOMER_DF['Churn'] = CUSTOMER_DF['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
	CUSTOMER_DF = CUSTOMER_DF.drop(['Attrition_Flag'], axis=1)

	test_encoder_helper(cl.encoder_helper)








