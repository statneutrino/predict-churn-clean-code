# Predict Customer Churn

**Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This a project to practice creating production-ready code, specific to ML contexts
by practising:
- Conforming to clean code principles (specifically PEP 8 coding style)
- Testing & Logging (e.g. unit testing, using pytest and using logging module)
The original jupyter notebook, ```churn_notebook.ipynb```, was designed to predict customer
churn. 
```churn_library.py``` takes the EDA, model fitting and model evaluation from the logistic
regression and random forest classifiers that were created in that notebook and creates
modular, concise and clean code so that it can tested properly and imported in production.

```churn_script_logging_and_tests.py``` is a script that tests each function that makes up
the ```churn_library.py``` module.

## Environment
In order to run the scripts, you can use conda to import the environment using the ```environment.yml``` file.

.. code-block:: bash

    $ conda env create -f environment.yml 

## Running Files
The churn library can be imported by including ```import churn_library```. It can be run from the command line with:

    $ ipython churn_library.py

Running the script will create EDA images, as well as produce feature importance plots, a [shap explainer](https://christophm.github.io/interpretable-ml-book/shap.html) 
plot, predictions for the train and test sets, and finally a classification report.

The churn library can be tested by running:

    $ ipython churn_script_logging_and_tests.py

A log of test information is saved in ```./log/churn_library.log```