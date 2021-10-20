"""
author: Alex Spiers
date: 19 Oct 2021
title: Testing and logging exercise
## STEPS TO COMPLETE ##
# 1. import logging
# 2. set up config file for logging called `test_results.log`
# 3. add try except with logging and assert tests for each function
#    - consider denominator not zero (divide_vals)
#    - consider that values must be floats (divide_vals)
#    - consider text must be string (num_words)
# 4. check to see that the log is created and populated correctly
#    should have error for first function and success for
#    the second
# 5. use pylint and autopep8 to make changes
#    and receive 10/10 pep8 score
"""

import logging

logging.basicConfig(
    filename='./Logs/test_results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


def divide_vals(numerator, denominator):
    '''
    Args:
        numerator: (float) numerator of fraction
        denominator: (float) denominator of fraction

    Returns:
        fraction_val: (float) numerator/denominator
    '''
    try:
        logging.info("Passed %s and %s as arguments", numerator, denominator)
        assert isinstance(numerator, (int, float))
        assert isinstance(numerator, (int, float))
        fraction_val = numerator / denominator
        logging.info(
            "SUCCESS: Passed %s and %s as arguments",
            numerator,
            denominator)
        return fraction_val
    except ZeroDivisionError:
        logging.error("%s passed as argument; cannot be zero", denominator)
        return "denominator cannot be zero"
    except AssertionError:
        logging.error(
            "Passed %s and %s as arguments; must be int or float",
            numerator,
            denominator)
        return "arguments must be of type int or float"


def eval_num_words(text):
    '''
    Args:
        text: (string) string of words

    Returns:
        num_words: (int) number of words in the string
    '''
    try:
        logging.info("Passed %s as argument", text)
        assert isinstance(text, str)
        num_words = len(text.split())
        logging.info("SUCCESS: Passed %s as argument", text)
        return num_words
    except (AttributeError, AssertionError):
        logging.error("Passed %s as argument; must be str", text)
        return "text argument must be a string"


if __name__ == "__main__":
    divide_vals(3.4, 0)
    divide_vals(4.5, 2.7)
    divide_vals(-3.8, 2.1)
    divide_vals(1, 2)
    eval_num_words(5)
    eval_num_words('This is the best string')
    eval_num_words('one')
