"""
author: Alex
date: 19 Oct 2021
Here the instructions for this exercise
## STEPS TO COMPLETE ##
# 1. import logging
# 2. set up config file for logging called `results.log`
# 3. add try except with logging for success or error
#    in relation to checking the types of a and b
# 4. check to see that log is created and populated correctly
#    should have error for first function and success for
#    the second
# 5. use pylint and autopep8 to make changes
#    and receive 10/10 pep8 score
"""

import logging

logging.basicConfig(
    filename='./Logs/results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


def sum_vals(num_one, num_two):
    '''
    Args:
        num_one: (int)
        num_two: (int)
    Return:
        num_one + num_two (int)
    '''
    try:
        logging.info("%s, %s", num_one, num_two)
        assert isinstance(num_one, int)
        assert isinstance(num_two, int)
        logging.info("You successfully summed the values: you rock!")
        return num_one + num_two

    except AssertionError:
        logging.info("%s, %s", num_one, num_two)
        logging.error("Object type other than float or int passed as arg")
    return None


if __name__ == "__main__":
    print(sum_vals('no', 'way'))
    print(sum_vals(300, 5))
    print(sum_vals(1, 2))
    print(sum_vals(1, 2.5))
