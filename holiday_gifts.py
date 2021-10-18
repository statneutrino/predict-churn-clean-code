'''
Holiday Gift refactoring code module

This is a Udacity task to clean this script in
# a way that uses the code as a single function
# that takes a path and returns the total_price variable
# that meets pep8 standards and receives a 10 score using pylint
'''

import numpy as np


def get_total_price_below_value(file_path, max_price_threshold):
    '''
    Calculates the total cost of all gifts under max_price_threshold (dollars) to
    see how much is spent on free gifts from a file_name_ txt file.

    Args:
    file_name: str. The path to the file containing gift prices
    max_price_threshold: int. Max gift price threshold to subset prices before calculating the sum.

    Returns:
    total_price: the sum of all prices below max_price_threshold.

    Returns
    '''

    # Opens file name
    with open(file_path, encoding="utf-8") as file_name:
        gift_costs = file_name.read().split('\n')

    # convert string to int
    gift_costs = np.array(gift_costs).astype(int)

    # Sums all gift prices below max_price_threshold
    total_price = (gift_costs[gift_costs < max_price_threshold]).sum() * 1.08
    return total_price


if __name__ == "__main__":
    TOTAL_PRICE_BELOW_25 = get_total_price_below_value('gift_costs.txt', 25)
    print(TOTAL_PRICE_BELOW_25)
