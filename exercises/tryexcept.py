def which_error(numerator, denominator):
    fraction_val = numerator / denominator
    return fraction_val


def divide_vals(numerator, denominator):
    '''
    Args:
        numerator: (float) numerator of fraction
        denominator: (float) denominator of fraction

    Returns:
        fraction_val: (float) numerator/denominator
    '''
    try:
        fraction_val = numerator / denominator
        return fraction_val

    except TypeError:
        print("Error: Must use floats as arguments.")
    except ZeroDivisionError:
        # try to return the fraction but if the denominator is zero
        # catch the error and return a string saying:
        # "denominator cannot be zero"
        print("Error: denominator cannot be zero.")



def num_words(text):
    '''
    Args:
        text: (string) string of words

    Returns:
        num_words: (int) number of words in the string
    '''
    # try to split based on spaces and return num of words
    try:
        num_words = len(text.split())
        return num_words
    except AttributeError:
        return "text argument must be a string"


if __name__ == "__main__":
    WORD_TEST = num_words(234343)
    print(WORD_TEST)