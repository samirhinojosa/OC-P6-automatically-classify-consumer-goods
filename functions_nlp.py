# General
import string

# Natural Language Processing
import nltk
from nltk.corpus import stopwords


def remove_punctuation(text):
    """
    Method used to remove the punctuation
    from de text.

    Parameters:
    -----------------
        text (str): text to clean

    Returns:
    -----------------
        text_without_punctuation (string): Text cleaned
    """
    text_without_punctuation = "".join([char for char in text
                                       if char not in string.punctuation])
    return text_without_punctuation


def remove_stop_words(words, language):
    """
    Method used to remove stop words

    Parameters:
    -----------------
        words (list): Words to transform to lowercase
        language (str): Language to use to remove stop words
                        ["english", "french", "spanish"]

    Returns:
    -----------------
        words (list): List of words without stop words
    """

    stop_words = set(stopwords.words(language))
    filtered_words = [word for word in words if word not in stop_words]

    return filtered_words


def lowercase_words(words):
    """
    Method used to transform words to lowercase

    Parameters:
    -----------------
        words (list): Words to transform to lowercase

    Returns:
    -----------------
        words (list): Lowercase words
    """

    lowercase_words = []

    for word in words:
        lowercase_words.append(word.lower())

    return lowercase_words


def cleaning_up_text(data):
    """
    Method used to call three methods to clean up
    the text

    Parameters:
    -----------------
        data (string): String to clean

    Returns:
    -----------------
        words (list): Lowercase words
    """
    
    data = remove_punctuation(data):
    data = lowercase_words(data)
    data = remove_stop_words(data, "english")

    return data