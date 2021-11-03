import re
import string

# Natural Language Processing
import nltk
from nltk.corpus import stopwords


def check_characters(text):
    """
    Method used to check the digit and special
    character in a text.

    Parameters:
    -----------------
        text (string): Text to clean

    Returns:
    -----------------
        characters (dict): Dictionary with digit 
                           and special characters
    """
    
    numerical, special = [[] for i in range(2)]
    
    for i in range(len(text)):
        
        if text[i].isalpha():
            pass
        
        elif text[i].isdigit():
            # adding only unique characters
            numerical = list(set(numerical + [text[i]]))
            
        elif not text[i].isspace():
            # adding only unique characters
            special = list(set(special + [text[i]]))
            
    characters = {
        "numerical" : numerical,
        "special" : special
    }
            
    return characters


def remove_newlines_tabs(text):
    """
    Method used to remove the occurrences of newlines, tabs,
    and combinations like: \\n, \\.
    
    Parameters:
    -----------------
        text (string): Text to clean
        
    Returns:
    -----------------
        text (string): Text after removing of newlines, tabs, etc.
    
    """
    
    # Replacing the occurrences with a space.
    formatted_text = text.replace('\\n', ' ')\
                        .replace('\n', ' ').replace('\r',' ')\
                        .replace('\t',' ').replace('\\', ' ')\
                        .replace('. com', '.com')
            
    return formatted_text


def remove_whitespace(text):
    """
    Method used to remove extra whitespaces from the text
    
    Parameters:
    -----------------
        text (string): Text to clean
        
    Returns:
    -----------------
        text (string): Text after removing extra whitespaces.
    
    """
    
    pattern = re.compile(r"\s+")
    without_whitespace = re.sub(pattern, " ", text)
                         
    # Adding space for some instance where there is space before and after.
    if not " ? " in text:
        text = without_whitespace.replace("?", " ? ")
    if not ") " in text:
        text = without_whitespace.replace(")", ") ")
                         
    return text

















# def remove_punctuation(text):
#     """
#     Method used to remove the punctuation
#     from de text.

#     Parameters:
#     -----------------
#         text (string): Text to clean

#     Returns:
#     -----------------
#         text_without_punctuation (string): Text cleaned
#     """

#     # adding space before punctuation
#     text = re.sub(r"(?<=[.,])(?=[^\s])", r" ", text)

#     text_without_punctuation = "".join([char for char in text
#                                        if char not in string.punctuation])
#     return text_without_punctuation


# def lowercase_words(text):
#     "Method used to transform text to lowercase"
    
#     return text.lower()


# def remove_non_alphabet(text):
#     "Method used to remove all non alphabet from text"

#     # removing all non alphabet chars
#     text = re.sub("[^a-zA-Z]+", " ", text)
    
#     return text


# def tokenizer(text):
#     """
#     Method used to tokenize a string.

#     Parameters:
#     -----------------
#         text (str): text to tokenize

#     Returns:
#     -----------------
#         tokens (list): Word into text tokenized
#     """

#     tokenizer = nltk.RegexpTokenizer(r"\w+")
#     tokens = tokenizer.tokenize(text)
#     return tokens


# def remove_stop_words(words, language):
#     """
#     Method used to remove stop words

#     Parameters:
#     -----------------
#         words (list): Words to transform to lowercase
#         language (str): Language to use to remove stop words
#                         ["english", "french", "spanish"]

#     Returns:
#     -----------------
#         words (list): List of words without stop words
#     """

#     stop_words = set(stopwords.words(language))
#     filtered_words = [word for word in words
#                       if word not in stop_words]

#     return filtered_words


# def cleaning_up_text(text):
#     """
#     Method used to clean up the text calling
#     the following methods
#     - remove_punctuation(text)
#     - lowercase_words(text)ç
#     - tokenize(text)
#     - remove_stop_words(words, language)

#     Parameters:
#     -----------------
#         text (string): Text to clean

#     Returns:
#     -----------------
#         words (list): Words cleaned
#     """

#     text = remove_punctuation(text)
#     text = lowercase_words(text)
# #     text = remove_non_alphabet(text)
#     words = tokenizer(text)
#     words = remove_stop_words(words, "english")

#     return words
