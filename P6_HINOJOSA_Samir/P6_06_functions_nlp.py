import re
import string
import unidecode

from bs4 import BeautifulSoup

# Natural Language Processing
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


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
        "numerical": numerical,
        "special": special
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
    text = text.replace("\\n", " ").replace("\n", " ")\
        .replace("\r", " ").replace("\t", " ")\
        .replace("\\", " ").replace(". com", ".com")

    return text


def remove_html_tags(text):
    """
    Method used to remove the occurrences of html tags from the text

    Parameters:
    -----------------
        text (string): Text to clean

    Returns:
    -----------------
        text (string): Text after removing html tags.

    """

    # Initiating BeautifulSoup object soup
    soup = BeautifulSoup(text, "html.parser")

    # Get all the text other than html tags.
    text = soup.get_text(separator=" ")

    return text


def remove_links(text):
    """
    Method used to remove the occurrences of links from the text

    Parameters:
    -----------------
        text (string): Text to clean

    Returns:
    -----------------
        text (string): Text after removing links.

    """

    # Removing all the occurrences of links that starts with https
    remove_https = re.sub(r"http\S+", "", text)

    # Remove all the occurrences of text that ends with .com
    # and start with http
    text = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)

    return text


def remove_extra_whitespace(text):
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
    if " ? " not in text:
        text = without_whitespace.replace("?", " ? ")

    if ") " not in text:
        text = without_whitespace.replace(")", ") ")

    return text


def remove_emails(text):
    """
    Method used to remove the occurrences of emails from the text

    Parameters:
    -----------------
        text (string): Text to clean

    Returns:
    -----------------
        text (string): Text after removing emails.

    """

    # Removing all the occurrences of emails
    text = re.sub(r"\S*@\S*\s?", "", text)

    return text


def remove_accented_characters(text):
    """
    Method used to remove accented characters from the text

    Parameters:
    -----------------
        text (string): Text to clean

    Returns:
    -----------------
        text (string): Text after removing accentes.

    """

    # Remove accented characters from text using unidecode.
    text = unidecode.unidecode(text)

    return text


def reduce_incorrect_character_repeatation(text):
    """
    Method used to reduce repeatition to two characters
    for alphabets and to one character for punctuations.

    Parameters:
    -----------------
        text (string): Text to clean

    Returns:
    -----------------
        text (string): Text after formatting.

    Example:
    -----------------
    Input : Realllllllllyyyyy!!!!....
    Output : Reallyy!.

    """

    # Pattern matching for all case alphabets
    pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)

    # Limiting all the  repeatation to two characters.
    formatted_text = pattern_alpha.sub(r"\1\1", text)

    # Pattern matching for all the punctuations that can occur
    pattern_punct = re.compile(r"([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}")

    # Limiting punctuations in previously formatted string to only one.
    combined_formatted = pattern_punct.sub(r"\1", formatted_text)

    # The below statement is replacing repeatation of spaces that occur
    # more than two times with that of one occurrence.
    final_formatted = re.sub(" {2,}", " ", combined_formatted)

    return final_formatted


def lowercase_words(text):
    """
    Method used to transform text to lowercase"

    Parameters:
    -----------------
        text (string): Text to clean

    Returns:
    -----------------
        text (string): Text after transforming to lowercase.

    """

    text = text.lower()

    return text


def expand_contractions(text, contractions):
    """
    Method used to expand the contractions from the text"

    Parameters:
    -----------------
        text (string): Text to clean
        contractions (dict): Dictionary of contractions with expansion
                             for each contraction

    Returns:
    -----------------
        text (string): Text after transforming the contractions.

    """

    # Tokenizing text into tokens.
    tokens = text.split(" ")

    for token in tokens:

        # Checking whether token is in contractions as a key
        if token in contractions:

            # Token is replace if is in dictionary and tokens
            tokens = [item.replace(token, contractions[token])
                      for item in tokens]

    # Transforming from list to string
    text = " ".join(str(token) for token in tokens)

    return text


def remove_punctuation(text):
    """
    Method used to remove the punctuation
    from de text.

    Parameters:
    -----------------
        text (string): Text to clean

    Returns:
    -----------------
        text (string): Text cleaned without punctuation

    """

    # adding space before punctuation
    text = re.sub(r"(?<=[.,])(?=[^\s])", r" ", text)

    text = "".join([char for char in text
                   if char not in string.punctuation])

    return text


def remove_non_alphabetic(text):
    """
    Method used to remove all non alphabet from text

    Parameters:
    -----------------
        text (string): Text to clean

    Returns:
    -----------------
        text (string): Text cleaned without punctuation

    """

    # removing all non alphabet chars
    text = re.sub("[^a-zA-Z]+", " ", text)

    return text


def tokenizer(text):
    """
    Method used to tokenize a string.

    Parameters:
    -----------------
        text (str): text to tokenize

    Returns:
    -----------------
        tokens (list): Word into text tokenized

    """

    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)

    return tokens


def cleaning_up_product_specifications(text):
    """
    Method used to clean up the feature
    product_specifications

    Parameters:
    -----------------
        text (str): text to tokenize

    Returns:
    -----------------
        text (str): Cleaned text

    """    
    text = re.findall(r"\"value\"=>\"(.*?)\"}", text)
    text = " ".join(text)
    
    return text
    

def cleaning_up_text(text, contractions):
    """
    Method used to clean up the text calling
    the following methods

    - remove_newlines_tabs(text)
    - remove_html_tags(text)
    - remove_links(text)
    - remove_extra_whitespace(text)
    - remove_emails(text)
    - remove_accented_characters(text)
    - reduce_incorrect_character_repeatation(text)
    - lowercase_words(text)
    - expand_contractions(text, contractions)
    - remove_punctuation(text)
    - remove_non_alphabet(text)
    - tokenizer(text)

    Parameters:
    -----------------
        text (string): Text to clean
        contractions (dict): Dictionary of contractions with expansion
                             for each contraction

    Returns:
    -----------------
        words (list): Words cleaned

    """

    text = remove_newlines_tabs(text)
    text = remove_html_tags(text)
    text = remove_links(text)
    text = remove_extra_whitespace(text)
    text = remove_emails(text)
    text = remove_accented_characters(text)
    text = reduce_incorrect_character_repeatation(text)
    text = lowercase_words(text)
    text = expand_contractions(text, contractions)
    text = remove_punctuation(text)
    text = remove_non_alphabetic(text)
    words = tokenizer(text)

    return words


def remove_stop_words(words, language):
    """
    Method used to remove stop words

    Parameters:
    -----------------
        words (list): Words to filter
        language (str): Language to use to remove stop words
                        ["english", "french", "spanish"]

    Returns:
    -----------------
        filtered_words (list): List of words without stop words

    """

    stop_words = stopwords.words(language)

    # extending stop words
    others_stop_words = ["cm", "inch", "g",
                         "com", "ml", "yes",
                         "rs"]
    stop_words.extend(others_stop_words)

    filtered_words = [word for word in words
                      if word not in stop_words]

    return filtered_words


def remove_non_english_words(words):
    """
    Method used to remove non english words

    Parameters:
    -----------------
        words (list): Words to filter

    Returns:
    -----------------
        filtered_words (list): List of words without non english words

    """

    english_words = set(nltk.corpus.words.words())

    filtered_words = [word for word in words
                      if word in english_words]

    return filtered_words


def keep_nouns(words):
    """
    Method used to keep only nouns in words

    NN   : noun, common, singular or mass
           common-carrier cabbage knuckle-duster Casino
    NNP  : noun, proper, singular
           Motown Venneboerger Czestochwa Ranzer Conchita
    NNPS : noun, proper, plural
           Americans Americas Amharas Amityvilles
    NNS  : noun, common, plural
           undergraduates scotches bric-a-brac products

    Parameters:
    -----------------
        words (list): Words to filtered

    Returns:
    -----------------
        filtered_words (list): List of words filtered by nouns

    """

    tags = nltk.pos_tag(words)

    filtered_words = [word for word, pos in tags
                      if (pos == "NN" or pos == "NNP" or
                          pos == "NNPS" or pos == "NNS")]

    return filtered_words


def remove_words(words, language):
    """
    Method used to remove words calling
    the following methods

    - remove_stop_words(words)
    - remove_non_english_words(words)
    - keep_nouns(words)

    Parameters:
    -----------------
        words (list): Words to treat
        language (str): Language to use to remove stop words
                        ["english", "french", "spanish"]

    Returns:
    -----------------
        words (list): Words cleaned

    """

    words = remove_stop_words(words, language)
    words = remove_non_english_words(words)
    words = keep_nouns(words)

    return words


def stem_words(words):
    """
    Method used to stem words using Snowball stemming (Porter2) algorithm

    Parameters:
    -----------------
        words (list): Words to transform to lowercase

    Returns:
    -----------------
        stemmed_words (list): List of words stemed

    """

    # Initializing an object of class PorterStemmer
    stemmer = SnowballStemmer("english")

    stemmed_words = [stemmer.stem(word) for word in words]

    return stemmed_words


def lemma_words(words):
    """
    Method used to stem words using Lemmatizer algorithm

    Parameters:
    -----------------
        words (list): Words to transform to lowercase

    Returns:
    -----------------
        lemma_words (list): Lema words list

    """

    # Initializing an object of class lemmatizer
    lemmatizer = WordNetLemmatizer()

    lemma_words = [lemmatizer.lemmatize(word) for word in words]

    return lemma_words


def display_topics(lda, feature_names, number_of_words):
    """
    Method used to display topics based on LDA Latent Dirichlet Allocation

    Parameters:
    -----------------
        lda (obj): Based on sklearn.decomposition import LatentDirichletAllocation
        feature_names (obj): 
        number_of_words (int): Number of word to show

    Returns:
    -----------------
        None.
        Print words based on topic

    """

    for topic_idx, topic in enumerate(lda.components_):
        print("- Topic %d:" % (topic_idx))
        print("  " + " ".join([feature_names[i]
                        for i in topic.argsort()[:-number_of_words - 1:-1]]))