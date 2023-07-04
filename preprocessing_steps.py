import pandas as pd
pd.set_option('display.max_colwidth', None)
import numpy as np

# NLTK
import nltk
from nltk import TweetTokenizer, word_tokenize
from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from nltk.stem import WordNetLemmatizer
# from nltk.tag import pos_tag
# from nltk import ne_chunk
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# SPACY
import spacy
nlp = spacy.load("en_core_web_sm")

# Others
import re
import string
from string import punctuation
import time
from spellchecker import SpellChecker

# Self-created reference files
from emoticon_unicodes import *
from slangs import *

# Init stopwords
STOPWORDS_LIST = list(stopwords.words('english'))
PUNCTUATION_LIST = list(string.punctuation)
# Add variations not captured by string.punctuation
PUNCTUATION_LIST.append("“")
PUNCTUATION_LIST.append("”")
PUNCTUATION_LIST.append("’")
PUNCTUATION_LIST.append("…")

class Preprocess:

    def convert_emoticons_to_words(text):
        processed_text = []
        tokenizer = TweetTokenizer()
        tokenized = tokenizer.tokenize(text)
        for token in tokenized: 
            # Convert emoticons to words
            if token in UNICODE_EMO:
                token = UNICODE_EMO[token].replace(":","")
            processed_text.append(token)
        return " ".join(processed_text)

    def convert_to_lowercase(text):
        return text.lower()

    def remove_urls(text):
        regex_url_pattern = r"https?://\S+|www\.\S+"
        text = re.sub(regex_url_pattern, "", text)
        return text

    def entity_recognition_handling(text):
        processed_txt = []
        tokenizer = TweetTokenizer()
        text_with_details = nlp(text)
#         print("\n---------------------------------------------------------------------------------------------------------------\n")
#         print(text_with_details)
#         print("\n---------------------------------------------------------------------------------------------------------------\n")
        for ent in text_with_details.ents:
            sub_text = ent.text

#             print("\nBEFORE: " + sub_text + " - " + ent.label_)

            # Spacy entity recognition does not handle punctuation well # e.g. the "(Ye Old Mill)" becomes "(Ye Old Mill"
            sub_text = Preprocess.remove_punctuation(sub_text)

            # Remove any articles that come before the entity
            # e.g. 'a', 'A', 'the', 'The', 'an', 'An'
            sub_text = re.sub(r'\b[tT]he\s\b', '', sub_text)
            sub_text = re.sub(r'\b[aA]\s\b', '',sub_text)
            sub_text = re.sub(r'\b[aAn]\s\b', '',sub_text)

            # Remove trailing 's from named entities e.g. "Donald Trump's" or  "Donald Trump 's" becomes "Donald Trump"
            sub_text = re.sub(r' \'s', '', sub_text)

#             print("AFTER: " + sub_text + " - " + ent.label_)

            concatenated_text = "_".join(sub_text.split())
#             text = re.sub(sub_text, concatenated_text, text)
            # used string replace instead of re.sub due to some python bug "nothing to repeat at position 0" for unknown symbols
            text = text.replace(sub_text, concatenated_text); 
        return text

    def remove_multiple_white_spaces(text):
        text = re.sub('\s+', ' ', text) # Remove multiple white spaces that are concatenated together
        text = re.sub('^\s', '', text) # Remove white space at the start of sentences
        return text

    def remove_digits(text):
        regex_digits_pattern = r"[1234567890]"
        text = re.sub(regex_digits_pattern, '', text)
        text = Preprocess.remove_multiple_white_spaces(text)
        return text

    def convert_slangs_to_words(text):
        processed_text = []
        tokenizer = TweetTokenizer()
        tokenized = tokenizer.tokenize(text)
        for token in tokenized: 
            # Convert emoticons to words
            if token in SLANGS:
                token = SLANGS[token]
            processed_text.append(token)
        return " ".join(processed_text)

    def correct_spelling_mistakes(text):
        corrected_text = []
        tokenizer = TweetTokenizer()
        tokenized = tokenizer.tokenize(text)
        spellcheck = SpellChecker()
        misspelled_words = spellcheck.unknown(tokenized)
        for token in tokenized:
            if token in misspelled_words:
                corrected_text.append(spellcheck.correction(token))
            else:
                corrected_text.append(token)
        return " ".join(corrected_text)

    def replace_hyphens_or_underscores_with_spaces(text):
        regex_hyphens_or_underscores_pattern = r"[-|_]"
        text = re.sub(regex_hyphens_or_underscores_pattern, " ", text)  
        return text

    def remove_stopwords(text):
        processed_text = []
        tokenizer = TweetTokenizer()
        tokenized = tokenizer.tokenize(text)
        for token in tokenized: 
            if token.lower() not in STOPWORDS_LIST:
                processed_text.append(token)
        return " ".join(processed_text)

    def remove_punctuation(text):
        processed_text = []
#         tokenizer = TweetTokenizer()
#         tokenized = tokenizer.tokenize(text)
        tokenized = word_tokenize(text) # Tokenizes punctuation too
        for token in tokenized: 
            if token not in PUNCTUATION_LIST:
                processed_text.append(token)
        return " ".join(processed_text)

    def prepend_NOT_to_handle_negation(text):
        negation_words = ["n't", "not", "no", "never"]
        prepend_NOT = False
        processed_text = []
#         tokenized = word_tokenize(text) # Tokenizes and split #HAPPY into #, HAPPY
        # Use TweetTokenizer as we should not append "NOT_" to hashtags
        tokenizer = TweetTokenizer() # Tokenizes and maintains hashtags, i.e. maintains #HAPPY
        tokenized = tokenizer.tokenize(text)
        for i in range(len(tokenized)):
            token = tokenized[i]

            # Check "prepend_NOT" flag, i.e. 1 of the previous word was a logical negation
            # Therefore, prepend "NOT_" until it reaches a punctuation symbol
            # IMPORTANT: Must NOT remove punctuation before this function
            if prepend_NOT:
                if prepend_NOT and token in PUNCTUATION_LIST:
                    prepend_NOT = False
                else:
                    # Do not append "NOT_" to hashtags
                    if not Preprocess.is_hashtag(token):                 
                        token = "NOT_" + token

            # Note: Case sensitive. Therefore "NOT_" should not be captured
            regex_negation_pattern = r"(" + negation_words[0] + ")$"
            if re.search(regex_negation_pattern, token) or token in negation_words:
                prepend_NOT = True

            processed_text.append(token)
        return " ".join(processed_text)

    def is_hashtag(token):
        is_hashtag = False
        regex_hashtag_pattern = r"^#\w+"
        if re.match(regex_hashtag_pattern, token):
            is_hashtag = True
        return is_hashtag

    def process_text(text):
#         print(text)
#         print("\n---------------------------------------------------------------------------------------------------------------\n")
        text = Preprocess.entity_recognition_handling(text)

#         text = Preprocess.convert_to_lowercase(text)
#         text = Preprocess.remove_urls(text)
#         text = Preprocess.convert_emoticons_to_words(text)
#         text = Preprocess.replace_hyphens_or_underscores_with_spaces(text)
#         text = Preprocess.convert_slangs_to_words(text)
#         text = Preprocess.prepend_NOT_to_handle_negation(text)
#     #     text = Preprocess.remove_stopwords(text)
#         text = Preprocess.remove_punctuation(text)
#     #     text = Preprocess.correct_spelling_mistakes(text)
        return text