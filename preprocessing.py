import re
# import nltk
# from nltk.corpus import stopwords
from tqdm import tqdm

# # Download necessary NLTK resources (if not already downloaded)
# import nltk
# nltk.download('punkt', download_dir=r'C:\nltk_data')
# nltk.download('stopwords', download_dir=r'C:\nltk_data')
# nltk.data.path.append(r'C:\nltk_data')


# def preprocess_text(text_data): 
#     """
#     Preprocesses a list of text reviews by:
#     1. Removing punctuation
#     2. Converting to lowercase
#     3. Removing stopwords
#     """
#     preprocessed_text = [] 

#     for sentence in tqdm(text_data, desc="Preprocessing Text"): 
#         # Remove punctuation
#         sentence = re.sub(r'[^\w\s]', '', sentence) 

#         # Tokenize, lowercase, and remove stopwords
#         cleaned_sentence = ' '.join(
#             token.lower() 
#             for token in nltk.word_tokenize(sentence) 
#             if token.lower() not in stopwords.words('english')
#         )

#         preprocessed_text.append(cleaned_sentence)

#     return preprocessed_text
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess_text(text_data):
    """
    Simple preprocessing without NLTK:
    1. Lowercase
    2. Remove punctuation
    3. Remove stopwords (from scikit-learn)
    """
    preprocessed_text = []

    for sentence in tqdm(text_data, desc="Preprocessing Text"):
        # Remove punctuation
        sentence = re.sub(r'[^\w\s]', '', sentence)
        # Lowercase and remove stopwords
        cleaned_sentence = ' '.join(
            word.lower() for word in sentence.split() if word.lower() not in ENGLISH_STOP_WORDS
        )
        preprocessed_text.append(cleaned_sentence)

    return preprocessed_text