from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from stop_words import get_stop_words
import nltk
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

en_stop = get_stop_words('en')
sw = stopwords.words("english")


stop_words = sw + en_stop
stop_words.append('let')
stop_words.append('gon')
stop_words.append('dhe')
stop_words.extend(['car', 'like', 'got',
                   'get', 'one', 'well',
                   'back', 'bit', 'drive',
                   'look', 'see', 'good',
                   'quite', 'think', 'little',
                   'right', 'know', 'thing', 'want'])
stop_words.extend(['put', 'yeah', 'lot''dot', 'le', "'ve", 'really', 'car', 'like', 'got', 'get', 'one', 'well',
                   'back', 'bit', 'drive', 'look', 'see', 'good', 'quite', 'think', 'little', 'right', 'know',
                   'thing', 'want', 'dhe', 'gon', 'let', 'get'])
stop_words.extend(["\'re", "n\'t", "n\'t", "'ve", "really"])


def preprocessing(docs: list, do_stemming: bool = False, do_lemmatizing: bool = False,
                  remove_low_freq: bool = False) -> (list, list):
    """
    Document for processing a list of documents.

    Returns: preprocessed docs, vocabulary
    """
    print("---------")
    print("do_stemming: " + str(do_stemming))
    print("do_lemmatizing: " + str(do_lemmatizing))
    print("---------")

    vocabulary = []
    new_docs = []
    for i, doc in enumerate(docs):

        doc = doc.lower()

        tkns = word_tokenize(doc)

        # remove all tokens that are <= 3
        tkns = [w for w in tkns if len(w) > 2]

        # remove all tokens that are just digits
        tkns = [w for w in tkns if w.isalpha()]

        # remove stop words before stemming/lemmatizing
        tkns = [w for w in tkns if w not in stop_words]

        # remove all words that are not nouns
        tkns = [w for (w, pos) in nltk.pos_tag(tkns) if pos in ['NN', 'NNP', 'NNS', 'NNPS']]

        # stemming
        if do_stemming:
            tkns = [PorterStemmer().stem(w) for w in tkns]

        # lemmatizing
        if do_lemmatizing:
            tkns = [WordNetLemmatizer().lemmatize(w) for w in tkns]

        if len(tkns) == 0:
            continue

        new_docs.append(tkns)
        vocabulary.extend(tkns)

    if remove_low_freq:
        # remove low-frequency terms

        temp_new_docs = []
        for d in new_docs:
            temp_new_docs.extend(d)
        counter = Counter(temp_new_docs)

        l_threshold = 1

        docs_threshold = []
        vocab_threshold = []
        for d in new_docs:

            d_threshold = [w for w in d if counter[w] > l_threshold]
            if len(d_threshold) > 0:
                docs_threshold.append(d_threshold)
                vocab_threshold.extend(d_threshold)

        print("vocab threshold len: " + str(len(vocab_threshold)))
        print("vocab withouth threshold len: " + str(len(vocabulary)))
        new_docs = docs_threshold
        vocabulary = vocab_threshold



    return new_docs, sorted(list(set(vocabulary)))

