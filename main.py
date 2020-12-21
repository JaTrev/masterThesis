from src.get_data import *
from src.preprocessing import *
from src.model import *
from src.evaluation import *


data, _ = get_data()

#TODO: create a main() function

if __name__ == "__main__":
    data_processed = preprocessing(data)
    topics = lda_topics(data_processed, tf_idf_flag=True, top_n_word=10)

    for l in topics:
        print(l)
        print("----------")

    print(coherence_score(data_processed, topics))
