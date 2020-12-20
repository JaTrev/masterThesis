from src.get_data import *
from src.preprocessing import *
from src.model import *


data, _ = get_data()


if __name__ == "__main__":
    data_processed = preprocessing(data)
    topics = lda_topics(data_processed, tf_idf_flag=True)

    for l in topics:
        print(l)