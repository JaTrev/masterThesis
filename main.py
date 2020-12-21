from src.get_data import *
from src.preprocessing import *
from src.model import *
from src.evaluation import *
from src.clustering import *
from src.vectorization import *


data, _ = get_data()

#TODO: create a main() function

if __name__ == "__main__":
    data_processed = preprocessing(data)



