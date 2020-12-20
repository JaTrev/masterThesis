# fetch data

import glob
import os
import pandas as pd
import collections
from tqdm import tqdm
import numpy as np


def get_partition(task_data_path, path="data/processed_tasks/metadata/partition.csv") \
        -> (dict, collections.defaultdict):
    """
    get_partition fetches information on what sample ID is for training/developing/testing

    :param task_data_path:
    :param path: csv file that maps each sample ID to a train/devel/test
    :return: dicts with mappings between the sample IDs and the proposal
    """

    # any label to collect filenames safely
    names = glob.glob(os.path.join(task_data_path, 'label_segments', 'arousal', '*.' + 'csv'))
    sample_ids = []
    for n in names:
        name_split = n.split(os.path.sep)[-1].split('.')[0]
        sample_ids.append(int(name_split))
    sample_ids = set(sample_ids)

    df = pd.read_csv(path, delimiter=",")
    data = df[["Id", "Proposal"]].values

    id_to_partition = dict()
    partition_to_id = collections.defaultdict(set)

    for i in range(data.shape[0]):
        sample_id = int(data[i, 0])
        partition = data[i, 1]

        if sample_id not in sample_ids:
            continue

        id_to_partition[sample_id] = partition
        partition_to_id[partition].add(sample_id)

    return id_to_partition, partition_to_id


def get_class_names(class_name, path="/data/processed_tasks/metadata/"):
    if class_name == 'topic':
        df = pd.read_csv(os.path.join(path, 'topic_class_mapping.csv'))
        return df['topic'].values.tolist()
    else:
        df = pd.read_csv(os.path.join(path, 'emotion_class_mapping.csv'))
        return df['emotion'].values.tolist()


def load_id2topic(save_path):
    df = pd.read_csv(save_path)
    df = df.values.tolist()
    id2topic = {row[0]: row[1] for row in df}
    return id2topic


def classid_to_classname(labels, save_path):
    id2topic = load_id2topic(save_path)
    return np.vectorize(id2topic.get)(labels)


def read_classification_classes(label_file):
    df = pd.read_csv(label_file, delimiter=",", usecols=['class_id'])
    y_list = df['class_id'].tolist()
    return y_list


def read_cont_scores(label_file):
    df = pd.read_csv(label_file, delimiter=",", usecols=['mean'])
    return df['mean'].tolist()


def sort_trans_files(elem):
    return int(elem.split('_')[-1].split('.')[0])


def prepare_data(task_data_path, transcription_path) -> dict:
    """
    prepare_data creates a dict for the segment-level transcripts and their topic label
    :param task_data_path:
    :param transcription_path:
    :return: dict that consists of transcripts and their topic label
    """
    # Reading transcriptions on SEGMENT-level, sep. in train, develop, test of the official challenge

    id_to_partition, partition_to_id = get_partition(task_data_path)

    data = {}

    # training with test labels available
    for partition in tqdm(partition_to_id.keys()):
        segment_txt = []
        ys_a, ys_v, ys_t = [], [], []

        for sample_id in tqdm(sorted(partition_to_id[partition])):
            transcription_files = glob.glob(os.path.join(transcription_path, str(sample_id), '*.' + 'csv'))

            for file in sorted(transcription_files, key=sort_trans_files):
                df = pd.read_csv(file, delimiter=',')
                words = df['word'].tolist()
                segment_txt.append(" ".join(words))

            # training without test labels available
            label_file_topic = os.path.join(task_data_path, 'label_segments', 'topic', str(sample_id) + ".csv")
            y_list_topic = read_classification_classes(label_file_topic)

            for y in y_list_topic:
                ys_t.append(y)

        data[partition] = {'text': segment_txt, 'labels_topic': ys_t}

    return data


def get_data(task_data_path='data/processed_tasks/c2_muse_topic',
             transcription_path='data/processed_tasks/c2_muse_topic/transcription_segments'):

    all_data = prepare_data(task_data_path=task_data_path, transcription_path=transcription_path)
    data = all_data['train']['text']
    data.extend(all_data['devel']['text'])
    test_data = all_data['test']['text']

    return data, test_data
