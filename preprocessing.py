__author__ = 'Jumabek Alikhan'
__data__ = 'Nov 26,2019'
import numpy as np
import pandas as pd
from os.path import join
import glob

def read_data(dataroot, file_ending='*.pcap_ISCX.csv'):
    """
    Read CSV files with specified file ending in the dataroot directory and concatenate them into a single Pandas DataFrame.
    """
    filenames = glob.glob(f"{dataroot}/{file_ending}")
    if not filenames:
        raise ValueError(f"No files found in {dataroot} with pattern {file_ending}")
    combined_csv = pd.concat([pd.read_csv(f, dtype=object) for f in filenames], sort=False)
    return combined_csv

 # reads csv file and returns np array of X,y -> of shape (N,D) and (N,1)
def load_data(dataroot):
    data = read_data(dataroot, '*.pcap_ISCX.csv')
    num_records, num_features = data.shape
    print(f"There are {num_records} flow records with {num_features} feature dimensions")

    data = data.rename(columns=lambda x: x.strip())  # Strip whitespace from column names

    df_label = data['Label']
    data = data.drop(columns=['Flow Packets/s', 'Flow Bytes/s', 'Label'])

    nan_count = data.isnull().sum().sum()
    if nan_count > 0:
        data.fillna(data.mean(), inplace=True)
        print('Filled NaN')

    X = normalize(data.astype(float).values)  # Normalize the input data
    y = encode_label(df_label.values)
    return X, y


#We balance data as follows:
#1) oversample small classes so that their population/count is equal to mean_number_of_samples_per_class
#2) undersample large classes so that their count is equal to mean_number_of_samples_per_class

def balance_data(X, y, seed):
    """
    Balance the data by randomly sampling the same number of examples from each class.
    """
    np.random.seed(seed)
    unique, counts = np.unique(y, return_counts=True)
    mean_samples_per_class = int(round(np.mean(counts)))
    new_X = np.empty((0, X.shape[1])) 
    new_y = np.empty((0), dtype=int)
    for i, c in enumerate(unique):
        temp_x = X[y == c]
        indices = np.random.choice(temp_x.shape[0], mean_samples_per_class)
        new_X = np.concatenate((new_X, temp_x[indices]), axis=0)
        temp_y = np.ones(mean_samples_per_class, dtype=int) * c
        new_y = np.concatenate((new_y, temp_y), axis=0)
    indices = np.arange(new_y.shape[0])
    np.random.shuffle(indices)
    new_X = new_X[indices, :]
    new_y = new_y[indices]
    return new_X, new_y


# chganges label from string to integer/index
def encode_label(Y_str):
    labels_d = make_value2index(np.unique(Y_str))
    Y = np.array([labels_d[y_str] for y_str in Y_str])
    return Y


def make_value2index(attacks):
    attacks = sorted(attacks)
    return {attack: i for i, attack in enumerate(attacks)}


def normalize(data):
    """
    Normalize the input data by zero-centering it and scaling it by the range of values.
    
    Args:
    data (numpy array): Input data to be normalized
    
    Returns:
    numpy array: Normalized data
    """
    data = data.astype(np.float32)
    eps = 1e-15
    mask = data == -1
    data[mask] = 0
    mean_i = np.mean(data, axis=0)
    min_i = np.min(data, axis=0)
    max_i = np.max(data, axis=0)
    r = max_i - min_i + eps
    data = (data - mean_i) / r
    data[mask] = 0
    return data