from preprocessing import read_data
import numpy as np
import pandas as pd

def load_data(dataroot):
    data = read_data(dataroot, '*.pcap_ISCX.csv')
    num_records, num_features = data.shape
    print(f"There are {num_records} flow records with {num_features} feature dimensions")

    data = data.rename(columns=lambda x: x.strip())  # Strip whitespace from column names
    data.to_csv('data.csv', index=False)

load_data('MachineLearningCVE/')