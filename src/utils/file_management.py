import pickle
import pandas as pd
import csv
import os

def dump_object(path, object):
    with open(path, 'wb') as fout:
        pickle.dump(object, fout)

def load_object(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_csv_as_df(path, **kwargs):
    return pd.read_csv(path, header=kwargs.get('header', None), names=kwargs.get('column_names'), usecols=kwargs.get('usecols'))

def create_csv_for_predictions(path, filename, header, data):
    with open(os.path.join(path, filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)

