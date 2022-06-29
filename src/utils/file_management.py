import pickle
import pandas as pd

def dump_object(path, object):
    with open(path, 'wb') as fout:
        pickle.dump(object, fout)

def load_object(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_dataset_as_df(path, **kwargs):
    return pd.read_csv(path, header=0, names=kwargs.get('column_names'), usecols=kwargs.get('usecols'))
    
    
