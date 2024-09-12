import pickle
import pathlib
from .printlog import print

def save_model_file(model, filepath):
    # if the path does not exist, create it
    pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"{model} saved as picled data into '{filepath}'", show=False)
    return

def load_model_file(filepath=''):
    with open(filepath, 'rb') as f:
        model = pickle.load(f, fix_imports=False)
    print(f"{model} retrieved from picled data @ {filepath}", show=False)
    return model