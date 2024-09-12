#%%
import pandas as pd
import pymongo, os
import pathlib as pl
import datetime as dt
import pymongo.collation
import pymongo.collection
import pymongo.database
from tqdm import tqdm, trange
from .printlog import print
from typing import List

def ShakToMongo(path: str, db: pymongo.database.Database, freq: int, chunk_size: int) -> List[pymongo.collection.Collection]:
    # find all the .csv - get the current working directory and join local path
    path = pl.Path(os.getcwd()).joinpath(path)
    files = list(path.glob('*.csv'))
    files.sort()    
    print(f'Found {len(files)} files in {path}', show=False)
    collection_list = [] # list of collections
    # Loop over the files
    for path in tqdm(files, desc=f'Loading files into database', position=0):
        #  Load the data
        df = pd.read_csv(path)
        df.head()
        print(f'The file {path} contains {df.shape[0]} samples', show=False)

        # Windowing
        # Create a window of 1 second
        n_samples = int(freq * chunk_size)
        n_windows = df.shape[0] // n_samples # number of windows - integer by default

        # Create a list of dataframes
        dfs = [df.iloc[i*n_samples:(i+1)*n_samples] for i in range(n_windows)]
        print(f'The file {path} contains {n_windows} windows of {n_samples} samples each', show=False)

        # get the file creation timestamp
        timestamp = path.stat().st_ctime
        print(f'The timestamp is created {pd.Timestamp(timestamp, unit="s")}', show=False)

        collection = db[path.stem+'_timeseries'] # use the filename as the collection name

        # %% Insert the data
        for i in trange(len(dfs), desc=f'Inserting data into {collection.name}', leave=True, position=1):
            df = dfs[i]
            dict = {}
            # timestamp
            dict['timestamp'] = dt.datetime.fromtimestamp(timestamp + i * chunk_size)
            # data
            for col in df.columns:
                dict[col] = {}
                dict[col]['sampFreq'] = freq
                dict[col]['timeSerie'] = df[col].values.tolist()
            # insert
            collection.insert_one(dict)
        collection_list.append(collection)
    return collection_list
