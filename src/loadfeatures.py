import pymongo
from tqdm import tqdm
import numpy as np
from .printlog import print

def load_features(collection: pymongo.collection ,config):
    X = [] # features matrix
    features_names = [] # features names
    first = True
    for record in collection.find().sort('timestamp', pymongo.ASCENDING):
        for sensor in config['Sensors']:
            for feature in record[sensor].keys():
                if feature in config['Sensors'][sensor]['features'].keys():
                    X.append(record[sensor][feature])
                    if first: features_names.append(sensor + "_" + feature)
                elif config['Sensors'][sensor]['features']['wavPowers']:
                    X.append(record[sensor][feature])
                    if first: features_names.append(sensor + "_Wavelet_" + feature)
        first = False
    return X, features_names