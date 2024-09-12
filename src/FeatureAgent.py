import pymongo.collection
import os, yaml, pymongo
from typing import List
import numpy as np
import pywt
import matplotlib.pyplot as plt
from .printlog import print
import scipy
from tqdm import tqdm as tdqm
from .printlog import print

def packTrasform(timeSerie: list,wavelet='db10', mode='symmetric',maxlevel=6, plot=False):
    '''perform the wavelet trasform of a time series:
    RETURN: coefs:  [list] coefficients of the decomposition
            pows:   [list] powers of all the coefficients
            nodes:  [list] names of the nodes'''
    _wp = pywt.WaveletPacket(data=timeSerie, wavelet=wavelet, mode=mode,maxlevel=maxlevel)   # perform the packet trasform
    _nodes=[node.path for node in _wp.get_level(_wp.maxlevel, 'natural')]                    # extract the lower level nodes
    _powers=[np.linalg.norm(_wp[index].data) for index in _nodes]                            # compute the l2 norm of coefs
    _coefs=[_wp[index].data for index in _nodes]
    return _coefs, _powers, _nodes

class FA(): # feature agent
    '''
    empty the RAW collection and populate the Unconsumed collection with extracted features:
    '''
    def __init__(self, 
                 configStr: str = None, 
                 URI: str = None, 
                 col_timeseries: pymongo.collection.Collection = None,
                 col_features: pymongo.collection.Collection = None,
                 order: int = 1):
        self.configStr = configStr    #  path to config file (json)
        try:
            with open(self.configStr,'r') as f:
                self.Config = yaml.safe_load(f)
                print(f'Loaded config file @ {self.configStr}', show=False)
        except:
            raise Exception(f'Error reading config file @ {self.configStr}')
        self.sensors  = list(self.Config['Sensors'].keys())       # list of sensors4
        self.features = {}                                                              # initialize the features dict
        self.features["timestamp"] = None                                               # initialize the features dict with timestamp
        self.features.update({key: {} for key in self.sensors})                         # initialize the features dict with sensors                              
        self.URI = URI                                                                  # URI of the MongoDB
        self.order = order                                                              # order of the sort
        self.col_t = col_timeseries                                                   # collection of the timeseries
        self.col_F = col_features                                                      # collection of the features
    def _readFromRaw(self):
        ''' Read the data from the RAW collection '''
        try:
            self.snaps    = list(self.col_t.find().sort('timestamp',self.order))     # oldest/newest record - sort gives a cursor, the [0] is the dict
            print(f"Imported snapshots from {self.col_t}", show=False)
            return True    
        except IndexError:
            print(f"No data in collection {self.col_t.full_name}, waiting for new data...", show=False)
            return False

    def _extractFeatures(self):
        ''' extract features from the data '''
        for sensor in self.sensors:                                  # for each sensor (names are keys of the dict)
            self.features["timestamp"] = self.snap["timestamp"]      # add the timestamp to the features
            self._extractTimeFeautures(sensor)                       # extract time domain features
            self._extractFreqFeautures(sensor)                       # extract frequency domain features

    def _extractTimeFeautures(self, sensor):
        ''' extract time domain features '''
        # if Mean Enabled
        if self.Config['Sensors'][sensor]['features']['mean']:
            self.features[sensor].update({'mean':np.mean(self.snap[sensor]['timeSerie'])})
            print(f"Mean extracted from [purple]{sensor}[/]", show=False)
        # if RMS Enabled
        if self.Config['Sensors'][sensor]['features']['rms']:
            self.features[sensor].update({'rms':np.sqrt(np.mean(np.square(self.snap[sensor]['timeSerie'])))})
            print(f"RMS extracted from [purple]{sensor}[/]", show=False)
        # if peak2peak Enabled
        if self.Config['Sensors'][sensor]['features']['peak']:
            self.features[sensor].update({'peak2peak':np.ptp(self.snap[sensor]['timeSerie'])})
            print(f"Peak2Peak extracted from [purple]{sensor}[/]", show=False)
        # if std Enabled
        if self.Config['Sensors'][sensor]['features']['std']:
            self.features[sensor].update({'std':np.std(self.snap[sensor]['timeSerie'],ddof=1)})
            print(f"Standard deviation extracted from [purple]{sensor}[/]", show=False)
        # if skewness Enabled
        if self.Config['Sensors'][sensor]['features']['skew']:
            self.features[sensor].update({'skewness':scipy.stats.skew(self.snap[sensor]['timeSerie'],bias=False)})
            print(f"Skewness extracted from [purple]{sensor}[/]", show=False)
        # if kurtosis Enabled
        if self.Config['Sensors'][sensor]['features']['kurt']:
            self.features[sensor].update({'kurtosis':scipy.stats.kurtosis(self.snap[sensor]['timeSerie'],bias=False)})
            print(f"Kurtosis extracted from [purple]{sensor}[/]", show=False)

    def _extractFreqFeautures(self, sensor):
        # if Wavelet is enabled
        if self.Config['Sensors'][sensor]['features']['wavPowers']:    
            _, pows, nodes = packTrasform(self.snap[sensor]['timeSerie'],     # perform the wavelet trasform
                                                wavelet=self.Config["wavelet"]["type"],
                                                mode=self.Config["wavelet"]["mode"],
                                                maxlevel=self.Config["wavelet"]["maxlevel"], 
                                                plot=False)
            self.features[sensor].update(dict(zip(nodes, pows)))  # create a dictionary with nodes as keys and powers as values
            print(f"Wavelet coefs extracted from [purple]{sensor}[/]", show=False)
                
    def _writeToUnconsumed(self):
        ''' write the extracted features to the Unconsumed collection '''
        __dummy=self.features.copy() # create a copy of the features dictionary
        self.col_F.insert_one(__dummy) # insert the features in the Unconsumed collection, without changing the dictionary

        