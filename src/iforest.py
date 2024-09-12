import numpy as np
import pymongo
import pathlib
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from .modelfiles import save_model_file, load_model_file
from sklearn.ensemble import IsolationForest
from .printlog import print

class IF():
    def __init__(self, db: pymongo.database.Database, detection_mode = 'novelty', random_state = 0, autosave = True, savefolder='framework/models'):
        self.db = db                                    # database object   
        self.random_state = random_state                # random state
        self.detection_mode = detection_mode            # 'novelty' or 'fault' detection mode
        self.autosave = autosave                        # autosave flag
        # Swap colleciton for consistency
        if detection_mode == 'novelty':
            self.db.col_train = db.HEALTHY_TRAIN
        else:
            self.db.col_train = db.FAULTY_TRAIN
        self.savefolder = savefolder            # save folder
        return
    def __repr__(self):
        return f"IForest model"
    def __str__(self):
        return 'IForest'
    def train(self, X_train: np.ndarray):
        # train the IForest model
        print("training IForest model...",show=False)
        model=IsolationForest(random_state=self.random_state,verbose=0,max_samples=np.shape(X_train)[0])
        y_pred_train = model.fit_predict(X_train)
        self.model = model # save the model in the object 
        # save the model
        if self.autosave or input('Save the model? (y/n)')=='y':
            path = pathlib.Path(self.savefolder,str(self)+'.pkl')
            self.save(path) # save the model
        # return the labels
        return y_pred_train
    def evaluate(self, X: np.ndarray):
        # compute the novelty metric
        return -self.model.score_samples(X)
    def predict(self, X: np.ndarray):
        return self.model.predict(X)
    def save(self,path=''):
        #save_model_mongo(self.model, self.db.col_train, str(self))
        save_model_file(self.model, path)
        print(f"{str(self)} saved as pickled data into '{path}'",show=False)
    def retrieve(self):
        path = pathlib.Path(self.savefolder,str(self)+'.pkl')
        self.model = load_model_file(path)
        print(f"{str(self)} retrieved from pickled data",show=False)
    