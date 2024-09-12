import numpy as np
import pymongo
import pathlib
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from .modelfiles import save_model_file, load_model_file
from .printlog import print
class LOF():
    def __init__(self, db: pymongo.database.Database, detection_mode = 'novelty', n_neighbors=20, novelty=True, contamination=0.005, autosave = True, savefolder='framework/models'):
        self.db = db                                    # database object   
        self.n_neighbors = n_neighbors                  # number of neighbors
        self.novelty = novelty                          # novelty flag
        self.contamination = contamination              # contamination
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
        return f"LOF model"
    def __str__(self):
        return 'LOF'
    def train(self, X_train: np.ndarray):
        # train the LOF model
        print("training LOF model...",show=False)
        model=LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=self.novelty, contamination=self.contamination)
        model.fit(X_train)
        y_pred_train = model.predict(X_train) 
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
    def retrieve(self):
        path = pathlib.Path(self.savefolder,str(self)+'.pkl')
        self.model = load_model_file(path)
