import numpy as np
import pymongo
import pathlib
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from .modelfiles import save_model_file, load_model_file
from sklearn.svm import OneClassSVM
from .printlog import print

class nuSVM():
    def __init__(self, db: pymongo.database.Database, detection_mode = 'novelty', kernel='rbf', nu=0.1, autosave = True, savefolder='framework/models'):
        self.db = db                                    # database object   
        self.kernel = kernel                            # kernel type
        self.nu = nu                                    # nu value
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
        return f"nuSVM model"
    def __str__(self):
        return 'nuSVM'
    def train(self, X_train: np.ndarray):
        # train the nuSVM model
        print("training nuSVM model...",show=False)
        model=OneClassSVM(kernel=self.kernel,nu=self.nu,verbose=0)
        y_pred_train = model.fit_predict(X_train)
        if model.fit_status_: # check if the model has converged
            raise Exception('The nuSVM model has not converged')
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
   