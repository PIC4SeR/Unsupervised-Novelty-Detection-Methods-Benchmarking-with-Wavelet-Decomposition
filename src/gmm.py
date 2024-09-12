import numpy as np
import pymongo
import pathlib
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from .modelfiles import save_model_file, load_model_file
from .printlog import print

class GMM():
    def __init__(self, db: pymongo.database.Database, detection_mode = 'novelty', covariance_type = 'full', autosave = True, savefolder='framework/models'):
        self.db = db                                    # database object   
        self.cova_type = covariance_type                # covariance type
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
        return f"GMM model"
    def __str__(self):
        return 'GMM'
    def train(self, X_train: np.ndarray):
        # train the GMM model
        model=BayesianGaussianMixture(n_components=np.shape(X_train)[0],covariance_type=self.cova_type,verbose=0)
        n_trial = 0
        while True: # check if the model has converged
            y_pred_train = model.fit_predict(X_train)
            n_trial += 1
            if not model.converged_ and n_trial < 50:
                pass
            else:
                break
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
 