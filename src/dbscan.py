import numpy as np
import pymongo
import pathlib
from sklearn.cluster import KMeans, DBSCAN as dbscan_skit
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from .modelfiles import save_model_file, load_model_file
from .printlog import print

class mydbscan(dbscan_skit):
    def __init__(self,
        eps=0.5,
        *,
        min_samples=5,
        metric="euclidean",
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        p=None,
        n_jobs=None):
        super().__init__(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params, algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=n_jobs)
    def predict(self, X: np.ndarray):
        nr_samples = X.shape[0]
        y = np.ones(shape=nr_samples, dtype=int) * -1   # initialize the labels
        for i in range(nr_samples):
            indx, dist = self.clst_smpl(X[i,:])         # index of the closest cluster instance
            if dist < self.eps:                   # if the distance is less than eps
                y[i] = self.labels_[indx]         # assign the label
        return y                                        # return the labels
    def clst_smpl(self, X: np.ndarray):
        diff = self.components_ - X               # NumPy broadcasting
        dist = np.linalg.norm(diff, axis=1)             # Euclidean distance
        try :
            index = np.argmin(dist)                         # index of the closest cluster instance
        except:
            index = None
            return index, np.inf
        return  index, dist[index]                      # index of the closest cluster instance, distance to closest cluster instance


class DBSCAN():
    def __init__(self, db: pymongo.database.Database, detection_mode ='Novelty', eps_lim = (1,20), eps_n = 100, min_samples=5, autosave = True, savefolder='framework/models'):
        self.db = db                                # database object   
        self.min_samples = min_samples              # number of samples in a neighborhood for a point to be considered as a core point
        self.detection_mode = detection_mode        # 'novelty' or 'fault' detection mode
        self.eps_iter = np.linspace(eps_lim[0],eps_lim[1],eps_n) # eps values
        self.autosave = autosave                    # autosave flag
        # Swap colleciton for consistency
        if detection_mode == 'novelty':
            self.db.col_train = db.HEALTHY_TRAIN
        else:
            self.db.col_train = db.FAULTY_TRAIN
        self.savefolder = savefolder            # save folder
        return
    def __repr__(self):
        return f"DBSCAN model with eps={self.eps} and min_samples={self.min_samples}"
    def __str__(self):
        return 'DBSCAN'
    def train(self, X_train: np.ndarray):
        #  train with different number of clusters - DBSCAN Part
        sil_score=[]
        for self.eps in tqdm(self.eps_iter,desc='Training DBSCAN with different eps', position=2, leave=False):
            model=mydbscan(eps=self.eps,min_samples=self.min_samples).fit(X_train)
            try:
                sil_score.append(silhouette_score(X_train, model.labels_))
            except:
                sil_score.append(0)
        # chose the best number of clusters
        self.eps = self.eps_iter[np.argmax(sil_score)]
        print('The best eps is',self.eps, show=False)

        # train the DBSCAN model
        model=mydbscan(eps=self.eps,min_samples=self.min_samples).fit(X_train)
        y_pred_train=model.fit_predict(X_train)
        print('DBSCAN model trained with',
              self.eps,'eps', 'number of features:',
              X_train.shape[1], 'number of samples:',
              X_train.shape[0],
              show=False)
        self.model = model # save the model in the object
    
        # save the model
        if self.autosave or input('Save the model? (y/n)')=='y':
            path = pathlib.Path(self.savefolder,str(self)+'.pkl')
            self.save(path) # save the model
        # return the labels
        return y_pred_train
    def evaluate(self, X: np.ndarray):
        NovMetric = np.array([])                        # array to store the distances to the closest cluster instances
        for snap in X:
            _, dist = self.model.clst_smpl(snap)              # index of the closest cluster instance
            NovMetric = np.append(NovMetric, dist/self.model.eps)      # store the distance
        return NovMetric                                # return the novelty metric

    def save(self,path=''):
        #save_model_mongo(self.model, self.db.col_train, str(self))
        save_model_file(self.model, path)
        print(f"{str(self)} saved as pickled data into '{path}'", show=False)
    def retrieve(self):
        path = pathlib.Path(self.savefolder,str(self)+'.pkl')
        self.model = load_model_file(path)
        print(f"{str(self)} retrieved from pickled data", show=False)

    def predict(self, X: np.ndarray):
        return self.model.predict(X)
