import numpy as np
import pymongo, os
import pathlib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from .modelfiles import save_model_file, load_model_file
from .printlog import print

def count_unique_elements(lst: np.ndarray):
    # Count the occurrences of each element in the list
    _, counts = np.unique(lst, return_counts=True)
    n_of_singlet = np.sum(counts==1)
    return n_of_singlet

class kmeans():
    def __init__(self, db: pymongo.database.Database, detection_mode = 'Novelty', max_clusters=100, max_iter=1000, n_init=10, autosave = True,savefolder='framework/models'):
        self.db = db                                # database object   
        self.max_clusters = max_clusters            # maximum number of clusters
        self.max_iter = max_iter                    # maximum number of iterations
        self.n_init = n_init                        # number of initializations
        self.detection_mode = detection_mode        # 'novelty' or 'fault' detection mode
        self.n_clust = None                         # number of clusters
        self.autosave = autosave                    # autosave flag
        # Swap colleciton for consistency
        if detection_mode == 'novelty':
            self.db.col_train = db.HEALTHY_TRAIN
        else:
            self.db.col_train = db.FAULTY_TRAIN            
        self.savefolder = savefolder            # save folder
        return
    def __repr__(self):
        return f"KMeans model with {self.n_clust} clusters"
    def __str__(self):
        return 'KMeans'
    def train(self, X_train: np.ndarray):
        #  train with different number of clusters - K-Means Part
        sil_score=[]
        inertia=[]
        for self.n_clust in tqdm(range(2,self.max_clusters+1),
                                 desc='Training K-Means with different number of clusters',
                                 position=2, leave=False):
            model=KMeans(self.n_clust,n_init='auto',max_iter=300)
            y_pred_train=model.fit_predict(X_train)
            sil_score.append(silhouette_score(X_train,y_pred_train))
            inertia.append(model.inertia_)
        # chose the best number of clusters
        self.n_clust = np.argmax(sil_score)+2
        # if there are clusters with one single sample, train with that much less clusters
        count_unique_elements(y_pred_train)
        self.n_clust = max(1,self.n_clust - count_unique_elements(y_pred_train)) # min one cluster

        print('The best number of clusters is',self.n_clust, show=False)

        # train the K-Means model
        model=KMeans(self.n_clust,n_init=10)
        y_pred_train=model.fit_predict(X_train)

        # compute the maximum distance from the centroid for each cluster in the train dataset
        cluster_distances=model.transform(X_train) # distance to each cluster in the train dataset
        model.max_dist=[] # maximum distance to eah cluster in the train dataset
        for cluster in range(0,self.n_clust):
            model.max_dist.append(max(cluster_distances[y_pred_train==cluster,cluster])) # maximum distance to each cluster in the train dataset seful for metric computation
        model.max_dist=model.max_dist # save the maximum distance to each cluster in the kmeans object
        self.model = model # save the model in the object

        # save the model
        if self.autosave or input('Save the model? (y/n)')=='y':
            path = pathlib.Path(self.savefolder,str(self)+'.pkl')
            self.save(path) # save the model
        # return the labels
        return y_pred_train
    def evaluate(self, X: np.ndarray):
        # compute the novelty metric
        NovMetric=np.array([]); 
        for snap in X:
            # print(np.shape(snap))
            snap=snap.reshape(1, -1)
            y=self.model.predict(snap) # predict the cluster for the new snap
            distance = self.model.transform(snap)[0,y[0]] # distance to the predicted cluster
            NovMetric = np.append(NovMetric,(distance-self.model.max_dist[int(y[0])])/self.model.max_dist[int(y[0])]*100) # novelty metric
        return NovMetric # return the novelty metric
    def predict(self, X: np.ndarray):
        return self.model.predict(X)
    def save(self,path=''):
        #save_model_mongo(self.model, self.db.col_train, str(self))
        save_model_file(self.model, path)
        print(f"{str(self)} saved as pickled data into '{path}'", show=False)
    def retrieve(self):
        path = os.path.join(self.savefolder,str(self)+'.pkl')
        self.model = load_model_file(path)
        print(f"{str(self)} retrieved from pickled data", show=False)