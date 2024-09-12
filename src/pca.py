import numpy as np
from sklearn.decomposition import PCA

def pca_obj_function(trial, X_train, X_test, model, n_samp_variance):
    n_feat = trial.suggest_int('n_princ_comp', 2, np.shape(X_train)[1]) # number of principal components minimum 2, maximum all the features
    PCA_model = PCA(n_components=n_feat)
    PCA_model.fit(X_train)

    # manage exceptions
    if n_samp_variance == 0:
        raise Exception("n_samp_variance = 0! ... nothing to compute the variance with!")
    if n_samp_variance > np.shape(X_test)[0]:
        raise Exception(f"n_samp_variance = {n_samp_variance} is larger than the number of samples in the test dataset ({np.shape(X_test)[0]})")
    
    try: # train the model with this specifica parameters
        model.train(np.array(PCA_model.transform(X_train))) # train the model with the PCA reduced representation
    except:
        return 100000 # if the model cannot be trained, return big variance to avoid it to be selected
    
    NovMetric = model.evaluate(np.array(PCA_model.transform(X_test))) # evaluate the model with the PCA reduced representation
    
    # compute the cost function to be minimized
    return np.var(NovMetric[:n_samp_variance])