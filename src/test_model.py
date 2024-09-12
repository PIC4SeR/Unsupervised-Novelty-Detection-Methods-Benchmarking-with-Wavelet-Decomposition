import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import types
import time
import src
from .printlog import print

def test_model(model, X_train = np.ndarray([]),
                X_test              = np.ndarray([]), *, 
                train_flag          = False, 
                n_samp_variance     = 100, 
                NormalizeNovMetric  = True):
    # train and evaluate the model with original dataset
    if train_flag:
        model.train(X_train)
    else:
        model.retrieve()
        model.predict(X_train)
    start = time.time()
    eval_result = model.evaluate(X_test)             # compute the novelty metric
    end = time.time()
    print(f"Time for {model} is {end-start} seconds to evaluate {np.shape(X_test)[0]} samples.")
    inference_time = (end - start)/np.shape(X_test)[0] # inference time per sample
    NovMetric = np.array(eval_result).reshape(-1, 1) # reshape the novelty metric
    if NormalizeNovMetric:
        NovMetric = MinMaxScaler().fit_transform(NovMetric) # normalize the novelty metric

    # compute the variance in the first chunk of the test dataset
    if n_samp_variance > 0:
        Variance = np.var(NovMetric[:n_samp_variance])
        print(f"Variance for {model} is {Variance} over the first {n_samp_variance} samples.")
        Mean_1 = np.mean(NovMetric[:n_samp_variance])
        print(f"Mean for {model} is {Mean_1} over the first {n_samp_variance} samples.")
        Mean_2 = np.mean(NovMetric[n_samp_variance:])
        print(f"Mean for {model} is {Mean_2} over the last {np.shape(NovMetric)[0]-n_samp_variance} samples.")

    # compute the metric for model evaluaiton
    return Variance, Mean_1, Mean_2, src.reractivity(NovMetric,n_samp_variance), inference_time, NovMetric #NovMetric, Variance, Mean, NovFlag