import numpy as np

def reractivity(novelty_metric: np.ndarray, slice_test_train: int):
    INITIAL_MEAN = np.mean(novelty_metric[:slice_test_train])
    FINAL_MEAN = np.mean(novelty_metric[slice_test_train:])
    METRIC = (FINAL_MEAN-INITIAL_MEAN) #/INITIAL_MEAN
    return float(METRIC)