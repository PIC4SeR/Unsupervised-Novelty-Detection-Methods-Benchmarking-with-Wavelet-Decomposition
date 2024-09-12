from .autoencoder import Autoencoder, train_nn, predict_nn
import numpy as np
from .printlog import print


def objective_reduced(trial, X_train, X_test, model, n_samp_variance):
    midlay = trial.suggest_int('midlay', 50, 65)
    latent = trial.suggest_int('latent', 10, 45)
    lr = trial.suggest_float('lr', 0.01, 0.1)
    batch_size = trial.suggest_categorical('batch_size', [32,64])
    aemodel = Autoencoder(midlay,latent)
    aemodel = train_nn(aemodel,X_train,epochs=150,batch_size=batch_size,lr=lr)
    _, train_latent_representation, _ = predict_nn(aemodel,X_train)
    _, test_latent_representation, _ = predict_nn(aemodel,X_test)
    try:
        model.train(train_latent_representation.cpu().numpy())
        NovMetric = model.evaluate(test_latent_representation.cpu().numpy())
        model.train(X_train)
        NovMetric_raw= model.evaluate(X_test)

        # compute the variance in the first chunk of the test dataset
        if n_samp_variance > 0:
            Variance = float(np.var(NovMetric[:n_samp_variance]))
            Variance_raw = float(np.var(NovMetric_raw[:n_samp_variance]))
            print(f"Variance for {model} is {Variance} over the first {n_samp_variance} samples.", show=False)
            print(f"Variance for {model} is {Variance_raw} over the first {n_samp_variance} raw samples.", show=False)

        return Variance
    except:
        return 10000  # return big variance if the model fails
    
def objective_augmented(trial, X_train, X_test, model, n_samp_variance):
    midlay = trial.suggest_int('midlay', 75, 80)
    latent = trial.suggest_int('latent', 85, 100)
    lr = trial.suggest_float('lr', 0.01, 0.1)
    batch_size = trial.suggest_categorical('batch_size', [32,64])
    aemodel = Autoencoder(midlay,latent)
    aemodel = train_nn(aemodel,X_train,epochs=150,batch_size=batch_size,lr=lr)
    _, train_latent_representation, _ = predict_nn(aemodel,X_train)
    _, test_latent_representation, _ = predict_nn(aemodel,X_test)
    try:
        model.train(train_latent_representation.cpu().numpy())
        NovMetric = model.evaluate(test_latent_representation.cpu().numpy())
        model.train(X_train)
        NovMetric_raw= model.evaluate(X_test)

        # compute the variance in the first chunk of the test dataset
        if n_samp_variance > 0:
            Variance = float(np.var(NovMetric[:n_samp_variance]))
            Variance_raw = float(np.var(NovMetric_raw[:n_samp_variance]))
            print(f"Variance for {model} is {Variance} over the first {n_samp_variance} samples.", show=False)
            print(f"Variance for {model} is {Variance_raw} over the first {n_samp_variance} raw samples.", show=False)

        return Variance
    except:
        return 10000  # return big variance if the model fails