# *************************************************************************************************** #
#                                        IMPORT LIBRARIES                                             #
# %%************************************************************************************************* #
import yaml, os, pymongo, logging, numpy as np, optuna, torch, warnings, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from optuna.exceptions import ExperimentalWarning
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import src
from pymongo.collection import Collection
from typing import List
from tqdm import tqdm, trange
from src import kmeans, DBSCAN, GMM, nuSVM, IF, LOF, Autoencoder, test_model
from src.autoencoder import best_aemodel
import src.globals
import src.optuna_ as myoptuna
from sklearn.decomposition import PCA
from rich import print
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
import matplotlib.patches as patches


src.set_matplotlib_params() # set matplotlib parameters to generate plots for the thesis

# *************************************************************************************************** #
#                                       DEMO CONFIGURATION                                            #       
# %%************************************************************************************************* #

data_code_train         = 1                     # select training data - (int or list - number of the
                                                # test, str - environment [IMS, SHAK])
data_code_test          = [1,2]                 # select test data - (int or list - number of the test, 
                                                # str - environment [IMS, SHAK]) 
                                                # e.g. [i for i in range(1,9)] means to use all the 
                                                # experimental data

slicing_train           = (0,100)               # slicing of the dataset for training (start, end)
slicing_test            = (100,-1)              # slicing of the dataset for testing (start, end)
n_samp_variance         = 106                   # number of samples to use for the variance computation
                                                # (to be minimized)
                                                # 106 because each test has 206 samples (100 for training and 106 for testing)

load_timeseries_flag    = True                  # if True, the timeseries are loaded to MongoDB
run_FA_flag             = True                  # if True, the FA is run
train_flag              = False                 # if True, the models are trained
NormalizeNovMetric      = True                  # if True, the novelty metric is normalized

Generate_PCA            = False                 # if True, the PCA models are generated
Optimize_Autoencoder    = False                 # if True, the Autoencoder models are optimized
Fit_Autoencoder         = False                  # if True, the Autoencoder models are fitted
plot_patches            = False                 # if True, the variance is visualized as a patch in the plot

modelfolder             = 'models/'             # folder where to save the models
detection_mode          = 'novelty'                         # 'novelty' or 'fault' detection mode
Shak_folder             = 'data/Experimental'               # folder where the shaker timeseries are stored
IMS_folder              = 'data/IMS'                        # folder where the IMS timeseries are stored
URI                     = 'mongodb://localhost:27017' # URI of the MongoDB
logfile                 = 'logs/demo.log'                   # path where to save the logs
optuna_storage          = 'sqlite:///db.sqlite3'            # path where to save the optuna study

figsize                 = (3.48,7.8)                        # size of the figures [inch]

# *************************************************************************************************** #
#                                SELECTING THE RIGHT CONFIG FILE                                      #
# %%************************************************************************************************* #

data_code_train = [data_code_train] if type(data_code_train) is int else data_code_train
data_code_test = [data_code_test] if type(data_code_test) is int else data_code_test


configfilepath = os.path.join(Path(__file__).parent, "config/config_SHAKER.yaml")

try:  # Load the configuration file
    with open(configfilepath,'r') as f:
        config = yaml.safe_load(f)
        print(f'Loaded config file @ \n{configfilepath}')
except:
    raise Exception(f'Error reading config file @ {configfilepath}')

# logging configuration
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, 
                    filemode='w', 
                    filename=os.path.join(Path(__file__).parent, logfile), )


# *************************************************************************************************** #
#                                      LOAD THE TIMESERIES                                            #
# %%************************************************************************************************* #

client = pymongo.MongoClient(URI) # connect to the MongoDB
db = client['Experiment'] # select the database
        

if load_timeseries_flag:
    # drop all the collections if starting from scratch
    for collection in db.list_collection_names():
        db.drop_collection(collection)
    timeSeriesCollection: List[Collection] = src.ShakToMongo(Shak_folder,db,1666,1)    
 

# *************************************************************************************************** #
#                                  RUN FEATURE EXTRACTION                                             #
# %%************************************************************************************************* #
if run_FA_flag:
    for timeserie_col in tqdm(timeSeriesCollection, desc='Extracting features from timeseries',
                              leave=True, position=0):
        # link to the features collection
        features_col_name = timeserie_col.name.removesuffix('_timeseries') + '_features'
        features_col = db.create_collection(features_col_name)
        FA = src.FA(configfilepath, URI, timeserie_col, features_col, 1)
        FA._readFromRaw()
        for i in trange(len(FA.snaps), desc=f"Feature extraction into {features_col.name}", 
                            position=1, ):  # run until there is data in the RAW collection
            FA.snap = FA.snaps[i]
            FA._extractFeatures()
            FA._writeToUnconsumed()

# *************************************************************************************************** #
#                                  GENERATE / LOAD MODELS                                             #
# %%************************************************************************************************* #

mod_KMeans          = kmeans(db, detection_mode,max_clusters=50,savefolder=modelfolder)
mod_DBSCAN          = DBSCAN(db, detection_mode,savefolder=modelfolder)
mod_GMM             = GMM(db, detection_mode,savefolder=modelfolder)
mod_nuSVM           = nuSVM(db, detection_mode,savefolder=modelfolder)
mod_IF              = IF(db, detection_mode,savefolder=modelfolder)
mod_LOF             = LOF(db, detection_mode,savefolder=modelfolder)

# *************************************************************************************************** #
#                                  LOAD DATASET IN RAM                                                #
# %%************************************************************************************************* #

X_train = []        # training dataset
X_test = []        # test dataset

# load the train dataset

print("loading the train dataset...")

for test_number in data_code_train:
    collection = db[f'test{test_number}_features']  # collection to read
    X_loaded, features_names = src.load_features(collection,config)
    X_train.append(X_loaded)

# slice the dataset
X_train = np.array(X_train)
X_train = X_train.reshape(-1,len(features_names))
X_train = X_train[slicing_train[0]:slicing_train[1],:]

# load the test dataset
for test_number in data_code_test:
            collection = db[f'test{test_number}_features']  # collection to read
            X_loaded, _ = src.load_features(collection,config)
            X_test.append(X_loaded)

# slice the dataset
X_test = np.array(X_test)
X_test = X_test.reshape(-1,len(features_names))
X_test = X_test[slicing_test[0]:slicing_test[1],:]

print("datasets loaded - train shape:",X_train.shape,"test shape:",X_test.shape)

# *************************************************************************************************** #
#                                  STANDARDIZE THE DATASET                                            #
# %%************************************************************************************************* #

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("datasets standardized")

# *************************************************************************************************** #
#                                  OPTIMIZE AUTOENCODERS                                              #
# %%************************************************************************************************* #
optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=ExperimentalWarning)
src.globals.globals_init() # initialize the global variable
if Optimize_Autoencoder: # if optimizing enabled
    best_hprprm = {}
    best_value= {}
    for model in tqdm([mod_KMeans, mod_DBSCAN, mod_GMM, mod_nuSVM, mod_IF, mod_LOF],
                      desc='Optimizing Autoencoders all UML models', leave=True, position=0):
        name = f"AE_{model}"
        best_hprprm[str(model)] = {} # best hyperparameters
        best_value[str(model)] = {}     # best value
        best_hprprm[str(model)]['reduced'] = [] # best hyperparameters for reduced
        best_value[str(model)]['augmented'] = [] # best hyperparameters for augmented
        src.globals.OPTUNA_BAR = tqdm(total=50, desc=f'Optimizing AE_reduced for {model} model', position=1, leave=False)
        study = src.recreate_study(name, optuna_storage)
        study.optimize(lambda trial: src.objective_reduced(trial, X_train, X_test, model, n_samp_variance),
                        n_trials=50,
                        callbacks=[src.optuna_callback])
        best_hprprm[str(model)]['reduced'] = study.best_params
        best_value[str(model)]['reduced'] = study.best_value
        src.globals.OPTUNA_BAR = tqdm(total=50, desc=f'Optimizing AE_augmented for {model} model', position=1, leave=False)
        study = src.recreate_study(name, optuna_storage) # recreate the study
        study.optimize(lambda trial: src.objective_augmented(trial, X_train, X_test, model, n_samp_variance),
                        n_trials=50,
                        callbacks=[src.optuna_callback])
        best_hprprm[str(model)]['augmented'] = study.best_params
        best_value[str(model)]['augmented'] = study.best_value
    with open(os.path.join(Path(__file__).parent, modelfolder,'best_hprprm.yaml'), 'w') as file:     # save the best parameters to file
        documents = yaml.dump(best_hprprm, file)
else:
    try:
        with open(os.path.join(Path(__file__).parent, modelfolder,'best_hprprm.yaml'), 'r') as file:    # load the best parameters from file
            best_hprprm = yaml.safe_load(file)
    except:
        print("No best hyperparameters found. Please run the optimization first (set Generate_Autoencoder to True).")

# *************************************************************************************************** #
#                               FIT AUTOENCODER WITH BEST HYPERPARAMETERS                             #                   
#                                  AUTOENCODER FEATURE TRANSFORMATION                                 #
# %%************************************************************************************************* #

X_train_autoencoder = {}
X_test_autoencoder = {} 

for model in tqdm([mod_KMeans, mod_DBSCAN, mod_GMM, mod_nuSVM, mod_IF, mod_LOF],
                    desc='Fitting Autoencoders and transforming the dataset', leave=True, position=0):
    X_train_autoencoder[str(model)] = {}
    X_test_autoencoder[str(model)] = {}
    type_map = ['reduced','augmented']
    for Type in [0,1]:
        ae_model = Autoencoder(best_hprprm[str(model)][type_map[Type]]['midlay'],best_hprprm[str(model)][type_map[Type]]['latent'])
        if Fit_Autoencoder:
            best_aemodel(model,
                         X_train,
                         X_test,
                         ae_model,
                         best_hprprm[str(model)][type_map[Type]]['lr'],
                         best_hprprm[str(model)][type_map[Type]]['batch_size'],
                         n_samp_variance,
                         Type,
                         savepath=os.path.join(Path(__file__).parent, modelfolder))
        ae_model.load_state_dict(torch.load(os.path.join(Path(__file__).parent, modelfolder,f'AE_Models/aemodel_{type_map[Type]}_{model}.pth')))
        output_train, X_train_autoencoder[str(model)][type_map[Type]], error_train = src.predict_nn(ae_model,X_train)
        output_test, X_test_autoencoder[str(model)][type_map[Type]], error_test = src.predict_nn(ae_model,X_test)
        X_train_autoencoder[str(model)][type_map[Type]] = X_train_autoencoder[str(model)][type_map[Type]].cpu().numpy()
        X_test_autoencoder[str(model)][type_map[Type]] = X_test_autoencoder[str(model)][type_map[Type]].cpu().numpy()

# *************************************************************************************************** #
#                                  OPTIMIZE PCA                                                       #
# %%************************************************************************************************* #
src.globals.globals_init() # initialize the global variable
if Generate_PCA: # to avoid commenting all cell 
    best_n_features = {}
    for model in tqdm([mod_KMeans, mod_DBSCAN, mod_GMM, mod_nuSVM, mod_IF, mod_LOF],
                      desc='Optimizing PCA on all UML models', leave=True, position=0):
        name = f"PCA_{model}"
        src.globals.OPTUNA_BAR = tqdm(total=50, desc=f'Optimizing PCA for {model} model', position=1, leave=False)
        study = src.recreate_study(name, optuna_storage)
        study.optimize(lambda trial: src.pca_obj_function(trial, X_train,X_test, model,n_samp_variance),
                       n_trials=50,
                       callbacks=[src.optuna_callback])
        best_n_features[str(model)] = study.best_params
    # save the best parameters to file
    with open(os.path.join(Path(__file__).parent, modelfolder,'best_n_features.yaml'), 'w') as file:
        documents = yaml.dump(best_n_features, file)
else:
    with open(os.path.join(Path(__file__).parent, modelfolder,'best_n_features.yaml'), 'r') as file:
        best_n_features = yaml.safe_load(file)

# *************************************************************************************************** #
#                                  PCA FEATURE REDUCTION                                              #
# %%************************************************************************************************* #

X_train_PCA = {}
X_test_PCA = {} 
for model in [mod_KMeans, mod_DBSCAN, mod_GMM, mod_nuSVM, mod_IF, mod_LOF]:
    x = best_n_features[str(model)]['n_princ_comp']
    PCA_model = PCA(n_components=x)
    PCA_model.fit(X_train)
    
    # reasform the dataset
    X_train_PCA[str(model)] = np.array(PCA_model.transform(X_train))
    X_test_PCA[str(model)] = np.array(PCA_model.transform(X_test))

# *************************************************************************************************** #
#                                  MAP THE TRAIN / TEST DATASET                                       #
# %%************************************************************************************************* #

X_train_dict = {
    'original'  : X_train,
    'AE_reduced':
    {
        'KMeans'    : X_train_autoencoder['KMeans']['reduced'],
        'DBSCAN'    : X_train_autoencoder['DBSCAN']['reduced'],
        'GMM'       : X_train_autoencoder['GMM']['reduced'],
        'nuSVM'     : X_train_autoencoder['nuSVM']['reduced'],
        'IForest'   : X_train_autoencoder['IForest']['reduced'],
        'LOF'       : X_train_autoencoder['LOF']['reduced']
    },
    'AE_augmented':
    {
        'KMeans'    : X_train_autoencoder['KMeans']['augmented'],
        'DBSCAN'    : X_train_autoencoder['DBSCAN']['augmented'],
        'GMM'       : X_train_autoencoder['GMM']['augmented'],
        'nuSVM'     : X_train_autoencoder['nuSVM']['augmented'],
        'IForest'   : X_train_autoencoder['IForest']['augmented'],
        'LOF'       : X_train_autoencoder['LOF']['augmented']
    },
    'PCA'      :
    {
        'KMeans'    : X_train_PCA['KMeans'],
        'DBSCAN'    : X_train_PCA['DBSCAN'],
        'GMM'       : X_train_PCA['GMM'],
        'nuSVM'     : X_train_PCA['nuSVM'],
        'IForest'   : X_train_PCA['IForest'],
        'LOF'       : X_train_PCA['LOF']
    }
}

X_test_dict = {
    'original'  : X_test,
    'AE_reduced': 
    {
        'KMeans'    : X_test_autoencoder['KMeans']['reduced'],
        'DBSCAN'    : X_test_autoencoder['DBSCAN']['reduced'],
        'GMM'       : X_test_autoencoder['GMM']['reduced'],
        'nuSVM'     : X_test_autoencoder['nuSVM']['reduced'],
        'IForest'   : X_test_autoencoder['IForest']['reduced'],
        'LOF'       : X_test_autoencoder['LOF']['reduced']
    },
    'AE_augmented':
    {
        'KMeans'    : X_test_autoencoder['KMeans']['augmented'],
        'DBSCAN'    : X_test_autoencoder['DBSCAN']['augmented'],
        'GMM'       : X_test_autoencoder['GMM']['augmented'],
        'nuSVM'     : X_test_autoencoder['nuSVM']['augmented'],
        'IForest'   : X_test_autoencoder['IForest']['augmented'],
        'LOF'       : X_test_autoencoder['LOF']['augmented']
    },
    'PCA'      :
    {
        'KMeans'    : X_test_PCA['KMeans'],
        'DBSCAN'    : X_test_PCA['DBSCAN'],
        'GMM'       : X_test_PCA['GMM'],
        'nuSVM'     : X_test_PCA['nuSVM'],
        'IForest'   : X_test_PCA['IForest'],
        'LOF'       : X_test_PCA['LOF']
    }
}

# *************************************************************************************************** #
#                                  TRAIN / EVAL THE MODELS                                            #
# %%************************************************************************************************* #

# train and evaluate the models

Results = {}
NovMetric = {}

n_features_original = np.shape(X_train)[1]

for model in tqdm([mod_KMeans, mod_DBSCAN, mod_GMM, mod_nuSVM, mod_IF, mod_LOF],
                  desc='(Training and) evaluating the models', leave=True, position=0):
    # initialize the dictionaries
    Results[str(model)] = {}
    NovMetric[str(model)] = {}
    
    for reduction in tqdm(['original','AE_reduced','AE_augmented','PCA'],
                          desc=f"(Training and) evaluating {model} performances", leave=False, position=1):

        model.savefolder = os.path.join(modelfolder, reduction)

        Results[str(model)][reduction] = {}

        match reduction:
            case 'original':
                X_train_red = X_train_dict[reduction]
                X_test_red = X_test_dict[reduction]
            case _:
                X_train_red = X_train_dict[reduction][str(model)]
                X_test_red = X_test_dict[reduction][str(model)]
        n_features_reduced = np.shape(X_train_red)[1]
        Variance, Mean_1, Mean_2, Reractivity, inference_time, NovMetric_result = test_model(model,
            X_train_red,
            X_test_red,
            train_flag = train_flag,  
            n_samp_variance = n_samp_variance, 
            NormalizeNovMetric = NormalizeNovMetric)
        
        NovMetric[str(model)][reduction] = NovMetric_result
        Results[str(model)][reduction]['n_red_features'] = n_features_reduced
        Results[str(model)][reduction]['Feat_red_factor'] = n_features_reduced/n_features_original*100
        Results[str(model)][reduction]['Variance'] = Variance
        Results[str(model)][reduction]['Mean_1'] = Mean_1
        Results[str(model)][reduction]['Mean_2'] = Mean_2
        Results[str(model)][reduction]['Reactivity'] = Reractivity
        Results[str(model)][reduction]['Inference_time'] = inference_time
    
# *************************************************************************************************** #
#                                  CPMPILE THE LATEX RESULTS                                          #
# %%************************************************************************************************* #

src.print_result_table(Results,os.path.join('results/preambolo.tex'),os.path.join('results/table.tex'))

df_res = pd.concat({k: pd.DataFrame(v).T for k, v in Results.items()}, axis=0)

# print in red 
print('Model performances evaluated!')
pd.set_option('display.precision', 2)
print(df_res)

# *************************************************************************************************** #
#                                       PLOT THE RESULTS                                              #
# %%************************************************************************************************* #


plt.close("all") # close all the previous plots

# set spines to be thinner as the minot ticks
cols = 1
rows = int(max(len(NovMetric.keys())/cols,cols))
fig, axs=plt.subplots(rows,cols,sharex=True)
fig.set_size_inches(figsize)
size = 0.5
colormap = ListedColormap(["blue", "red", "green", "orange", "cyan", "purple"])

# create the custom legend
dot1 = Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor=colormap(0), markersize=10)
dot2 = Line2D([0], [0], marker='o', color='w', label='Faulty', markerfacecolor=colormap(1), markersize=10)
dot3 = Line2D([0], [0], marker='o', color='w', label='Novelty', markerfacecolor=colormap(2), markersize=10)
dot4 = Line2D([0], [0], marker='o', color='w', label='Novelty', markerfacecolor=colormap(3), markersize=10)

# convert to 2D
axs_2d = np.atleast_2d(axs).reshape(rows,cols)

# adjust the layout
plt.subplots_adjust(hspace=0.0, wspace=0.0, top=.925, bottom=0.053, left=0.145, right=0.987)

# loop over the models
for i, model in enumerate(NovMetric.keys()):
    # first ax index - second ax index
    ax = axs_2d[i//cols, i%cols]
    for j, reduction in enumerate(['original', 'AE_reduced', 'AE_augmented', 'PCA']):
    # plot the novelty metric for the all the combinations
        ax.scatter(range(len(NovMetric[model][reduction])), NovMetric[model][reduction], c=colormap(j), s=size, marker='.')

    # plot the mean and variance
    if n_samp_variance > 0 and plot_patches:
        for j, reduction in enumerate(['original', 'AE_reduced', 'AE_augmented', 'PCA']):
            rectangle = patches.Rectangle((0, (Results[model][reduction]['Mean_1'] - np.sqrt(Results[model][reduction]['Variance']))), 
                                            n_samp_variance,
                                            2*np.sqrt(Results[model][reduction]['Variance']),
                                            alpha=0.3,
                                            edgecolor=None,
                                            facecolor=colormap(j),
                                            linewidth=0)
            ax.add_patch(rectangle)
    ax.set_ylim(-0.1,1.1)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    ax.set_title(model,x=0.99, y=0.05, pad=0, loc='right', fontsize=10)
ax.set_xlabel('Samples')
fig.supylabel('Normalized novelty metric')

# Getting the position of the first ax
pos1 = axs_2d[0,0].get_position()
pos1 = [pos1.x0, pos1.y0, pos1.width, pos1.height*1.35]

# Adding the custom legend
fig.legend([dot1, dot2, dot3, dot4], ['OF', 'AER', 'AEA', 'PCA'], loc='upper center',
           bbox_to_anchor=[0.49,1], ncol=4, borderaxespad=0.,frameon=False)
fig.suptitle('Novelty metric for the different models',y=0.96,x=pos1[0]+pos1[2]/2,ha='center',va='top',fontsize=12)

fig.savefig(os.path.join('results','Novelty_metric.pdf'))

# plot hystogram

fig, axs = plt.subplots(5,sharex=True)
fig.set_size_inches((figsize[0],figsize[1]/2))

# reduce the xtick font size

for ax_ind, metric in enumerate(['Variance','Mean_1','Mean_2','Reactivity','Inference_time']):

    # map titles
    titles = {
        'Variance' : 'Variance',
        'Mean_1' : 'Mean Train',
        'Mean_2' : 'Mean Test',
        'Reactivity' : 'Reactivity',
        'Inference_time' : 'Inf. time [s]'
    }

    xticklabels = []
    tick = []
    gap = 1
    width = 0.5
    pos = gap + width/2
    #axs[ax_ind].set_yticks([])
    axs[ax_ind].set_yscale('log')
    axs[ax_ind].set_title(titles[metric], pad=5)
    #axs[ax_ind].set_yscale('log')
    for model in Results.keys():
        for i, reduction in enumerate(Results[model].keys()):
            axs[ax_ind].bar(pos,Results[model][reduction][metric],width=width,label=model + ' ' + reduction, color=colormap(i))
            tick.append(pos)
            pos += (width)
            xticklabels.append(reduction) 
        pos += gap
    xticklabels = [xticklabels[i].replace('original', 'None').replace('_reduced','R').replace('_augmented', 'A') for i in range(0,len(xticklabels))]
    axs[ax_ind].set_xticks(tick)
    axs[ax_ind].set_xticklabels([]) #(xticklabels,rotation=90, fontsize=8)
    axs[ax_ind].set_xlim(width/2,pos-gap)
    axs[ax_ind].yaxis.set_major_locator(ticker.LogLocator())
    axs[ax_ind].yaxis.set_minor_locator(ticker.LogLocator())
    axs[ax_ind].yaxis.set_minor_formatter(ticker.NullFormatter())

    # ylim = axs[ax_ind].get_ylim()
    # if ylim[0] > 0.1:
    #     axs[ax_ind].set_ylim(0.09,ylim[1])
    
fig.legend([dot1, dot2, dot3, dot4], ['OF', 'AER', 'AEA', 'PCA'], loc='upper center',
           bbox_to_anchor=[0.49,1], ncol=4, borderaxespad=0.,frameon=False)

fig.text(0.2, 0.02, 'Kmeans', ha='center', va='center', fontsize=9)
fig.text(0.347, 0.02, 'DBSCAN', ha='center', va='center', fontsize=9)
fig.text(0.495, 0.02, 'GMM', ha='center', va='center', fontsize=9)
fig.text(0.6375, 0.02, 'nuSVM', ha='center', va='center', fontsize=9)
fig.text(0.79, 0.02, 'IForest', ha='center', va='center', fontsize=9)
fig.text(0.936, 0.02, 'LOF', ha='center', va='center', fontsize=9)
fig.subplots_adjust(top=0.88,bottom=0.0625,left=0.1125,right=0.992,hspace=0.8)

plt.savefig(os.path.join('results','Metrics.pdf'))

import seaborn as sns
fig, ax = plt.subplots()
fig.set_size_inches(figsize[0],figsize[1]/2)
sns.heatmap(df_res.iloc[:,1:].corr(), annot=True, cmap='coolwarm', ax=ax)
plt.title('Correlation matrix')
plt.subplots_adjust(top=0.94,
bottom=0.26,
left=0.295,
right=0.9,
hspace=0.2,
wspace=0.2)
plt.savefig(os.path.join('results','Correlation.pdf'))

# show the table plot
fig, ax = plt.subplots()
fig.set_size_inches(figsize)
ax.axis('off')

# # print the table.tex file
# ax.text(0, 0, open(os.path.join('results','table.tex')).read())
# plt.savefig(os.path.join('results','Table.pdf'))
