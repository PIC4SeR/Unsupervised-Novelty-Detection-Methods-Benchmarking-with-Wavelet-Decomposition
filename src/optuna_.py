import optuna
from tqdm import tqdm
from . import globals
def recreate_study(name, optuna_storage="sqlite:///db.sqlite3"):
    try:
        optuna.delete_study(study_name=name, storage=optuna_storage)
    except:
        pass
    return optuna.create_study(storage=optuna_storage,
                                study_name=name,
                                direction='minimize',
                                sampler=optuna.samplers.GPSampler(),
                                )

def optuna_callback(study:optuna.study.Study, trial:optuna.trial.FrozenTrial):
    globals.OPTUNA_BAR.update(1)
    return None