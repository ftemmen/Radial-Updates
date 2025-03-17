import numpy as np
import pandas as pd
from typing import Optional

from utils import get_NSL_lattices_from_h5


def compute_and_save_observables(cfgs: np.ndarray, 
                                 obs_dict: dict, 
                                 csv_path: str):
    """
    obs_dict should have strings as keys with the name of the observable and a callable function that takes the array and computes
    the respective observable, i.e. returns an array of length [N_samples]
    """
    full_dict = {}
    for obs_name, obs_func in obs_dict.items():
        full_dict[obs_name] = obs_func(cfgs)
    df = pd.DataFrame(full_dict)
    df.to_csv(csv_path, index = False)
    
def load_comp_save_obs(path: str, 
                       obs_dict: dict, 
                       Nconf_max: Optional[int] = None, 
                       csv_path: Optional[str] = None):
    if Nconf_max==None:
        Nconf_max = int(1e18)
    cfgs = get_NSL_lattices_from_h5(path, Nconf_max)
    if csv_path == None:
        csv_path = path[:-3]+"_obs.csv"
    compute_and_save_observables(cfgs, obs_dict, csv_path)
    
def mean_obs(cfgs: np.ndarray) -> np.ndarray:
    return np.mean(cfgs, axis = 1)
    
def abs_mean_obs(cfgs: np.ndarray) -> np.ndarray:
    return np.mean(np.fabs(cfgs), axis = 1)

def compute_time_slice_radius(data: np.ndarray, Nt: int, Nx: int) -> np.ndarray:
    Nconf = data.shape[0]
    reshaped_data = data.reshape(Nconf, Nt, Nx)
    tmp = np.sqrt(np.sum(np.power(np.sum(reshaped_data, axis = 1), 2), axis = 1))
    return tmp

def compute_radius(data: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.power(data, 2), axis = 1))
