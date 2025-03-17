import numpy as np
from typing import Union, Callable, Tuple

def mean_s(x: np.ndarray) -> Union[np.ndarray, np.float64]:
    """
    Estimator of the expection value x. Intended to be used with the function allmighty_bootstrap. 
    """
    return np.mean(x, axis = (0, 1))

def meff_s(x: np.ndarray) -> np.ndarray:
    """
    Estimator for the effective mass. Intended to be used with the function allmighty_bootstrap. 
    """
    C = np.mean(x, axis = (0, 1)) # correlator with shape (Nt) for the corresponding bootstrap samples
    logC = np.log(C)
    meff = np.subtract(logC, np.roll(logC, shift = -1, axis = 0))[:-1]
    return meff  # return meff with shape (Nt-1)

def allmighty_bootstrap(s: Callable[[np.ndarray], Union[np.ndarray, np.float64]], 
                        data: np.ndarray, 
                        N_boot: int, 
                        binsize: int, 
                        seed = 1337
                       ) -> Tuple[Union[np.ndarray, np.float64], Union[np.ndarray, np.float64]]:
    """
    General block-bootstrap analysis to estimate the standard error of an estimator $\hat{\Theta} = s(x)$ for the parameter $\Theta = t(F)$, 
    where F is the probability distribution from which the data x was drawn.
    For further details on the method the reader is referred to the book "An Introduction to the Bootstrap" by Bradley Efron and Robert J. Tibshirani. 
    args: 
        s (function) : function that is the estimator of $\Theta$. Takes as input the boostrapped data with shape [N_bins, binsize, ...], 
            where the remaining dimensions are the same as for the input data. The output of the function does NOT have to have the same shape but can be arbitrary. 
        data (np.array) : data set drawn from F. Should be provided in the shape [N_samples, ...]. Note: The same random numbers will be used for each observable element of the data.
        N_boot (int) : Number of bootstrap samples
        binsize (int) : binsize for block-bootstrapping
        seed (int) : seed for random number generator
    """
    rng = np.random.default_rng(seed = seed)  
    N_samples = data.shape[0]
    N_bins = N_samples//binsize
    if N_samples%binsize!=0:
        print("Trimming data because of mismatch between number of samples and binsize")
        data = data[:N_bins*binsize]
    shape = [N_bins, binsize] if len(data.shape)==0 else [N_bins, binsize, *data.shape[1:]]
    reshaped_data = np.reshape(data, shape)
    rng_numbers = rng.integers(low = 0, high=N_bins, size=(N_bins))
    bootstrap_sample = reshaped_data[rng_numbers]
    first_replica = s(bootstrap_sample)
    replica_shape = first_replica.shape
    br_shape = (N_boot) if len(replica_shape)==0 else (N_boot, *replica_shape)
    bootstrap_replicas = np.zeros(br_shape)
    bootstrap_replicas[0] = first_replica
    for i in range(1, N_boot):
        rng_numbers = rng.integers(low = 0, high=N_bins, size=(N_bins))
        bootstrap_sample = reshaped_data[rng_numbers]
        bootstrap_replicas[i,...] = s(bootstrap_sample)
    return np.mean(bootstrap_replicas, axis = 0), np.std(bootstrap_replicas, axis = 0)