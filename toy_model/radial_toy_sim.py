import numpy as np
import time
from typing import Union, Optional, Tuple

from radial_toy_HMC import Scalar_HMC
from radial_toy_action import Radial_toy_action
from radial_toy_ac import autocorrelation_analysis
from radial_toy_utils import save_arrays, sep

def simulate_radial_toy_model(L: int, 
                              omega: float, 
                              beta: float, 
                              N_MD: int, 
                              t_MD: float, 
                              radial_scale: float, 
                              N_therm: int, 
                              N_MC: int, 
                              N_skip: int, 
                              save_data_fname: str = "temp.txt", 
                              ac_analysis: bool = True, 
                              plot_ac: bool = False, 
                              save_ac_plot: str = None, 
                              true_mean: Optional[Union[list, float]] = None, 
                              S: float = 1.
                             ) -> Tuple[Optional[np.ndarray], 
                                        Optional[np.ndarray], 
                                        Optional[np.ndarray], 
                                        Optional[np.ndarray]]:
    """
    Simulates the toy model with probability distribution p(x) = cos(w*x^2)*exp(-beta*x**2) using HMC and radial updates.
    args:
            L (int)                       :  Lattice size, i.e. dimensionality of the x-vector
            omega (float)                 :  frequency used in cosine term of the distribution
            beta (float)                  :  prefactor in the exponential term in the distribution
            N_MD (int)                    :  Number of molecular dynamics (MD) steps used in leapfrog evolution
            t_MD (float)                  :  trajectory length used in MD evolution
            radial_scale (float)          :  Standard deviation of the proposal distribution used in the radial updates
            N_therm (int)                 :  Number of thermalization steps
            N_MC (int)                    :  Number of configurations recorded in the Monte Carlo (MC) simulation 
            N_skip (int)                  :  Save frequency in simulation
            plot_ac (bool)                :  Determines whether the autocorrelation analysis should be plotted. 
            save_ac_plot (string)         :  Filename if the ac plot is supposed to be saved. If it shouldnt be saved set it to None (default)
            true_mean (list, float, None) : If float: The value is used as the true mean in the correlation analysis. If data is multidimensional the value is used for EVERY time series. 
                                            If None: The estimated mean is used. 
                                            If list: Dimension of list must match number of observables of data. List is flattened if multidimensional.  
            S (float)                     : Proportionality factor S used in eq. (50-51) in Wolff paper. Reasonable choices are S = 1 ... 2
            
    returns: (Note: Shape of outputs is always (N_obs) if np.ndarray)
    
            mean (np.ndarray)               :  Estimated expectation values of the observables given in the toy model action. 
            var (np.ndarray)                :  Estimated variance of the observables given in the toy model action. 
            stderr (np.ndarray)             :  Estimated error of mean of the observables given in the toy model action. 
            err_err (np.ndarray)            :  Estimated error of the error of the observables given in the toy model action. 
            t_int (np.ndarray)              :  Integrated autocorrelation time of the observables given in the toy model action. 
            t_int_err (np.ndarray)          :  Error of the integrated autocorrelation time of the observables given in the toy model action. 
            W (np.ndarray)                  :  Truncation point used in determining the integrated autocorrelation time of the observables given in the toy model action. 
    """
    model = Radial_toy_action(omega, beta)
    HMC = Scalar_HMC(model = model, 
                     t_MD = t_MD, 
                     N_MD = N_MD, 
                     cfg_shape = [L], 
                     radial_scale = radial_scale, 
                     seed = 1337)
    N_obs = len(HMC.calc_obs()) if type(HMC.calc_obs())==tuple else 1 # Get number of observables defined in the toy model action
    
    print(f"Begin simulation with L={L}; omega={omega}; beta={beta}; N_MD={N_MD}; t_MD={t_MD}; radial_scale={radial_scale}")
    t1 = time.time()
    if radial_scale: #Check if radial_scale = 0 i.e. no radial updates
        for i in range(N_therm):
            HMC.radial_update()
            HMC.step()
    else:
        for i in range(N_therm):
            HMC.step()
    t2 = time.time()
    print(f"Thermalization finished after {np.round(t2-t1, 2)}s!")
    
    HMC.reset_acc()
    obs = np.zeros((N_obs, N_MC)) #empty array to store measured observables
    j = 0
    if radial_scale: #Check if radial_scale = 0 i.e. no radial updates
        for i in range(int(N_MC*N_skip)):
            HMC.radial_update()
            HMC.step()
            if i % N_skip == 0:
                obs[:, j] = HMC.calc_obs()
                j += 1
    else:
        for i in range(int(N_MC*N_skip)):
            HMC.step()
            if i % N_skip == 0:
                obs[:, j] = HMC.calc_obs()
                j += 1
    print(f"Generation finished after {np.round(time.time()-t2, 2)}s!")
    print(f"Acceptance rate {HMC.out_acc()*100}%")
    if radial_scale: 
        print(f"Radial Acceptance rate {HMC.out_radial_acc()*100}%")
    print(f"Simulation finished after {np.round(time.time()-t2, 2)}s!")    
    if save_data_fname:
        save_arrays(save_data_fname, None, False, *obs)
    if ac_analysis:
        print("Observable estimates")
        mean, var, stderr, err_err, t_int, t_int_err, W = autocorrelation_analysis(data = obs.transpose(), 
                                                                                   use_fft = True, 
                                                                                   plot_results = plot_ac, 
                                                                                   save_plot = save_ac_plot, 
                                                                                   true_mean = true_mean, 
                                                                                   S = S)
        sep()
        return mean, var, stderr, err_err, t_int, t_int_err, W
    else:
        return None, None, None, None, None, None, None
