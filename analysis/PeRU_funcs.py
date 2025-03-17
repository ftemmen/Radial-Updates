import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple

from autocorrelation_funcs import single_autocorrelation_analysis


def get_PeRU_correlators_from_h5(path: str, Nconf_max: int = int(1e5)):
    """
    Function to get correlators for Perylene Radial Updates simulations obtained from NSL. 
    """
    t1 = time.time()
    shape = (96, 2, 2)
    full_corr = np.zeros((Nconf_max, 96, 2, 2), dtype = np.complex128)
    with h5py.File(path, "r") as f:
        name = str(*f.keys())
        Nconf = len(f[name+"/markovChain"])
        for i in range(Nconf_max):
            corr = np.reshape(f[name+f"/markovChain/{i}/correlators/single/particle/k0"][:], shape)
            full_corr[i, :, :, :] = corr
    print(f"Loaded data in {round(time.time()-t1, 2)}s")
    return full_corr


def PeRU_correlator_autocorr_analysis(full_corr: np.ndarray, 
                                      Nt: int = 96, 
                                      S: float  = 1.0, 
                                      c: float = 0.0, 
                                      use_first_zero_crossing: bool = False, 
                                      save_plot: Optional[str] = None, 
                                      suptitle: str = "", 
                                      corr_ind: int = 0
                                     ) -> Tuple[np.ndarray, 
                                                np.ndarray, 
                                                np.ndarray, 
                                                np.ndarray, 
                                                np.ndarray]:
    """
    Full autocorrelation analysis for Perylene Radial Updates (PeRU) simulations. 
    args: 
        full_corr (np.array) : Numpy array of shape (Nconf, 2, 2) with the full correlators. 
        Nt (int) : Number of time slices
        S (float) : S value used for Wolff criterion in autocorrelation analysis. Is overwritten if use_first_zero_crossing=True or c!=0.0
        c (float) : c value used for Sokal criterion in autocorrelation analysis. Is overwritte if use_first_zero_crossing=True
        use_first_zero_crossing (bool) : Bool to decide whether the first zero crossing of the autocorrelation function should be used as cutting criterion in autocorrelation analysis. 
        save_plot (None or string) : Whether to save the autocorrelation analysis plots. If None plots are not saved. If there is a string specified, this name will be used as the file path and appended with some number indicating which plot it is and filetype. 
        suptitle (string) : Suptitle used in the plots. 
        corr_ind (int) : Either 0 or 1. The index of the correlator the analysis should be performed for. 
    """
    
    fs = 14
    
    tint_, dtint_, W_, obs_, stderr_ = np.zeros((Nt)), np.zeros((Nt)), np.zeros((Nt)), np.zeros((Nt)), np.zeros((Nt))
    N = full_corr.shape[0]
    mod_n = 16
    split_n1, split_n2 = 4, 4
    for t in range(Nt):
        mod_ind = t%mod_n
        i, j = (t%mod_n)//split_n2, (t%mod_n)%split_n1
        i2 = i*2
        if mod_ind==0:
            fig, ax = plt.subplots(split_n1*2, split_n2, figsize = (split_n1*4, split_n2*4), sharex = True)
            fig.subplots_adjust(left=None,
                    bottom=None, 
                    right=None, 
                    top=0.95, 
                    wspace=None, 
                    hspace=None)
            
        data = full_corr[:, t, corr_ind, corr_ind]
        Gamma, tau_int_W, obs_[t], _, stderr_[t], _, tint_[t], tmp, W = single_autocorrelation_analysis(data = data, 
                                                                                                            S = S, 
                                                                                                            c = c, 
                                                                                                            use_first_zero_crossing = use_first_zero_crossing)
        W_[t], dtint_[t] = W, np.sum(tmp)
        ax[i2+0, j].set_title(f"{t=}", fontsize = fs, y= 0.8)
        ax[i2+0, j].plot(np.arange(1, len(Gamma)+1, 1), Gamma/Gamma[0]) #, marker = "o", linestyle = "--", markersize = "3"
        ax[i2+1, j].plot(np.arange(1, len(tau_int_W)+2, 1), np.concatenate((np.array([0.5]), tau_int_W*(1+(2*W+1)/N)), axis = 0))
        ax[i2+0, j].axvline(W+1, color = "k", linestyle = "--", linewidth = 1.5), ax[i2+1, j].axvline(W+1, color = "k", linestyle = "--", linewidth = 1.5)
        if j == 0:
            ax[i2+0, j].set_ylabel(r"$\Gamma(t)$", fontsize = fs), ax[i2+1, j].set_ylabel(r"$\tau_{\mathrm{int}}(W)$", fontsize = fs)
        ax[i2+0, j].set_xscale("log"), ax[i2+0, j].set_xlim(1, len(Gamma)), ax[i2+0, j].set_ylim(-0.1, 1.), ax[i2+0, j].grid()
        ax[i2+1, j].set_xscale("log"), ax[i2+1, j].set_xlim(1, len(tau_int_W)), ax[i2+1, j].set_ylim(bottom = -0.1), ax[i2+1, j].grid()
        fig.suptitle(suptitle, fontsize = fs)
        
        if ((t%mod_n)+1)//16==1:
            for j3 in range(split_n2):
                ax[-1, j3].set_xlabel(r"$t+1$", fontsize = fs)
            if save_plot:
                plt.savefig(save_plot+f"_{t//mod_n}.png", dpi = 300, bbox_inches = "tight")
            plt.show()
    return obs_, stderr_, tint_, dtint_, W_