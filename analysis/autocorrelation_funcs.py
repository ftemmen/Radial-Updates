import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple



def single_autocorrelation_analysis(data: np.ndarray, 
                                    data_axis: int = 0, 
                                    use_fft: bool = True, 
                                    true_mean: Optional[float] = None, 
                                    S: float = 1.0, 
                                    use_first_zero_crossing: bool = False, 
                                    c: float = 0.0
                                   ) -> Tuple[np.ndarray, np.ndarray, 
                                             float, float, float, float, float, float, float]:
    """
    Function to perform autocorrelation analysis of a sequence of data. 
    The analysis is based on the paper "Monte Carlo errors with less errors" by Ulli Wolff, 
    the lecture notes "Monte Carlo Methods in Statistical Mechanics: Foudnations and New Algorithms" by Alan Sokal, 
    and the comp-avg tool by Johann Ostmeyer (https://github.com/j-ostmeyer/comp-avg). 
    
    args:       data (np.array)               : One-dimensional numpy array with time series data. 
                use_fft (bool)                : Determines whether a fast fourier transform (fft) should be used instead of a direct computation of the correlations.
                true_mean (float, None)       : If float: The value is used as the true mean in the correlation analysis. (And Bessel-like correction removed) 
                                                If None: The estimated mean is used. 
                S (float)                     : Proportionality factor S used in eq. (50-51) in Wolff paper. Reasonable choices are S = 1 ... 2
                c (float)                     : c-factor used in Sokal lecture notes at the very end of section 3. Should be chosen to be in the interval [4., 10.], 
                                                where c\approx 4.0 would correspond to a purely exponential decay. 
                use_first_zero_crossing (bool): If true use first zero crossing of the autocorrelation function as criterion to determine W. 
                                            
    returns:    Gamma (array)                 : Autocorrelation function
                tau_int_W (array)             : Integrated autocorrelation time as a function of the cutoff point W
                O_bar (float)                 : Estimated mean of the data
                var (float)                   : Estimated variance of the data
                stderr (float)                : Standard error of O_bar taking into account correlations
                err_err (float)               : Estimate error of the error of the estimated mean
                tau_int (float)               : Integrated autocorrelation time
                tau_int_err (list)            : List with the estimated statistical (index 0) and systematic error (index 1) of the integrated autocorrelation time
                W (int)                       : Cutoff point W (aka t_max) in computation of integrated autocorrelation time
    """
    method = "fft" if use_fft else "direct"
    N = len(data)
    if not true_mean == None:
        O_bar = true_mean
    O_bar = np.mean(data) if true_mean == None else true_mean
    signal_arr = data - O_bar
    Gamma = signal.correlate(signal_arr, signal_arr, mode = "full", method = method)[N-1:] / np.arange(N, 0, -1) # autocorrelation function (Wolff eq. 31)
    var = Gamma[0] # Variance (Wolff eq. 34)
    C = Gamma[0] + 2*np.cumsum(Gamma[1:]) # C_F(W) for W = 1...N (Wolff eq. 35)
    tau_int_W = C / (2*var)
    # print(tau_int_W)
    if use_first_zero_crossing: # first zero crossing of Gamma; If this is used then the error estimate of the integrated autocorrelation time is likely not correct
        W = np.amin(np.argwhere(Gamma <= 0)) + 1
        method = "zc_or_Sokal"
    elif c!=0: # Sokal method
        # print("Sokal method")
        W_tilde = np.arange(1, N, 1) # all possible stopping points W
        arg = c * tau_int_W - W_tilde
        neg_inds = np.argwhere(arg <= 0)
        method = "zc_or_Sokal"
        if len(neg_inds) == 0: ### Check if the condition is fulfilled for atleast one W, else take last value
            W = N-1
            print("The c-condition was never fulfilled. ")
        else:
            W = np.amin(neg_inds) + 1
    else: # Wolff method
        method = "Wolff"
        W_tilde = np.arange(1, N, 1) # all possible stopping points W
        arg = (2*tau_int_W+1)/(2*tau_int_W-1)
        arg = np.where(tau_int_W<=0.5, 1.0000000001, arg) # leads to tiny positive value for tau_W if tau_int_W<=0.5 
        # S = 1.0 if S == 0 else S
        tau_W = np.nan_to_num(S/np.log(arg), nan = 1e-10, posinf = 1e-10, neginf = 1e-10)
        g_W = np.exp(-np.divide(W_tilde, tau_W)) - np.divide(tau_W, np.sqrt(W_tilde*N)) # W derivative (Wolff eq. 52)
        neg_inds = np.argwhere(g_W < 0)
        if len(neg_inds) == 0: ### Check if g_W negative for atleast one W, else take last value
            W = N-1
            print("The gradient never turned negative. ")
        else:
            W = np.amin(neg_inds) + 1
    C_opt = C[W-1]
    ### Alternative bias correction from Wolff code
    # Gamma_prime = Gamma + C_opt/N
    # C2 = Gamma_prime[0] + 2*np.cumsum(Gamma_prime[1:])
    # C_opt_prime2 = C2[W-1]
    C_opt_corrected = C_opt*(1+(2*W+1)/N) if true_mean == None else C_opt 
    tau_int = C_opt_corrected / (2*var) # integrated autocorrelation time (Wolff eq. 41)
    if method == "Wolff":
        stat_dtau_int = tau_int * 2 * np.sqrt((W + 0.5 - tau_int)/N)  # statistical error of tau_int (Wolff eq. 42)
        tau = 0.5*tau_int*(1+np.exp(-2*W/tau_int)) # Where does this come from? I would think \tau \approx S*tau_int with S = 1..2
        sys_dtau_int = tau_int*np.exp(-W/tau) # systematic error of tau_int (?Wolff eq. 36?)
    elif method == "zc_or_Sokal":
        stat_dtau_int = np.sqrt(2*(2*W+1)/N)*tau_int
        sys_dtau_int = 0.0 #-0.5 * np.divide(np.subtract(C[-1], C_opt), var)
    dtau_int = [stat_dtau_int, sys_dtau_int]
    stderr = np.sqrt(2*tau_int*var/N) # standard error of the mean taking into account correlations (Wolff eq. 23)
    # stderr = np.sqrt(C_opt_corrected/N) # Alternative standard error (Wolff eq. 44)
    err_err = stderr * 0.5 * np.sum(dtau_int)/tau_int # error of the error (Wolff eq. 43)
    return Gamma, tau_int_W, np.mean(data), var, stderr, err_err, tau_int, dtau_int, W

def autocorrelation_analysis(data: np.ndarray, 
                             use_fft: bool = True, 
                             true_mean: Optional[Union[float, list]] = None, 
                             S: float = 1.0, 
                             c: float = 0.0, 
                             use_first_zero_crossing: bool = False, 
                             plot_results: bool = False, 
                             save_plot: Optional[str] = None, 
                             suptitle: str = ""
                            ) -> Tuple[np.ndarray, 
                                      np.ndarray, 
                                      np.ndarray, 
                                      np.ndarray, 
                                      np.ndarray, 
                                      np.ndarray, 
                                      np.ndarray]:
    """
    Function to perform autocorrelation analysis of multiple sequences of data. 
    The analysis is based on the paper "Monte Carlo errors with less errors" by Ulli Wolff, 
    the lecture notes "Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms" by Alan Sokal, 
    and the comp-avg tool by Johann Ostmeyer (https://github.com/j-ostmeyer/comp-avg). 
    
    args:       data (np.array)               : d-dimensional numpy array with time series data. 
                                                If d>1 it is assumed that axis = 0 is the time series dimension. All other dimensions are flattened to yield an array of shape [N, N_obs]
                use_fft (bool)                : Determines whether a fast fourier transform (fft) should be used instead of a direct computation of the correlations.
                true_mean (list, float, None) : If float: The value is used as the true mean in the correlation analysis. If data is multidimensional the value is used for EVERY time series. 
                                                If None: The estimated mean is used. 
                                                If list: Dimension of list must match number of observables of data. List is flattened aswell if multidimensional.  
                S (float)                     : Proportionality factor S used in eq. (50-51) in Wolff paper. Reasonable choices are S = 1 ... 2
                c (float)                     : c value used in Sokal criterion. A reasonable choice is 6...10. 
                use_first_zero_crossing (bool): If true use first zero crossing of the autocorrelation function as criterion to determine W. 
                plot_results (bool)           : If True then the intermediate results of the autocorrelation function $\Gamma$ and $\tau_int(W)$ are plotted. 
                                                This is only done if the number of observables is below the arbitrarily chosen threshhold of 7 to avoid accidentally creating huge plots. 
                save_plot (string or None)    : If a string is provided the plot is saved to that string. Has to be an actual directory and requires e.g. ".png" ending. If None, plot is not saved. 

                                            
    returns:    O_bar (np.array)                 : Array with estimated means of the data
                var (np.array)                   : Array with estimated variances of the data
                stderr (np.array)                : Array with standard errors of O_bar taking into account correlations
                err_err (np.array)               : Array with estimated errors of the error of the estimated means
                tau_int (np.array)               : Array with integrated autocorrelation times
                tau_int_err (np.array)           : Array with with the estimated errors of the integrated autocorrelation times (statistical + systematical)
                W (np.array)                     : Array with cutoff points W (aka t_max) in computation of integrated autocorrelation time
    """
    shape = data.shape
    if len(shape)>1:
        N, N_obs = shape[0], np.prod(shape[1:])
    else:
        N, N_obs = len(data), 1
    data = np.reshape(data, (N, N_obs))
    ### Construct true_mean list for analyses
    if true_mean == None: 
        tm_ = N_obs*[None]
    elif type(true_mean) == float or type(true_mean) == int:
        tm_ = N_obs*[float(true_mean)]
    elif type(true_mean) == list:
        dim = np.prod(np.shape(true_mean))
        if dim == 1 and (type(true_mean[0])==float or type(true_mean[0])==int):
            tm_ = N_obs*[float(true_mean[0])]
        assert N_obs == dim, "The list length does not match the number of observables. "
        tm_ = np.reshape(true_mean, dim)
    elif type(true_mean) == tuple:
        dim = len(true_mean)
        if dim == 1 and (type(true_mean[0])==float or type(true_mean[0])==int):
            tm_ = N_obs*[float(true_mean[0])]
        else:
            assert N_obs == dim, "The tuple length does not match the number of observables. "
            tm_ = [*true_mean]
    else:
        print("I have no idea what your true_mean input is supposed to be. I set it to None for all observables. Try again next time. ")
        tm_ = N_obs*[None]
        
    if plot_results and N_obs<=6: # initialize plot
        fig, ax = plt.subplots(2, N_obs, figsize = (N_obs*4, 2*4))
        if N_obs == 1:
            ax = np.reshape(ax, (2, N_obs))
    
    O_bar_, var_, stderr_, err_err_, tau_int_, dtau_int_, W_ = np.zeros(N_obs), np.zeros(N_obs), np.zeros(N_obs), np.zeros(N_obs), np.zeros(N_obs), np.zeros(N_obs), np.zeros(N_obs)
    
    tab_space = 22
    print(f"%-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %s" % (tab_space, "Mean", 
                                                             tab_space, "Variance", 
                                                             tab_space, "Stderr", 
                                                             tab_space, "Error of error", 
                                                             tab_space, "t_int", 
                                                             2*tab_space, "t_int_err", "W") )
    
    for i in range(N_obs):
        Gamma, tau_int_W, O_bar, var, stderr, err_err, tau_int, dtau_int, W = single_autocorrelation_analysis(data = data[: ,i], 
                                                                                                              use_fft = use_fft, 
                                                                                                              true_mean = tm_[i], 
                                                                                                              S = S, 
                                                                                                              c = c, 
                                                                                                              use_first_zero_crossing = use_first_zero_crossing)
        ##### PRINT RESULTS #####
        print(f"%-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %s" % (tab_space, O_bar, 
                                                                 tab_space, var, 
                                                                 tab_space, stderr, 
                                                                 tab_space, err_err, 
                                                                 tab_space, tau_int, 
                                                                 2*tab_space, str(dtau_int[0])+" + "+str(dtau_int[1]), W) )
        O_bar_[i], var_[i], stderr_[i], err_err_[i], tau_int_[i], dtau_int_[i], W_[i] = O_bar, var, stderr, err_err, tau_int, np.sum(dtau_int), W
        if plot_results and N_obs<=6:
            fs = 16
            ax[0, i].plot(np.arange(1, len(Gamma)+1, 1), Gamma/Gamma[0]) #, marker = "o", linestyle = "--", markersize = "3"
            ax[1, i].plot(np.arange(1, len(tau_int_W)+2, 1), np.concatenate((np.array([0.5]), tau_int_W*(1+(2*W+1)/N)), axis = 0))
            ax[0, i].axvline(W+1, color = "k", linestyle = "--", linewidth = 1.5), ax[1, i].axvline(W+1, color = "k", linestyle = "--", linewidth = 1.5)
            if c!=0.0:
                ax[1, i].plot(np.arange(1, len(tau_int_W)+2, 1), np.divide(np.arange(0, len(tau_int_W)+1, 1), c), color = "tab:grey", linestyle = "--", linewidth = 1.5)#, ax[1, i].axvline(W+1, color = "k", linestyle = "--", linewidth = 1.5)
                ax[1, i].set_ylim(0.5, np.amax(tau_int_W))
            if i == 0:
                ax[0, i].set_ylabel(r"$\Gamma(t)$", fontsize = fs), ax[1, i].set_ylabel(r"$\tau_{\mathrm{int}}(W)$", fontsize = fs)
            ax[0, i].set_xlabel(r"$t+1$", fontsize = fs), ax[1, i].set_xlabel(r"$W+1$", fontsize = fs)
            ax[0, i].set_xscale("log"), ax[0, i].set_xlim(1, len(Gamma)), ax[0, i].set_ylim(-0.1, 1.), ax[0, i].grid()
            ax[1, i].set_xscale("log"), ax[1, i].set_xlim(1, len(tau_int_W)), ax[1, i].set_ylim(bottom = -0.1), ax[1, i].grid()
            
    if plot_results and N_obs<=6:
        fig.suptitle(suptitle, fontsize = fs)
        fig.subplots_adjust(left=None,
                    bottom=None, 
                    right=None, 
                    top= 0.9, 
                    wspace=None, 
                    hspace=0.2)
        if save_plot:
            plt.savefig(save_plot, dpi = 300, bbox_inches = "tight")
        plt.show()
    return O_bar_, var_, stderr_, err_err_, tau_int_, dtau_int_, W_