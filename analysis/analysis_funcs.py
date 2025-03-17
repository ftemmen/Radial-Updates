import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.optimize import curve_fit, leastsq
from typing import Tuple, Optional, Callable, Union, Sequence

from autocorrelation_funcs import single_autocorrelation_analysis, autocorrelation_analysis
from utils import sep, round_to_n_of_err, get_parentheses_error_notation, save_arrays
from fitting import bootstrap_fit, bootstrap_fit_and_sigmin, ac_fit_f, smin_fit_f, logsmin_fit_f


def Nt_autocorrelation_analysis(Nt: int, 
                                sig_: list, 
                                analysis_name: str, 
                                postpro_path: str, 
                                plot_path: str, 
                                plot_results: bool = True):
    """
    Autocorrelation analysis for "standard" observables for two-site and four-site model for a given value number of time slices Nt and a range of proposal standard deviations. 
    Results are saved to csv files. 
    args: 
        Nt (int) : Number of time slices. 
        sig_ (list) : List of proposal standard deviations. 
        analysis_name (str) : Identifier for the current analysis used to locate the files. ("R2S" for two-site model and "Sq4S" for Square four-site model). 
        postpro_path (str) : Path to post-processing folder. 
        plot_path (str) : Path to plots folder
        plot_results (bool) : If True, results are plotted. 
    """
    for i, sig in enumerate(sig_):
        csv_path = postpro_path + f"Nt{Nt}/" + analysis_name + f"_Nt{Nt}_sig"+str(sig).replace(".", "p")+"_obs.csv"
        df = pd.read_csv(csv_path)
        data_head = df.columns
        data = df.to_numpy() #array of shape [N_samples, N_obs]
        if i == 0:
            ac_data_array = np.zeros((2, len(data_head), len(sig_), 7))
        ac_ind = 0
        ac_data_array[ac_ind, :, i, 0], \
        ac_data_array[ac_ind, :, i, 1], \
        ac_data_array[ac_ind, :, i, 2], \
        ac_data_array[ac_ind, :, i, 3], \
        ac_data_array[ac_ind, :, i, 4], \
        ac_data_array[ac_ind, :, i, 5], \
        ac_data_array[ac_ind, :, i, 6] = autocorrelation_analysis(data = data, 
                                                          use_first_zero_crossing = False, 
                                                          plot_results = plot_results, 
                                                          save_plot = plot_path + f"ac/Nt{Nt}/ac_S1_Nt{Nt}_sig"+str(sig).replace(".", "p")+".png", 
                                                          suptitle = rf"$N_t=${Nt}; $\sigma = ${sig}; $S=1$")

        ac_ind = 1
        ac_data_array[ac_ind, :, i, 0], \
        ac_data_array[ac_ind, :, i, 1], \
        ac_data_array[ac_ind, :, i, 2], \
        ac_data_array[ac_ind, :, i, 3], \
        ac_data_array[ac_ind, :, i, 4], \
        ac_data_array[ac_ind, :, i, 5], \
        ac_data_array[ac_ind, :, i, 6] = autocorrelation_analysis(data = data, 
                                                          use_first_zero_crossing = True, 
                                                          plot_results = plot_results, 
                                                          save_plot = plot_path + f"ac/Nt{Nt}/ac_zc_Nt{Nt}_sig"+str(sig).replace(".", "p")+".png", 
                                                          suptitle = rf"$N_t=${Nt}; $\sigma = ${sig}; zero-crossing")

    header = ["sigma", "mean", "std", "stderr", "err_err", "tint", "tint_err", "W"]
    ac_types = ["S1", "zc"]
    for ac_ind, ac_name in enumerate(ac_types):
        for i, obs_name in enumerate(data_head):
            ac_dict = {}
            for j, col_name in enumerate(header):
                if j == 0:
                    ac_dict[col_name] = sig_
                else:  
                    ac_dict[col_name] = ac_data_array[ac_ind, i, :, j-1]
            df = pd.DataFrame(ac_dict)
            df.to_csv(postpro_path + f"Nt{Nt}/" + analysis_name + f"_Nt{Nt}_"+obs_name+"_ac_"+ac_name+".csv", index = False)
    print("Autocorrelation analysis done")



def full_ac_fits(Nt_: list, 
                 use_obs_dict: dict, 
                 acfit_dict: dict, 
                 plot_path: str, 
                 analysis_name: str, 
                 plot_obs: bool = False, 
                 n_bootstrap: int = 1000, 
                 weight_fits: bool = False, 
                 save_plots: bool = False
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Function to make autocorrelation plots with corresponding $N_t$ value.
    args:
        Nt_ (list) : List of all values of Nt for the corresponding analysis. 
        use_obs_dict (dict) : Dictionary that specifies for which of the observables fits should be performed. 
        acfit_dict (dict) : Dictionary with fit details. 
        plot_path (str) : Path to plots folder. 
        analysis_name (str) : Identifier for the current analysis used to locate the files. ("R2S" for two-site model and "Sq4S" for Square four-site model). 
        plot_obs (bool) : If True, observables are included in plots in addition to the integrated autocorrelation times. Does not work for correlators. 
        n_bootstrap (int) : Number of bootstrap samples utilized in the bootstrap fitting routine. 
        weight_fits (bool) : If True, residuals in the least square fits are weighted with the inverse std error. 
        save_plots: (bool) : If True, plots are saved to plot_path. 
    returns: 
        smin_boot_ (np.array) : Estimates of the position of the minimum in the fit ansatz. 
        smin_err_ (np.array) : Estimates of the error of the position of the minimum in the fit ansatz. 
        tintmin_boot_ (np.array) : Estimates of the integrated autocorrelation time at the respective minimum. 
        tintmin_err_ (np.array) : Estimates of the error of the integrated autocorrelation time at the respective minimum. 
    """
    N_obs = 0
    for key, val in use_obs_dict.items():
        if val:
            N_obs+=1
    if plot_obs:
        n_rows, n_cols = 2, N_obs
    else:
        break_bool = False
        for i in range(1, N_obs+1):
            for j in range(1, i+1):
                if i*j==N_obs:
                    n_rows, n_cols = j, i
                    break_bool = True
                    break
            if break_bool:
                break

    # fs, fs2, ms, lw = 24, 24, 7, 2 #with latex font
    fs, fs2, ms, lw = 20, 20, 8, 2
    marker_ = ["o", "s", "*", "v", "^", "x", "p"]
    cs = [(2/255, 61/255, 107/255), (217/255, 85/255, 20/255), (163/255, 201/255, 0/255), (100/255, 2/255, 107/255), (107/255, 47/255, 2/255), "tab:blue", "tab:green", "tab:orange", "tab:red"]
    
    shape = (N_obs, len(Nt_))
    smin_boot_, smin_err_ = np.zeros(shape), np.zeros(shape)
    tintmin_boot_, tintmin_err_ = np.zeros(shape), np.zeros(shape)
    for Nt_count, Nt in enumerate(Nt_):
        fig, ax = plt.subplots(n_rows, n_cols, figsize = (n_cols*7., n_rows*6.4))
        if n_rows == 1:
            ax = np.reshape(ax, (n_rows, n_cols))
        for i in range(n_rows):
            ax[i, 0].set_ylabel(r"$\tau_{\mathrm{int}}$", fontsize = fs2)
            # ax[i, 0].set_ylabel(r"$\tau_{\mathrm{int},\mathcal{O}}$", fontsize = fs2)
        if plot_obs:
            ax[1, 0].set_ylabel(r"$\mathcal{O}$", fontsize = fs2)
        for i in range(n_cols):
            ax[-1, i].set_xlabel(r"$\sigma_R$", fontsize = fs)
        
        
        ax = ax.flatten()
        count = 0
        for key, val in use_obs_dict.items():
            if val:
                cax = ax[count]
                c_dict = acfit_dict[f"Nt{Nt}"][key]
                sigma, obs_mean, obs_stderr, tau_int, tau_int_err = c_dict["data"]
                sort_ind = np.argsort(sigma)
                psig_, ptint_, ptinterr_ = sigma[sort_ind], tau_int[sort_ind], tau_int_err[sort_ind]          
                current_outlier_inds = c_dict["outlier_inds"]
                psig_rm_, ptint_rm_, ptinterr_rm_ = np.delete(psig_, current_outlier_inds), np.delete(ptint_, current_outlier_inds), np.delete(ptinterr_, current_outlier_inds)
                sep()
                cax.errorbar(psig_rm_, ptint_rm_, ptinterr_rm_, color = cs[count], marker = marker_[count], markersize = ms, linestyle = "", capsize = 4, label = r"data")      

                ### FIT BEGIN ###
                curve_fit_params, curve_fit_cov = curve_fit(ac_fit_f, psig_rm_, ptint_rm_, sigma = ptinterr_rm_)

                fit_succeeded, \
                x_temp, mean_f, err_f, \
                fit_params, fit_stderr, \
                smin_boot, smin_err, \
                tintmin_boot_[count, Nt_count], tintmin_err_[count, Nt_count] = bootstrap_fit_and_sigmin(fit_f = ac_fit_f, 
                                                                                                       x_data = psig_rm_, 
                                                                                                       y_data = ptint_rm_, 
                                                                                                       y_data_err = ptinterr_rm_, 
                                                                                                       n_bootstrap = n_bootstrap, 
                                                                                                       weight_fits = weight_fits)
                print("Sigma min estimate", smin_boot_[count, Nt_count], smin_err_[count, Nt_count])
                print("Curve fit", curve_fit_params, np.sqrt(np.diagonal(curve_fit_cov)))
                print("Bootstrap fit", fit_params, fit_stderr)
                smin_boot_[count, Nt_count], smin_err_[count, Nt_count] = smin_boot, smin_err
                
                if fit_succeeded:
                    smin_boot_, smin_err_, tintmin_boot_, tintmin_err_
                    if c_dict["xlim"]:
                        cax.set_xlim(*c_dict["xlim"])
                    else:
                        cax.set_xlim(np.amin(x_temp), np.amax(x_temp))
                    if c_dict["ylim"]:
                        cax.set_ylim(*c_dict["ylim"])

                    val1, err1 = round_to_n_of_err(fit_params[0], fit_stderr[0], n = 2)
                    val2, err2 = round_to_n_of_err(fit_params[1], fit_stderr[1], n = 2)
                    val3, err3 = round_to_n_of_err(fit_params[2], fit_stderr[2], n = 2)

                    cax.plot(x_temp, mean_f, color = cs[2], linewidth = lw, 
                                  label = r"$\tau_{\mathrm{int}}(\sigma_R)$ fit"+f"\na="+get_parentheses_error_notation(val1, err1)+"\nb="+get_parentheses_error_notation(val2, err2)+"\nc="+get_parentheses_error_notation(val3, err3))
                    cax.fill_between(x_temp, np.add(mean_f, err_f), np.subtract(mean_f, err_f), color = cs[2], alpha = 0.5)
                    val1, err1 = round_to_n_of_err(smin_boot, smin_err, n = 2)
                    cax.axvline(val1, color = "tab:grey", linestyle = "--", linewidth = lw, label = r"$\sigma_R^{(\mathrm{min})}=$"+get_parentheses_error_notation(val1, err1))
                cax.set_title(c_dict["obs_name"], fontsize = fs, pad = 25)
                cax.set_xscale("log"), cax.set_yscale("log")
                
                if fit_succeeded:
                    handles, labels = cax.get_legend_handles_labels()
                    handles_tmp = [handles[2], handles[0], handles[1]]
                    labels_tmp = [labels[2], labels[0], labels[1]]
                    cax.legend(handles = handles_tmp, labels = labels_tmp, fontsize = fs-3)
                else:
                    cax.legend(fontsize = fs-3)
                cax.grid()
                cax.tick_params(axis = "both", which = "both", labelsize = fs)
                if c_dict["xticks"]:
                    cax.set_xticks(c_dict["xticks"][0], labels = c_dict["xticks"][1], fontsize = fs)
                if c_dict["yticks"]:
                    cax.set_yticks(c_dict["yticks"][0], labels = c_dict["yticks"][1], fontsize = fs)

                if plot_obs and type(obs_mean)!=type(None):
                    obs_mean, obs_stderr = obs_mean[sort_ind], obs_stderr[sort_ind]  
                    cax = ax[count+N_obs]
                    mean_rm_, stderr_rm_ = np.delete(obs_mean, current_outlier_inds), np.delete(obs_stderr, current_outlier_inds)
                    cax.errorbar(psig_rm_, mean_rm_, stderr_rm_, color = cs[count], marker = marker_[count], markersize = ms, linestyle = "", capsize = 4)
                    cax.set_xscale("log")
                    cax.grid()
                    cax.tick_params(axis = "both", labelsize = fs)
                    if c_dict["xlim"]:
                        cax.set_xlim(*c_dict["xlim"])
                    if c_dict["xticks"]:
                        cax.set_xticks(c_dict["xticks"][0], labels = c_dict["xticks"][1], fontsize = fs)
                    if c_dict["yticks"]:
                        cax.set_xticks(c_dict["yticks"][0], labels = c_dict["yticks"][1], fontsize = fs)
                count+=1

        fig.suptitle(rf"$N_t=${Nt}", fontsize = fs2)
        fig.subplots_adjust(left=None,
                            bottom=None, 
                            right=None, 
                            top = 0.87 if plot_obs else 0.8, 
                            wspace=0.3, 
                            hspace=0.28)
        if save_plots:
            add_str1 = "_weighted" if weight_fits else ""
            add_str2 = "_obs" if plot_obs else ""
            save_fname = plot_path + analysis_name + f"_ac_fit_Nt{Nt}"+add_str1+add_str2+".pdf"
            plt.savefig(save_fname, dpi = 300, bbox_inches = "tight")                          
        plt.show()
    return smin_boot_, smin_err_, tintmin_boot_, tintmin_err_


def ac_analysis_detM(Nt: int, 
                     data_str_f: Callable[[float], str], 
                     sig_: list, 
                     analysis_name: str, 
                     postpro_path: str, 
                     plot_path: Optional[str] = None, 
                     plot_results: bool = True) -> np.ndarray:
    """
    Autocorrelation analysis for sgnf (sgn of projected fermion determinant misnomered as detM observable) observable.
    args: 
        Nt (int) : Number of time slices. 
        data_str_f (function) : Takes a value of the proposal standard deviation and return the path to the corresponding file. 
        sig_ (list) : List of proposal standard deviations. 
        analysis_name (str) : Identifier for the current analysis used to locate the files. ("R2S" for two-site model and "Sq4S" for Square four-site model). 
        postpro_path (str) : Path to post-processed data folder.
        plot_path (str) : Path to plots folder.
        plot_results (bool) : If True, results are plotted. 
    returns: 
        ac_data_array (np.array) : Array shaped (n_ac_crits, n_obs, n_sigma, 7) with autocorrelation data for 
                                    - Wolff and zero-crossing autocorrelation criterion 
                                    - all observables 
                                    - all values of sigma_R. 
                                    - the autocorrelation results, i.e. (mean, variance, stderr, error of the error, tau_int, error of tau_int, cutoff point W). 
    """
    for i, sig in enumerate(sig_):
        path = data_str_f(sig)
        real, imag = np.loadtxt(path, usecols = (0, 1), skiprows = 1, unpack = True)
        assert np.allclose(np.divide(imag, real), np.zeros_like(imag)), "Imaginary parts are not zero. " # Is this working in the way it is intended?
        data = np.sign(real)
        if i == 0:
            ac_data_array = np.zeros((2, 1, len(sig_), 7))
        ac_ind = 0
        ac_data_array[ac_ind, :, i, 0], \
        ac_data_array[ac_ind, :, i, 1], \
        ac_data_array[ac_ind, :, i, 2], \
        ac_data_array[ac_ind, :, i, 3], \
        ac_data_array[ac_ind, :, i, 4], \
        ac_data_array[ac_ind, :, i, 5], \
        ac_data_array[ac_ind, :, i, 6] = autocorrelation_analysis(data = data, 
                                                          use_first_zero_crossing = False, 
                                                          plot_results = plot_results, 
                                                          save_plot = (plot_path + f"ac/Nt{Nt}/"+analysis_name+f"_Nt{Nt}_sig"+str(sig).replace(".","p")+"_S1_detM.png") if plot_path!=None else None, 
                                                          suptitle = rf"$N_t=${Nt}; $\sigma = ${sig}; $S=1$")
        

        ac_ind = 1
        ac_data_array[ac_ind, :, i, 0], \
        ac_data_array[ac_ind, :, i, 1], \
        ac_data_array[ac_ind, :, i, 2], \
        ac_data_array[ac_ind, :, i, 3], \
        ac_data_array[ac_ind, :, i, 4], \
        ac_data_array[ac_ind, :, i, 5], \
        ac_data_array[ac_ind, :, i, 6] = autocorrelation_analysis(data = data, 
                                                          use_first_zero_crossing = True, 
                                                          plot_results = plot_results, 
                                                          save_plot = (plot_path + f"ac/Nt{Nt}/"+analysis_name+f"_Nt{Nt}_sig"+str(sig).replace(".","p")+"_zc_detM.png") if plot_path!=None else None, 
                                                          suptitle = rf"$N_t=${Nt}; $\sigma = ${sig}; zero-crossing")

    header = ["sig", "obs", "var", "stderr", "errerr", "tautint", "dtauint", "W"]
    ac_ind = 0
    save_arrays(postpro_path + f"Nt{Nt}/"+analysis_name+f"_Nt{Nt}_detM_ac_S1.txt", header, False, 
                sig_, 
                ac_data_array[ac_ind, 0, :, 0], 
                ac_data_array[ac_ind, 0, :, 1], 
                ac_data_array[ac_ind, 0, :, 2], 
                ac_data_array[ac_ind, 0, :, 3], 
                ac_data_array[ac_ind, 0, :, 4], 
                ac_data_array[ac_ind, 0, :, 5], 
                ac_data_array[ac_ind, 0, :, 6])
    
    ac_ind = 1
    save_arrays(postpro_path + f"Nt{Nt}/"+analysis_name+f"_Nt{Nt}_detM_ac_zc.txt", header, False, 
                sig_,  
                ac_data_array[ac_ind, 0, :, 0], 
                ac_data_array[ac_ind, 0, :, 1], 
                ac_data_array[ac_ind, 0, :, 2], 
                ac_data_array[ac_ind, 0, :, 3], 
                ac_data_array[ac_ind, 0, :, 4], 
                ac_data_array[ac_ind, 0, :, 5], 
                ac_data_array[ac_ind, 0, :, 6])
    
    return ac_data_array


def plot_and_fit_sigmin(ax: Axes, 
                        d: Union[list, np.ndarray], 
                        smin: np.ndarray, 
                        smin_err: np.ndarray, 
                        n_bootstrap: int, 
                        color: Union[str, Sequence[Union[float, int]]], 
                        marker: str, 
                        obs_name: str, 
                        logfit: bool = False,
                        weight_fits: bool = False, 
                        n_start: int = 0, 
                        lw: int = 2, 
                        linestyle: str = "-", 
                        capsize: int = 6, 
                        markersize: int = 6):
    """
    Function to make scaling plots and the corresponding fits.
    args:
        ax (Matplot Axes) : Axes object on which to plot. 
        d (list or np.array) : Sequence with all dimensionalities. 
        smin (np.ndarray) : Array with corresponding values at the d values. 
        smin_err (np.ndarray) : Array with corresponding errors for the values at the d values.
        n_boostrap (int) : Number of bootstrap samples used in bootstrap fitting routine. 
        color (str, RGB) : color used for lines and marker. Either string or RGB tuple. 
        marker (str) : Marker used, e.g. "o", "s", "v" ...
        obs_name (str) : String identifying the observable and used in legend. 
        logfit (bool) : If True, the log data is fitted via a linear fit. 
        weight_fits (bool) : If True, the residuals in the least square fit are weighted with the inverse standard error. 
        n_start (int) : At which d to start with the fitting. 
        lw (int) : Linewidth in plotting. 
        linestyle (str) : Linestyle in plotting.
        capsize (int) : Capsize in plotting.
        markersize (int) : Markersize in plotting.
    """
    if logfit:
        d = np.log10(d)
        smin_err = np.divide(smin_err, smin*np.log(10))
        smin = np.log10(smin)
        d_plot, mean, mean_err, popt, perr = bootstrap_fit(logsmin_fit_f, d[n_start:], smin[n_start:], smin_err[n_start:], n_bootstrap = n_bootstrap, weight_fits = weight_fits, seed = 1337, n_plot_points = 1000, x_min = -5, x_max = 7)
    else:
        d_plot, mean, mean_err, popt, perr = bootstrap_fit(smin_fit_f, d[n_start:], smin[n_start:], smin_err[n_start:], n_bootstrap = n_bootstrap, weight_fits = weight_fits, seed = 1337, n_plot_points = 1000, x_min = 1, x_max = 100)
    ax.errorbar(d, smin, smin_err, capsize = capsize, markersize = markersize, alpha = 0.5, marker = marker, linestyle = "", color = color)
    points = ax.errorbar(d[n_start:], smin[n_start:], smin_err[n_start:], capsize = capsize, markersize = markersize, label = obs_name, marker = marker, linestyle = "", color = color)
    val1, err1 = round_to_n_of_err(popt[0], perr[0], n = 2)
    val2, err2 = round_to_n_of_err(popt[1], perr[1], n = 2)
    if logfit:
        fit = ax.plot(d_plot, mean, linestyle = linestyle, linewidth = lw, color = color, label = "Fit with\n"+r"$\alpha'=$"+get_parentheses_error_notation(val1, err1)+"\n"+r"$\beta=$"+get_parentheses_error_notation(val2, err2))
    else:
        fit = ax.plot(d_plot, mean, linestyle = linestyle, linewidth = lw, color = color, label = "Fit with\n"+r"$\alpha=$"+get_parentheses_error_notation(val1, err1)+"\n"+r"$\beta=$"+get_parentheses_error_notation(val2, err2))
    # fit = plt.plot(d_plot, mean, linestyle = "-", linewidth = lw, color = color, label = r"$\alpha=$"+get_parentheses_error_notation(val1, err1)+"\n"+r"$\beta=$"+get_parentheses_error_notation(val2, err2))
    ax.fill_between(d_plot, np.add(mean, mean_err), np.subtract(mean, mean_err), alpha = 0.4, color = color)
   

def R2S_correlator_autocorr_analysis(full_corr: np.ndarray, 
                                      Nt: int, 
                                      split_n1: int, split_n2: int, 
                                      S: float = 1.0, 
                                      c: float = 0.0, 
                                      use_first_zero_crossing: bool = False, 
                                      save_plot: Optional[str] = None, 
                                      suptitle: str = "", 
                                      corr_ind: int = 0
                                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Radial two-site correlator autocorrelation analyis. 
    args: 
        full_corr (np.array) : Full correlator data. 
        Nt (int) : Number of time slices. 
        split_n1, split_n2 (int,int) : How to split the plots for the corresponding components of the correlators. Their product should equal Nt. 
        S (float) : Proportionality factor S used in eq. (50-51) in Wolff paper. Reasonable choices are S = 1 ... 2
        c (float) : c value used in Sokal criterion. A reasonable choice is 6...10. 
        use_first_zero_crossing (bool) : If true use first zero crossing of the autocorrelation function as criterion to determine W. 
        save_plot (str) : If not None, path (without filetype) to save the plot to. 
        suptitle (str) : Suptitle in plot. 
        corr_ind (int) : Either 0 or 1, indicating the component of the correlator that is being analyzed. 
    returns: 
        obs_ (np.array) : Values of the components of the correlators. 
        stderr_ (np.array) : Errors of the components of the correlators. 
        tint_ (np.array) : Integrated autocorrelation times of the components of the correlators. 
        dtint_ (np.array) : Errors of the integrated autocorrelation times of the components of the correlators. 
        W_ (np.array) : Cutoff points of the components of the correlators. 
    """
    fs = 14
    tint_ = np.zeros((Nt))
    dtint_ = np.zeros((Nt))
    W_ = np.zeros((Nt))
    obs_ = np.zeros((Nt))
    stderr_ = np.zeros((Nt))
    N = full_corr.shape[0]
    fig, ax = plt.subplots(split_n1*2, split_n2, figsize = (max(split_n1, split_n2)*4, max(split_n1, split_n2)*4), sharex = True)
    if split_n2 == 1:
        ax = np.expand_dims(ax, -1)
    fig.subplots_adjust(left=None,
            bottom=None, 
            right=None, 
            top=0.9, 
            wspace=None, 
            hspace=None)
    for t in range(Nt):
        i, j = t//max(split_n1, split_n2), t%max(split_n1, split_n2)
        i2 = i*2
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
        
    for j3 in range(split_n2):
        ax[-1, j3].set_xlabel(r"$t+1$", fontsize = fs)
    if save_plot:
        plt.savefig(save_plot+f".png", dpi = 300, bbox_inches = "tight")
    plt.close()
    return obs_, stderr_, tint_, dtint_, W_


def get_NSL_Sq4S_correlators_from_h5(path: str, Nt: int, Nconf_max: int = int(1e5)):
    """
    Function to load correlators for Square 4 site model for data generated with NSL. 
    """
    t1 = time.time()
    shape = (Nt)
    full_corr = np.zeros((Nconf_max, Nt, 4), dtype = np.complex128)
    with h5py.File(path, "r") as f:
        name = str(*f.keys())
        Nconf = len(f[name+"/markovChain"])
        for i in range(Nconf_max):
            # print(f[name+f"/markovChain/{i}/correlators/single/particle/k0"][:].shape)
            corr = np.reshape(f[name+f"/markovChain/{i}/correlators/single/particle/k0"][:], shape)
            full_corr[i, :, 0] = corr
            corr = np.reshape(f[name+f"/markovChain/{i}/correlators/single/particle/k1"][:], shape)
            full_corr[i, :, 1] = corr
            corr = np.reshape(f[name+f"/markovChain/{i}/correlators/single/particle/k2"][:], shape)
            full_corr[i, :, 2] = corr
            corr = np.reshape(f[name+f"/markovChain/{i}/correlators/single/particle/k3"][:], shape)
            full_corr[i, :, 3] = corr
    print(f"Loaded data in {round(time.time()-t1, 2)}s")
    return full_corr


    
def Sq4S_correlator_autocorr_analysis(full_corr: np.ndarray, 
                                      Nt: int, 
                                      split_n1: int, split_n2: int, 
                                      S: float = 1.0, 
                                      c: float = 0.0, 
                                      use_first_zero_crossing: bool = False, 
                                      save_plot: Optional[str] = None, 
                                      suptitle: str = ""
                                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Radial two-site correlator autocorrelation analyis. 
    args: 
        full_corr (np.array) : Full correlator data. 
        Nt (int) : Number of time slices. 
        split_n1, split_n2 (int,int) : How to split the plots for the corresponding components of the correlators. Their product should equal Nt. 
        S (float) : Proportionality factor S used in eq. (50-51) in Wolff paper. Reasonable choices are S = 1 ... 2
        c (float) : c value used in Sokal criterion. A reasonable choice is 6...10. 
        use_first_zero_crossing (bool) : If true use first zero crossing of the autocorrelation function as criterion to determine W. 
        save_plot (str) : If not None, path (without filetype) to save the plot to. 
        suptitle (str) : Suptitle in plot. 
    returns: 
        obs_ (np.array) : Values of the components of the correlators. 
        stderr_ (np.array) : Errors of the components of the correlators. 
        tint_ (np.array) : Integrated autocorrelation times of the components of the correlators. 
        dtint_ (np.array) : Errors of the integrated autocorrelation times of the components of the correlators. 
        W_ (np.array) : Cutoff points of the components of the correlators. 
    """
    fs = 14
    tint_ = np.zeros((Nt))
    dtint_ = np.zeros((Nt))
    W_ = np.zeros((Nt))
    obs_ = np.zeros((Nt))
    stderr_ = np.zeros((Nt))
    N = full_corr.shape[0]
    fig, ax = plt.subplots(split_n1*2, split_n2, figsize = (max(split_n1, split_n2)*4, max(split_n1, split_n2)*4), sharex = True)
    if split_n2 == 1:
        ax = np.expand_dims(ax, -1)
    fig.subplots_adjust(left=None,
            bottom=None, 
            right=None, 
            top=0.9, 
            wspace=None, 
            hspace=None)
    for t in range(Nt):
        i, j = t//max(split_n1, split_n2), t%max(split_n1, split_n2)
        i2 = i*2
        data = full_corr[:, t]
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
        
    for j3 in range(split_n2):
        ax[-1, j3].set_xlabel(r"$t+1$", fontsize = fs)
    if save_plot:
        plt.savefig(save_plot+f".png", dpi = 300, bbox_inches = "tight")
    plt.close()
    return obs_, stderr_, tint_, dtint_, W_