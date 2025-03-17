import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from typing import TypedDict, Tuple, Optional, Union

from utils import sort_arrays_according_to_array, delete_indices_from_arrays, get_parentheses_error_notation, round_to_n_of_err, save_arrays
from fitting import bootstrap_fit_and_sigmin, ac_fit_f
from autocorrelation_funcs import autocorrelation_analysis


def get_Rtoy_autocorrelations(d: int, 
                              sim_dict: dict, 
                              eval_dict: dict, 
                              data_path: str, 
                              postpro_path: str, 
                              plot_path: str, 
                              obs_indices: tuple = (0, 1, 2, 3, 4, 5), 
                              append_data: bool = False):
    """
    Performs autocorrelation analysis for a given value of dimensionality d and saves the data, given the respective paths.
    args: 
        d (int) : Dimensionality of the system
        sim_dict (dict) : Corresponding simulation dictionary with information on NMD, tMD, sig, and Nconf
        eval_dict (dict) : Dictionary with autocorrelation criterion dictionaries. Each autocorrelation dict should contain "S", "use_zc", "c", "supt_name"
        data_path (str) : path to raw data
        postpro_path (str) : path to post-processed data
        plot_path (str) : path to plots folder
        obs_indices (tuple) : tuple with integers 0-5, indicating which of the observables should be analyzed
        append_data (bool) : Boolean deciding if the recorded data should be appended to already existing files. 
    """
    
    N_MD = sim_dict["NMD"]
    t_MD = sim_dict["tMD"]
    sig_ = sim_dict["sig"]
    Nconf_str = sim_dict["Nconf"]
    t1 = time.time()
    for i in range(len(sig_)):
        fname = data_path + f"d{d}/Rtoy_b0p125_d{d}_s"+str(sig_[i]).replace(".", "p")+f"_NMD{N_MD}_tMD"+str(t_MD)+".txt"
        loaded_data = np.loadtxt(fname, usecols = obs_indices, max_rows = 10)
    print("All files can be loaded: Analysis begins")
    N_obs = len(obs_indices)
    N_crits = len(eval_dict)
    
    results = np.zeros((N_crits, N_obs, len(sig_), 7))
    for i, sig in enumerate(sig_):
        fname = data_path + f"d{d}/Rtoy_b0p125_d{d}_s"+str(sig_[i]).replace(".", "p")+f"_NMD{N_MD}_tMD"+str(t_MD)+".txt"
        loaded_data = np.loadtxt(fname, usecols = obs_indices)
        for j, (crit_name, crit_dict) in enumerate(eval_dict.items()):
            save_plot_fname = plot_path + f"ac/d{d}/Rtoy_b0p125_d{d}_s"+str(sig).replace(".", "p")+f"_NMD{N_MD}_tMD"+str(t_MD)+"_"+crit_name+".png"
            results[j, :, i, 0], \
            results[j, :, i, 1], \
            results[j, :, i, 2], \
            results[j, :, i, 3], \
            results[j, :, i, 4], \
            results[j, :, i, 5], \
            results[j, :, i, 6] = autocorrelation_analysis(data = loaded_data, 
                                             S = crit_dict["S"], 
                                             use_first_zero_crossing = crit_dict["use_zc"], 
                                             c = crit_dict["c"], 
                                             plot_results = True, 
                                             save_plot = save_plot_fname, 
                                             suptitle = rf"$d=${d}; $\sigma_R=${sig}; "+crit_dict["supt_name"])
    
    obs_names_ = ["mean", "absmean", "radius", "mean_x0", "absmean_x0", "radius_x0"]
    header = ["sigma", "mean", "var", "stderr", "err_err", "tint", "tint_err", "W"]
    for j, (crit_name, crit_dict) in enumerate(eval_dict.items()):
        for i, obs_ind in enumerate(obs_indices):
            save_fname = postpro_path + f"d{d}/Rtoy_b0p125_"+obs_names_[obs_ind]+f"_d{d}"+f"_NMD{N_MD}_tMD"+str(t_MD)+"_"+crit_name+".txt"
            save_arrays(save_fname, header, append_data, 
                        sig_, 
                        results[j, i, :, 0], 
                        results[j, i, :, 1], 
                        results[j, i, :, 2], 
                        results[j, i, :, 3], 
                        results[j, i, :, 4], 
                        results[j, i, :, 5], 
                        results[j, i, :, 6])

    print(f"FULLY DONE AFTER {time.time()-t1}s")




def make_rtoy_ac_fits(d: int, 
                      sim_dict: dict, 
                      crit_dict: dict, 
                      fit_dict: dict, 
                      postpro_path: str, 
                      plot_path: str, 
                      obs_indices: list = (0, 1, 2, 3, 4, 5), 
                      suptitle: str = "", 
                      plot_obs: bool = False, 
                      save_plots: bool = False, 
                      close_plot: bool = False, 
                      weight_fits: bool = False, 
                      seed: int = 1337
                     ) -> Tuple[float, float, float, float]:
    """
    Function to make autocorrelation plots for the radial toy model. 
    args: 
        d (int) : Dimensionality of the system
        sim_dict (dict) : Corresponding simulation dictionary with information on NMD, tMD, sig, and Nconf
        crit_dict (dict) : Dictionary with autocorrelation criterion dictionaries. Each autocorrelation dict should contain "S", "use_zc", "c", "supt_name"
        fit_dict (dict) : Dictionary with information on fitting procedure for each autocorrelation criterion. 
        postpro_path (str) : path to post-processed data
        plot_path (str) : path to plots folder
        obs_indices (tuple) : tuple with integers 0-5, indicating which of the observables should be analyzed
        suptitle (str) : Deprecated. Should be removed. 
        plot_obs (bool) : If True, computed observables are shown in addition to the integrated autocorrelation times. Does not work for correlators.
        save_plots (bool) : If True, plots are saved. 
        close_plot (bool) : If True, plots are closed such that they are not shown in the jupyter notebook. 
        weight_fits (bool) : If True, the residuals in the least square fit are weighted with the inverse standard error. 
        seed (int) : Seed for random number generator used in bootstrap fitting. 
    returns: 
        smin_ (float) : Estimated position of the minimum of the fit ansatz. 
        smin_err_ (float) : Estimated error of the position of the minimum of the fit ansatz. 
        tint_at_smin_ (float) : Estimated integrated autocorrelation time at the respective minimum. 
        tint_at_smin_err_ (float) : Estimated error of the integrated autocorrelation time at the respective minimum. 
    """
    N_obs = len(obs_indices)
    N_crits = len(crit_dict)
    n_cols = N_obs//2
    
    fs, fs2, ms, lw = 16, 20, 7, 2
    marker_ = ["o", "s", "x"] # TODO: add 3
    cs = [(2/255, 61/255, 107/255), (217/255, 85/255, 20/255), (163/255, 201/255, 0/255), (100/255, 2/255, 107/255), (107/255, 47/255, 2/255), "tab:blue", "tab:green", "tab:orange", "tab:red"]
        
    obs_names_ = ["mean", "absmean", "radius", "mean_x0", "absmean_x0", "radius_x0"]   
    obs_names = [r"$\sum_{i} x_i/d$", r"$\sum_{i} |x_i|/d$", r"$\sum_{i} x_i^2/d$", r"$x_0$", r"$|x_0|$", r"$x_0^2$"]
    true_mean = [0., 0.3810436956417012, 0.23105857863000492, 0., 0.3810436956417012, 0.23105857863000492] #true mean for specific choice of beta; has to be changed when choice is changed
    N_MD, t_MD, sig_, Nconf_str = sim_dict["NMD"], sim_dict["tMD"], sim_dict["sig"], sim_dict["Nconf"]
    
    smin_, smin_err_, tint_at_smin_, tint_at_smin_err_ = np.zeros((N_crits, N_obs)), np.zeros((N_crits, N_obs)), np.zeros((N_crits, N_obs)), np.zeros((N_crits, N_obs))
    
    for crit_j, (crit_name, crit_dict) in enumerate(crit_dict.items()):
        # tmp_str = f"_NMD{N_MD}_tMD"+str(t_MD)+"_wb0p125_N"+Nconf_str+"_"+crit_name
        
        c_fit_dict = fit_dict[crit_name]
        n_bootstrap= c_fit_dict["n_bootstrap"]
        custom_xlim = c_fit_dict["custom_xlim"]
        custom_xticks = c_fit_dict["custom_xticks"] 
        custom_ylim = c_fit_dict["custom_ylim"]
        custom_yticks = c_fit_dict["custom_yticks"]
        loglog = c_fit_dict["loglog"]
        outlier_inds = [c_fit_dict["outlier_inds"]]
        
        fig, ax = plt.subplots(2, n_cols, figsize = (n_cols*8., 2*6.4), sharex = True)
        if n_cols == 1:
            ax = np.reshape(ax, (2, n_cols))
        if plot_obs:
            fig2, ax2 = plt.subplots(2, n_cols, figsize = (n_cols*8., 2*6.4), sharex = True)
        
        if len(outlier_inds)==1:
            outlier_inds = tuple(N_obs*[outlier_inds[0]])
        elif len(outlier_inds)!=N_obs:
            outlier_inds = tuple(N_obs*[[0]])
        for obs_i, obs_ind in enumerate(obs_indices):
            j, i = obs_i%n_cols, obs_i//n_cols
            # fname = postpro_path + f"d{d}/Rtoy_"+obs_names_[obs_ind]+f"_d{d}"+tmp_str+".txt"
            fname = postpro_path + f"d{d}/Rtoy_b0p125_"+obs_names_[obs_ind]+f"_d{d}"+f"_NMD{N_MD}_tMD"+str(t_MD)+"_"+crit_name+".txt"
            sig, mean, var, err, err_err, tint, tint_err, W = np.loadtxt(fname, usecols = (0, 1, 2, 3, 4, 5, 6, 7), skiprows = 1, unpack = True)
            sig, mean, var, err, err_err, tint, tint_err, W = sort_arrays_according_to_array(sig, mean, var, err, err_err, tint, tint_err, W)
            _, sig, mean, var, err, err_err, tint, tint_err, W = delete_indices_from_arrays(outlier_inds[obs_i], sig, mean, var, err, err_err, tint, tint_err, W)
            cax = ax[i, j]
            cax.errorbar(sig, tint, tint_err, color = cs[0], marker = "o", markersize = ms, linestyle = "", capsize = 4, label = r"$\tau_{\mathrm{int}}$")
            if plot_obs:
                cax2 = ax2[i, j]
                cax2.errorbar(sig, mean, err, color = cs[0], marker = "o", markersize = ms, linestyle = "", capsize = 4, label = r"$\mathcal{O}$")
                cax2.axhline(true_mean[obs_ind], color = "k", linestyle = "--", linewidth = lw, label = "True")
                cax2.legend(fontsize = fs), ax2[i, 0].set_ylabel(r"$\mathcal{O}$", fontsize = fs2)
                ax2[-1, j].set_xlabel(r"$\sigma_R$", fontsize = fs2)
                cax2.set_title(r"$\mathcal{O}=$"+obs_names[obs_ind], fontsize = fs, pad = 25)
                if loglog:
                    cax2.set_xscale("log")
                cax2.grid()
                fig2.suptitle(f"d={d}; "+crit_dict["supt_name"], fontsize = fs2)
            if n_bootstrap:
                # Standard bootstrap fit to determine sigma
                fit_succeeded, \
                sig_cont, tint_cont, tint_err_cont, \
                fit_params, fit_stderr, \
                smin_boot, smin_err, \
                tauintmin, tauintminerr = bootstrap_fit_and_sigmin(fit_f = ac_fit_f, 
                                                                   x_data = sig, 
                                                                   y_data = tint, 
                                                                   y_data_err = tint_err, 
                                                                   n_bootstrap = n_bootstrap, 
                                                                   weight_fits = weight_fits)
                tint_at_smin_[crit_j, obs_i], tint_at_smin_err_[crit_j, obs_i] = tauintmin, tauintminerr
                smin_[crit_j, obs_i], smin_err_[crit_j, obs_i] = smin_boot, smin_err
                if fit_succeeded: #Check if fitting failed, i.e. attempted number of fits exceeded the threshhold
                    cax.set_xlim(np.amin(sig_cont), np.amax(sig_cont))
                    val1, err1 = round_to_n_of_err(fit_params[0], fit_stderr[0], n = 2)
                    val2, err2 = round_to_n_of_err(fit_params[1], fit_stderr[1], n = 2)
                    val3, err3 = round_to_n_of_err(fit_params[2], fit_stderr[2], n = 2)
                    cax.plot(sig_cont, tint_cont, color = cs[2], linewidth = lw, 
                          label = r"$\tau_{\mathrm{int}}(\sigma_R)$ fit"+f"\na="+get_parentheses_error_notation(val1, err1)+"\nb="+get_parentheses_error_notation(val2, err2)+"\nc="+get_parentheses_error_notation(val3, err3))
                    cax.fill_between(sig_cont, np.add(tint_cont, tint_err_cont), np.subtract(tint_cont, tint_err_cont), color = cs[2], alpha = 0.5)
                    val1, err1 = round_to_n_of_err(smin_boot, smin_err, n = 2)
                    cax.axvline(val1, color = "tab:grey", linestyle = "--", linewidth = lw, label = r"$\sigma_R^{(\mathrm{min})}=$"+get_parentheses_error_notation(val1, err1))

            cax.set_title(r"$\mathcal{O}=$"+obs_names[obs_ind], fontsize = fs, pad = 25)
            if loglog:
                cax.set_xscale("log"), cax.set_yscale("log")
            ax[-1, j].set_xlabel(r"$\sigma_R$", fontsize = fs2), ax[i, 0].set_ylabel(r"$\tau_{\mathrm{int}}$", fontsize = fs2)
            cax.legend(fontsize = fs-2)
            cax.grid()
            ########################## OLD FUNCTION ###############################
            if custom_xlim:
                cax.set_xlim(*custom_xlim)
                if plot_obs:
                    cax2.set_xlim(*custom_xlim)
            if custom_xticks:
                cax.set_xticks(custom_xticks[0], labels = custom_xticks[1])
                if plot_obs:
                    cax2.set_xticks(custom_xticks[0], labels = custom_xticks[1])
            if custom_ylim:
                cax.set_ylim(*custom_ylim[obs_i])
            if custom_yticks:
                tmp = custom_yticks[obs_i]
                cax.set_yticks(tmp[0], labels = tmp[1])
            cax.tick_params(axis='x', labelsize=fs), cax.tick_params(axis='y', labelsize=fs)
            if plot_obs:
                cax2.tick_params(axis='x', labelsize=fs), cax2.tick_params(axis='y', labelsize=fs)
        fig.suptitle(crit_name, fontsize = fs2)
        fig.subplots_adjust(left=None,
                            bottom=None, 
                            right=None, 
                            top= 0.9, 
                            wspace=None, 
                            hspace=0.2)
        if save_plots:
            add_str1 = "_weighted" if weight_fits else ""
            save_fname = plot_path + f"Rtoy_d{d}"+"_"+crit_name+add_str1+".png"
            fig.savefig(save_fname, dpi = 300, bbox_inches = "tight")
        if close_plot:
            plt.close(fig)
        else:
            fig.show()
        if plot_obs:
            fig2.subplots_adjust(left=None,
                            bottom=None, 
                            right=None, 
                            top= 0.9, 
                            wspace=None, 
                            hspace=0.2)
            if save_plots:
                add_str1 = "_weighted" if weight_fits else ""
                save_obs_fname = plot_path + f"Rtoy_obs_d{d}"+"_"+crit_name+add_str1+".png"
                fig2.savefig(save_obs_fname, dpi = 300, bbox_inches = "tight") 
            if close_plot:
                plt.close(fig2)
            else:
                fig2.show()
    return smin_, smin_err_, tint_at_smin_, tint_at_smin_err_



def compare_rtoy_ac_crits(d: int, 
                          sim_dict: dict, 
                          crit_dict: dict, 
                          postpro_path: str, 
                          plot_path: str, 
                          obs_indices = (0, 1, 2, 3, 4, 5), 
                          save_plots: bool = False, 
                          loglog: bool = True, 
                          seed: int = 1337):
    """
    Function to make autocorrelation plots for the radial toy model. 
    """
    N_obs = len(obs_indices)
    N_crits = len(crit_dict)
    n_cols = N_obs//2
    
    fs, fs2, ms, lw = 16, 20, 7, 2
    marker_ = ["o", "s", "x"] # TODO: add 3
        
    obs_names_ = ["mean", "absmean", "radius", "mean_x0", "absmean_x0", "radius_x0"]   
    obs_names = [r"$\sum_{i} x_i/d$", r"$\sum_{i} |x_i|/d$", r"$\sum_{i} x_i^2/d$", r"$x_0$", r"$|x_0|$", r"$x_0^2$"]
    true_mean = [0., 0.3810436956417012, 0.23105857863000492, 0., 0.3810436956417012, 0.23105857863000492]
    N_MD, t_MD, sig_, Nconf_str = sim_dict["NMD"], sim_dict["tMD"], sim_dict["sig"], sim_dict["Nconf"]
    
    fig, ax = plt.subplots(2, n_cols, figsize = (n_cols*8., 2*6.4), sharex = True)
    if n_cols == 1:
        ax = np.reshape(ax, (2, n_cols))
    for obs_i, obs_ind in enumerate(obs_indices):
        j, i = obs_i%n_cols, obs_i//n_cols
        cax = ax[i, j]
        for crit_j, (crit_name, crit_dict) in enumerate(eval_dict.items()):
            tmp_str = f"_NMD{N_MD}_tMD"+str(t_MD)+"_wb0p125_N"+Nconf_str+"_"+crit_name
            fname = postpro_path + f"d{d}/Rtoy_"+obs_names_[obs_ind]+f"_d{d}"+tmp_str+".txt"
            sig, mean, var, err, err_err, tint, tint_err, W = np.loadtxt(fname, usecols = (0, 1, 2, 3, 4, 5, 6, 7), skiprows = 1, unpack = True)
            sig, mean, var, err, err_err, tint, tint_err, W = sort_arrays_according_to_array(sig, mean, var, err, err_err, tint, tint_err, W)
            cax.errorbar(sig, tint, tint_err, color = cs[crit_j], marker = marker_[crit_j], markersize = ms, linestyle = "", capsize = 4, label = crit_name)
            cax.set_title(r"$\mathcal{O}=$"+obs_names[obs_ind], fontsize = fs, pad = 25)
        if loglog:
            cax.set_xscale("log"), cax.set_yscale("log")
        ax[-1, j].set_xlabel(r"$\sigma_R$", fontsize = fs2), ax[i, 0].set_ylabel(r"$\tau_{\mathrm{int}}$", fontsize = fs2)
        cax.legend(fontsize = fs-2)
        cax.grid()
    fig.suptitle(f"{d=}", fontsize = fs2)
    fig.subplots_adjust(left=None,
                        bottom=None, 
                        right=None, 
                        top= 0.9, 
                        wspace=None, 
                        hspace=0.2)
    if save_plots:
        save_fname = plot_path + f"ac/d{d}/Rtoy_ac_comp_d{d}.png"
        fig.savefig(save_fname, dpi = 300, bbox_inches = "tight")
    fig.show()
    
    
def get_tunneling_rate(fname: str, 
                       bins: list[float], 
                       labels: list[int], 
                       col: int = 3, 
                       nvals: Optional[int] = None):
    """
    Compute tunneling rate of $x_0$ component for radial toy model. 
    """
    x0_hist = np.loadtxt(fname, usecols = (col), max_rows = nvals)
    cut = pd.cut(x0_hist, bins = bins, labels = labels)
    df = pd.crosstab(cut[1:], cut[:-1], dropna = False)
    tm = np.divide(df.to_numpy(), len(x0_hist))
    tmp_tm = np.copy(tm)
    np.fill_diagonal(tmp_tm, 0)
    return np.sum(tmp_tm), tm

def Rtoy_dist_2D(X: np.ndarray, 
                 Y: np.ndarray, 
                 beta: Union[int, float]):
    """
    Function to compute the two-dimensional toy model distribution for a given $\beta$ value. 
    args: 
        X, Y (np.ndarray) : two-dimensional numpy array obtained with np.meshgrid
        beta (int, float) : beta value in toy model action. 
    """
    p_X = np.cos(X)**2 * np.exp(-beta*np.power(X, 2))
    p_Y = np.cos(Y)**2 * np.exp(-beta*np.power(Y, 2))
    full_P = np.multiply(p_X, p_Y)
    x = np.arange(-20, 20+0.001, 0.001)
    px = np.cos(x)**2 * np.exp(-beta*np.power(x, 2))
    Z = np.trapz(px, x)
    return np.divide(full_P, np.power(Z, 2))

def get_margin_dist(beta: Union[int, float], 
                    lim: Union[int, float] = 20, 
                    dx: Union[int, float] = 0.01):
    """
    Function to get the marginal toy model distribution. 
    Careful, even though the toy model is solvable analytically, this function determines the partition function numerically so "lim" and "dx" should be chosen wisely. The provided values suffice for a very good estimate. 
    args:
        beta (int, float) : beta value in toy model action. 
        lim (int, float) : interval limits for determining the toy model action. 
        dx (int, float) : spacing in numpy array. 
    returns:
    
    """
    x = np.arange(-lim, lim+dx, dx)
    px = np.cos(x)**2 * np.exp(-beta*np.power(x, 2))
    Z = np.trapz(px, x)
    return x, px/Z