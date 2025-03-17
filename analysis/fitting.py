import numpy as np
from scipy.optimize import curve_fit, leastsq
import math

from typing import Any, Optional, Tuple, Union, Callable, Sequence

def get_bootstrapped_samples(x_mean: np.ndarray, x_stderr: np.ndarray, rng: Any) -> np.ndarray:
    """
    Function to get a bootstrap sample. The return is sampled from the x_mean using the x_stderr and the given random number generator. 
    args:
        x_mean (np.array) : Mean of the quantity. 
        x_stderr (np.array) : Standard error of the quantity. 
        rng (Numpy rng) : Numpy random number generator class. 
    returns: 
        (np.array) : Bootstrap sample. 
    """
    return np.add(x_mean, np.multiply(rng.standard_normal(size=np.shape(x_mean)), x_stderr))

def bootstrap_fit(fit_f: Callable[[np.ndarray,float,...], np.ndarray], 
                  x_data: np.ndarray, 
                  y_data: np.ndarray, 
                  y_data_err: np.ndarray, 
                  n_bootstrap: int = 100, 
                  weight_fits: bool = False, 
                  seed: int = 1337, 
                  n_plot_points: int = 1000, 
                  x_min: Optional[float] = None, 
                  x_max: Optional[float] = None, 
                  p0: Optional[Sequence[float]] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Boostrap fitting routine.
    Adapted from https://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i
    args: 
        fit_f (function) : Callable fit function. Takes as input the x data and some number of parameters. 
        x_data (np.array) : Input x data for fit. 
        y_data (np.array) : Input y data for fit. 
        y_data_err (np.array) : Input y error data for fit. 
        n_bootstrap (int) : Number of bootstrap samples used in bootstrap fitting routine. 
        weight_fits (bool) : If True, residual in the least square fit are weighted with the inverse of the standard error. 
        seed (int) : Seed for random number generator. 
        n_plot_points (int) : Number of points for plotting. 
        x_min (float) : Lower x limit. 
        x_max (float) : Upper x limit. 
        p0 (Sequence) : Optional inital fit parameters. 
    returns: 
        plot_interval (np.array) : Range of x values for plotting
        mean_f (np.array) : Corresponding mean values obtained from bootstrap fitting. 
        err_f (np.array) : Corresponding standard error values obtained from bootstrap fitting. 
        mean_pfit (np.array) : Mean of fit parameters. 
        err_pfit (np.array) : Error of fit parameters. 
    """
    errfunc = lambda p, x, y: fit_f(x, *p) - y
    if weight_fits:
        errfunc = lambda p, x, y, yerr: (fit_f(x, *p) - y)/yerr
    rng = np.random.default_rng(seed = seed)
    if not p0:
        p0, pcov = curve_fit(fit_f, x_data, y_data, sigma = y_data_err)
    ps, AICs = [], []
    i = 0
    if x_min and x_max:
        plot_interval = np.linspace(x_min, x_max, n_plot_points)
    else:
        min_val, max_val = np.amin(x_data), np.amax(x_data)
        plot_interval = np.linspace(min_val - 10**(math.floor(np.log10(min_val-1e-8))), max_val + 10**(math.floor(np.log10(max_val-1e-8))), n_plot_points)
    bootstrapped_fits = np.zeros((n_bootstrap, n_plot_points))
    while i < n_bootstrap:
        sampled_y_data = np.add(y_data, np.multiply(rng.standard_normal(size=np.shape(y_data)), y_data_err))
        if weight_fits:
            randomfit, randomcov = leastsq(errfunc, p0, args=(x_data, sampled_y_data, y_data_err), full_output=False)
        else:
            randomfit, randomcov = leastsq(errfunc, p0, args=(x_data, sampled_y_data), full_output=False)
        bootstrapped_fits[i, :] = fit_f(plot_interval, *randomfit)
        ps.append(randomfit)
        i += 1
    ps = np.array(ps)
    mean_f = np.mean(bootstrapped_fits, axis = 0)
    err_f = np.std(bootstrapped_fits, axis = 0)
    mean_pfit = np.mean(ps, 0)
    err_pfit = np.std(ps, 0)
    return plot_interval, mean_f, err_f, mean_pfit, err_pfit

def bootstrap_fit_and_sigmin(fit_f: Callable[[np.ndarray,float,...], np.ndarray], 
                             x_data: np.ndarray, 
                             y_data: np.ndarray, 
                             y_data_err: np.ndarray, 
                             n_bootstrap: int = 100, 
                             weight_fits: bool = False, 
                             seed: int = 1337, 
                             skip_thresh: Union[float, int] = 1e5, 
                             n_plot_points: int = 1000, 
                             custom_plot_interval: Optional[list] = None
                            ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], 
                                      Optional[np.float64], Optional[np.float64], Optional[np.float64], Optional[np.float64]]:
    """
    Adapted from https://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i
    Adjusted for the fit y = a*x^{-2} + b + c*x. 
    Additionally determines minimum x_min and the value of y_min at the x_min. 
    In the context of radial updates y is the integrated autocorrelation time and x the proposal standard deviation of the radial updates. 
    args: 
        fit_f (function) : Callable fit function. Takes as input the x data and some number of parameters. 
        x_data (np.array) : Input x data for fit. 
        y_data (np.array) : Input y data for fit. 
        y_data_err (np.array) : Input y error data for fit. 
        n_bootstrap (int) : Number of bootstrap samples used in bootstrap fitting routine. 
        weight_fits (bool) : If True, residual in the least square fit are weighted with the inverse of the standard error. 
        seed (int) : Seed for random number generator. 
        skip_thresh (float,int) : Threshold for maximal number of skipped fits. Fits are skipped if they do not allow for an estimation of the minimum.  
        n_plot_points (int) : Number of points for plotting. 
        custom_plot_interval (list) : List (or tuple) with atleast two number, which are used as lower and upper limit for plot interval. 
    returns: 
        (bool) : If True, fits succeeded. Else, fits failed because threshold was passed. 
        plot_interval (np.array) : Range of x values for plotting
        mean_f (np.array) : Corresponding mean values obtained from bootstrap fitting. 
        err_f (np.array) : Corresponding standard error values obtained from bootstrap fitting. 
        mean_pfit (np.array) : Mean of fit parameters. 
        err_pfit (np.array) : Error of fit parameters. 
        mean_smin (np.float) : Estimated minimum of the fit function. 
        err_smin (np.float) : Error of the estimated minimum of the fit function. 
        tautintmin (np.float) : Estimated integrated autocorrelation time at the minimum. 
        tautintminerr (np.float) : Estimated error of the integrated autocorrelation time at the minimum. 
    """
    errfunc = lambda p, x, y: fit_f(x, *p) - y
    if weight_fits:
        errfunc = lambda p, x, y, yerr: (fit_f(x, *p) - y)/yerr
    rng = np.random.default_rng(seed = seed)

    p0, pcov = curve_fit(fit_f, x_data, y_data, sigma = y_data_err)
    # pfit, perr = leastsq(errfunc, p0, args=(x_data, y_data), full_output=False)
    ps, smins = [], []
    i, fits_skipped = 0, 0
    if custom_plot_interval:
        plot_interval = np.linspace(custom_plot_interval[0], custom_plot_interval[1], n_plot_points)
    else:
        min_val, max_val = np.amin(x_data), np.amax(x_data)
        plot_interval = np.linspace(min_val - 10**(math.floor(np.log10(min_val-1e-8))), max_val + 10**(math.floor(np.log10(max_val-1e-8))), n_plot_points)
    bootstrapped_fits = np.zeros((n_bootstrap, n_plot_points))
    ps = np.zeros((n_bootstrap, len(p0)))
    smins = np.zeros(n_bootstrap)
    while i < n_bootstrap and fits_skipped < skip_thresh:
        sampled_y_data = np.add(y_data, np.multiply(rng.standard_normal(size=np.shape(y_data)), y_data_err))
        if weight_fits:
            randomfit, randomcov = leastsq(errfunc, p0, args=(x_data, sampled_y_data, y_data_err), full_output=False)
        else:
            randomfit, randomcov = leastsq(errfunc, p0, args=(x_data, sampled_y_data), full_output=False)
        ### Compute chi squared and weight fit: BEGIN ###
        if randomfit[0]<0 or randomfit[2]<=0:
            fits_skipped += 1
        else:
            bootstrapped_fits[i, :] = fit_f(plot_interval, *randomfit)
            ps[i, :] = randomfit
            smins[i] = np.power(2*randomfit[0]/randomfit[2], 1/3) 
            i += 1
    print("Fits skipped", fits_skipped)
    if fits_skipped>=skip_thresh:
        return False, None, None, None, None, None, None, None, None, None
    else:
        mean_f, err_f = np.mean(bootstrapped_fits, axis = 0), np.std(bootstrapped_fits, axis = 0)
        mean_pfit, err_pfit = np.mean(ps, 0), np.std(ps, 0)
        mean_smin, err_smin = np.mean(smins), np.std(smins)
        tauintmin_ = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            tauintmin_[i] = fit_f(mean_smin, *ps[i, :])
        tauintmin, tauintminerr = np.mean(tauintmin_), np.std(tauintmin_)
        return True, plot_interval, mean_f, err_f, mean_pfit, err_pfit, mean_smin, err_smin, tauintmin, tauintminerr


def ac_fit_f(sigma: np.ndarray, a: float, b: float, c: float):
    """
    Fit function for integrated autocorrelation time as a function of proposal standard deviation. 
    $\tau_{int} (\sigma) = a\sigma^{-2} + b + c\sigma$
    """
    return np.divide(a, np.power(sigma, 2)) + b + np.multiply(c, sigma)

def smin_fit_f(d: np.ndarray, alpha: float, beta: float):
    """
    Fit for leading order scaling with dimensionality d. 
    $\sigma (d) = \alpha d^{\beta}$
    """
    return np.multiply(alpha, np.power(d, beta))

def logsmin_fit_f(d: np.ndarray, alpha: float, beta: float):
    """
    Log-Fit for leading order scaling with dimensionality d. 
    $\log \sigma (d) = \alpha' + \beta \log d$
    """
    return np.multiply(beta, d) + alpha
