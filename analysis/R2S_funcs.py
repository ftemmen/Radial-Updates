import numpy as np

def fermi_matrix_exp(phi1, phi2, kappa, beta, Nt):
    kappa_tilde = kappa * beta / Nt
    return 2*(np.cosh(0.5*np.add(phi1, phi2)) + np.cosh(0.5*np.subtract(phi1, phi2))*np.cosh(kappa_tilde))

def two_site_dist(phi1, phi2, Nt, kappa, U, beta):
    phi1_ = phi1*1j
    phi2_ = phi2*1j
    return fermi_matrix_exp(phi1_, phi2_, kappa, beta, Nt)*fermi_matrix_exp(-phi1_, -phi2_, kappa, beta, Nt)*np.exp(np.add(np.power(phi1_, 2), np.power(phi2_, 2))/(2*U*beta))

def one_site_dist(phi1, phi2, U, beta):
    Ub = U*beta
    num1 = np.power(np.cos(0.5*phi1), 2)
    num2 = np.power(np.cos(0.5*phi2), 2)
    num3 = np.exp(Ub - np.add(np.power(phi1, 2), np.power(phi2, 2))/(2*Ub) )
    denom = 30*np.pi*(np.exp(0.5*Ub)+1)**2
    return (num1 * num2 * num3)/denom

def one_site_margin(phi, U, beta):
    Ub = U*beta
    num1 = np.exp(0.25*Ub - np.power(phi, 2)/(2*Ub))
    num2 = np.divide(1, np.cosh(0.25*Ub))
    num3 = np.power(np.cos(0.5*phi), 2)
    denom = np.sqrt(2*np.pi*Ub)
    return (num1 * num2 * num3)/denom


def full_two_site_dist(lim, dx, U, beta, Nt, kappa):
    ### unnormalized
    x = y = np.arange(-lim, lim+dx, dx)
    X_twosite, Y_twosite = np.meshgrid(x, y)
    Z_twosite = two_site_dist(X_twosite, Y_twosite, Nt = Nt, kappa = kappa, U = U, beta = beta)

    b = np.ones_like(x)
    marginal_dist = np.zeros_like(x)

    for i, val in enumerate(x):
        out = np.real(two_site_dist(x, b*val, Nt = Nt, kappa = kappa, U = U, beta = beta))
        marginal_dist[i] = np.trapz(out, dx = dx)

    norm = np.trapz(marginal_dist, dx = dx)
    normed_marginal_dist = np.divide(marginal_dist, norm)
    
    return X_twosite, Y_twosite, np.divide(Z_twosite, norm), x, normed_marginal_dist

def full_one_site_dist(lim, dx, U, beta):
    x = y = np.arange(-lim, lim+dx, dx)
    X_onesite, Y_onesite = np.meshgrid(x, y)
    Z_onesite = one_site_dist(X_onesite, Y_onesite, U = U, beta = beta)
    norm = np.trapz(np.trapz(Z_onesite, y, axis = 1), x, axis = 0)
    
    margin_dist_os = one_site_margin(x, U, beta)
    norm_const = np.trapz(margin_dist_os, x)
    normed_marginal_dist = np.divide(margin_dist_os, norm_const)
    
    return X_onesite, Y_onesite, np.divide(Z_onesite, norm), x, normed_marginal_dist