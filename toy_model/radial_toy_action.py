import numpy as np
from typing import Tuple, Union

class Radial_toy_action():
    """
    Toy action for radial updates with: 
        p[x] \propto \prod_{i=1}^d cos^2(w * x_i) * exp( -\beta * x_i^2)
    The action is given by
        S[x] = \beta \sum_i x_i^2 - \sum_i ln(cos^2(w * x_i))
    and the force term by
        dS/dx_i = 2 (\beta x_i -  w * tan(w * x_i))
    """
    def __init__(self, omega: Union[float, int] = 1., beta: Union[float, int] = 1.):
        """
        Initialization of toy model action. While one can specifiy omega and beta, one can absorb either in the field, which basically leads modifies the other variable. Therefore, in our study, we assumed omega = 1. and changed only beta. 
        args: 
            omega (float) : frequency parameter in cosine terms
            beta (float)  : "inverse width" of gaussian terms
        """
        self.omega, self.beta = omega, beta

    def get_action_grad(self, cfg: np.ndarray) -> np.ndarray:
        """
        Compute gradient of the action for HMC. 
        args: 
            cfg (np.ndarray) : Field configuration.
        returns:
            gradient (np.ndarray) : Gradient of the action with same shape as cfg. 
        """
        return 2*(self.beta * cfg + self.omega * np.tan(self.omega * cfg))

    def get_action(self, cfg: np.ndarray) -> float:
        """
        Compute action of the field configuration cfg. 
        args: 
            cfg (np.ndarray) : Field configuration.
        returns: 
            action (float) : Action of the configuration cfg. 
        """
        r_sq = np.sum(np.power(cfg, 2))
        log_cos = 2*np.log(np.fabs(np.cos(self.omega*cfg)))
        return self.beta * r_sq - np.sum(log_cos)

    def get_obs(self, cfg: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        """
        Compute observables for the toy model. In total we considered six observables: average field, L0 norm of the field and mean of field squared, as well as the respective counterparts for the 0th component of the field. 
        args: 
            cfg (np.ndarray) : Field configuration.
        returns: 
            obs (tuple of floats) : Values for the observables. Six in total. 
        """
        return np.mean(cfg), np.mean(np.fabs(cfg)), np.mean(np.power(cfg, 2)), cfg[0], np.fabs(cfg[0]), np.power(cfg[0], 2)
