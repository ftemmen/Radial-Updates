import numpy as np
from typing import Any, Tuple, Union

float_dtype = np.float64

class Scalar_HMC():
    """
    Hybrid Monte Carlo (HMC) algorithm for real scalar fields. 
    """
    def __init__(self, 
                 model: Any, 
                 t_MD: float, 
                 N_MD: int, 
                 cfg_shape: Union[tuple, list], 
                 radial_scale: Union[int, float] = 1., 
                 seed: int = 1337):
        """
        args: 
            model : Action class as specified in radial_toy_action; Needs member functions "get_action", "get_action_grad", "get_obs"
            t_MD (float) : HMC trajectory length. 
            N_MD (int) : Number of molecular dynamics (MD) steps. 
            cfg_shape (tuple, list) : Shape of a configuration matching the model. 
            radial_scale (int, float) : Proposal standard deviation for radial updates. 
            seed (int) : Seed for random number generator. 
        """
        self.model = model
        self.N_MD = N_MD
        self.dt, self.dt_half = t_MD/self.N_MD, 0.5*t_MD/self.N_MD
        self.cfg_shape =  cfg_shape
        self.d = np.prod(self.cfg_shape)
        self.rng = np.random.default_rng(seed)
        self.cfg = self.rng.normal(loc = 0., scale = 1/self.d, size = self.cfg_shape).astype(float_dtype)
        self.mom = self.rng.standard_normal(self.cfg_shape).astype(float_dtype)
        self.action = self.model.get_action(self.cfg)
        self.steps, self.acc = 0, 0
        self.radial_scale = radial_scale
        self.radial_steps, self.radial_acc = 0, 0
        
    def step(self):
        """
        HMC step. 
        """
        self.mom = self.rng.standard_normal(self.cfg_shape).astype(float_dtype)
        old_cfg = np.copy(self.cfg) 
        old_H = self.action + 0.5 * np.sum(np.power(self.mom, 2))

        ### Leapfrog integration
        self.mom -= self.dt_half*self.model.get_action_grad(self.cfg)
        for _ in range(self.N_MD-1):
           self.cfg += self.dt*self.mom
           self.mom -= self.dt*self.model.get_action_grad(self.cfg)
        self.cfg += self.dt*self.mom
        self.mom -= self.dt_half*self.model.get_action_grad(self.cfg)

        old_action = self.action
        self.action = self.model.get_action(self.cfg)
        new_H = self.action + 0.5 * np.sum(np.power(self.mom, 2))
        p = np.exp(old_H - new_H)
        if p <= self.rng.random(): # rejected
            self.cfg = old_cfg
            self.action = old_action
        else: # accepted
            self.acc += 1
        self.steps += 1

    def radial_update(self):
        """
        Radial update step. 
        """
        gamma = self.rng.normal(loc = 0., scale = self.radial_scale)
        new_cfg = np.exp(gamma)*self.cfg
        new_action = self.model.get_action(new_cfg)
        p = np.exp(self.action - new_action + self.d*gamma)
        if p >= self.rng.random(): # accepted
            self.cfg = new_cfg
            self.action = new_action
            self.radial_acc += 1
        self.radial_steps += 1

    def out_radial_acc(self) -> float:
        """
        Output current radial acceptance rate. 
        """
        return self.radial_acc/self.radial_steps

    def reset_radial_acc(self):
        """
        Reset the current radial acceptance rate, e.g. after thermalization. 
        """
        self.radial_acc = self.radial_steps = 0

    def calc_obs(self) -> Tuple[float,...]:
        """
        Compute observables from model. 
        """
        return self.model.get_obs(self.cfg)

    def out_acc(self):
        """
        Output current HMC acceptance rate. 
        """
        return self.acc/self.steps

    def reset_acc(self):
        """
        Reset the current HMC acceptance rate, e.g. after thermalization. 
        """
        self.acc = self.steps = 0

    def out_cfg(self) -> np.ndarray:
        """
        Output the current configuration.
        """
        return self.cfg
    
    