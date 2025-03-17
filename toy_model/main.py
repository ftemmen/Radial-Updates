import numpy as np
import yaml
import sys

from radial_toy_utils import save_arrays
from radial_toy_sim import simulate_radial_toy_model

### TEST SIMULATION
config_file = sys.argv[1]
print(config_file)

with open (config_file, "r") as file:
    config = yaml.safe_load(file)


### MODEL
name = config["model"]["name"]
d = config["model"]["d"]
omega = config["model"]["omega"]
beta = config["model"]["beta"]


### SIMULATION
N_MD = config["HMC"]["N_MD"]
t_MD = config["HMC"]["t_MD"]
N_therm = config["HMC"]["N_therm"]
N_MC = config["HMC"]["N_MC"]
N_skip = config["HMC"]["N_skip"]

radial_scale_ = config["HMC"]["rscale"]

### ANALYSIS
S = config["analysis"]["S"]
    
save_data = config["analysis"]["save_data"]
save_plots = config["analysis"]["save_plots"]
ac_analysis = config["analysis"]["ac_analysis"]
# save_dir = config["analysis"]["save_dir"]

N_obs = 6
shape = (N_obs, len(radial_scale_))
total_mean, total_var, total_err, total_err_err = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
total_t_int, total_t_int_err, total_W = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    
tmp_str = f"_NMD{N_MD}_tMD"+str(t_MD).replace(".", "p")+"_wb"+str(beta).replace(".","p")+"_N"+str(N_MC)[0]+f"e{int(np.log10(N_MC))}"
for j in range(len(radial_scale_)):
    save_data_fname = "data/"+name+f"_d{d}_sig"+str(radial_scale_[j]).replace(".", "p")+tmp_str+".txt" if save_data else None
    # save_data_fname = config["analysis"]["save_dir"]+"raw_data/"+name+f"d{d}_sig"+str(radial_scale_[j]).replace(".", "p")+tmp_str+".txt" if save_data else None
    save_plots_fname = "ac_plots/"+name+f"_d{d}_sig"+str(radial_scale_[j]).replace(".", "p")+tmp_str+".png" if save_plots else None
    # save_plots_fname = config["analysis"]["save_dir"]+"ac_plots/"+name+f"d{d}_sig"+str(radial_scale_[j]).replace(".", "p")+tmp_str+".png" if save_plots else None

    total_mean[:, j], \
    total_var[:, j], \
    total_err[:, j], \
    total_err_err[:, j], \
    total_t_int[:, j], \
    total_t_int_err[:, j], \
    total_W[:, j] = simulate_radial_toy_model(L = d, 
                                             omega = omega, 
                                             beta = beta, 
                                             N_MD = N_MD, 
                                             t_MD = t_MD, 
                                             radial_scale = radial_scale_[j], 
                                             N_therm = N_therm, 
                                             N_MC = N_MC, 
                                             N_skip = N_skip, 
                                             save_data_fname = save_data_fname, 
                                             ac_analysis = ac_analysis, 
                                             plot_ac = save_plots, 
                                             save_ac_plot = save_plots_fname, 
                                             S = S)

if ac_analysis:
    save_fname = "data/"+name+f"_d{d}"+tmp_str+".txt"
    # save_fname = config["analysis"]["save_dir"]+name+f"d{d}_sig"+tmp_str+".txt"
    header = ["sigma_r", 
              "mean0", "mean1", "mean2", "mean3", "mean4", "mean5", 
              "var0", "var1", "var2", "var3", "var4", "var5", 
              "stderr0", "stderr1", "stderr2", "stderr3", "stderr4", "stderr5", 
              "err_err0", "err_err1", "err_err2", "err_err3", "err_err4", "err_err5", 
              "t_int0", "t_int1", "t_int2", "t_int3", "t_int4", "t_int5", 
              "t_int_err0", "t_int_err1", "t_int_err2", "t_int_err3", "t_int_err4", "t_int_err5", 
              "W0", "W1", "W2", "W3", "W4", "W5", ]
    save_arrays(save_fname, header, True, radial_scale_, *total_mean, *total_var, *total_err, *total_err_err, *total_t_int, *total_t_int_err, *total_W)