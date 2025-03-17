# Radial-Updates
This repository contains simulation and analysis code used in the preparation of our paper on Radial Updates. It includes the implementation of Radial Updates in a toy model, along with data analysis code to reproduce key results and figures.  

This code requires the following Python packges: `numpy`, `matplotlib`, `scipy`, `pandas`, and `h5py`. 
You can install them using:
<pre>```bash pip install numpy matplotlib scipy pandas h5py```</pre>

### Toy model
The directory `toy_model/` contains code to simulate the toy model using Hybrid Monte Carlo and Radial Updates.  
The script `main.py` can be used to perform simulations detailed in the configuration file `toy.yaml`.  
To run the code navigate into `toy_model/` and execute: 
<pre>```bash python main.py toy_model.yaml```</pre>


### Analysis code
The analysis code for post-processing simulation data and generating plots is provided as Jupyter notebooks in the `analysis` directory.  
To run the notebooks, make sure the required data is available. Also, you'll need to set the global path to the data folder at the top of each notebook. 