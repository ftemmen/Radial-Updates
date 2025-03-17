# Radial Updates
This repository contains simulation and analysis code used in the preparation of our paper on Radial Updates. It includes the implementation of Radial Updates in a toy model, along with data analysis code to reproduce key results and figures.  

This code requires the following Python packges: `numpy`, `matplotlib`, `scipy`, `pandas`, and `h5py`. 
You can install them using:
```
pip install numpy matplotlib scipy pandas h5py
```

### Toy model
The directory `toy_model/` contains code to simulate the toy model using Hybrid Monte Carlo and Radial Updates.  
The script `main.py` can be used to perform simulations detailed in the configuration file `radial_toy.yaml`.  
To run the code navigate into `toy_model/` and execute: 
```
python main.py radial_toy.yaml
```


### Analysis code
The analysis code for post-processing simulation data and generating plots is provided as Jupyter notebooks in the `analysis` directory.  
To run the notebooks, make sure the required data is available. Also, you'll need to set the global path to the data folder at the top of each notebook. 
