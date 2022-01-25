# IOTA Access Control Simulator
Python simulator for the IOTA Access/Congestion Control Algorithm (ICCA).

# Installation
Install Conda 4.10.3 or greater.
If you are using a Mac, you can create the environment from the provided yaml file.
```console
~$ conda env create -f environment.yml
~$ conda activate iota
```
On Windows machines, it seems that you will need to create the environment from scratch using the following commands.
```console
~$ conda create --name iota python=3.9
~$ conda activate iota
~$ conda install numpy=1.21
~$ conda install matplotlib=3.4
~$ conda install networkx=2.6
~$ conda install dash=1.19
~$ conda install pandas=1.3
```

# Running Dash app
Go to the `global_params.py` file and ensure that `DASH=True`.

Run `main.py` and a browser window should open showing the live plots from the simulation.