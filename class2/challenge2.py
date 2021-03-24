import sys
import os
_dir = os.path.dirname( __file__ )
sys.path.append(os.path.abspath(os.path.join(_dir, '..')))

from pyprotosyn import *
from class1_solutions.challenge1 import *
import numpy as np
from copy import deepcopy
import torch
import torchani

periodic_table = {"H": 1, "C": 6, "N": 7, "O": 8}

# Ex 1) Monte Carlo simulation
# 1.1) Load the rotamer library

# 1.2) Load the starting molecule

# 1.3) Set-up the Monte Carlo settings

# 1.4) Measure the starting energy and print initial results

    # 1.5) Copy the molecule as a backup

    # 1.6) Measure the chosen aminoacid phi and psi angles

    # 1.7) Sample a random Rotamer from the correct RotamerStack, based on
    # measured phi and psi angles

    # 1.8) Apply the rotamer to the molecule

    # 1.9) Measure the energy of the new conformation

    # 1.10) Choose wether to keep the new conformation

    # 1.11) Export the current conformation every 10 steps