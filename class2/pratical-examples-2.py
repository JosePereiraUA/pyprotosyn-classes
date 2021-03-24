import sys
import os
_dir = os.path.dirname( __file__ )
sys.path.append(os.path.abspath(os.path.join(_dir, '..')))

from pyprotosyn import *

# Ex 1)
# Create a function that receives a Molecule as input and outputs each atom’s
# species number, in a list.

periodic_table = {"H": 1, "C": 6, "N": 7, "O": 8}

def species_numbers(mol):
    sn = []
    for atom in mol.atoms:
        sn.append(periodic_table[atom.symbol])
    return np.array(sn)

mol = Molecule()
mol.load("mol1.pdb")

species = species_numbers(mol)
print("Species Numbers:", species)

# Ex 2)
# Create a function that receives a Molecule as input and outputs each atom’s
# cartesian coordinates, in a list.

def cartesian_coordinates(mol):
    cc = []
    for atom in mol.atoms:
        cc.append(atom.cartesian)
    return np.array(cc)

cartesian = cartesian_coordinates(mol)
print("Cartesian coordinates:", cartesian)

# Ex 3)
# Create and test a new function in order to calculate the energy of a Molecule.

import torch
import torchani

device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model       = torchani.models.ANI2x(periodic_table_index=True).to(device)
coordinates = torch.tensor([cartesian], requires_grad=True, device=device).float()
species     = torch.tensor([species], device=device, dtype=torch.int64)
energy      = model((species, coordinates)).energies[0].detach().numpy()
print("Energy:", energy)