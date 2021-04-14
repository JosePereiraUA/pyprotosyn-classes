import sys
import os
_dir = os.path.dirname( __file__ )
sys.path.append(os.path.abspath(os.path.join(_dir, '..')))

from pyprotosyn import *
from class1_solutions.challenge1_solutions import *
import numpy as np
from copy import deepcopy
import torch
import torchani
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = torchani.models.ANI2x(periodic_table_index=True).to(device)
periodic_table = {"H": 1, "C": 6, "N": 7, "O": 8}
print("TorchANI model loaded.")

def species_numbers(mol):
    sn = []
    for atom in mol.atoms:
        sn.append(periodic_table[atom.symbol])
    return np.array(sn)

def cartesian_coordinates(mol):
    cc = []
    for atom in mol.atoms:
        cc.append(atom.cartesian)
    return np.array(cc)

def measure_dihedral(atom):
    return atom.dihedral + atom.parent.d_dihedral

# Ex 1) Monte Carlo simulation
# 1.1) Load the rotamer library
filename = "../class1_solutions/dunbrack_rotamers.lib"
rot_lib  = create_rot_lib_matrix(filename)
rot_lib  = fill_rot_lib_matrix(filename, rot_lib)
print("Rotamer library loaded.")

# 1.2) Load the starting molecule
mol = Molecule()
mol.load("mol1.pdb")

# 1.3) Set-up the Monte Carlo settings
N            = 500 # Number of steps
T            = 0.5 # Temperature
export_every = 10
n_aminoacids = len(set([at.res_id for at in mol.atoms]))

# 1.4) Measure the starting energy and print initial results
cartesian   = cartesian_coordinates(mol)
species     = species_numbers(mol)
coordinates = torch.tensor([cartesian], requires_grad=True, device=device).float()
species     = torch.tensor([species], device=device, dtype=torch.int64)
energy      = model((species, coordinates)).energies[0].detach().numpy()
print("Molecule model loaded. Energy: %7.3f" % (energy))
mol.export("mol2.pdb", "w")

energies = [energy]
for step in range(N):
    # 1.4) Copy the molecule as a backup
    backup = deepcopy(mol)

    # 1.5) Choose a random animoacid
    res            = np.random.randint(1, n_aminoacids)
    aminoacid      = list(filter(lambda at: at.res_id == res, mol.atoms))
    aminoacid_next = list(filter(lambda at: at.res_id == res + 1, mol.atoms))
    name           = aminoacid[0].res_name
    if not name in rot_lib.keys():
        if step % export_every == 0:
            mol.export("mol2.pdb", "a", model = step)
        print("Step %4d | Energy: %7.3f" % (step, energy))
        energies.append(energy)
        continue

    # 1.6) Measure the chosen aminoacid phi and psi angles
    C  = list(filter(lambda at: at.name == "C", aminoacid))[0]
    phi = measure_dihedral(C)
    N   = list(filter(lambda at: at.name == "N", aminoacid_next))[0]
    psi = measure_dihedral(N)

    # 1.7) Sample a random Rotamer from the correct RotamerStack, based on
    # measured phi and psi angles
    rotamer_stack = rot_lib[name].get_rotamer_stack(phi, psi)
    norm_weights = np.array(rotamer_stack.weights) / sum(rotamer_stack.weights)
    rotamer = np.random.choice(rotamer_stack.rotamers, 1, p = norm_weights)[0]

    # 1.8) Apply the rotamer to the molecule
    rotamer.apply_to_aminoacid(aminoacid)
    mol.i2c()

    # 1.9) Measure the energy of the new conformation
    cartesian   = cartesian_coordinates(mol)
    species     = species_numbers(mol)
    coordinates = torch.tensor([cartesian], requires_grad=True, device=device).float()
    species     = torch.tensor([species], device=device, dtype=torch.int64)
    new_energy  = model((species, coordinates)).energies[0].detach().numpy()

    # 1.10) Choose wether to keep the new conformation
    metropolis = np.exp(-((new_energy - energy) / T))
    if new_energy < energy or np.random.rand() < metropolis:
        # Accept (save new energy and structure for comparison/backup)
        energy = new_energy
        backup = deepcopy(mol)
    else:
        # Reject (return to backup)
        mol = deepcopy(backup)

    # 1.11) Export the current conformation every 10 steps
    if step % export_every == 0:
        mol.export("mol2.pdb", "a", model = step)

    print("Step %4d | Energy: %7.3f" % (step, energy))
    energies.append(energy)

plt.plot([x for x in range(1, 502)], energies)
plt.show()