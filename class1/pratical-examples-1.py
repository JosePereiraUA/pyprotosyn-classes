import sys
import os
_dir = os.path.dirname( __file__ )
sys.path.append(os.path.abspath(os.path.join(_dir, '..')))

from pyprotosyn import *
import numpy as np

mol = Molecule()
mol.load("mol1.pdb")
mol.export("mol2.pdb")

# Ex 2)
# Rotate the PHI dihedral on the second aminoacid by +360 degrees, in 10 degree
# interval, appending each new conformation as a new frame to the PDB file.

# 2.1 The third atom of a phi dihedral is always CA. Which means we need to
# change d_dihedral in the CA to affect the positioning of all children atoms.
# First step is finding all CA atoms.
CAs = list(filter(lambda at: at.name == "CA", mol.atoms))

# 2.2 We can now select the second CA (second aminoacid), and change the
# d_dihedral value
for i in range(1, 36):
    CAs[1].d_dihedral += np.deg2rad(10)

    # 2.3 We now need to update the cartesian coordinates base on the new
    # internal coordinates, since export only prints the current coordinates
    mol.i2c()
    mol.export("mol2.pdb", "a", 1 + i)

# Ex. 3)
# Rotate the PHI dihedral on the second aminoacid to by +X degrees, where X is a
# random number between 50 and 150. Create a function to measure the current
# value of the dihedral, based on the internal coordinates. Verify the correct
# value using the "measure" functionality in PyMOL.

# 3.1 Using the same strategy as in Ex. 1 we can rotate the dihedral by a random
# amount
x = np.random.randint(50, 150)
CAs[1].d_dihedral += np.deg2rad(x)
mol.i2c()

# 3.2 The current value of the dihedral is, therefore, the value of dihedral of
# the last atom of the dihedral plus (+) the d_dihedral value of its parent.
def measure_dihedral(atom):
    return atom.dihedral + atom.parent.d_dihedral

Cs = list(filter(lambda at: at.name == "C", mol.atoms))
dihedral_value = measure_dihedral(Cs[1])

print("Current dihedral: %7.3f" % (np.rad2deg(dihedral_value) % 360))
mol.export("mol2.pdb", "a", 37)

# Ex. 4)
# Rotate the PHI dihedral of the second aminoacid to the specific value of -65
# degrees.

# 4.1 In order to rotate to a specific value we must first measure the current
# value, calculate the necessary rotation, and apply it.
dihedral_value = measure_dihedral(Cs[1])
rot = np.deg2rad(-65) - dihedral_value
Cs[1].parent.d_dihedral += rot
mol.i2c()
mol.export("mol2.pdb", "a", 38)

# Ex 5)
# Set the value of CHI-4 dihedral on aminoacid 2 (ARG) to 90 degreees.

# 5.1. We can filter the list of atoms based on the residue number, in order to
# find the NE atom of aminoacid 2.
aminoacid2 = list(filter(lambda at: at.res_id == 2, mol.atoms))
NE = list(filter(lambda at: at.name == "NE", aminoacid2))[0]

# Once we have the NE atom, simply setting it's d_dihedral to the correct value
# will rotate all children atoms to the requested dihedral
dihedral_value = measure_dihedral(NE)
rot = np.deg2rad(90) - dihedral_value
NE.d_dihedral += rot
mol.i2c()

mol.export("mol2.pdb", "a", 39)