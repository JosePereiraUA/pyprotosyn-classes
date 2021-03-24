import sys
import os
_dir = os.path.dirname( __file__ )
sys.path.append(os.path.abspath(os.path.join(_dir, '..')))

from pyprotosyn import *
import numpy as np

aminoacid_sc_heavy_atoms = {
    "ALA": [],
    "CYS": ["CB"],
    "ASP": ["CB", "CG"],
    "GLU": ["CB", "CG", "CD"],
    "PHE": ["CB", "CG"],
    "GLY": [],
    "HIS": ["CB", "CG"],
    "HIE": ["CB", "CG"],
    "ILE": ["CB", "CG1"],
    "LYS": ["CB", "CG", "CD", "CE"],
    "LEU": ["CB", "CG"],
    "MET": ["CB", "CG", "SD"],
    "ASN": ["CB", "CG"],
    "PRO": ["CB", "CG", "CD"],
    "GLN": ["CB", "CG", "CD"],
    "ARG": ["CB", "CG", "CD", "NE"],
    "SER": ["CB"],
    "THR": ["CB"],
    "VAL": ["CB"],
    "TRP": ["CB", "CG"],
    "TYR": ["CB", "CG"]
}

class Rotamer:
    """
    Hold the information relative to a single Rotamer, and is able to be applied
    to a Molecule object (given a list of atoms).
    """


class RotamerStack:
    """
    Hold a list of Rotamers (at a specific phi and psi value) and the
    corresponding natural occurrence probability weights.
    """


class RotamerLib:
    """
    Hold a matrix of RotamerStacks, for each phi and psi values read from a
    file. Is specific for a certain aminoacid type.
    """


if __name__ == "__main__":

    # Ex 1)
    # Read and load the Dunbrack rotamer library in such a way that any of the
    # rotamers can be applied to a Molecule (changing the dihedral angles of all
    # chi angles).


    # Ex 2)
    # Sample a rotamer for aminoacid “ARG”, based on the measured phi and psi
    # angles of aminoacid 2 on “mol1.pdb” structure.

    # 2.1) Load "mol1.pdb" and measure the phi and psi angles at aminoacid 2.

    # 2.2) Get the RotamerStack for that specific pair of phi and psi values.

    # 2.3) Sample a Rotamer from the RotamerStack, based on the respective
    # weights (normalized so that sum(weights) = 1.0)


    # Ex 3)
    # Apply a sampled rotamer (from ex.2) to the aminoacid 2 on “mol1.pdb”.
    # Export the changed structure to “mol2.pdb” and observe the changes on
    # PyMOL.

    # Ex 4)
    # For all aminoacids of “mol1.pdb” structure, sample a new rotamer based on
    # the measured phi and psi angles (taking the natural occurrence probability
    # into account) and apply them to the corresponding aminoacid. Export the
    # changed structure to “mol2.pdb”. Perform this loop 10 times, appending a
    # new frame to the “mol2.pdb” trajectory. Observe the results on PyMOL.

            # 4.1) Get list of atoms and name of aminoacid N.

            # 4.2) Get the RotamerStack for the current phi and psi value. 

            # 4.3) Sample Rotamer from RotamerStack based on normalized weights

            # 4.4) Apply the rotamer