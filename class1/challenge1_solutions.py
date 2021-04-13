import sys
import os
_dir = os.path.dirname( __file__ )
sys.path.append(os.path.abspath(os.path.join(_dir, '..')))

from pyprotosyn import *
import numpy as np

aminoacid_sc_heavy_atoms = {
    "ALA": ["HB1"],
    "CYS": ["CB", "SG"],
    "ASP": ["CB", "CG", "OD1"],
    "GLU": ["CB", "CG", "CD", "OE1"],
    "PHE": ["CB", "CG", "CD1"],
    "GLY": [],
    "HIS": ["CB", "CG", "ND1"],
    "HIE": ["CB", "CG", "ND1"],
    "ILE": ["CB", "CG1", "CD", "HD1"],
    "LYS": ["CB", "CG", "CD", "CE", "NZ"],
    "LEU": ["CB", "CG", "CD1"],
    "MET": ["CB", "CG", "SD", "CE", "NZ"],
    "ASN": ["CB", "CG", "OD1", "HD21"],
    "PRO": ["CB", "CG", "CD", "HD1"],
    "GLN": ["CB", "CG", "CD", "NE2"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ"],
    "SER": ["CB", "OG"],
    "THR": ["CB", "OG1"],
    "VAL": ["CB", "CG1"],
    "TRP": ["CB", "CG", "CD1"],
    "TYR": ["CB", "CG", "CD1"]
}

class Rotamer:
    """
    Hold the information relative to a single Rotamer, and is able to be applied
    to a Molecule object (given a list of atoms).
    """

    def __init__(self, name, chis):
        self.name = name
        self.chis = chis

    """
    Apply this Rotamer to an aminoacid. `aminoacid_atoms` should be a list of
    atoms belonging to a single aminoacid, retieved, for example, using
    `aminoacid2 = list(filter(lambda at: at.res_id == 2, mol.atoms))`
    """
    def apply_to_aminoacid(self, aminoacid_atoms):
        atoms = []
        for h_atom_name in aminoacid_sc_heavy_atoms[self.name][1:]:
            h_atom = list(filter(lambda at: at.name == h_atom_name, aminoacid_atoms))[0]
            atoms.append(h_atom)
        self.apply_to(atoms)

    """
    Apply this Rotamer to a specific set of atoms. Can be any atoms, as long as
    the same number of available chis. Will set the dihedral angle of each atom
    in the list to match the N'th chi value in the Rotamer (therefore atoms
    should be the 3rd atom of each chi dihedral in the sidechain). Requires
    a mol.i2c() after to apply the changes made to the internal coordinates.
    """
    def apply_to(self, atoms):
        if not len(atoms) == len(self.chis):
            print("The number of provided atoms doesn't match the available chis to apply!")
            return
        
        for (chi, atom) in zip(self.chis, atoms):
            value = np.random.randn() * chi[1] + chi[0]
            dihedral_value = atom.dihedral + atom.parent.d_dihedral
            rot = value - dihedral_value
            atom.parent.d_dihedral += rot


class RotamerStack:
    """
    Hold a list of Rotamers (at a specific phi and psi value) and the
    corresponding natural occurrence probability weights.
    """
    def __init__(self):
        self.weights  = []
        self.rotamers = []

    def __str__(self):
        return str(self.weights) + "\n" + str(self.rotamers)

    def __repr__(self):
        return str(self)


class RotamerLib:
    """
    Hold a matrix of RotamerStacks, for each phi and psi values read from a
    file. Is specific for a certain aminoacid type.
    """
    def __init__(self, name, phis, psis, rot_stacks):
        self.name       = name
        self.phis       = phis
        self.psis       = psis
        self.rot_stacks = rot_stacks

    def get_rotamer_stack(self, phi, psi):
        """
        Search for the RotamerStack for a particular phi and psi pair of values.
        Note: Always search in radians.
        """
        phi_index = self.phis[self.phis <= phi].argmax()
        psi_index = self.psis[self.psis <= psi].argmax()
        return self.rot_stacks[phi_index, psi_index]



def create_rot_lib_matrix(filename):
    """
    Read a rotamer library file and create the necessary/found matrices (empty).
    """

    matrices = {}

    with open(filename, "r") as file_in:
        for line in file_in:
            if line.startswith("# phi interval, deg"):
                elem = line.split()
                phi_lower_bound = np.deg2rad(float(elem[4][1:-2]))
                phi_upper_bound = np.deg2rad(float(elem[5][:-2]))

            if line.startswith("# phi step, deg"):
                elem = line.split()
                phi_step = np.deg2rad(float(elem[4]))

            if line.startswith("# psi interval, deg"):
                elem = line.split()
                psi_lower_bound = np.deg2rad(float(elem[4][1:-2]))
                psi_upper_bound = np.deg2rad(float(elem[5][:-2]))

            if line.startswith("# psi step, deg"):
                elem = line.split()
                psi_step = np.deg2rad(float(elem[4]))

            if line.startswith("# Input data taken from"):
                name = str(line.split()[5])
                phis = np.arange(phi_lower_bound, phi_upper_bound, phi_step)
                psis = np.arange(psi_lower_bound, psi_upper_bound, psi_step)
                
                rot_stacks = np.empty((len(phis), len(psis)), dtype = object)
                rot_stacks.flat = [RotamerStack() for _ in rot_stacks.flat]
                np.append(phis, phi_upper_bound)
                np.append(psis, psi_upper_bound)

                matrices[name] = RotamerLib(name, phis, psis, rot_stacks)
    return matrices


def fill_rot_lib_matrix(filename, matrices):
    """
    Read a rotamer library file and created empty matrices (for example, from
    `create_rot_lib_matrix` and fill them with the found values.
    """
    with open(filename, "r") as file_in:
        for line in file_in:
            if line.startswith("#"):
                continue
            elem = line.split()
            
            # Create Rotamer containing its name and chi values + standard dev.
            name = elem[0]
            chis = []
            for index in range(4):
                if elem[index + 5] == "0":
                    continue
                value = np.deg2rad(float(elem[index + 9]))
                sd    = np.deg2rad(float(elem[index + 13]))
                chis.append((value, sd))
            
            weight  = float(elem[8])
            rotamer = Rotamer(name, chis)

            # Place this Rotamer in the correct bin of the correct RotamerLib
            # (based on the name of the aminoacid, phi and psi values)
            rot_lib   = matrices[name]
            phi       = np.deg2rad(float(elem[1]))
            psi       = np.deg2rad(float(elem[2]))
            rot_stack = rot_lib.get_rotamer_stack(phi, psi)

            rot_stack.weights.append(weight)
            rot_stack.rotamers.append(rotamer)

    return matrices

if __name__ == "__main__":

    # Ex 1)
    # Read and load the Dunbrack rotamer library in such a way that any of the
    # rotamers can be applied to a Molecule (changing the dihedral angles of all
    # chi angles).

    # 1.1) For this solution we will read the file twice: the first to create
    # all empty matrices, and the second to fill them.
    rot_lib = create_rot_lib_matrix("dunbrack_rotamers.lib")
    rot_lib = fill_rot_lib_matrix("dunbrack_rotamers.lib", rot_lib)
    print("Rotamer library loaded.")

    # Ex 2)
    # Sample a rotamer for aminoacid “ARG”, based on the measured phi and psi
    # angles of aminoacid 2 on “mol1.pdb” structure.

    # 2.1) Load "mol3.pdb" and measure the phi and psi angles at aminoacid 2.
    mol = Molecule()
    mol.load("mol3.pdb")

    def measure_dihedral(atom):
        return atom.dihedral + atom.parent.d_dihedral

    aminoacid2 = list(filter(lambda at: at.res_id == 2, mol.atoms))

    # Third atom of phi angle is a CA.
    CA = list(filter(lambda at: at.name == "CA", aminoacid2))[0]
    phi = measure_dihedral(CA)

    # Third atom of psi angle is a C.
    C = list(filter(lambda at: at.name == "C", aminoacid2))[0]
    psi = measure_dihedral(C)

    # 2.2) Get the RotamerStack for that specific pair of phi and psi values.
    rotamer_stack = rot_lib[aminoacid2[0].res_name].get_rotamer_stack(phi, psi)

    # 2.3) Sample a Rotamer from the RotamerStack, based on the respective
    # weights (normalized so that sum(weights) = 1.0)
    norm_weights = np.array(rotamer_stack.weights) / sum(rotamer_stack.weights)
    rotamer = np.random.choice(rotamer_stack.rotamers, 1, p = norm_weights)[0]

    # Ex 3)
    # Apply a sampled rotamer (from ex.2) to the aminoacid 2 on “mol1.pdb”.
    # Export the changed structure to “mol2.pdb” and observe the changes on
    # PyMOL.
    rotamer.apply_to_aminoacid(aminoacid2)
    mol.i2c()
    mol.export("mol4.pdb")

    # Ex 4)
    # For all aminoacids of “mol1.pdb” structure, sample a new rotamer based on
    # the measured phi and psi angles (taking the natural occurrence probability
    # into account) and apply them to the corresponding aminoacid. Export the
    # changed structure to “mol2.pdb”. Perform this loop 10 times, appending a
    # new frame to the “mol2.pdb” trajectory. Observe the results on PyMOL.
    n_aminoacids = len(set([at.res_id for at in mol.atoms]))
    for step in range(1, 11):
        print("Step %3d" % step)
        for res in range(1, n_aminoacids + 1):

            # 4.1) Get list of atoms and name of aminoacid N.
            aminoacid_atoms = list(filter(lambda at: at.res_id == res, mol.atoms))
            name = aminoacid_atoms[0].res_name
            if not name in rot_lib.keys() or name == "PRO":
                continue

            # 4.2) Get the RotamerStack for the current phi and psi value. 
            CA = list(filter(lambda at: at.name == "CA", aminoacid_atoms))[0]
            phi = measure_dihedral(CA)
            C = list(filter(lambda at: at.name == "C", aminoacid_atoms))[0]
            psi = measure_dihedral(C)
            rotamer_stack = rot_lib[name].get_rotamer_stack(phi, psi)

            # 4.3) Sample Rotamer from RotamerStack based on normalized weights
            norm_weights = np.array(rotamer_stack.weights) / sum(rotamer_stack.weights)
            rotamer = np.random.choice(rotamer_stack.rotamers, 1, p = norm_weights)[0]

            # 4.4) Apply the rotamer
            rotamer.apply_to_aminoacid(aminoacid_atoms)
            mol.i2c()
        mol.export("mol4.pdb", "a", model = step + 1)