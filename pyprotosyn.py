# Built and tested on Python 3.8.8
# Developed by Zé
import numpy as np

# ------------------------------------------------------------------------------
# The following functions and classes are a starting point for your work on the
# subject of LVCC. these are provided as is, and can (and should) be improved
# and built on. Some bugs and constraints are to be expected (and fixed!).
# Good luck. For any questions please contact: jose.manuel.pereira@ua.pt
# ------------------------------------------------------------------------------

# --- Auxiliary functions

def flatten(l, ltypes=(list, tuple)):
    """
    Return the given list/tuple of lists/tuples flattened to a 1D tuple/list.

    # Example
    In [1]: a = (1, (2, (3, (4))))

    In [2]: flatten(a)
    Out[2]: (1, 2, 3, 4)
    """

    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def ceil(value, precision=0):
    """
    Return the given value rounded up, to the given precision (number of decimals).

    # Example
    In [1]: ceil(1.56756237862354238582353, precision = 5)
    Out[1]: 1.56757
    """

    return np.true_divide(np.ceil(value * 10**precision), 10**precision)


# --- Atom class

class Atom(object):
    """
    An Atom contains all information pertaining to a single atom in the molecule.
    This includes the name, symbol (chemical element), the cartesian and internal
    coordinates as well as the parenthood relationships with other atoms of the
    molecule graph.
    """

    def __init__(self, name = "X", symbol = "X", index = -1, res_name = "UNK",
        res_id = -1, chain_id = "A", cartesian = np.zeros(3)):
        
        # Properties
        self.name     = name
        self.index    = index
        self.symbol   = symbol
        self.res_name = res_name
        self.res_id   = res_id
        self.chain_id = chain_id

        # Cartesian coordinates
        self.cartesian = cartesian

        # Internal coordinates
        self.b          = 0.0 # self <-> parent (bond length, angstrom)
        self.angle      = 0.0 # self <-> parent <-> grandparent (angle, radians)
        self.dihedral   = 0.0 # self <-> parent <-> grandparent <-> grand-grandparent (dihedral, radians)
        self.d_dihedral = 0.0 # dihedral rotation to be applied downstream (radians)
        self.r          = np.eye(3) # Local to global rotation matrix

        # Check current status of AtomState
        self.internal_2_cartesian = False
        if not np.all(cartesian == 0.0):
            self.cartesian_2_internal = True
        else:
            self.cartesian_2_internal = False

        # Graph parenthood
        self.parent     = None
        self.children   = []
        self.bonds      = [] # For visualization purposes only (i.e. cyclical aminoacids)


    def __setattr__(self, name, value):
        if name == "cartesian":
            self.cartesian_2_internal = True
        elif name in ["b", "angle", "dihedral", "d_dihedral"]:
            self.internal_2_cartesian = True
        return super().__setattr__(name, value)

    def __str__(self):
        s = f''
        s += f'<Atom: {self.name} | {self.symbol} | {self.index}\n'
        s += f' Cartesian: {self.cartesian}\n'
        s += f' Internal: b={self.b} | angle={self.angle} | dihedral={self.dihedral} | d_dihedral={self.d_dihedral}\n'
        s += f' i2c: {self.internal_2_cartesian}, c2i: {self.cartesian_2_internal})>\n---\n'
        return s

    def __repr__(self):
        return str(self)

    def ascendents(self, n = 3):
        """
        Return a tuple with all the ascendents up to N parents/grandparents.
        """

        if self.parent == None:
            raise AssertionError("Tried to return atom parent but no parent was found.")
        if n == 1:
            return (self.parent)
        res = (self.parent, self.parent.ascendents(n = n - 1))
        return flatten(res)

    def setparent(self, parent):
        """
        Set the parent of self to be the given atom, while automatically setting
        the parent.children and both atom bond records correctly.
        """

        if not self.parent == None:
            raise AssertionError("Tried to set parent of a non-orphan atom.")
        self.parent = parent
        parent.children.append(self)
        self.bond(parent)

    def bond(self, atom_2):
        if self not in atom_2.bonds:
            atom_2.bonds.append(self)
        if atom_2 not in self.bonds:
            self.bonds.append(atom_2)

    def i2c(self):
        """
        Calculate and update the cartesian coordinates of this atom (and all
        subsequent children atoms) based on the current internal coordinates.
        """
        
        if not self.internal_2_cartesian:
            return

        (j, k, l) = self.ascendents()

        # Calculate this atom new cartesian coordinates
        Ri = self.r
        Rj = j.r

        b = self.b
        s_angle, c_angle = np.sin(self.angle), np.cos(self.angle)
        d = self.dihedral + j.d_dihedral
        s_dihedral, c_dihedral = np.sin(d), np.cos(d)
        x_1 = -b * c_angle
        x_2 =  b * c_dihedral * s_angle
        x_3 =  b * s_dihedral * s_angle

        vji = np.zeros(3)
        for u in range(3):
            vji[u] = Rj[u, 0] * x_1 + Rj[u, 1] * x_2 + Rj[u, 2] * x_3

        vjk = k.cartesian - j.cartesian
        for u in range(3):
            Ri[u, 0] = vji[u]/b

        n = np.cross(vji, vjk)
        dn = np.sqrt(np.dot(n, n))

        for u in range(3):
            Ri[u, 2] = n[u]/dn

        Ri[:, 1] = np.cross(Ri[:, 2], Ri[:, 0])

        # Apply new cartesian coordinates
        self.cartesian = vji + j.cartesian

        # Update children position
        for child in self.children:
            child.internal_2_cartesian = True
            child.i2c()

        self.internal_2_cartesian = False
        
    def c2i(self):
        """
        Calculate and update the internal coordinates of this atom (and all
        subsequent children atoms) based on the current cartesian coordinates.
        """

        if not self.cartesian_2_internal:
            return

        (j, k, l) = self.ascendents()

        # Calculate this atom new internal coordinates
        self.b          = self.measure_distance(j)
        self.angle      = self.measure_angle(j, k)
        self.dihedral   = self.measure_dihedral(j, k, l)
        self.d_dihedral = 0.0

        # Update children atoms
        for child in self.children:
            child.cartesian_2_internal = True
            child.c2i()

        self.cartesian_2_internal = False

    def measure_distance(self, atom_2):
        """
        Measure the distance (in Angstrom) between this and the provided atom 2.
        """

        return np.linalg.norm(self.cartesian - atom_2.cartesian)

    def measure_angle(self, atom_2, atom_3):
        """
        Measure the angle (in radians) between this and the provided atom 2 and
        atom 3.
        """

        v21 = self.cartesian - atom_2.cartesian
        v23 = atom_3.cartesian - atom_2.cartesian
        a = np.dot(v21, v23) / (np.linalg.norm(v21) * np.linalg.norm(v23))
        return np.arccos(ceil(a, precision = 15)) # ceil prevents errors at 180 degree angles

    def measure_dihedral(self, atom_2, atom_3, atom_4):
        """
        Measure the dihedral angle (in radians) between this and the provided
        atom 2, 3 and 4.
        """

        b1 = atom_2.cartesian - self.cartesian
        b2 = atom_3.cartesian - atom_2.cartesian
        b3 = atom_4.cartesian - atom_3.cartesian
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        x = np.dot(np.cross(n1, n2), b2) / np.sqrt(np.dot(b2, b2))
        y = np.dot(n1, n2)
        return np.arctan2(x, y)


# --- Molecule class

class Molecule:
    """
    A Molecule contains all the information relative to a set of atoms. It
    includes a `root` (3 virtual atoms that act as an anchor for all the
    subsequent atoms of the molecule, in order to calculate initial internal
    coordinates).
    """

    def __init__(self):
        self.root = self.create_root()
        self.atoms = []

    def __str__(self):
        s = f''
        for atom in self.atoms:
            s += str(atom)
        return s

    def __repr__(self):
        return str(self)

    def create_root(self):
        """
        Return a set of 3 virtual atoms.
        """

        a0 = Atom(name = "R0", cartesian = np.array([ 0.0, 0.0, 0.0]))
        a1 = Atom(name = "R1", cartesian = np.array([-1.0, 0.0, 0.0]))
        a2 = Atom(name = "R2", cartesian = np.array([-1.0, 1.0, 0.0]))
        a1.setparent(a2)
        a0.setparent(a1)
        return [a2, a1, a0]

    def reindex(self):
        """
        Loop over all atoms of the molecule and reindex, starting at index 1.
        """
        for (index, atom) in enumerate(self.atoms, start = 1):
            atom.index = index

    def i2c(self):
        """
        Force call i2c() on all children of the molecule root.
        """

        for child in self.root[-1].children:
            child.internal_2_cartesian = True
            child.i2c()

    def c2i(self):
        """
        Force call c2i() on all children of the molecule root.
        """

        for child in self.root[-1].children:
            child.cartesian_2_internal = True
            child.c2i()

    def load(self, filename):
        """
        Assumes that atoms in the provided filename are ordered is ascedent
        order based on index.
        """

        if not filename[-3:] == "pdb":
            raise AssertionError("File format ." + filename[-3:] + " not recognised. Please use .pdb files.")

        self.atoms = []
        with open(filename, "r") as file_input:
            for line in file_input:
                if line.startswith("ATOM"):
                    elem = line.split()
                    atom = Atom(elem[2], elem[9], int(elem[1]), elem[3],
                        int(elem[5]), elem[4],
                        np.array([float(elem[6]), float(elem[7]), float(elem[8])]))
                    self.atoms.append(atom)
                    if len(self.atoms) == 1:
                        self.atoms[0].setparent(self.root[-1])

                elif line.startswith("CONECT"):
                    elem   = line.split()
                    parent = self.atoms[int(elem[1]) - 1]
                    for atom_id in elem[2:]:
                        self.atoms[int(atom_id) - 1].bond(parent)
                        try:
                            self.atoms[int(atom_id) - 1].setparent(parent)
                        except AssertionError:
                            continue
        self.c2i()

    def export(self, filename, mode = "w", model = 1):
        if not filename[-3:] == "pdb":
            raise AssertionError("File format ." + filename[-3:] + " not recognised. Please use .pdb files.")

        PDB = "ATOM  %5d %4s %3s %1s %3d     %7.3f %7.3f %7.3f %23s\n"

        with open(filename, mode) as file_out:
            file_out.write("MODEL %8s\n" % (model))
            for atom in self.atoms:
                file_out.write(PDB % (atom.index, atom.name, atom.res_name,
                    atom.chain_id, atom.res_id, atom.cartesian[0],
                    atom.cartesian[1], atom.cartesian[2], atom.symbol))

            for atom in self.atoms:
                conect_record = "CONECT %4d" % (atom.index)
                for bonded_atom in atom.bonds:
                    conect_record += " %4d" % (bonded_atom.index)
                file_out.write(conect_record+"\n")

            file_out.write("ENDMDL\n")