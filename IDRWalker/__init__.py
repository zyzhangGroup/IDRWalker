import os

import random
import numpy as np
from copy import deepcopy
from kearsley import Kearsley

from tqdm import tqdm

bonds = np.array((1.34, 1.45, 1.52))
angles = np.array((116.56, 121.87, 111.08))

data_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_psi.txt')
phi_psi = np.loadtxt(data_file_path)
N_phi_psi = len(phi_psi)
phi_psi_p = np.zeros(N_phi_psi)
for i in range(N_phi_psi):
    phi, psi = phi_psi[i]
    phi_psi_p[i] = (phi*phi+psi*psi)**0.5
phi_psi_p /= np.sum(phi_psi_p)

def cal_r_n_v(atom1, atom2, atom3):
    v1 = atom1 - atom2
    v2 = atom3 - atom2
    n = np.cross(v1, v2)
    n /= np.linalg.norm(n)
    v = v2
    v /= np.linalg.norm(v)
    r = atom3
    return r, n, v

def Rotate_Matrix_around_axis(a, b, c, theta):
    theta = np.deg2rad(theta)
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array((
        ((b*b+c*c)*cos+a*a, a*b*(1-cos)-c*sin, a*c*(1-cos)+b*sin),
        (a*b*(1-cos)+c*sin, b*b+(1-b*b)*cos, b*c*(1-cos)-a*sin),
        (a*c*(1-cos)-b*sin, b*c*(1-cos)+a*sin, c*c+(1-c*c)*cos)
    ))

def atom_generator(r0, n1, v1, bond, angle, dihedral):
    n2 = Rotate_Matrix_around_axis(*v1, dihedral) @ n1
    v2 = Rotate_Matrix_around_axis(*n2, angle) @ (-v1)
    return r0 + v2*bond, n2, v2

def main_chain_generator(r, n, v, bonds, angles, phi, psi, omega=180.0):
    main_chain = np.zeros((3, 3))
    r, n, v = atom_generator(r, n, v, bonds[0], angles[0], psi)
    main_chain[0] = r
    r, n, v = atom_generator(r, n, v, bonds[1], angles[1], omega)
    main_chain[1] = r
    r, n, v = atom_generator(r, n, v, bonds[2], angles[2], phi)
    main_chain[2] = r
    return main_chain, n, v

def main_chain_generator_reverse(r, n, v, bonds, angles, phi, psi, omega=180.0):
    main_chain = np.zeros((3, 3))
    r, n, v = atom_generator(r, n, v, bonds[0], angles[1], phi)
    main_chain[2] = r
    r, n, v = atom_generator(r, n, v, bonds[2], angles[0], omega)
    main_chain[1] = r
    r, n, v = atom_generator(r, n, v, bonds[1], angles[2], psi)
    main_chain[0] = r
    return main_chain, n, v

def xyz2grid(xyz, l=1.2, edge=None):
    if not edge is None:
        return (xyz//l).astype(int)%edge
    return (xyz//l).astype(int)

C_N_distance_limit =  np.array((
    1.30, 1.38,
    2.70, 4.80,
    4.40, 7.80,
    5.45, 10.60,
    6.10, 13.15,
    6.30, 15.40,
    6.49, 17.74,
    6.27, 20.00,
    6.26, 21.88,
    6.58, 23.63,
    6.78, 25.41
)).reshape(-1, 2)

def C_N_distance_limit_check(d, C_ter_id, N_ter_id):
    N = N_ter_id - C_ter_id - 2
    if N > 10:
        if d * d > 65 * N + 1.34*1.34:
            return False
    elif d < C_N_distance_limit[N, 0] or d > C_N_distance_limit[N, 1]:
        return False
    return True

kearsley = Kearsley()

class Residue:
    
    def __init__(self, resname, resid, atom_name_list, coordinates):
        if len(atom_name_list) != len(coordinates):
            raise ValueError
        self.resname = resname
        self.resid = resid
        self.names = atom_name_list
        self.coordinates = coordinates
    
    def __repr__(self):
        return f"Residue {self.resid} {self.resname} with {len(self.names)} atoms."
    
    #def main_chain_coordinates(self):
    #    return self.coordinates.take((0, 1, -2), axis=0)

AA_templates = {}
ALA_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['A'] = Residue('ALA', 0, ['N', 'CA', 'CB', 'C', 'O'] , ALA_coordinates)
ARG_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    4.274, 5.01, 1.195, 
    3.881, 5.818, 2.427, 
    4.54, 7.143, 2.424, 
    4.364, 8.041, 3.389, 
    3.575, 7.808, 4.434, 
    5.006, 9.201, 3.287, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['R'] = Residue('ARG', 0, ['N', 'CA', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', 'C', 'O'] , ARG_coordinates)
ASN_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    4.254, 5.017, 1.232, 
    5.005, 5.34, 0.315, 
    3.985, 5.818, 2.266, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['N'] = Residue('ASN', 0, ['N', 'CA', 'CB', 'CG', 'OD1', 'ND2', 'C', 'O'] , ASN_coordinates)
ASP_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    4.275, 5.011, 1.195, 
    3.669, 5.955, 0.62, 
    5.408, 5.092, 1.741, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['D'] = Residue('ASP', 0, ['N', 'CA', 'CB', 'CG', 'OD1', 'OD2', 'C', 'O'] , ASP_coordinates)
CYS_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    4.31, 5.304, 1.366, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['C'] = Residue('CYS', 0, ['N', 'CA', 'CB', 'SG', 'C', 'O'] , CYS_coordinates)
GLN_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    4.274, 5.01, 1.195, 
    3.907, 5.848, 2.41, 
    3.139, 5.408, 3.263, 
    4.459, 7.062, 2.488, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['Q'] = Residue('GLN', 0, ['N', 'CA', 'CB', 'CG', 'CD', 'OE1', 'NE2', 'C', 'O'] , GLN_coordinates)
GLU_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    4.267, 4.996, 1.195, 
    3.874, 5.805, 2.429, 
    4.595, 5.679, 3.454, 
    2.856, 6.542, 2.334, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['E'] = Residue('GLU', 0, ['N', 'CA', 'CB', 'CG', 'CD', 'OE1', 'OE2', 'C', 'O'] , GLU_coordinates)
GLY_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    5.484, 2.687, -0.0, 
    5.993, 1.568, -0.0, 
)).reshape(-1, 3)
AA_templates['G'] = Residue('GLY', 0, ['N', 'CA', 'C', 'O'] , GLY_coordinates)
HIS_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    4.201, 5.026, 1.321, 
    3.943, 5.885, 2.383, 
    4.624, 6.998, 2.183, 
    5.294, 6.891, 1.062, 
    5.059, 5.679, 0.492, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['H'] = Residue('HIS', 0, ['N', 'CA', 'CB', 'CG', 'ND1', 'CE1', 'NE2', 'CD2', 'C', 'O'] , HIS_coordinates)
ILE_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.552, 3.621, 1.245, 
    3.97, 2.846, 2.49, 
    4.23, 4.987, 1.245, 
    3.812, 5.762, 2.49, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['I'] = Residue('ILE', 0, ['N', 'CA', 'CB', 'CG2', 'CG1', 'CD1', 'C', 'O'] , ILE_coordinates)
LEU_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    4.274, 5.01, 1.195, 
    3.853, 5.763, -0.063, 
    3.881, 5.818, 2.427, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['L'] = Residue('LEU', 0, ['N', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'C', 'O'] , LEU_coordinates)
LYS_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    4.274, 5.01, 1.195, 
    3.881, 5.818, 2.427, 
    4.578, 7.173, 2.389, 
    4.199, 7.952, 3.577, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['K'] = Residue('LYS', 0, ['N', 'CA', 'CB', 'CG', 'CD', 'CE', 'NZ', 'C', 'O'] , LYS_coordinates)
MET_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    4.274, 5.01, 1.195, 
    3.817, 5.981, 2.652, 
    4.753, 7.463, 2.341, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['M'] = Residue('MET', 0, ['N', 'CA', 'CB', 'CG', 'SD', 'CE', 'C', 'O'] , MET_coordinates)
PHE_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    4.201, 5.026, 1.321, 
    3.912, 5.857, 2.41, 
    4.49, 7.13, 2.492, 
    5.358, 7.571, 1.486, 
    5.647, 6.739, 0.397, 
    5.068, 5.467, 0.315, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['F'] = Residue('PHE', 0, ['N', 'CA', 'CB', 'CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2', 'C', 'O'] , PHE_coordinates)
PRO_coordinates = np.array((
    3.327, 1.557, -0.0, 
    3.934, 2.871, -0.105, 
    4.302, 0.477, 0.08, 
    5.547, 1.172, 0.545, 
    5.369, 2.628, 0.185, 
    3.505, 3.526, -1.41, 
    2.754, 2.939, -2.185, 
)).reshape(-1, 3)
AA_templates['P'] = Residue('PRO', 0, ['N', 'CA', 'CD', 'CG', 'CB', 'C', 'O'] , PRO_coordinates)
SER_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    4.231, 4.925, 1.197, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['S'] = Residue('SER', 0, ['N', 'CA', 'CB', 'OG', 'C', 'O'] , SER_coordinates)
THR_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    2.066, 3.859, 1.244, 
    3.972, 2.947, 2.411, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['T'] = Residue('THR', 0, ['N', 'CA', 'CB', 'CG2', 'OG1', 'C', 'O'] , THR_coordinates)
TRP_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    4.201, 5.026, 1.321, 
    4.023, 5.931, 2.293, 
    4.812, 7.074, 1.95, 
    5.427, 6.842, 0.817, 
    6.297, 7.689, 0.12, 
    6.814, 7.187, -1.069, 
    6.483, 5.953, -1.505, 
    5.604, 5.117, -0.786, 
    5.083, 5.623, 0.412, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['W'] = Residue('TRP', 0, ['N', 'CA', 'CB', 'CG', 'CD1', 'NE1', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3', 'CD2', 'C', 'O'] , TRP_coordinates)
TYR_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    4.267, 4.996, 1.195, 
    4.06, 5.919, 2.227, 
    4.7, 7.164, 2.193, 
    5.547, 7.486, 1.126, 
    6.169, 8.695, 1.092, 
    5.755, 6.563, 0.094, 
    5.115, 5.318, 0.128, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['Y'] = Residue('TYR', 0, ['N', 'CA', 'CB', 'CG', 'CD1', 'CE1', 'CZ', 'OH', 'CE2', 'CD2', 'C', 'O'] , TYR_coordinates)
VAL_coordinates = np.array((
    3.326, 1.548, -0.0, 
    3.97, 2.846, -0.0, 
    3.577, 3.654, 1.232, 
    3.998, 2.9, 2.49, 
    4.274, 5.01, 1.195, 
    5.486, 2.705, -0.0, 
    6.009, 1.593, -0.0, 
)).reshape(-1, 3)
AA_templates['V'] = Residue('VAL', 0, ['N', 'CA', 'CB', 'CG1', 'CG2', 'C', 'O'] , VAL_coordinates)

class Gap:
    
    def __init__(self, N_ter_id=0, C_ter_id=0, N_r_n_v=None, C_r_n_v=None):
        self.N_ter_id = N_ter_id
        self.C_ter_id = C_ter_id
        self.N_r_n_v = N_r_n_v
        self.C_r_n_v = C_r_n_v
    
    def __repr__(self):
        if self.N_ter_id == 0:
            return f"C-terminal at Residue {self.C_ter_id}"
        if self.C_ter_id == 0:
            return f"N-terminal at Residue {self.N_ter_id}"
        return f"Gap between Residue {self.C_ter_id} and {self.N_ter_id}"
    
    def check(self, start, end):
        if self.N_ter_id == 0 and self.C_ter_id == end:
            return False
        if self.C_ter_id == 0 and self.N_ter_id == start:
            return False
        if self.N_ter_id - self.C_ter_id == 1:
            return False
        return True

class Chain:
    
    def __init__(self, seq, PDB, chainID, tail=True):
        self.chainID = chainID
        if seq[-6:] == ".fasta":
            with open(seq) as fasta:
                fasta.readline()
                self.seq = fasta.readline()
                if self.seq[-1] == '\n':
                    self.seq = self.seq[:-1]
        else:
            self.seq = seq
        self.start = 1
        self.end = len(self.seq)
        self.residue_list = []
        for i, s in enumerate(self.seq):
            self.residue_list.append(deepcopy(AA_templates[s]))
            self.residue_list[-1].resid = i + 1
        resids= []
        with open(PDB, 'r') as pdb:
            while (line := pdb.readline()):
                if line[:4] != "ATOM":
                    continue
                name = line[12:16].strip()
                resid = int(line[22:26])
                resids.append(resid)
                if not name in self.residue_list[resid-1].names:
                    continue
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                self.residue_list[resid-1].coordinates[self.residue_list[resid-1].names.index(name)] = np.array((x, y, z))
        resids = np.unique(resids)
        #self.N_ter_id = resids[0]
        #self.C_ter_id = resids[-1]
        self.gap_list = []
        #N_r_n_v = cal_r_n_v(*self.residue_list[resids[0]-1].coordinates.take((0, 1, -2), axis=0)[::-1])
        #self.gap_list.append(Gap(N_ter_id=resids[0], N_r_n_v=N_r_n_v))
        self.gap_list.append(Gap(N_ter_id=resids[0]))
        for i in range(len(resids)-1):
            if resids[i+1] - resids[i] != 1:
                #C_r_n_v = cal_r_n_v(*self.residue_list[resids[i]-1].coordinates.take((0, 1, -2), axis=0))
                #N_r_n_v = cal_r_n_v(*self.residue_list[resids[i+1]-1].coordinates.take((0, 1, -2), axis=0)[::-1])
                #self.gap_list.append(Gap(N_ter_id=resids[i+1], C_ter_id=resids[i], N_r_n_v=N_r_n_v, C_r_n_v=C_r_n_v))
                self.gap_list.append(Gap(N_ter_id=resids[i+1], C_ter_id=resids[i]))
        #C_r_n_v = cal_r_n_v(*self.residue_list[resids[-1]-1].coordinates.take((0, 1, -2), axis=0))
        #self.gap_list.append(Gap(C_ter_id=resids[-1], C_r_n_v=C_r_n_v))
        self.gap_list.append(Gap(C_ter_id=resids[-1]))
        if tail == False:
            self.start = resids[0]
            self.end = resids[-1]
    
    def __repr__(self):
        return f"Chain {self.chainID} with {self.end-self.start+1} residues"
        
    def C_forward(self, occ_matrix, xyz2grid, gap):
        for attempt in range(1000):
            phi, psi =  phi_psi[np.random.choice(N_phi_psi, p=phi_psi_p)]
            #if (phi*phi/180/180 + psi*psi/180/180)/2 < random.random():
            #    continue
            main_chain, n, v = main_chain_generator(*gap.C_r_n_v, bonds, angles, phi, psi)
            if gap.N_ter_id != 0:
                d = np.linalg.norm(main_chain[-1] - self.residue_list[gap.N_ter_id-1].coordinates[0])
                if not C_N_distance_limit_check(d, gap.C_ter_id, gap.N_ter_id):
                    continue
            kearsley.fit(main_chain, AA_templates[self.seq[gap.C_ter_id]].coordinates.take((0, 1, -2), axis=0))
            New_coordinates = kearsley.transform(AA_templates[self.seq[gap.C_ter_id]].coordinates[:-1])
            ijk_list = [xyz2grid(xyz) for xyz in New_coordinates[1:-1]]
            for i, j, k in ijk_list:
                if occ_matrix[i, j, k] == True:
                    break
            else:
                for i, j, k in ijk_list:
                    occ_matrix[i, j, k] = True
                gap.C_ter_id += 1
                self.residue_list[gap.C_ter_id-1].coordinates[:-1] = New_coordinates
                n /= np.linalg.norm(n)
                v /= np.linalg.norm(v)
                gap.C_r_n_v = np.array((main_chain[-1], n, v))
                #print('C_forward %s %d'%(self.chainID,gap.C_ter_id))
                return True
        return False
    
    def C_backward(self, occ_matrix, xyz2grid, gap):
        ijk_list = [xyz2grid(xyz) for xyz in self.residue_list[gap.C_ter_id-1].coordinates[1:-2]]
        for i, j, k in ijk_list:
            occ_matrix[i, j, k] = False
        gap.C_ter_id -= 1
        gap.C_r_n_v = cal_r_n_v(*self.residue_list[gap.C_ter_id-1].coordinates.take((0, 1, -2), axis=0))
        #print('C_backward %s %d'%(self.chainID,gap.C_ter_id))
        
    def N_forward(self, occ_matrix, xyz2grid, gap):
        for attempt in range(1000):
            phi, psi =  phi_psi[np.random.choice(N_phi_psi, p=phi_psi_p)]
            #if (phi*phi/180/180 + psi*psi/180/180)/2 < random.random():
            #    continue
            main_chain, n, v = main_chain_generator_reverse(*gap.N_r_n_v, bonds, angles, phi, psi)
            if gap.C_ter_id != 0:
                d = np.linalg.norm(main_chain[0] - self.residue_list[gap.C_ter_id-1].coordinates[-2])
                if not C_N_distance_limit_check(d, gap.C_ter_id, gap.N_ter_id):
                    continue
            kearsley.fit(main_chain, AA_templates[self.seq[gap.N_ter_id-2]].coordinates.take((0, 1, -2), axis=0))
            New_coordinates = kearsley.transform(AA_templates[self.seq[gap.N_ter_id-2]].coordinates[:-1])
            ijk_list = [xyz2grid(xyz) for xyz in New_coordinates[1:-1]]
            for i, j, k in ijk_list:
                if occ_matrix[i, j, k] == True:
                    break
            else:
                for i, j, k in ijk_list:
                    occ_matrix[i, j, k] = True
                gap.N_ter_id -= 1
                self.residue_list[gap.N_ter_id-1].coordinates[:-1] = New_coordinates
                n /= np.linalg.norm(n)
                v /= np.linalg.norm(v)
                gap.N_r_n_v = np.array((main_chain[0], n, v))
                #print('N_forward %s %d'%(self.chainID,gap.N_ter_id))
                return True
        return False
    
    def N_backward(self, occ_matrix, xyz2grid, gap):
        ijk_list = [xyz2grid(xyz) for xyz in self.residue_list[gap.N_ter_id-1].coordinates[1:-2]]
        for i, j, k in ijk_list:
            occ_matrix[i, j, k] = False
        gap.N_ter_id += 1
        gap.N_r_n_v = cal_r_n_v(*self.residue_list[gap.N_ter_id-1].coordinates.take((0, 1, -2), axis=0)[::-1])
        #print('N_backward %s %d'%(self.chainID,gap.N_ter_id))
        
    def run(self, occ_matrix, xyz2grid):
        if len(self.gap_list) == 0:
            return True
        gap = random.choice(self.gap_list)
        if not gap.check(self.start, self.end):
            self.gap_list.remove(gap)
            return False
        if gap.C_r_n_v is None:
            if not self.N_forward(occ_matrix, xyz2grid, gap):
                self.N_backward(occ_matrix, xyz2grid, gap)
        elif gap.N_r_n_v is None:
             if not self.C_forward(occ_matrix, xyz2grid, gap):
                self.C_backward(occ_matrix, xyz2grid, gap)
        elif random.randint(0, 1):
            if not self.C_forward(occ_matrix, xyz2grid, gap):
                self.C_backward(occ_matrix, xyz2grid, gap)
        else:
            if not self.N_forward(occ_matrix, xyz2grid, gap):
                self.N_backward(occ_matrix, xyz2grid, gap)
        return False
    
    def write(self, file_name):
        if file_name[-4:] != ".pdb":
            file_name += ".pdb"
        ix = 1
        with open(file_name, 'w') as pdb:
            for residue in self.residue_list:
                if residue.resid < self.start or residue.resid > self.end:
                    continue
                for i in range(len(residue.names)):
                    line = f"ATOM  {ix%100000:5d} {residue.names[i]:^4s} {residue.resname:s} {self.chainID[:1]}{residue.resid:4d}    {residue.coordinates[i, 0]:8.3f}{residue.coordinates[i, 1]:8.3f}{residue.coordinates[i, 2]:8.3f}\n"
                    pdb.write(line)
                    ix += 1
            pdb.write('TER\n')
            
class Box:
    
    def __init__(self, box_size, grid_size):
        self.size = np.array(box_size)
        self.grid_size = grid_size
        self.shape = (self.size//self.grid_size).astype(int)
        self.occ_matrix = np.zeros(self.shape, dtype=bool)
        self.task_list = []
        self.done_list = []
        
    def xyz2grid(self, xyz):
        return (xyz//self.grid_size).astype(int)%self.shape
        
    def add_chain(self, chain):
        self.task_list.append(chain)
        C_ter_id = []
        N_ter_id = []
        for gap in chain.gap_list:
            C_ter_id.append(gap.C_ter_id)
            N_ter_id.append(gap.N_ter_id)
        for N, C in zip(N_ter_id[:-1], C_ter_id[1:]):
            for ri in range(N-1, C):
                for xyz in chain.residue_list[ri].coordinates:
                    i, j, k = self.xyz2grid(xyz)
                    self.occ_matrix[i, j, k] = True
        
    def task_done(self, chain):
        self.done_list.append(chain)
        self.task_list.remove(chain)
        for i in range(chain.start-1, chain.end-1):
            CA, C = chain.residue_list[i].coordinates.take((1, -2), axis=0)
            N = chain.residue_list[i+1].coordinates[0]
            O = -0.757*CA - 0.886*N + 2.643*C
            chain.residue_list[i].coordinates[-1] = O
        N, CA, C = chain.residue_list[chain.end-1].coordinates.take((0, 1, -2), axis=0)
        O = 0.783*N -1.461*CA + 1.678*C
        chain.residue_list[chain.end-1].coordinates[-1] = O

        
    def run(self):
        for chain_i in self.task_list:
            for gap_i in chain_i.gap_list:
                if gap_i.N_ter_id != 0:
                    gap_i.N_r_n_v = cal_r_n_v(*chain_i.residue_list[gap_i.N_ter_id-1].coordinates.take((0, 1, -2), axis=0)[::-1])
                if gap_i.C_ter_id != 0:
                    gap_i.C_r_n_v = cal_r_n_v(*chain_i.residue_list[gap_i.C_ter_id-1].coordinates.take((0, 1, -2), axis=0))
        pbar = tqdm(total=len(self.task_list))
        pbar.set_description('Processing')
        while self.task_list:
            chain_i = random.choice(self.task_list)
            if chain_i.run(self.occ_matrix, self.xyz2grid):
                self.task_done(chain_i)
                pbar.update(1)
        pbar.close()