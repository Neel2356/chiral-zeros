import numpy as np
import os
from biopandas.pdb import PandasPdb
import pandas as pd
import numpy as np
import os
from biopandas.pdb import PandasPdb
import pandas as pd
import random
import itertools
import copy
import scipy
import math
import sys
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import directed_hausdorff, pdist, squareform, cdist
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import directed_hausdorff, cdist
from scipy.optimize import minimize
from itertools import product
from joblib import Parallel, delayed
import multiprocessing
import tqdm

def CA_coord(pdb_name, chain1, chain2):

    coord1 = PandasPdb()
    coord1.fetch_pdb(pdb_name)
    # coord1.read_pdb(os.path.join(protein_dir, pdb_name + '.pdb'))
    prot1_df = coord1.df['ATOM']
    prot1_df = prot1_df[(prot1_df['alt_loc'] == "") | (prot1_df['alt_loc'] == "A")]

    c1_ = [[] for _ in range(len(chain1))]
    for ii in range(len(chain1)):
        c1_[ii] = prot1_df[prot1_df['chain_id'] == chain1[ii]]
    c1 = pd.concat(c1_).reset_index(drop=True)

    c1_all_res = c1[['chain_id', 'residue_number', 'insertion']].drop_duplicates().reset_index(drop=True)
    c1_ca_res = c1[c1['atom_name'] == 'CA'][['chain_id', 'residue_number', 'insertion']]
    c1_no_cas = pd.merge(c1_all_res, c1_ca_res, how='left', indicator=True)['_merge']=='left_only'

    if sum(c1_no_cas) != 0:
        c1_incomplete_res = np.squeeze(np.where(c1[['chain_id', 'residue_number', 'insertion']].astype(str).agg('_'.join, axis=1).to_numpy() ==
               				        c1_all_res[c1_no_cas].astype(str).agg('_'.join, axis=1).to_numpy()))
        c1 = c1.drop(np.atleast_1d((c1_incomplete_res))).reset_index(drop=True)
    else:
        c1_incomplete_res = np.array([])

    c2_ = [[] for _ in range(len(chain2))]
    for ii in range(len(chain2)):
        c2_[ii] = prot1_df[prot1_df['chain_id'] == chain2[ii]]
    c2 = pd.concat(c2_).reset_index(drop=True)

    c2_all_res = c2[['chain_id', 'residue_number', 'insertion']].drop_duplicates().reset_index(drop=True)
    c2_ca_res = c2[c2['atom_name'] == 'CA'][['chain_id', 'residue_number', 'insertion']]
    c2_no_cas = pd.merge(c2_all_res, c2_ca_res, how='left', indicator=True)['_merge'] == 'left_only'

    if sum(c2_no_cas) != 0:
        c2_incomplete_res = np.squeeze(np.where(c2[['chain_id', 'residue_number', 'insertion']].astype(str).agg('_'.join, axis=1).to_numpy() ==
                     c2_all_res[c2_no_cas].astype(str).agg('_'.join, axis=1).to_numpy()))
        c2 = c2.drop(np.atleast_1d((c2_incomplete_res))).reset_index(drop=True)
    else:
        c2_incomplete_res = np.array([])

    c1_CA = pd.concat([c1[c1['atom_name'] == 'CA']['x_coord'],
                       c1[c1['atom_name'] == 'CA']['y_coord'],
                       c1[c1['atom_name'] == 'CA']['z_coord']],
                      axis=1).to_numpy()
    c2_CA = pd.concat([c2[c2['atom_name'] == 'CA']['x_coord'],
                       c2[c2['atom_name'] == 'CA']['y_coord'],
                       c2[c2['atom_name'] == 'CA']['z_coord']],
                      axis=1).to_numpy()

    return c1, c2, c1_CA, c2_CA, c1_incomplete_res, c2_incomplete_res
    
    
def osipov_whole(coord):
    n = 2
    m = 1

    N = len(coord)

    if N >= 4:
        P = np.array(list(itertools.permutations(np.arange(N), 4))) # Get permutations

        coords_P = coord[P]
        r = coords_P - np.roll(coords_P, -1, axis=1)
        r[:, 3] = -r[:, 3]
        r_mag = np.linalg.norm(r, axis=-1)

        cross_vecs = np.cross(r[:, 0], r[:, 2])

        G_p_up = np.einsum('ij,ij->i', cross_vecs, r[:, 3]) * np.einsum('ij,ij->i', r[:, 0], r[:, 1]) * np.einsum('ij,ij->i', r[:, 1], r[:, 2])
        G_p_down = np.power(np.prod(r_mag[:,0:3], axis=-1), n) * np.power(r_mag[:, 3], m)


        G_p = (1 / 3) * np.sum(G_p_up / G_p_down)

        G_os = (24)/(N ** 4) *  G_p

    else:
        G_os = 0

    return G_os


DB5 = pd.read_csv('DB5.csv')


def opd_chains(ii):
    pdb_name = DB5['pdb_id'].iloc[ii]
    chain1 = DB5['chain1'].iloc[ii]
    chain2 = DB5['chain2'].iloc[ii]
    c1, c2, _, _, _, _ = CA_coord(pdb_name, chain1, chain2)
    c1 = c1[c1['element_symbol'] != 'H']
    c2 = c2[c2['element_symbol'] != 'H']

    int_locs = np.where(cdist(c1[['x_coord', 'y_coord', 'z_coord']].to_numpy(), c2[['x_coord', 'y_coord', 'z_coord']].to_numpy()) <= 4.5)

    chain1_opd = osipov_whole(c1[['x_coord', 'y_coord', 'z_coord']].to_numpy()[np.unique(int_locs[0])])
    chain2_opd = osipov_whole(c2[['x_coord', 'y_coord', 'z_coord']].to_numpy()[np.unique(int_locs[1])])

    return np.array([chain1_opd, chain2_opd])

num_cores = multiprocessing.cpu_count()
output = Parallel(n_jobs=num_cores)(delayed(opd_chains)(i) for i in range(len(DB5)))

pd.DataFrame(output).to_csv('opd.csv', index=False)
