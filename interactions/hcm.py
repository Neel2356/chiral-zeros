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


def min_haus(degs, L, D):

    x,y,z,a,b,c = degs

    r = R.from_euler('xyz', [a,b,c])
    D = r.apply(D)

    D += [x,y,z]
    haus, _, _ = directed_hausdorff(L, D)

    return haus

def haus_opt(x0, L, D):

    res = minimize(min_haus, x0, method='SLSQP', args=(L,D), options={'ftol': 1e-3, 'disp': False})

    return res['fun'], res['x']

def hcm(L):
    D = -L
    perm = product(np.linspace(-np.pi, np.pi, 24), repeat=3) #from -pi to pi rads, 24 even groups in between them
    inits_abc = np.array([i for i in perm])
    # iter_xyz = product(np.linspace(-np.max(cdist(c1_xyz, c1_xyz)) /50, np.max(cdist(c1_xyz, c1_xyz)) /50, 3), repeat=3)
    # inits_xyz = np.array([i for i in iter_xyz])
    inits_xyz = np.zeros([1,3])
    inits = np.append(np.tile(inits_xyz, [len(inits_abc), 1]), np.repeat(inits_abc, len(inits_xyz), axis=0), axis=1)

    num_cores = multiprocessing.cpu_count()
    output = Parallel(n_jobs=num_cores)(delayed(haus_opt)(i, L, D) for i in inits)
    values = [_[0] for _ in output]
    return min(values) / np.max(cdist(L,L))



DB5 = pd.read_csv('DB5.csv')

hcms = [[],[]]

for ii in np.arange(0, len(DB5)):
# for ii in range(5):
    pdb_name = DB5['pdb_id'].iloc[ii]
    chain1 = DB5['chain1'].iloc[ii]
    chain2 = DB5['chain2'].iloc[ii]
    c1, c2, _, _, _, _ = CA_coord(pdb_name, chain1, chain2)
    c1 = c1[c1['element_symbol'] != 'H']
    c2 = c2[c2['element_symbol'] != 'H']
    
    int_locs = np.where(cdist(c1[['x_coord', 'y_coord', 'z_coord']].to_numpy(), c2[['x_coord', 'y_coord', 'z_coord']].to_numpy()) <= 4.5)
    
    hcms[0].append(hcm(c1[['x_coord', 'y_coord', 'z_coord']].to_numpy()[np.unique(int_locs[0])]))
    hcms[1].append(hcm(c2[['x_coord', 'y_coord', 'z_coord']].to_numpy()[np.unique(int_locs[1])]))
    
pd.DataFrame(hcms).T.to_csv('hcm.csv', index=False)




