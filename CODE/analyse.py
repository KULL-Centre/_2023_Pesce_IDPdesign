import pandas as pd
import numpy as np
import mdtraj as md
import itertools
import os

aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
residues = pd.read_csv('/home/projects/ku_10001/people/frapes/IDP_EVO/MAIN/single_chains/residues.csv')
residues = residues.set_index('one')

parameters = pd.DataFrame(columns=['eps_factor','temp','pH','ionic'])
parameters.loc['default'] = dict(eps_factor=0.2,temp=293,pH=7.4,ionic=0.2)
parameters.loc['fus'] = dict(eps_factor=0.2,temp=298,pH=5.5,ionic=0.15)
parameters.loc['laf1'] = dict(eps_factor=0.2,temp=298,pH=7.5,ionic=0.15)
parameters.loc['a1'] = dict(eps_factor=0.2,temp=298,pH=7.0,ionic=0.15)
parameters.loc['asyn'] = dict(eps_factor=0.2,temp=293,pH=7.4,ionic=0.2)
parameters.loc['v_a1_25'] = dict(eps_factor=0.2,temp=298,pH=7.0,ionic=0.15)
parameters.loc['v_a1_34'] = dict(eps_factor=0.2,temp=307,pH=7.0,ionic=0.15)

def genParamsLJ(df,params,seq):
    fasta = seq.copy()
    r = df.copy()
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    r.loc['X','MW'] += 2
    r.loc['Z','MW'] += 16
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    types = list(np.unique(fasta))
    lj_eps = params.eps_factor*4.184
    lj_sigma = pd.DataFrame((r.sigmas.values+r.sigmas.values.reshape(-1,1))/2,
                            index=r.sigmas.index,columns=r.sigmas.index)
    lj_lambda = pd.DataFrame((r.lambdas.values+r.lambdas.values.reshape(-1,1))/2,
                             index=r.lambdas.index,columns=r.lambdas.index)
    return lj_eps, lj_sigma, lj_lambda, fasta, types

def genParamsDH(df,params,seq):
    kT = 8.3145*params.temp*1e-3
    fasta = seq.copy()
    r = df.copy()
    # Set the charge on HIS based on the pH of the protein solution
    r.loc['H','q'] = 1. / ( 1 + 10**(params.pH-6) )
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    r.loc['X','q'] = r.loc[seq[0],'q'] + 1.
    r.loc['Z','q'] = r.loc[seq[-1],'q'] - 1.
    # Calculate the prefactor for the Yukawa potential
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(params.temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    yukawa_eps = [r.loc[a].q*np.sqrt(lB*kT) for a in fasta]
    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*params.ionic*6.022/10)
    return yukawa_eps, yukawa_kappa

def calcRg(df,name,seq):
    t = md.load_dcd("{:s}/traj.dcd".format(name),"{:s}/top.pdb".format(name))
    masses = df.loc[seq,'MW'].values
    rgarray = md.compute_rg(t,masses=masses)
    return rgarray

def genDCD(name, eqsteps=1000):
    """ 
    Generates coordinate and trajectory
    in convenient formats
    """
    traj = md.load("{:s}/pretraj.dcd".format(name), top="{:s}/top.pdb".format(name))
    traj = traj.image_molecules(inplace=False, anchor_molecules=[set(traj.top.chain(0).atoms)], make_whole=True)
    traj.center_coordinates()
    traj.xyz += traj.unitcell_lengths[0,0]/2
    tocut = eqsteps #10 ns (eq)
    traj[int(tocut):].save_dcd("{:s}/traj.dcd".format(name))
