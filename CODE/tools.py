import pandas as pd
from localcider.sequenceParameters import SequenceParameters
import sys
import numpy as np
import MDAnalysis
from MDAnalysis import transformations

def fix_topology(t,seq):
    cgtop = md.Topology()
    cgchain = cgtop.add_chain()
    for res in seq:
        cgres = cgtop.add_residue(res, cgchain)
        cgtop.add_atom('CA', element=md.element.carbon, residue=cgres)
    traj = md.Trajectory(t.xyz, cgtop, t.time, t.unitcell_lengths, t.unitcell_angles)
    traj = traj.superpose(traj, frame=0)
    return traj

def calc_scd_shd(aa_params,fasta,pH=7.4,model='M1',beta=-1):
    # set histidine charge based on pH
    aa_params.loc['H','q'] = 1. / ( 1 + 10**(pH-6) )
    # set lambda parameters to a given model
    #aa_params.lambdas = aa_params[model]
    N = len(fasta)
    pairs = np.array(list(itertools.combinations(fasta,2)))
    pairs_indices = np.array(list(itertools.combinations(range(N),2)))
    # calculate sequence separations
    ij_dist = np.diff(pairs_indices,axis=1).flatten().astype(float)
    # calculate charge products
    qq = aa_params.q.loc[pairs[:,0]].values*aa_params.q.loc[pairs[:,1]].values
    # calculate lambda sums
    ll = aa_params.lambdas.loc[pairs[:,0]].values+aa_params.lambdas.loc[pairs[:,1]].values
    scd = np.sum(qq*np.sqrt(ij_dist))/N
    shd = np.sum(ll*np.power(np.abs(ij_dist),beta))/N
    return scd,shd

def get_omega_aro(seq):
    SeqObj = SequenceParameters(seq)
    return SeqObj.get_kappa_X(grp1=['Y','F','W'])

def get_kappa(seq):
    SeqObj = SequenceParameters(seq)
    return SeqObj.get_kappa()

def calc_seq_params(pkl):
    df = pd.read_pickle(pkl)
    params = []
    for s in df.fasta.values:
        tmp = np.array([0.0,0.0,0.0,0.0])
        tmp[0], tmp[1] = calc_scd_shd(residues.set_index('one'),s,pH=7.)
        tmp[2] = get_omega_aro(''.join(s))
        tmp[3] = get_kappa(''.join(s))
        params.append(tmp)
    np.savetxt(pkl[:-3]+'params', np.array(params))

def rg(t, seq):
    residues = pd.read_csv('/storage1/francesco/PROJECTS/IDP_EVO/MAIN/residues.csv')
    residues = residues.set_index('one')
    masses = residues.loc[seq,'MW'].values
    rgarray = md.compute_rg(t,masses=masses)
    return rgarray

def calc_zpatch(z,h):
    cutoff = 0
    ct = 0.
    ct_max = 0.
    zwindow = []
    hwindow = []
    zpatch = []
    hpatch = []
    for ix, x in enumerate(h):
        if x > cutoff:
            ct += x
            zwindow.append(z[ix])
            hwindow.append(x)
        else:
            if ct > ct_max:
                ct_max = ct
                zpatch = zwindow
                hpatch = hwindow
            ct = 0.
            zwindow = []
            hwindow = []
    zpatch = np.array(zpatch)
    hpatch = np.array(hpatch)
    return zpatch, hpatch

def center_slab(name,start=None,end=None,step=1,input_pdb='top.pdb'):
    path = f'{name}'
    u = MDAnalysis.Universe(f'{path}/{input_pdb}',path+f'/{name}.dcd',in_memory=True)
    n_frames = len(u.trajectory[start:end:step])
    ag = u.atoms
    n_atoms = ag.n_atoms

    L = u.dimensions[0]/10
    lz = u.dimensions[2]
    edges = np.arange(0,lz+1,1)
    dz = (edges[1] - edges[0]) / 2.
    z = edges[:-1] + dz
    n_bins = len(z)
    hs = np.zeros((n_frames,n_bins))
    with MDAnalysis.Writer(path+'/traj.dcd',n_atoms) as W:
        for t,ts in enumerate(u.trajectory[start:end:step]):
            # shift max density to center
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos,bins=edges)
            zmax = z[np.argmax(h)]
            ag.translate(np.array([0,0,-zmax+0.5*lz]))
            ts = transformations.wrap(ag)(ts)
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos, bins=edges)
            zpatch, hpatch = calc_zpatch(z,h)
            zmid = np.average(zpatch,weights=hpatch)
            ag.translate(np.array([0,0,-zmid+0.5*lz]))
            ts = transformations.wrap(ag)(ts)
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos,bins=edges)
            hs[t] = h
            W.write(ag)
    np.save(f'{path}/{name}.npy',hs,allow_pickle=False)
