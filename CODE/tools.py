import pandas as pd
from localcider.sequenceParameters import SequenceParameters

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

def rh_kirk(conf):
    n=len(list(conf.top.residues))
    invrij = (1-1/n)*(1/md.compute_distances(conf,conf.top.select_pairs('all','all'))).mean(axis=1)
    return 1/invrij
