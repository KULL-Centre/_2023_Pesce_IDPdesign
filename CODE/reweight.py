from analyse import *
import pymbar
from numba import jit

def genParamsDHrew(df,params,seq):
    kT = 8.3145*params.temp*1e-3
    r = df.copy()
    # Set the charge on HIS based on the pH of the protein solution
    r.loc['H','q'] = 1. / ( 1 + 10**(params.pH-6) )
    r.loc['X','q'] = r.loc[seq[0],'q'] + 1.
    r.loc['Z','q'] = r.loc[seq[-1],'q'] - 1.
    # Calculate the prefactor for the Yukawa potential
    qq = pd.DataFrame(r.q.values*r.q.values.reshape(-1,1),index=r.q.index,columns=r.q.index)
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(params.temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    yukawa_eps = qq*lB*kT
    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*params.ionic*6.022/10)
    return yukawa_eps, yukawa_kappa

def calcD(name):
    traj = md.load_dcd("{:s}/traj.dcd".format(name),"{:s}/top.pdb".format(name))
    pairs = traj.top.select_pairs('all','all')
    mask = np.abs(pairs[:,0]-pairs[:,1])>1 # exclude bonds
    pairs = pairs[mask]
    d = md.compute_distances(traj,pairs).astype(np.float32)
    d[d>4.] = np.inf
    return d, mask


@jit(nopython=True)
def accE(lj_eps, l, sig, dh_eps, yukawa_kappa, d):
    #en_tot = np.apply_along_axis(lambda a: ( ah(a) + yukawa(a) ).sum(), 1, d)
    en_tot = []
    for x in d:
        lj = 4*lj_eps*((sig/x)**12-(sig/x)**6)
        ah = np.where(x<=np.power(2.,1./6)*sig,lj+lj_eps*(1-l),l*lj)
        yukawa = dh_eps*(np.exp(-x*yukawa_kappa)/x - np.exp(-4.*yukawa_kappa)/4.)
        en_tot.append( (ah + yukawa).sum() )
    return np.array(en_tot)

def calcEtot(df, d, params, seq, mask):
    yukawa_eps, yukawa_kappa = genParamsDHrew(df,params,seq)
    lj_eps, lj_sigma, lj_lambda, fasta, _ = genParamsLJ(df,params,seq)
    pairs = np.array(list(itertools.combinations(fasta,2)))
    pairs = np.core.defchararray.add(pairs[:,0],pairs[:,1])
    pairs = pairs[mask]
    dflambda = lj_lambda.unstack()
    dflambda.index = dflambda.index.map('{0[0]}{0[1]}'.format)
    dfsigma = lj_sigma.unstack()
    dfsigma.index = dfsigma.index.map('{0[0]}{0[1]}'.format)
    sig = dfsigma.loc[pairs].values
    l = dflambda.loc[pairs].values
    dfyukawa = yukawa_eps.unstack()
    dfyukawa.index = dfyukawa.index.map('{0[0]}{0[1]}'.format)
    dh_eps = dfyukawa.loc[pairs].values
    en_tot = accE(lj_eps, l, sig, dh_eps, yukawa_kappa, d)
    return en_tot


def calcWeights(e1,e2,params):
    kT = 8.3145*params.temp*1e-3
    weights = np.exp((e1-e2)/kT)
    weights /= weights.sum()
    idxs = np.where( weights > 1e-50 )
    srel = np.sum( weights[idxs] * np.log( weights[idxs] * weights.size ) )
    eff  = np.exp( -srel )
    return weights, eff




#### MBAR FUNCTIONS


def calcPhi(weights):
    weights /= weights.sum()
    idxs = np.where( weights > 1e-50 )
    srel = np.sum( weights[idxs] * np.log( weights[idxs] * weights.size ) )
    eff  = np.exp( -srel )
    return eff

def update_emat(emat, pool_ndx, pool_d, pool_mask, max_pool_size, add_ndx, params, residues, fasta, len_pool_i):
    d, mask = calcD('g'+str(add_ndx))
    if len(pool_ndx) == max_pool_size:
        del pool_ndx[0]
        del pool_d[0]
        del pool_mask[0]
        emat = emat[1:, len_pool_i:]
    newu = np.array([])
    for dd, mm in zip(pool_d, pool_mask):
         Etot = calcEtot(residues, dd, params, fasta[add_ndx], mm)
         newu = np.concatenate((newu, Etot))
    emat = np.vstack((emat, newu))
    pool_ndx.append(add_ndx)
    pool_d.append(d)
    pool_mask.append(mask)
    newu = []
    for i in pool_ndx:
        Etot = calcEtot(residues, pool_d[-1], params, fasta[i], pool_mask[-1])
        newu.append(Etot)
    emat = np.hstack((emat, np.array(newu)))
    return emat, pool_ndx, pool_d, pool_mask


def MBAR(emat, residues, params, pool_d, pool_mask, len_pool_i, newseq):
    kT = 8.3145*params.temp*1e-3
    b = 1/kT
    nj = [len_pool_i for i in pool_d]
    nj.append(0)
    newu = np.array([])
    for dd, mm in zip(pool_d, pool_mask):
        Etot = calcEtot(residues, dd, params, newseq, mm)
        newu = np.concatenate((newu, Etot))
    emat = np.vstack((emat, newu))

    mbar = pymbar.MBAR(b*emat, np.array(nj))
    w = mbar.W_nk[...,-1]
    phieff = calcPhi(w)
    #av = mbar.computeExpectations(rgpool, output='averages')[0][-1]
    neff = phieff * np.sum(nj)
    return w, neff
