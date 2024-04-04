from simtk import openmm, unit
from simtk.openmm import app
from simtk.openmm import XmlSerializer
import time
import os
import sys
import pandas as pd
import numpy as np
import mdtraj as md
import itertools
import MDAnalysis
from MDAnalysis import transformations

aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
residues = pd.read_csv('/home/projects/ku_10001/people/frapes/IDP_EVO/MAIN/single_chains/residues.csv')
residues = residues.set_index('one', drop=False)
eps_factor=0.2

def genParamsLJ(df,name,fasta):
    fasta = fasta.copy()
    r = df.copy()
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    r.loc['X','MW'] += 2
    r.loc['Z','MW'] += 16
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    types = list(np.unique(fasta))
    MWs = [r.loc[a,'MW'] for a in types]
    lj_eps = eps_factor*4.184
    return lj_eps, fasta, types, MWs

def genParamsDH(df,name,fasta,temp,pH,ionic):
    kT = 8.3145*temp*1e-3
    fasta = fasta.copy()
    r = df.copy()
    # Set the charge on HIS based on the pH of the protein solution
    r.loc['H','q'] = 1. / ( 1 + 10**(pH-6) )
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    r.loc['X','q'] = r.loc[fasta[0],'q'] + 1.
    r.loc['Z','q'] = r.loc[fasta[-1],'q'] - 1.
    # Calculate the prefactor for the Yukawa potential
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    yukawa_eps = [r.loc[a].q*np.sqrt(lB*kT) for a in fasta]
    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*ionic*6.022/10)
    return yukawa_eps, yukawa_kappa

def simulate(residues,name,seq,temp,pH=7.0,ionic=0.15,cutoff=2.0,nsteps=5e8):
    os.mkdir(name)

    lj_eps, fasta, types, MWs = genParamsLJ(residues,name,seq)
    yukawa_eps, yukawa_kappa = genParamsDH(residues,name,seq,temp,pH,ionic)

    N = len(fasta)

    # set parameters
    L = 15.
    margin = 2
    if N > 350:
        L = 25.
        Lz = 300.
        margin = 8
        Nsteps = int(2e7)
    elif N > 200:
        L = 17.
        Lz = 300.
        margin = 4
        Nsteps = int(6e7)
    else:
        Lz = 10*L
        Nsteps = int(6e7)

    system = openmm.System()

    # set box vectors
    a = unit.Quantity(np.zeros([3]), unit.nanometers)
    a[0] = L * unit.nanometers
    b = unit.Quantity(np.zeros([3]), unit.nanometers)
    b[1] = L * unit.nanometers
    c = unit.Quantity(np.zeros([3]), unit.nanometers)
    c[2] = Lz * unit.nanometers
    system.setDefaultPeriodicBoxVectors(a, b, c)
    
    # initial config
    xy = np.empty(0)
    xy = np.append(xy,np.random.rand(2)*(L-margin)-(L-margin)/2).reshape((-1,2))
    for x,y in np.random.rand(1000,2)*(L-margin)-(L-margin)/2:
        x1 = x-L if x>0 else x+L
        y1 = y-L if y>0 else y+L
        if np.all(np.linalg.norm(xy-[x,y],axis=1)>.7):
            if np.all(np.linalg.norm(xy-[x1,y],axis=1)>.7):
                if np.all(np.linalg.norm(xy-[x,y1],axis=1)>.7):
                    xy = np.append(xy,[x,y]).reshape((-1,2))
        if xy.shape[0] == 100:
            break

    n_chains = xy.shape[0]

    pdb_file = name+'/top.pdb'

    if os.path.isfile(pdb_file):
        pdb = app.pdbfile.PDBFile(pdb_file)
    else:
        top = md.Topology()
        pos = []
        for x,y in xy:
            chain = top.add_chain()
            pos.append([[x,y,Lz/2+(i-N/2.)*.38] for i in range(N)])
            for resname in fasta:
                residue = top.add_residue(resname, chain)
                top.add_atom(resname, element=md.element.carbon, residue=residue)
            for i in range(chain.n_atoms-1):
                top.add_bond(chain.atom(i),chain.atom(i+1))
        md.Trajectory(np.array(pos).reshape(n_chains*N,3), top, 0, [L,L,Lz], [90,90,90]).save_pdb(pdb_file)
        pdb = app.pdbfile.PDBFile(pdb_file)

    for _ in range(n_chains):
        system.addParticle((residues.loc[seq[0]].MW+2)*unit.amu)
        for a in seq[1:-1]:
            system.addParticle(residues.loc[a].MW*unit.amu) 
        system.addParticle((residues.loc[seq[-1]].MW+16)*unit.amu)

    hb = openmm.openmm.HarmonicBondForce()

    energy_expression = 'select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6-shift),4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))'
    ah = openmm.openmm.CustomNonbondedForce(energy_expression+'; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6')

    ah.addGlobalParameter('eps',lj_eps*unit.kilojoules_per_mole)
    ah.addGlobalParameter('rc',cutoff*unit.nanometer)
    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')
    
    print('rc',cutoff*unit.nanometer)
 
    yu = openmm.openmm.CustomNonbondedForce('q*(exp(-kappa*r)/r-shift); q=q1*q2')
    yu.addGlobalParameter('kappa',yukawa_kappa/unit.nanometer)
    yu.addGlobalParameter('shift',np.exp(-yukawa_kappa*4.0)/4.0/unit.nanometer)
    yu.addPerParticleParameter('q')

    for j in range(n_chains):
        begin = j*N
        end = j*N+N
       
        for a,e in zip(seq,yukawa_eps):
            yu.addParticle([e*unit.nanometer*unit.kilojoules_per_mole])
            ah.addParticle([residues.loc[a].sigmas*unit.nanometer, residues.loc[a].lambdas*unit.dimensionless])

        for i in range(begin,end-1):
            hb.addBond(i, i+1, 0.38*unit.nanometer, 8033.0*unit.kilojoules_per_mole/(unit.nanometer**2))
            yu.addExclusion(i, i+1)
            ah.addExclusion(i, i+1)

    yu.setForceGroup(0)
    ah.setForceGroup(1)
    yu.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    ah.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    hb.setUsesPeriodicBoundaryConditions(True)
    yu.setCutoffDistance(4*unit.nanometer)
    ah.setCutoffDistance(cutoff*unit.nanometer)

    system.addForce(hb)
    system.addForce(yu)
    system.addForce(ah)

    print(ah.usesPeriodicBoundaryConditions())
    print(yu.usesPeriodicBoundaryConditions())
    print(hb.usesPeriodicBoundaryConditions())
    print(ah.getSwitchingDistance())

    integrator = openmm.openmm.LangevinIntegrator(temp*unit.kelvin,0.01/unit.picosecond,0.01*unit.picosecond)

    print(integrator.getFriction(),integrator.getTemperature())

    platform = openmm.Platform.getPlatformByName('CUDA')

    simulation = app.simulation.Simulation(pdb.topology, system, integrator, platform, dict(CudaPrecision='mixed'))

    check_point = name+'/restart.chk'

    if os.path.isfile(check_point):
        print('Reading check point file')
        simulation.loadCheckpoint(check_point)
        simulation.reporters.append(app.dcdreporter.DCDReporter(name+'/{:s}.dcd'.format(name),int(1e4),append=True))
    else:
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        simulation.reporters.append(app.dcdreporter.DCDReporter(name+'/{:s}.dcd'.format(name),int(1e4)))

    simulation.reporters.append(app.statedatareporter.StateDataReporter(name+'/{:s}.log'.format(name),100000,
             step=True,speed=True,elapsedTime=True,separator='\t'))

    #simulation.runForClockTime(20*unit.hour, checkpointFile=check_point, checkpointInterval=2*unit.hour)
    simulation.step(nsteps)
    simulation.saveCheckpoint(check_point)

    #genDCD(residues,name,seq,n_chains)
