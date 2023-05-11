from simtk import openmm, unit
from simtk.openmm import app
from simtk.openmm import XmlSerializer
from analyse import *
import time
import os
import sys

def simulate(residues,name,seq,params,nsteps,stride=1e3, eqsteps=1000, platform='CUDA'):
    os.mkdir(name)

    lj_eps, _, _, fasta, types= genParamsLJ(residues,params,seq)
    yukawa_eps, yukawa_kappa = genParamsDH(residues,params,seq)

    N = len(fasta)

    L = 200

    system = openmm.System()

    # set box vectors
    a = unit.Quantity(np.zeros([3]), unit.nanometers)
    a[0] = L * unit.nanometers
    b = unit.Quantity(np.zeros([3]), unit.nanometers)
    b[1] = L * unit.nanometers
    c = unit.Quantity(np.zeros([3]), unit.nanometers)
    c[2] = L * unit.nanometers
    system.setDefaultPeriodicBoxVectors(a, b, c)
    
    top = md.Topology()
    pos = []
    chain = top.add_chain()
    pos.append([[0,0,L/2+(i-N/2.)*.38] for i in range(N)])
    for resname in fasta:
        residue = top.add_residue(resname, chain)
        top.add_atom(resname, element=md.element.carbon, residue=residue)
    for i in range(chain.n_atoms-1):
        top.add_bond(chain.atom(i),chain.atom(i+1))
    md.Trajectory(np.array(pos).reshape(N,3), top, 0, [L,L,L], [90,90,90]).save_pdb('{:s}/top.pdb'.format(name))

    pdb = app.pdbfile.PDBFile('{:s}/top.pdb'.format(name))

    system.addParticle((residues.loc[seq[0]].MW+2)*unit.amu)
    for a in seq[1:-1]:
        system.addParticle(residues.loc[a].MW*unit.amu) 
    system.addParticle((residues.loc[seq[-1]].MW+16)*unit.amu)

    hb = openmm.openmm.HarmonicBondForce()
    energy_expression = 'select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6),4*eps*((s/r)^12-(s/r)^6)+eps*(1-l))'
    ah = openmm.openmm.CustomNonbondedForce(energy_expression+'; s=0.5*(s1+s2); l=0.5*(l1+l2)')
    yu = openmm.openmm.CustomNonbondedForce('q*(exp(-kappa*r)/r - exp(-kappa*4)/4); q=q1*q2')
    yu.addGlobalParameter('kappa',yukawa_kappa/unit.nanometer)
    yu.addPerParticleParameter('q')

    ah.addGlobalParameter('eps',lj_eps*unit.kilojoules_per_mole)
    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')
 
    for a,e in zip(seq,yukawa_eps):
        yu.addParticle([e*unit.nanometer*unit.kilojoules_per_mole])
        ah.addParticle([residues.loc[a].sigmas*unit.nanometer, residues.loc[a].lambdas*unit.dimensionless])

    for i in range(N-1):
        hb.addBond(i, i+1, 0.38*unit.nanometer, 8033*unit.kilojoules_per_mole/(unit.nanometer**2))
        yu.addExclusion(i, i+1)
        ah.addExclusion(i, i+1)

    yu.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    ah.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    hb.setUsesPeriodicBoundaryConditions(True)
    yu.setCutoffDistance(4*unit.nanometer)
    ah.setCutoffDistance(4*unit.nanometer)
 
    system.addForce(hb)
    system.addForce(yu)
    system.addForce(ah)

    print(ah.usesPeriodicBoundaryConditions())
    print(yu.usesPeriodicBoundaryConditions())
    print(hb.usesPeriodicBoundaryConditions())

    serialized_system = XmlSerializer.serialize(system)
    outfile = open('system.xml','w')
    outfile.write(serialized_system)
    outfile.close()

    integrator = openmm.openmm.LangevinIntegrator(params.temp*unit.kelvin,0.01/unit.picosecond,0.010*unit.picosecond) #10 fs timestep

    print(integrator.getFriction(),integrator.getTemperature())

    platform = openmm.Platform.getPlatformByName(platform)

    simulation = app.simulation.Simulation(pdb.topology, system, integrator, platform)#, dict(CudaPrecision='mixed')) 

    check_point = '{:s}/restart.chk'.format(name)

    if os.path.isfile(check_point):
        print('Reading check point file')
        simulation.loadCheckpoint(check_point)
        simulation.reporters.append(app.dcdreporter.DCDReporter('{:s}/pretraj.dcd'.format(name),int(stride),append=True))
    else:
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        simulation.reporters.append(app.dcdreporter.DCDReporter('{:s}/pretraj.dcd'.format(name),int(stride)))

    simulation.reporters.append(app.statedatareporter.StateDataReporter('{:s}/traj.log'.format(name),int(stride),
             potentialEnergy=True,temperature=True,step=True,speed=True,elapsedTime=True,separator='\t'))

    simulation.step(nsteps)

    simulation.saveCheckpoint(check_point)

    genDCD(name,eqsteps)
