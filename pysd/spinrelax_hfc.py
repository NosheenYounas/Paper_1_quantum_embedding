import numpy as np
from qutip import *
from numpy import *
import sys
import numpy.linalg as LA
import copy

from modeprojection import *
from collapseops import *
from readfiles import *
from T1steady import *
from unit import *

#---------------------------------------
# (TODO) read parameters from inputs
#---------------------------------------

breakline = "-"*75

nspin = 1
Se = 0.5
Sn = 0.5
#Sn = 3.5
dim_e = int(2 * Se + 1)
dim_n = int(2 * Sn + 1)

# unit operator of electron spin subspace
unit_e = qeye(dim_e)
# unit operator of nuclear spin subspace
unit_n = qeye(dim_n)

funq = 0

if len(sys.argv) > 1:
    funq = int(sys.argv[1])
    print('funq=', funq)

eS = jmat(Se) # electronic spin operator
nS = jmat(Sn) # nuclear spin operator
#-------------------------------------
# this format only works for one spin
# (TODO) extension to multple spin 
#-------------------------------------

print('electronic S')
for i,spin in enumerate(eS):
   print(i, spin.data)

print('Nuclear spin S')
for i,spin in enumerate(nS):
   print(i, spin.data)

#==========================
# read gtensor, hfc, zfs
#==========================

# (TODO) which units 
#unit of hfc (MHZ)

fname = 'gtensor.dat'
gtensor, hfc, zfs = read_g(fname)
print('\n gtensor=\n', gtensor)
print('\n hfc =\n', hfc)

Bvector = np.zeros(3)
Bvector[0] = 0.e1
Bvector[1] = 0.e1
Bvector[2] = 1.e1

Bvector /= np.linalg.norm(Bvector)
print('B field = ', Bvector)

# if more than one spin, get dipole dipole interactions
Dtensor = np.zeros((nspin, nspin))

#==================
## define spin Hamiltonian
#==================

H = None
# 0) Add ZFS term S_i D_i S_i (TODO)

# 1) Zeeman interaction
for i in range(3):
  for j in range(3):
   if H is None:
       H = gtensor[i][j] * tensor(eS[i], unit_n) * Bvector[j]
   else:
       H += gtensor[i][j] * tensor(eS[i], unit_n) * Bvector[j]
    
   H+= zfs[i,j] * tensor(eS[i], unit_n) * tensor(eS[j], unit_n)

# 2) hfc coupling
for i in range(3):
    for j in range(3):
        H += hfc[i][j] * tensor(eS[i], nS[j])
# 3) dipole-dipole interaction (todo)

print('\nHamiltonian=\n',H.data)

dim_en = tensor(eS[0], nS[0]).shape[0]

#======================================
# get phonon and spin-phonon couplings
#======================================
fname = 'spin_phonon.dat'
f2name = 'spin_phonon_hpc.dat'
#f3name = 'spin_phonon_dip.dat'
freq, dgx = get_phonon(fname)
freq_A, dAx = get_phonon(f2name)

h = 6.62607015e-34
c = 2.99792458e10
e = 1.60217663e-19
me = 9.1093837015e-31
mp = 1.4106067974e-26
pi = 3.1415926e0

cm2ev = c*h/e
ev2au = e/4.3597447222071e-18
cm2au = cm2ev * ev2au

# transfer Hamiltonain into au (check unit of H and transfer the unit)
H = H * cm2au

#
freq = freq * cm2au

Np = freq.shape[0]
print(H.shape[0], Np)

projection = True
if projection:
    # g-phonon coupling constant in the presence of external B field
    # eS * dgx * B[j]
    # System = tensor(electron, nuclearspin)
    gcoup = np.zeros((Np, dim_en, dim_en),dtype=complex)

    for k in range(Np):
        for i in range(3):
            for j in range(3):
                # gtensor component
                gtmp = dgx[k,i,j] * tensor(eS[i], unit_n) * Bvector[j]
                gcoup[k,:,:] += gtmp.data #tensor(eS[i], unit_n) * dgx[k,i,j] * Bvector[j]

                # hcp term
                gtmp = dAx[k,i,j] * tensor(eS[i], nS[j])
                gcoup[k,:,:] += gtmp.data

    # get eigenstates of spin system
    eigvals, eigvecs = H.eigenstates()
    gcoup = gcoup * cm2au #(todo: check the unit of original gcoup, may not be cm-1)

    print('\nEigen energies of the spin Hamiltonian:\n', eigvals)
    print('\nEigen states   of the spin Hamiltonian:\n', eigvecs)

    Us = np.zeros((dim_en,dim_en),dtype=complex)
    print('')
    # transform H in spin eigenstate representation
    for k in range(dim_en):
        print('eigvecs of H', k, eigvecs[k].full())
        Us[:,k] = eigvecs[k].full()[:,0]
    eigvecs = Us


    # V_sb is the coupling between system and bath phonons
    print('\n',breakline, '\n mode projection begins\n',breakline,'\n')
    omega_s, omega_b, newdgx, Vsb, Usys = modeprojection_svd2(eigvals, eigvecs, freq, gcoup)
    print('\n==============mode projection ends ==============\n')

    print('\n=========== trivial mode cutoff ========= \n')
    gcoup_cut = copy.deepcopy(gcoup)
    gmax = 0.0
    for i in range(Np):
        gtmp = gcoup[i,:,:].flatten()
        gtmp_norm = np.dot(gtmp, gtmp)
        if gtmp_norm > gmax: gmax = gtmp_norm

    count = 0
    for i in range(Np):
        gtmp = gcoup[i,:,:].flatten()
        #if np.dot(gtmp, gtmp) < 2.e-1*gmax:
        if np.dot(gtmp, gtmp) < 4.45e-1*gmax:
        #if np.dot(gtmp, gtmp) < 5.5e-1*gmax:
            gcoup_cut[i,:,:] = 0.0
        else:
            count += 1
            print(' selected omega=', freq[i]/cm2au, np.dot(gtmp, gtmp))
    print(' effective mode after thresholding=', count)
    print('==============================================\n')

    # calculate the T1 time using HF formalism
    Templist = np.linspace(20, 300, 20)
    T1 = np.zeros(len(Templist))
    T1cut = np.zeros(len(Templist))
    T1new = np.zeros(len(Templist))
    for k ,Tmp in enumerate(Templist):
        T1[k] = T1steady(gcoup, freq, Tmp)
        T1cut[k] = T1steady(gcoup_cut, freq, Tmp)
        T1new[k] = T1steady(newdgx, omega_s, Tmp)
        print('Temp, T1: %6.2f %14.6e %14.6e %14.6e %9.4f %9.4f' % (Tmp, 
                T1[k], T1new[k], T1cut[k], T1new[k]/T1[k], T1cut[k]/T1[k]))

    print('\n',breakline)
    print('Compute the R tensor within the mode projection scheme')
    print(breakline,'\n')
    # since we have sys-bath coupling Vsb,
    # the gamma can be calculated from Vsb
    Ns = omega_s.shape[0]
    Nb = omega_b.shape[0]
    
    #gamma = np.zeros(Ns)
    #for i in range(Ns):
    #    tmp = 0.0
    #    for j in range(Nb):
    #        tmp += Vsb[i,j]*Vsb[i,j].conj() 

    a_ops = []
    gamma = 0.1 # unit cm
    T = 300.0  # unit K
    
    eigvals, eigvecs = H.eigenstates()
    for i in range(Ns):
        ops = operator_projection(omega_s[i], newdgx[i,:,:], eigvecs, nS,
                Bvector, Dtensor, Vsb[i,:], omega_b, gamma, T)
        
        a_ops.append([ops.a, ops.spectrum])

    #=========================================
    # calculate eigen states and R tensor
    #=========================================
    R, ekets = bloch_redfield_tensor(H, a_ops)
    
    print('ekets=',ekets)
    print('R    =',R)

    # ------------------------------------
    # add the script of TD propagation 
    # ------------------------------------
    
    ## time-dependent expectation values to be computed
    ## <O> = Tr[rho * O]
    #, tensor(eS[1], qeye(dim_n)), tensor(eS[2], qeye(dim_n))]

    # expectatin value calculation is not working (to be fixed)
    # error message is: "can't convert complex to float"

    # time steps
    tend = 1.0e1 #fs
    dt = 0.01  #fs
    outdata_freq = int(1.0/dt)
    nt   = int(tend/dt)
    tlist = np.linspace(0, tend, nt)
    
    # define initial spin state
    psi0 = tensor(basis(dim_e,1), basis(dim_n,0))
    print('psi0=',psi0)
    
    e_ops = [psi0*psi0.dag()] #, tensor(eS[0],qeye(dim_n)),tensor(eS[1],qeye(dim_n)),tensor(eS[2],qeye(dim_n))]
    print(e_ops)

    print('\n',breakline)
    print(' solving Redfield equation==')
    print(breakline,'\n')
    
    expt_list_svd = bloch_redfield_solve(R, ekets, psi0, tlist, e_ops)


   # TD dynamics
    #if funq == 2:
    # ================================
    # definie A operator for each mode
    # ================================

    print('\n',breakline)
    print('--------- QD without mode projection ------------')
    print(breakline,'\n')

    a_ops = []
    eigvals, eigvecs = H.eigenstates()
    
    for i in range(len(freq)):
       ops = operator(freq[i], gcoup[i], eigvecs, nS, Bvector, Dtensor, gamma, T)
       #ops = operator(freq[i], dgx[i], eigvecs, nS, Bvector, Dtensor, gamma, T)
    
       # the coupling between spin and phonon mode q is:
       # eS * dg^q * nS * (a^\dag_q + a_q)
       # == A * (a^\dag_a + a_q)
       # where  A = eS * dg_alpha * nS
       #
       # so the A operator for the redfield equation (eq 7 of 
       # http://qutip.org/docs/latest/guide/dynamics/dynamics-bloch-redfield.html) is:
       # A = eS * dg^q * nS * S_q, where S_q is
       # the spectrum function of the mode q.
       a_ops.append([ops.a, ops.spectrum])
    
    #=========================================
    # calculate eigen states and R tensor
    #=========================================
    R, ekets = bloch_redfield_tensor(H, a_ops)
    
    print('ekets=',ekets)
    print('R    =',R)
    
    
    ## time-dependent expectation values to be computed
    ## <O> = Tr[rho * O]

    #e_ops = [tensor(eS[0],qeye(dim_n)), tensor(eS[1], qeye(dim_n)), tensor(eS[2], qeye(dim_n))]
    
    print('\n',breakline)
    print(' solving Redfield equation==')
    print(breakline,'\n')
    
    expt_list = bloch_redfield_solve(R, ekets, psi0, tlist, e_ops)
    
    for k,expt in enumerate(expt_list):
      print('test', k)
      #print(expt)
      for i, value in enumerate(expt):
          if (i+1) % outdata_freq == 0 or i ==0:
              #print('%15.6f %20.9e' % (tlist[i], value))
              print('%15.6f %20.9f %20.9f' % (tlist[i], value, expt_list_svd[k][i]))

