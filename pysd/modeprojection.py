
import numpy as np
from numpy import *
import sys
import numpy.linalg as LA
import copy

h = 6.62607015e-34
c = 2.99792458e10
e = 1.60217663e-19
me = 9.1093837015e-31
mp = 1.4106067974e-26
pi = 3.1415926e0

cm2ev = c*h/e
ev2au = e/4.3597447222071e-18
cm2au = cm2ev * ev2au

def modeprojection_svd(eigvals, eigvecs, vibeng, gcoup):

    Ne = eigvals.shape[0]     # number of electronic (spin) states
    Np = vibeng.shape[0]      # number of vib modes

    gcoup0 = copy.deepcopy(gcoup)

    # x = sqrt(2/omega_a) * (a^\dag_a + a_a)
    # p = sqrt(omega_a/2) * (a^\dag_a - a_a)

    g2d = gcoup0.reshape(Np,Ne*Ne)
    #print(' test gcoup', gcoup.shape)
    #print(' test g2d', g2d.shape)

    # U = (Np, Ns)
    # |a> -right U_{ja} |a> 
    # L = (Ns)
    # V = (Ns, Ns)
    U, L, V = LA.svd(g2d, full_matrices=False)

    newomega = np.einsum('ki,k,kj->ij', U, vibeng, U)

    #print('U shape',U.shape)
    #print('L shape',L.shape)
    #print('V shape',V.shape)
    #print(L)

    # g --> U D v^h

    # sum_{ij} U{ia} L_a V_{aj} -> g_{ij}
    # U_{ia} L_a V_{aj} A_j B_i
    #= [L_a V_{aj}] [ (U_{ia} B_i) ]
    #= g_{aj}  \sum_i [ (U_{ia} B_i  ]

    # U_{ia} x_alpha = U_{ia} (a^\dag_a + a_a) --> X_i
    # U_{ia} p_alpha = U_{ia} ((a^\daga - a_a) --> P_i
    
    threshold = 1.e-6
    Lnonzero = np.where(L > threshold)[0]
    print(Lnonzero)
    Np_new = len(Lnonzero)
    Unew = np.zeros((Np,Np_new))

    newg = np.zeros((Np_new,Ne*Ne))
    for i, k in enumerate(Lnonzero):
        newg[i,:] = L[k] * V[:,k]
        #Unew[:,i] = (U[:,k] + U[:,k].conj())/2.0
        #Unew[:,i] = U[:,k].conj()
        Unew[:,i] = U[:,k]
        #for j in range(Np):
        #    Unew[j,i] = Unew[j,i] * np.sqrt(2.0/1.0/vibeng[j])
    newg = newg.reshape(Np_new,Ne,Ne)

    print(newg)

    # new omega
    # |a><a| -> U_{ia}|a><a| U_{aj}
    Omega = np.zeros(Np)
    for i in range(Np):
        Omega[i] = vibeng[i] * 2.0
        #Omega[i] = vibeng[i] * vibeng[i] 
    #Omega_p = np.einsum('ai,a,ja->ij', Unew, vibeng, Unew.conj().T)
    #Omega_p = np.einsum('ia,a,aj->ij', Unew.conj().T, Omega, Unew)
    Omega_p = np.einsum('ai,a,aj->ij', Unew, Omega, Unew)

    print('\n omega_p=', Omega_p)

    vals, Mp = LA.eigh(Omega_p)

    newomega = np.zeros(Np_new)
    for i in range(Np_new):
        newomega[i] = np.sqrt(2.0*vals[i])
        #newomega[i] = vals[i]

    #newg = np.einsum('kab,ki->iab', newg, Mp)
    #newg = np.einsum('ki,kab->iab', Mp,newg)
    newg = np.einsum('ki,kab,ki->iab', Mp, newg,Mp)

    print(newomega)
    Vsb = None
    omega_b = None
    return newomega, omega_b, newg, Vsb

# eigvecs are not used, to be removed 
def modeprojection_svd2(eigvals, eigvecs, vibeng, gcoup):

    Ne = eigvals.shape[0]     # number of electronic (spin) states
    Np = vibeng.shape[0]      # number of vib modes

    gcoup0 = copy.deepcopy(gcoup)
    for i in range(Np):
        gcoup0[i,:,:] /= np.sqrt(2.0*vibeng[i])

    # x = sqrt(2/omega_a) * (a^\dag_a + a_a)
    # p = sqrt(omega_a/2) * (a^\dag_a - a_a)

    # covnert gcoup into 2D array
    g2d = gcoup0.reshape(Np,Ne*Ne)

    print('\ntest gcoup.shape g2d.shape=', gcoup0.shape, g2d.shape)

    # ---------------------------------------------------------------
    # |a> -right U_{ja} |a> 
    # U = (Np, Ns) 
    # L = (Ns)  . If we do thresholding, then the effective number of
    #             nonzeros in L is Na=len(L>threshold)
    # V = (Ns, Ns)
    # svd of gcoup: g_{ij} = U_{ia} L_a V_{ja}
    #                      = U_{ia} gsvd_{ja)
    #----------------------------------------------------------------
    #
    U, L, V = LA.svd(g2d, full_matrices=False)
    #print('\nU/L/V shape',U.shape, L.shape, V.shape)

    Omega = np.zeros(Np)
    for i in range(Np):
        #print('test:omega=', vibeng[i]), 
        Omega[i] = vibeng[i] * vibeng[i]

    ## this is the new w^2 in U presentation, no longer diagonal
    #Omega_all = np.einsum('ki,k,kj->ij', U, Omega, U)
    #print('\n test: Omega_all.shape=', Omega_all.shape)
    #vals, evecs = LA.eigh(Omega_all.copy())
    #print('\n system phonon mode=', np.sqrt(vals)/cm2au)

    # ----------------------------------------------------------------
    # In the next, we diagonal Omega_all in subspace S and B (=1-S),
    # ----------------------------------------------------------------

    threshold = 1.e-12
    Lnonzero = np.where(L > threshold)[0]
    Np_new = len(Lnonzero)

    print('\n non-zero indices=', Lnonzero)
    print('\n Number of system phonon mode=', Np_new)
    
    # get new transformation into S subspace
    # U(j,alpha) 
    Usys = U[:,Lnonzero]

    # assume Vnorm (3*N, 3N-6)
    #Vnorm_proj = np.einsum('ik,jk->ij', Usys.conj(), Vnorm)
    #(3*N, Np_enw)

    Imat = np.identity(Np)
    Pmat = np.einsum('ik,jk->ij', Usys.conj(), Usys)
    Qmat = Imat - Pmat

    print('\n Pmat.shape=',Pmat.shape)

    # construct subspace
    Omega_s2= np.einsum('ki,k,kj->ij', Usys, Omega, Usys)
    # (P + Q) omega (P+Q)
    # --> 
    Omega_s = np.einsum('ki,k,kj->ij', Pmat, Omega, Pmat)
    Omega_b = np.einsum('ki,k,kj->ij', Qmat, Omega, Qmat)

    #print('\n Omega_all after transformaiton')
    #Omega_all = Omega_s + Omega_b
    #for m in range(Np):
    #    for n in range(Np):
    #       if abs(Omega_all[m,n])> 1.e-8: print(m,n,Omega_all[m,n])

    val_s, evecs_s = LA.eigh(Omega_s.copy())
    val_b, evecs_b = LA.eigh(Omega_b.copy())
   

    #print('\n check diagonalization')
    ## U e U^\dag = A
    #tmp = np.einsum('ik,k,jk->ij', evecs_s.conj(), val_s, evecs_s)
    #print('U^+ Omega U - Omega_s=', np.linalg.norm(Omega_s - tmp))
    
    ## U^\dag A U
    #tmp = np.einsum('ik,ij,jk->k', evecs_s, Omega_s, evecs_s)
    #for w in tmp:
    #    if abs(w) > 1.e-8:print(np.sqrt(w)/cm2au)

    Pnonzero = np.where(val_s > 1.e-9)[0]
    Qnonzero = np.where(val_b > 1.e-9)[0]
    
    omega_s = np.sqrt(val_s[Pnonzero])
    omega_b = np.sqrt(val_b[Qnonzero])
    
    print('\n number of sys  phonon mode=', len(Pnonzero))
    print('\n number of bath phonon mode=', len(Qnonzero))
    #print('\n sys  phonon mode=', omega_s/cm2au)
    #print('\n bath phonon mode=', omega_b/cm2au)

    print('evecs.shape', evecs_s.shape)
    print('L[Lnonzero].shape', L.shape, L[Lnonzero].shape)
    print('omega_s.shape    ', omega_s.shape)
    print('V[Lnonzero].shape', V[Lnonzero].shape)

    gsvd = np.einsum('i,ji,jk->ik', np.sqrt(omega_s)*np.sqrt(1.0), evecs_s[:,Pnonzero], g2d)

    #gsvd = np.einsum('j,j,jk->jk', L[Lnonzero], np.sqrt(omega_s), V[Lnonzero,:])
    #gsvd = np.einsum('ij,j,j,jk->ik', evecs_s, L[Lnonzero], np.sqrt(omega_s), V[Lnonzero,:])

    #gsvd = np.einsum('i,i,ij->ij',L[Lnonzero],np.sqrt(omega_s),V[Lnonzero,:])
    #for i in range(Np_new):
    #    gsvd[i,:] *= np.sqrt(omega_s[i]*2.0)
    #gsvd = np.einsum('ik,kj->ij', evecs, gsvd)
    
    #gsvd = np.einsum('ik,kl,lj->ij', evecs, gsvd,evecs)

    gsvd = gsvd.reshape(Np_new,Ne,Ne)

    #3) get Vsb
    Ks = evecs_s[:,Pnonzero]
    Kb = evecs_b[:,Qnonzero]
    #print('ks.shape=', Ks.shape)
    #print('kb.shape=', Kb.shape)

    Ksb = np.zeros((Np, Np), dtype=complex)
    Ksb[:,:Np_new] = Ks
    Ksb[:,Np_new:] = Kb

    ## check if the matrix is block-diagonal,
    ## the off-diagonal is the sys-bath coupling
    #Omega_sb = np.einsum('ki, k, kj->ij', (Pmat+Qmat).conj(), Omega, Pmat+Qmat)
    #gamma_sb = np.einsum('ik, kl, lj', Ksb.conj().T, Omega_sb, Ksb)
    #
    #for i in range(Np):
    #    for j in range(Np):
    #        tmp = np.sqrt(gamma_sb[i,j])/cm2au
    #        if abs(tmp) > 1.e-3:
    #            print(i, j, tmp.real, tmp.imag)

    # gamma = P omega^2 Q  + Q omega^2 P
    Omega_sb = np.einsum('ki, k, kj->ij', Pmat.conj(), Omega, Qmat)
    #Omega_sb+= np.einsum('ki, k, kj->ij', Qmat.conj(), Omega, Pmat)

    Vsb = np.einsum('ik, kl, lj', Ks.conj().T, Omega_sb, Kb)

    for i in range(len(Pnonzero)):
        for j in range(len(Qnonzero)):
            #print("i, j, vsb[i,j]", omega_s[i], omega_b[j], Vsb[i,j]/omega_s[i]/omega_b[j])
            Vsb[i,j] = Vsb[i,j]/omega_s[i]/omega_b[j]

    #sys.exit()

    ## Vsb2 should be Vsb.conj().T
    #Vsb2= np.einsum('ik, kl, lj', Kb.conj().T, Omega_sb, Ks)
    #for i in range(Np_new):
    #    for j in range(Np-Np_new):
    #        print(i, j, Vsb[i,j], Vsb2[j,i])

    return omega_s, omega_b, gsvd, Vsb, Usys

