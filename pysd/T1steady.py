

import numpy as np
from unit import *

def T1steady(gcoup, vib, T, Utransform=None):

    '''
    calculate the T1 time using the formula:

    T1 =  \sum_i (dg/dx)^2 exp(omega_v/kt)/(exp{omega/kt)-1)^2

    T: Kelvin
    omega: wave number

    '''
    

    Np = gcoup.shape[0]
    T1 = 0.0
    if Utransform is None:
        for i in range(Np):
            tmp = np.exp(vib[i] / (T*K2au))
            tmp2 = 1.0 # without the occoupation part, the results with agree with all-mode calculation
            tmp2 = tmp /(tmp - 1.0) #/(tmp-1.0)
        
            if abs(tmp-1.0) < 1.e-15:
                print('warning: tmp-1.0 is too small', abs(tmp-1.0))


            tmp2 = 1.0 # for tests
            g = gcoup[i,:,:].flatten()
            tmp2 = tmp2 * np.dot(g, g.conj().T).real
            T1 += tmp2
    else:
        # wrong!!!!
        # U(j,i) K(j)
        # where K(j) = exp(omega_v/kt)/(exp{omega/kt)-1)^2
        Np = Utransform.shape[1]
        Np0 = vib.shape[0]
        #print('Usys.shape=', Utransform.shape, Np, Np0)
        for i in range(Np):
            tmp2 =0.0
            for k in range(Np0):
                tmp = np.exp(vib[k] / (T*K2au))
                tmp2 += tmp /(tmp - 1.0) # * Utransform[i] #*Utransform[k,i]

            if abs(tmp-1.0) < 1.e-15:
                print('warning: tmp-1.0 is too small', abs(tmp-1.0))

            tmp2 = 1.0 # for tests
            g = gcoup[i,:,:].flatten()
            tmp2 = tmp2 * np.dot(g, g.conj().T).real
            T1 += tmp2

    #print('T1 time = ', T1, 1.0/T1)
    return 1.0/T1
