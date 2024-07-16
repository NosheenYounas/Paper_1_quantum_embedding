
import numpy as np
from qutip import *
import math

class operator_projection():
   
   """
   class used to define A operator in equation 7 of 
   http://qutip.org/docs/latest/guide/dynamics/dynamics-bloch-redfield.html

   with phonon mode projeciton method
   """
   def __init__(self, freq, dg, eS, nS, Bvector, Dtensor, Vsb, omega_b, gamma, T):
      '''
      Vsb: system-bath couplings

      '''
      # spin-phonon coupling
      # dg * es * (a^\dag_q + a_q)
      #

      dim = eS[0].shape[0]
      unit = qeye(dim)

      a = Qobj(dg.real)
      #
      #for i in range(dim):
      #   for j in range(i,dim):
      #      if i == 0 and j == 0:
      #         a = dg[i][j] * eS[i] * eS[i].dag()  #unit
      #         #a = dg[i][j] * tensor(eS[i], nS[j])
      #      else:
      #         a = a + dg[i][j] * eS[i] * eS[j].dag() #unit
      #         #a = a + dg[i][j] * tensor(eS[i], nS[j])
      self.a = a + a.dag()

      self.freq = freq
      self.dg = dg
      self.nS = nS
      self.eS = eS
      self.gamma0 = gamma
      self.Vsb = Vsb
      self.omega_b = omega_b
      self.Nb = omega_b.shape[0]

      # since we have sys-bath coupling Vsb,
      # the gamma can be calculated from Vsb

      # ev to wavelength
      # K2ev = 1.38064852e-23 * T / 1.602176634e-19 ev
      K2ev = 1.38064852e-4 / 1.602176634
      K2au = K2ev/27.211
      # hc =  4.135667516 e-15 ev*s *  2.99792458e10 cm/s
      hc  = 4.135667516 * 2.99792458e-5 # ev * cm
      K2cm = K2ev / hc
      #print('ev to cm, and k to cm=',1.0/hc,K2cm)

      #self.T = T * K2au #K2cm
      self.T = T * K2ev*1000.0


   def a_ops(self):
      return self.a

   def spectrum(self,w):
      print('test-freq', self.freq)
      #

      gammas = 0.0
      gamma0 = self.gamma0
      self_eng1= 0.0
      self_eng2= 0.0
      
      if (w/self.T) < 1.e-6:
          nocc = 0.0
      else:
          nocc = 1.0 / (math.exp(w/self.T) - 1.0)

      for j in range(self.Nb):
          wq = self.omega_b[j]
          if (wq/self.T) < 1.e-6:
              nq = 0.0
          else:
              nq = 1.0 / (math.exp(wq/self.T) - 1.0)
          de = w - self.omega_b[j]
          spec = gamma0/(gamma0*gamma0 + de * de) * self.Vsb[j] * self.Vsb[j]*(wq*self.freq)
          #de = w + self.omega_b[j]
          #spec +=gamma0/(gamma0*gamma0 + de * de) * self.Vsb[j] * self.Vsb[j]#*(nq+1.0)
          gammas += spec /(2.0*np.pi)
          
          '''
          gammas += self.Vsb[j] * self.Vsb[j].conj() 
          
          tmp= nocc #1.0 / (math.exp(self.omega_b[j]/self.T) - 1.0)
          
          self_eng1 += tmp / gamma0  #*np.pi #* self.Vsb[j] * self.Vsb[j].conj()
          self_eng2 += (tmp+1.0) / gamma0 #*np.pi# * self.Vsb[j] * self.Vsb[j].conj()
          '''
      if self.T < 1.e-6:
          nq = 0.0
      else:
          nq = 1.0 / (math.exp(self.freq/self.T) - 1.0)
      
      self_eng1 += nq #*2.0*np.pi  #/gamma0 #* spec
      self_eng2 += (nq+1.0)#*2.0*np.pi  #/gamma0 #* spec
      print('gamma due to sys-bath coupling is:', gammas)
      print('self_eng1 due to sys-bath coupling is:', w, self_eng1)
      print('self_eng2 due to sys-bath coupling is:', w, self_eng2, np.dot(self.Vsb, self.Vsb))
      #
      # spec = 1/pi *{[n_q    *gamma / (gamma * gamma + (w-w_q)^2 +
      #               [(n_q+1)*gamma / (gamma * gamma + (w+w_q)^2]}
      #
      #gammas = self.gamma0*1.e-1 #3.0/198.0

      de = w - self.freq
      spec = gammas / (gammas * gammas + de * de)  * self_eng1 #nq
      de = w + self.freq
      spec = spec + gammas / (gammas * gammas + de * de) * self_eng2 #(nq + 1.0)
      spec = spec / np.pi 

      #print('spectrum of w is %12.4f %20.9f %20.9f %20.9f %20.9f' % (w, self.freq, self.T, nq, spec))
      return spec

   def spectrum_old(self,w):
      #print('test-ferq', self.freq)
      #
      if self.T < 1.e-6:
          nq = 0.0
      else:
          nq = 1.0 / (math.exp(self.freq/self.T) - 1.0)

      gammas = 0.0
      gamma0 = self.gamma0
      for j in range(self.Nb):
          de = w - self.omega_b[j]
          gammas += gamma0 / (gamma0 * gamma0 + de * de) * self.Vsb[j] * self.Vsb[j].conj()
      
      print('gamma due to sys-bath coupling is:', gammas)
      #
      # spec = 1/pi *{[n_q    *gamma / (gamma * gamma + (w-w_q)^2 +
      #               [(n_q+1)*gamma / (gamma * gamma + (w+w_q)^2]}
      #

      de = w - self.freq
      spec = gammas / (gammas * gammas + de * de) # * nq
      de = w + self.freq
      spec = spec + gammas / (gammas * gammas + de * de) #* (nq + 1.0)
      spec = spec / np.pi 
      #print('spectrum of w is %12.4f %20.9f %20.9f %20.9f %20.9f' % (w, self.freq, self.T, nq, spec))
      return spec



class operator():
   """
   class used to define A operator in equation 7 of 
   http://qutip.org/docs/latest/guide/dynamics/dynamics-bloch-redfield.html
   """
   def __init__(self, freq, dg, eS, nS, Bvector, Dtensor, gamma, T):
      #
      # spin-phonon coupling
      # dg * (es, nS)* (a^\dag_q + a_q)
      #
      dim = eS[0].shape[0]
      unit = qeye(dim)

      a = Qobj(dg)
      #print('dg =', dg)
      #print('dg + dg.H =', dg+dg.conj().T)
      print('a ops=',a)

      #for i in range(dim):
      #   for j in range(i,dim):
      #      if i == 0 and j == 0:
      #         a = dg[i][j] * eS[i] * eS[i].dag()  #unit
      #         #a = dg[i][j] * tensor(eS[i], nS[j])
      #      else:
      #         a = a + dg[i][j] * eS[i] * eS[j].dag() #unit
      #         #a = a + dg[i][j] * tensor(eS[i], nS[j])

      self.a = a + a.dag()

      self.freq = freq
      self.dg = dg
      self.nS = nS
      self.eS = eS
      self.gamma = gamma

      # ev to wavelength
      # K2ev = 1.38064852e-23 * T / 1.602176634e-19 ev
      K2ev = 1.38064852e-4 * T / 1.602176634
      K2au = K2ev/27.211
      # hc =  4.135667516 e-15 ev*s *  2.99792458e10 cm/s
      hc  = 4.135667516 * 2.99792458e-5 # ev * cm
      K2cm = K2ev / hc

      #print('ev to cm, and k to cm=',1.0/hc,K2cm)

      #self.T = T * K2au
      self.T = T * K2ev*1.e3 # to mev 

   def a_ops(self):
      return self.a

   def spectrum(self,w):
      #print('test-ferq', self.freq)
      if self.T < 1.e-6:
          nq = 0.0
      else:
          nq = 1.0 / (math.exp(self.freq/self.T) - 1.0)

      if (w/self.T) < 1.e-6:
          nq = 0.0
      else:
          nq = 1.0 / (math.exp(w/self.T) - 1.0)

      #
      # spec = 1/pi *{[n_q    *gamma / (gamma * gamma + (w-w_q)^2 +
      #               [(n_q+1)*gamma / (gamma * gamma + (w+w_q)^2]}
      #
      de = w - self.freq
      spec = self.gamma / (self.gamma * self.gamma + de * de) * nq
      de = w + self.freq
      spec = spec + self.gamma / (self.gamma * self.gamma + de * de) * (nq + 1.0)
      spec = spec / np.pi
      #print('spectrum of w is %12.4f %20.9f %20.9f %20.9f %20.9f' % (w, self.freq, self.T, nq, spec))

      return spec


