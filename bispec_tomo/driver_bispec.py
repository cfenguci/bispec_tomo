from scipy.interpolate import interp1d
from numpy import pi, cos, sin
from matplotlib.patches import Ellipse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from bispec_tomo import *
from perturb import *
from nonlinear import *


from scipy.integrate import odeint

def pend(y, t, b, c):
  theta, omega = y
  dydt = [omega, -b*omega - c*np.sin(theta)]
  return dydt

def test():
  b = 0.25
  c = 5.0
  y0 = [np.pi - 0.1, 0.0]
  t = np.linspace(0, 10, 101)
  sol = odeint(pend, y0, t, args=(b, c))

  fig = plt.figure(figsize=(10,9))
  plt.clf()
  plt.plot(t, sol[:, 0], 'b', label='theta(t)')
  plt.legend(loc="upper left", ncol=1, shadow=True,  fancybox=True, frameon=False,prop={'size':15});
  plt.savefig('test.pdf')


if __name__=='__main__':

  ch=cosmo_history()



  print ch.get_G(0.0+1e-6)
  #print ch.get_G(0.2)
  print ch.prepare_G()

  #mp=matter_power(ch)

  nl=nonlinear(ch)
  #print 'k_sigma=',nl.k_sigma
  #print 'C=',nl.C
  #print nl.allcoef

  #print nl.Delta2_Q(0.01)
  #print nl.Delta2_H(0.01)


  #print k,dq,dh


  #nl.map_k_kNL()
  
