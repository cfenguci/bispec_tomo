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


if __name__=='__main__':

  params={'omegam':0.25,'sigma8':0.8,'w0':-1.0,'wa':0.0,'Gamma':0,'h':0.72}
  ch=cosmo_history(params)
  ch.prepare_FisherMat()

  nl=nonlinear(ch)
  mp=matter_power(ch,nl)


  mp.test_cl_kappa()
  mp.test_bispec_proj()
  #nl.test_Delta_pk()
  #print nl.get_sigma8()
