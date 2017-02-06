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

  ch=cosmo_history()



