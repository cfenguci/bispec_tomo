from scipy.interpolate import interp1d
from numpy import pi, cos, sin
from matplotlib.patches import Ellipse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from bispec_tomo import *
from perturb import *
from nonlinear import *
import sys

from scipy.integrate import odeint
from utils_bispec import test_interpolate

if __name__=='__main__':
  k= float(sys.argv[1])
  test_interpolate(k)

