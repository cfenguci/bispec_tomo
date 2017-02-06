import numpy as np
from scipy.interpolate import interp1d
from numpy import pi, cos, sin
from matplotlib.patches import Ellipse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import pyfits as pf
import healpy as hp
from math import pi
import pprint
#from math import atanh
import math

import timeit

import os.path
import os
import subprocess
import argparse
import re

from multiprocessing.dummy import Pool as ThreadPool

def get_all_range(k,x,y):
  x1=[]
  fx=interp1d(x[:],y[:],kind='linear')
  for i in range(len(k)):
    if k[i]>=x[0] and k[i]<=x[len(x)-1]:
      #print z[i],fx(z[i])
      v=fx(k[i])
    else:
      #print z[i],0
      v=0.0

    x1.append(v)
  return np.asarray(x1)

def fnu_sz(nu):
  x=nu/56.84 #GHz
  #v=x/math.atanh(x/2.0)-4.0
  v=x*(np.exp(x)+1.0)/(np.exp(x)+1.0)-4.0
  return v

def lambda_ell(l):
  return l*(l+1.0)

def run_C_exe(command):
  proc = subprocess.Popen(command, stdout=subprocess.PIPE,stderr=subprocess.PIPE, shell=True)
  (out, err) = proc.communicate()
  return out,err


def wigner3j(aj,am):
  v3j=0.0
  file_exe='/data-3/cmb/newpros/phitau/wigxjpf-1.5/bin/wigxjpf'
  if os.path.exists(file_exe):
    command='%s --3j=%d,%d,%d,%d,%d,%d'%(file_exe,aj[0],aj[1],aj[2],am[0],am[1],am[2])
    out,err=run_C_exe(command)
    numbers=re.findall("[-+]?\d+[\.]?\d+[eE]?[-+]?\d*",out)
    v3j=numbers[-1]
  else:
    print 'wigner 3j exe does not exist!'
    v3j=0

  return float(v3j)



def logfac(n):
  v=0.0
  if n<=0:
    v=0.0
  else:
    v=n*np.log(n)-n+1.0/6.0*np.log(n*(1+4*n*(1+2*n)))+0.5*np.log(pi)

  return v



def wig3j_fast(aj):
  [l1,l2,l3]=aj
  L=l1+l2+l3
  LL=int(L)

  if LL % 2 ==0:
    logvalue=0.5*(logfac(L-2*l1)+logfac(L-2*l2)+logfac(L-2*l3)-logfac(L+1))+logfac(L/2)-logfac(L/2-l1)-logfac(L/2-l2)-logfac(L/2-l3);
    result=np.power(-1.0,L/2)*np.exp(logvalue);
  else:
    result=0.0

  return result

def hfac(aj,am):
  #v3j=wigner3j(aj,am)
  v3j=wig3j_fast(aj)
  [l1,l2,l3]=aj
  fac=np.sqrt((2.0*l1+1.0)*(2.0*l2+1.0)*(2.0*l3+1.0)/4.0/pi)

  return v3j*fac

def selection_rules(aj):

  [l1,l2,l3]=aj

  flag=[0,0,0,0]

  lmin=abs(l1-l2)
  lmax=l1+l2
  if l3>=lmin and l3<=lmax:
    flag[0]=1

  lmin=abs(l2-l3)
  lmax=l2+l3
  if l1>=lmin and l1<=lmax:
    flag[1]=1

  lmin=abs(l1-l3)
  lmax=l1+l3
  if l2>=lmin and l2<=lmax:
    flag[2]=1

  sum=int(l1+l2+l3)
  if sum % 2 == 0:
    flag[3]=1

  status=flag[0]*flag[1]*flag[2]*flag[3]
  return status

def outputfile(name,data):
  (ncol,num)=np.shape(data)
  #num=len(data[:,0])

  fp=open('%s'%name,'w')
  for i in range(num):
    for k in range(ncol):
      fp.write(str(data[k,i]))
      fp.write('\t')
    fp.write('\n')
  fp.close()




def test_interpolate(kk):
  from scipy.interpolate import InterpolatedUnivariateSpline
  data=np.loadtxt('../test_matterpower.dat')
  k=data[:,0]
  Pk=data[:,1]
  print k[0],k[-1]
  it_mp1=InterpolatedUnivariateSpline(k,Pk,k=1)
  it_mp2=interp1d(k,Pk,fill_value='extrapolate')
  print it_mp1(kk),it_mp2(kk)