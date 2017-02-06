import sys
sys.path.append('/opt/scipy/lib/python2.7/site-packages')
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import itertools as iter
import numpy
import numpy as np
from matplotlib import rc
import matplotlib.cm as cm

import pyfits as pf
import healpy as hp
from math import pi
import sys
from copy import copy
from scipy.stats import norm
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 5}
#rc('text', usetex=True,'font',**font)
rc('font',**font)

rc('text', usetex=True)
matplotlib.rcParams['axes.linewidth'] = 0.1


current_wk_dir='/data-3/cmb/gamma_halo/'
fig = plt.figure(figsize=(8, 6))
fig.subplots_adjust(wspace=0.0)
fig.subplots_adjust(hspace=0.0)

range=[[0.04,0.06],[5.0,7.0],[0.8,1.2]]

xCL=[[0.040, 0.045,0.050,0.055,0.060],[5.0,5.5,6.0,6.5,7.0],[0.8,0.9,1.0,1.1,1.2]]
xlabelsCL=[['0.040','0.045','0.050','0.055',''],['5.0','5.5','6.0','6.5',''],['0.8','0.9','1.0','1.1','1.2']]

best=[0.05,6.0,1.0]


def normalize_like(like):
  num=len(like[:,0])
  x=like[:,0]
  y=like[:,1]
  area=0.0

  area=y.max()

  print 'Area=',area
  y/=area
  nlike=np.zeros((num,2))
  nlike[:,0]=x
  nlike[:,1]=y
  return nlike

#forecast_CL_summary.cl
def load_stat_summary(filename):
  report=np.loadtxt(filename)
  return np.asarray(report)


def plot_legend(nrow,ncol,ix,iy,idplot,z):
  ax=fig.add_subplot(nrow,ncol,idplot,frameon=False)
  ax.xaxis.set_visible(False)
  ax.yaxis.set_visible(False)
  #fg.set_visible(False)
  #ax.axis('off')
  plt.xlim(range[ix][0],range[ix][1])
  #plt.ylim(range[iy][0],range[iy][1])
  plt.ylim(0,1)
  #plt.tick_params(which='both', width=0.5,labelsize=20)

  plt.annotate(r'$z=%0.2f$'%z,xy=(0.85,0.6), xytext=(0.85,0.6), fontsize=25,color='black')
  plt.show()

def plot_1d(nrow,ncol,ix,iy,idplot,like):

  ax=fig.add_subplot(nrow,ncol,idplot)
  plt.xticks(xCL[ix], xlabelsCL[ix])
  plt.tick_params(which='both', width=0.5,labelsize=10)
  plt.xlim(range[ix][0],range[ix][1])

  plt.title(r'$%s=%0.3f\pm%0.3f$'%(list_tex[ix],report[ix,0],report[ix,1]),fontsize=15)
  if ix==nrow-1 and iy==ncol-1:
    plt.xlabel(r'$%s$'%list_tex[ix],fontsize=20)
    ax.yaxis.set_visible(False)
  elif ix==0 and iy==0:
    plt.ylabel(r'$%s$'%list_tex[iy],fontsize=20)
    ax.xaxis.set_visible(False)
  else:
    if iy<2:
      ax.xaxis.set_visible(False)
    if ix>0:
      ax.yaxis.set_visible(False)


  plt.plot(like[:,0],like[:,1],label=r'${%s}$'%list_tex[ix],color='red',lw=1)
  #plt.plot(like[:,0],y[:],label=r'${%s}$'%list_tex[ix],color='green',lw=1)
  #plt.legend(loc="upper right", bbox_to_anchor=[1, 1],  ncol=1, shadow=True,  fancybox=True, frameon=False,prop={'size':15})


def plot_2d(nrow,ncol,ix,iy,idplot,like2):
  ax=fig.add_subplot(nrow,ncol,idplot)

  if iy==ncol-1:
    plt.xlabel(r'$%s$'%list_tex[ix],fontsize=20)
  if ix==0:
    plt.ylabel(r'$%s$'%list_tex[iy],fontsize=20)

  #plt.xlim(range[ix][0],range[ix][1])
  #plt.ylim(range[iy][0],range[iy][1])
  #plt.xticks(xCL[ix], xlabelsCL[ix])
  plt.tick_params(which='both', width=0.5,labelsize=10)

  if iy<2:
    ax.xaxis.set_visible(False)

  if ix>0:
    ax.yaxis.set_visible(False)
  #print idplot,best[ix],best[iy],xCL[ix][0],xCL[ix][-1]
  #plt.plot((xCL[ix][0],xCL[ix][-1]),(best[iy],best[iy]),ls=':',color='green',lw=0.1)
  #plt.plot((best[ix],best[ix]),(xCL[iy][0],xCL[iy][-1]),ls=':',color='green',lw=0.1)
  plt.plot(like2[:,0], like2[:,1],lw=1,color='blue')




if __name__=='__main__':



  path_likedata=''

  list_parameter=['Omega_m','sigma_8','w_0','w_a','Gamma','h']
  list_tex=['\\Omega_m','\\sigma_8','w_0','w_a','\\Gamma','h']

  num_para=len(list_parameter)

  z=0
  for i,stri in enumerate(list_parameter):
    for j,strj in enumerate(list_parameter):
      if j<=i:

        idplot=j*num_para+i+1

        if i!=j:
          print 'likelihood2d_%d-%d'%(i,j)
          like2=np.loadtxt(path_likedata+'forecast_CL_%s-%s_z%0.2f.cl'%(stri,strj,z))
          plot_2d(num_para,num_para,i,j,idplot,like2)


  #plot_legend(num_para,num_para,2,0,3,z)

  plt.savefig('Fisher.pdf')





