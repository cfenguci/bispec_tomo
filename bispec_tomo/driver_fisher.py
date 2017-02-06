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
from fishermat import *


if __name__=='__main__':
  list_params=['omegam','sigma8','w0','wa','Gamma','h']
  params={'omegam':0.25,'sigma8':0.8,'w0':-1.0,'wa':0.0,'Gamma':0,'h':0.72}

  bl,cov,deriv,deltaL=loadhdf5('run_tomo.h5')

  (nparams,nL,nperms)=np.shape(deriv)
  fm=FisherMatrix(bl,cov,deriv)

  Fmat=np.zeros((nparams,nparams))
  for i in range(nparams):
    for j in range(nparams):
      Fmat[i][j]=fm.do(i,j)

  print Fmat

  Fmat*=(deltaL**3)

  imat=inv(Fmat)
  for i in range(nparams):
    print np.sqrt(imat[i,i])



  z=0
  for i in range(nparams):
    for j in xrange(i+1,nparams):
      s1=imat[i,i]
      s12=imat[i,j]
      s2=imat[j,j]
      best1,best2=params[list_params[i]],params[list_params[j]]

      get_CL(z,j,i,best2,best1,s1,s2,s12,1.52)


  fig = plt.figure(figsize=(8, 6))
  fig.subplots_adjust(wspace=0.0)
  fig.subplots_adjust(hspace=0.0)
  path_likedata=''

  list_parameter=['Omega_m','sigma_8','w_0','w_a','Gamma','h']
  list_tex=['\\Omega_m','\\sigma_8','w_0','w_a','\\Gamma','h']

  num_para=len(list_parameter)

  z=0
  for i,stri in enumerate(list_parameter):
    if i>0:
      for j,strj in enumerate(list_parameter):
        if j<i:
          idplot=j*num_para+i+1
          print 'likelihood2d_%d-%d'%(i,j)
          like2=np.loadtxt(path_likedata+'forecast_CL_%s-%s_z%0.2f.cl'%(stri,strj,z))
          print i,j,idplot,list_params[i],list_params[j]
          plot_2d(fig,list_tex,num_para,num_para,i,j,idplot,like2)
  #plot_legend(num_para,num_para,2,0,3,z)

  plt.savefig('Fisher.pdf')
