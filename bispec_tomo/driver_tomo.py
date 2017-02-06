from scipy.interpolate import interp1d
from numpy import pi, cos, sin
from matplotlib.patches import Ellipse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from bispec_tomo import *
from perturb import *
from nonlinear import *
from fishermat import *


from scipy.integrate import odeint
import operator
import h5py

def combine_dicts(a, b, op=operator.add):
  return dict(a.items()+b.items()+[(k, op(a[k], b[k])) for k in set(b) & set(a)])

def run(params):
  ch=cosmo_history(params)
  nl=nonlinear(ch)
  mp=matter_power(ch,nl)
  bl,cov=mp.prepare_report()
  return bl,cov,ch




if __name__=='__main__':


  list_params=['omegam','sigma8','w0','wa','Gamma','h']
  dparams={'omegam':0.01,'sigma8':0.01,'w0':0.2,'wa':1.0,'Gamma':0.1,'h':0.05}
  params={'omegam':0.25,'sigma8':0.8,'w0':-1.0,'wa':0.0,'Gamma':0,'h':0.72}

  #params_1 = {x: params.get(x, 0) + dparams.get(x, 0) for x in set(params).union(dparams)}
  #params_2 = {x: params.get(x, 0) - dparams.get(x, 0) for x in set(params).union(dparams)}

  Bl,Cov,ch=run(params)
  (nL,nperms)=np.shape(Bl)
  print np.shape(Cov)

  print '------------------------------------------------'

  deriv_pool=[]
  for p in list_params:
    print p,params[p]
    params_1=params.copy()
    params_2=params.copy()
    delta=dparams[p]
    params_1[p]=params[p]+delta
    params_2[p]=params[p]-delta
    Bl_1,Cov_1,ch_1=run(params_1)
    Bl_2,Cov_2,ch_2=run(params_2)
    #n=len(Bl_1)
    print np.shape(Bl_1)
    deriv=(Bl_1-Bl_2)/(2.0*delta)
    deriv_pool.append(deriv)

    print 'Deriv ',p,' is done!'
    print '------------------------------------------------'

  print np.shape(deriv_pool)

  savedata(Bl,Cov,deriv_pool,ch.Delta_ell,'run_tomo.h5')

  '''
  (nparams,nL,nperms)=np.shape(deriv_pool)
  fm=FisherMatrix(Bl,Cov,deriv_pool)

  fp=open('mat_tomo','w')
  for i in range(nparams):
    for j in range(nparams):
      F=fm.do(i,j)
      fp.write(str(F))
      fp.write('\t')
    fp.write('\n')
  fp.close()
  '''
  
