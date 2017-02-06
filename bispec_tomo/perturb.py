import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import pi, cos, sin
from matplotlib.patches import Ellipse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math

from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.optimize import fsolve

from bispec_tomo import *
from bispec_tomo import z2a,a2z,tophat
#from bispec_tomo import integrate
#from bispec_tomo import cosmo_history
from nonlinear import nonlinear
from utils_bispec import wig3j_fast, hfac
import multiprocessing
import time

def getKey(item):
  return item[0]

def get_angle(a,b,c):
  ratio=(a**2+b**2-c**2)/(2.0*a*b)
  if ratio>=1.0:
    ratio=1.0
  if ratio<=-1.0:
    ratio=-1.0
  return math.degrees(math.acos(ratio)),ratio

def triangle(k):
  [k1,k2,k3]=k
  #print k1,k2,k3
  theta12,x12=get_angle(k1,k2,k3)
  theta23,x23=get_angle(k2,k3,k1)
  theta13,x13=get_angle(k1,k3,k2)

  #print theta12,theta23,theta13,theta12+theta23+theta13
  pair=[]
  pair.append([k1,k2,theta12,x12])
  pair.append([k2,k3,theta23,x23])
  pair.append([k1,k3,theta13,x13])
  pair=np.asarray(pair)

  return pair

class matter_power():
  def __init__(self,cosmo_history,nonlinear):
    self.z=cosmo_history.z
    self.copy_cosmo_params(cosmo_history)
    self.method=cosmo_history.interp_method
    self.interp_order=cosmo_history.interp_order

    self.a_growth,self.growth=cosmo_history.a_growth,cosmo_history.growth
    self.it_H_z,self.it_chi,self.it_dchidz=cosmo_history.it_H_z,cosmo_history.it_chi,cosmo_history.it_dchidz
    self.it_rho_m_z=cosmo_history.it_rho_m_z
    self.it_G=cosmo_history.it_G
    self.it_W=cosmo_history.it_W
    self.shot_tomo=cosmo_history.shot_tomo
    self.it_growth=InterpolatedUnivariateSpline(self.a_growth,self.growth,k=self.interp_order)
    self.z_growth=1.0/self.a_growth-1.0

    self.z_eff=[self.z_growth[-1],5.0]
    #self.z_eff=[self.z_growth[-1],self.z_growth[0]]
    print 'matter power -> eff z=',self.z_eff
    self.LoopZBandCl,self.LoopZBandBl,self.Combi_ell=cosmo_history.LoopZBandCl,cosmo_history.LoopZBandBl,cosmo_history.Combi_ell



    self.it_2d_NLPower=nonlinear.it_2d_NLPower

    self.it_mt_camb=nonlinear.it_mt_camb
    self.k_range=nonlinear.k_range
    self.numk=nonlinear.numk

    k,self.nk=self.n_slope()
    self.it_nk=InterpolatedUnivariateSpline(k,self.nk,k=self.interp_order)
    listkNL,self.z_min_kNL,self.z_max_kNL,self.it_kNL=self.prepare_kNL()
    print 'Get kNL within: ',self.z_min_kNL,self.z_max_kNL

  def copy_cosmo_params(self,cosmo_history):
    self.h=cosmo_history.h
    self.H0=cosmo_history.H0
    self.Gamma=cosmo_history.Gamma
    self.w0=cosmo_history.w0
    self.wa=cosmo_history.wa
    self.Omega_m=cosmo_history.Omega_m
    self.Omega_phi=cosmo_history.Omega_phi
    self.sigma8=cosmo_history.sigma8
    self.Omega_b=cosmo_history.Omega_b
    self.As=cosmo_history.As
    self.ns=cosmo_history.ns
    self.k0=cosmo_history.k0

    self.rho_crit_0=cosmo_history.rho_crit_0

    #weak lensing
    self.z0=cosmo_history.z0
    self.beta=cosmo_history.beta
    self.p0=cosmo_history.p0
    self.z_med=cosmo_history.z_med
    self.ngal=cosmo_history.ngal
    self.sigma_shot=cosmo_history.sigma_shot
    self.fsky=cosmo_history.fsky

    self.lmin=cosmo_history.lmin
    self.lmax=cosmo_history.lmax
    self.Delta_ell=cosmo_history.Delta_ell


  def pklin(self,k):
    return self.it_mt_camb(k)


  def n_slope(self,makeplot=1):
    [kmin,kmax]=self.k_range
    numk=self.numk
    k=np.logspace(np.log10(kmin),np.log10(kmax),numk)
    #get Pk
    fac=k**3/2.0/pi**2
    mp=self.it_mt_camb(k)/fac

    nk=np.zeros(np.shape(k))
    for i in range(len(k)-1):
      k1=k[i]
      k2=k[i+1]
      p1=mp[i]
      p2=mp[i+1]
      deriv=(np.log(p2)-np.log(p1))/(np.log(k2)-np.log(k1))
      nk[i]=deriv

    #assume the last one has the same slope as the previous 
    nk[-1]=nk[len(k)-2]

    if makeplot==1:
      fig = plt.figure(figsize=(10,9))
      plt.clf()
      plt.xscale('log')
      #plt.yscale('log')
      #plt.xlim(1e-3,1e2)
      #plt.ylim(1e-6,1e3)
      plt.plot(k, nk, 'r', label=r'$n(k)$')

      #plt.plot(p1h[:,0],p1h[:,1],label='paper 1h')
      #plt.plot(p2h[:,0],p2h[:,1],label='paper 2h')

      plt.legend(loc="upper left", ncol=1, shadow=True,  fancybox=True, frameon=False,prop={'size':15});
      plt.savefig('test_nk.pdf')

    return k,nk


  def root_kNL(self,kNL,a):
    D=self.it_growth(a)
    fac=kNL**3/2.0/pi**2
    Pk=self.it_mt_camb(kNL)/fac
    return 4.0*pi*kNL**3*D**2*Pk-1.0

  def get_kNL(self,a):
    sol=fsolve(self.root_kNL,0.1, args=(a))
    kNL=sol[0]
    return kNL

  def prepare_kNL(self,makeplot=1):
    a=self.a_growth
    list_kNL=[]

    [zmin,zmax]=self.z_eff
    num=1000
    z=np.logspace(np.log10(zmin),np.log10(zmax),num)
    a=z2a(z)
    for i in range(len(a)):
      kNL=self.get_kNL(a[i])
      list_kNL.append([z[i],kNL])

    list_kNL=np.asarray(list_kNL)
    it_kNL=InterpolatedUnivariateSpline(list_kNL[:,0], list_kNL[:,1],k=self.interp_order)

    if makeplot==1:
      fig = plt.figure(figsize=(10,9))
      plt.clf()
      #plt.xscale('log')
      #plt.yscale('log')
      #plt.xlim(1e-3,1e2)
      #plt.ylim(1e-6,1e3)
      plt.plot(list_kNL[:,0], list_kNL[:,1], 'r', label=r'$k_{\rm{NL}}$')

      #plt.plot(p1h[:,0],p1h[:,1],label='paper 1h')
      #plt.plot(p2h[:,0],p2h[:,1],label='paper 2h')

      plt.legend(loc="upper left", ncol=1, shadow=True,  fancybox=True, frameon=False,prop={'size':15});
      plt.savefig('test_kNL.pdf')

    return list_kNL,list_kNL[0,0],list_kNL[-1,0],it_kNL



  def Qn(self,n):
    q=(4.0-np.power(2.0,n))/(1.0+np.power(2.0,n+1.0))
    return q

  def coef_abc(self,k,z):
    n=self.it_nk(k)
    D=self.it_growth(z2a(z))
    s8=D*self.sigma8
    Q=self.Qn(n)
    kNL=self.it_kNL(z)
    q=k/kNL

    #print n,D,s8,Q,kNL,q

    x=1.0+np.power(s8,-0.2)*np.sqrt(0.7*Q)*np.power(q/4.0,n+3.5)
    y=1.0+np.power(q/4.0,n+3.5)
    ca=x/y

    x=1.0+0.4*(n+3.0)*np.power(q,n+3.0)
    y=1.0+np.power(q,n+3.5)
    cb=x/y


    x=1.0+4.5/(1.5+(n+3.0)**4)*np.power(2.0*q,n+3.0)
    y=1.0+np.power(2.0*q,n+3.5)
    cc=x/y

    return ca,cb,cc


  def Mkk(self,k,z):
    pair=triangle(k)
    m=[]
    for i in range(3):
      ki,kj,theta,x=pair[i]
      cai,cbi,cci=self.coef_abc(ki,z)
      caj,cbj,ccj=self.coef_abc(kj,z)
      v=10.0/7.0*cai*caj+cbi*cbj*(ki/kj+kj/ki)*x+4.0/7.0*cci*ccj*x**2
      m.append([ki,kj,v])
      #print v,x
    return np.asarray(m)

  def bispec(self,k,z):
    m=self.Mkk(k,z)
    #print np.shape(m)
    D=self.it_growth(z2a(z))

    inte=0.0
    for i in range(3):
      [ki,kj,v]=m[i]
      pki=self.it_2d_NLPower(ki,z)
      pkj=self.it_2d_NLPower(kj,z)
      faci=ki**3/2.0/pi**2
      facj=kj**3/2.0/pi**2
      pki/=faci
      pkj/=facj

      inte+=v*pki*pkj
    return inte


  def bispec_proj_integrand(self,ell,z,dz,id_zband):
    chi=self.it_chi(z)
    dchidz=self.it_dchidz(z)
    k=ell/chi
    #print k
    [i_z,j_z,k_z]=id_zband
    wi=self.it_W[i_z](z)
    wj=self.it_W[j_z](z)
    wk=self.it_W[k_z](z)
    #the nonlinear power spectrum is already z-dependent
    D=1.0
    b=self.bispec(k,z)
    return 1.0/chi**4*D**4*b*dchidz*dz*wi*wj*wk

  def bispec_proj(self,ell,id_zband):
    L1,L2,L3=ell
    [zmin,zmax]=self.z_eff
    numz=50
    #z=np.linspace(zmin,zmax,numz)
    z=np.logspace(np.log10(zmin),np.log10(zmax),numz)
    dz=[z[i+1]-z[i] for i in range(numz-1)]
    dz.append(0.0)
    dz=np.asarray(dz)

    #dz=(zmax-zmin)/numz
    #inte=self.bispec_proj_integrand(ell,z,dz,id_zband)
    inte=[self.bispec_proj_integrand(ell,z[i],dz[i],id_zband) for i in range(len(z))]
    result=np.sum(np.asarray(inte))
    return result*hfac(ell,[0.0,0.0,0.0])




  def test_bispec_proj(self,makeplot=1):
    [ell_min,ell_max]=[10.0,1000.0]
    num=10
    ell=np.linspace(ell_min,ell_max,num)

    data=[]
    for i in range(len(ell)):
      L=ell[i]
      fac=L*(L+1.0)/2.0/pi
      b=self.bispec_proj([L,L,L],[0,1,0])
      b=abs(b)
      power=fac**2*b
      data.append([L,power,b])
      print L,power
    data=np.asarray(data)

    if makeplot==1:
      #digi=np.loadtxt('../digi_paper_data/clkappa_lcdm')
      fig = plt.figure(figsize=(10,9))
      plt.clf()
      #plt.xscale('log')
      #plt.yscale('log')
      plt.plot(data[:,0], data[:,1], 'b', label='bispec')
      #plt.plot(digi[:,0], digi[:,1], 'r', label=r'$paper$')
      plt.legend(loc="upper left", ncol=1, shadow=True,  fancybox=True, frameon=False,prop={'size':15});
      plt.savefig('test_bl.pdf')



  def cl_kappa_integrand(self,case):
    [i_z,j_z,ell,z,dz]=case
    a=z2a(z)

    chi=self.it_chi(z)
    dchidz=self.it_dchidz(z)
    k=ell/chi
    #print k
    wi=self.it_W[i_z](z)
    wj=self.it_W[j_z](z)


    #get Pk
    fac=k**3/2.0/pi**2
    mp=self.it_mt_camb(k)/fac
    D=self.it_growth(z2a(z))

    return dchidz*dz*D**2*mp*(wi/chi)*(wj/chi)

  '''
  def cl_kappa(self,ell):
    [zmin,zmax]=self.z_eff
    numz=10000
    z=np.linspace(zmin,zmax,numz)
    dz=(zmax-zmin)/numz

    inte=[self.cl_kappa_integrand(ell,z[i],dz) for i in range(len(z))]
    result=np.sum(np.asarray(inte),axis=0)
    return result
  '''
  def cl_kappa(self,ell,id_zband):
    [i_z,j_z]=id_zband
    [zmin,zmax]=self.z_eff
    numz=50
    z=np.logspace(np.log10(zmin),np.log10(zmax),numz)
    #dz=(zmax-zmin)/numz
    dz=[z[i+1]-z[i] for i in range(numz-1)]
    dz.append(0.0)
    dz=np.asarray(dz)
    allcase=[[i_z,j_z,ell,z[i],dz[i]] for i in range(numz)]

    inte=[self.cl_kappa_integrand(allcase[i]) for i in range(len(z))]
    result=np.sum(np.asarray(inte))
    return result


  def get_WL_nl(self,id_zband):
    shot=self.shot_tomo
    nl=0.0
    [i,j]=id_zband
    if i!=j:
      nl=0.0
    else:
      nl=shot[i]
    return nl

  def test_cl_kappa_pre(self,id_zband,ell_min,ell_max):
    num=20
    ell=np.logspace(np.log10(ell_min),np.log10(ell_max),num)


    data=[]
    for i in range(len(ell)):
      fac=ell[i]*(ell[i]+1.0)/2.0/pi
      cl=self.cl_kappa(ell[i],id_zband)
      nl=self.get_WL_nl(id_zband)
      data.append([ell[i],fac*cl,fac*nl,cl,nl,cl+nl])
      print ell[i],fac*cl,fac*nl
    data=np.asarray(data)
    return data


  def test_cl_kappa(self,makeplot=1):
    [ell_min,ell_max]=[10.0,1000.0]
    id_zband=[0,0]
    data00=self.test_cl_kappa_pre(id_zband,ell_min,ell_max)
    id_zband=[0,1]
    data01=self.test_cl_kappa_pre(id_zband,ell_min,ell_max)
    id_zband=[1,1]
    data11=self.test_cl_kappa_pre(id_zband,ell_min,ell_max)

    if makeplot==1:
      digi=np.loadtxt('../digi_paper_data/clkappa_lcdm')
      fig = plt.figure(figsize=(10,9))
      plt.clf()
      plt.xscale('log')
      plt.yscale('log')
      plt.plot(data00[:,0], data00[:,1], 'b', label='00')
      plt.plot(data00[:,0], data00[:,2], 'b', label='00',linestyle='--')
      plt.plot(data01[:,0], data01[:,1], 'g', label='01')
      plt.plot(data11[:,0], data11[:,1], 'r', label='11')
      plt.plot(data11[:,0], data11[:,2], 'r', label='11',linestyle='--')
      plt.plot(digi[:,0], digi[:,1], 'm', label=r'$paper$')
      plt.tick_params('both',length=12,width=0.5,labelsize=26,which='major')
      plt.tick_params('both',length=6,width=0.5,labelsize=26,which='minor')

      plt.legend(loc="upper left", ncol=1, shadow=True,  fancybox=True, frameon=False,prop={'size':15});
      plt.savefig('test_cl_kappa.pdf')



  def delta_cov(self,ell):
    [l1,l2,l3]=ell
    w=1.0
    if l1==l2 and l1==l3:
      w=6.0
    if l1==l2 and l1!=l3:
      w=2.0
    if l1==l3 and l1!=l2:
      w=2.0
    return w


  def cov3p(self,ell,id_zband_1,id_zband_2):
    [l1,l2,l3]=ell
    [i,j,k]=id_zband_1
    [l,m,n]=id_zband_2
    cl_A=self.cl_kappa(l1,[i,l])
    nl_A=self.get_WL_nl([i,l])

    cl_B=self.cl_kappa(l2,[j,m])
    nl_B=self.get_WL_nl([j,m])

    cl_C=self.cl_kappa(l3,[k,n])
    nl_C=self.get_WL_nl([k,n])

    d=self.delta_cov(ell)

    return d/self.fsky*(cl_A+nl_A)*(cl_B+nl_B)*(cl_C+nl_C)


  #Now Fisher Matrix calculation
  def prepare_report(self):
    LoopZBandCl=self.LoopZBandCl
    LoopZBandBl=self.LoopZBandBl
    Combi_ell=self.Combi_ell
    ell_min,ell_max=self.lmin,self.lmax

    npair_cl,xx=np.shape(LoopZBandCl)
    npair_bl_zband,xx=np.shape(LoopZBandBl)
    npair_bl_ell,xx=np.shape(Combi_ell)

    #print 'preparing cl....'
    #PoolCl=[self.test_cl_kappa_pre(LoopZBandCl[i],ell_min,ell_max) for i in range(npair_cl)]

    print 'preparing Bl....'
    PoolBl=[]
    PoolCov=[]
    for i in range(npair_bl_ell):
      t1=time.time()
      bl=[self.bispec_proj(Combi_ell[i],LoopZBandBl[j]) for j in range(npair_bl_zband)]
      cov=[self.cov3p(Combi_ell[i],LoopZBandBl[j],LoopZBandBl[k]) for j in range(npair_bl_zband) for k in range(npair_bl_zband)]
      t2=time.time()
      print Combi_ell[i]#, bl, np.reshape(cov,(npair_bl_zband,npair_bl_zband))
      PoolBl.append(bl)
      PoolCov.append(np.reshape(cov,(npair_bl_zband,npair_bl_zband)))
    PoolBl=np.asarray(PoolBl)
    PoolCov=np.asarray(PoolCov)
    return PoolBl,PoolCov

