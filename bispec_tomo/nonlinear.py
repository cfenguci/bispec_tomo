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

from bispec_tomo import cosmo_history
from bispec_tomo import z2a,a2z,tophat

from scipy.interpolate import RectBivariateSpline

class nonlinear():
  def __init__(self,cosmo_history):
    #WMAP 5
    self.name='matter powerspectrum'
    self.copy_cosmo_params(cosmo_history)
    self.method=cosmo_history.interp_method
    self.interp_order=cosmo_history.interp_order


    self.a_growth,self.growth=cosmo_history.a_growth,cosmo_history.growth
    self.it_growth=InterpolatedUnivariateSpline(self.a_growth,self.growth,k=self.interp_order)
    self.z_eff=cosmo_history.z_eff

    self.matter_norm=1.0
    #by doing this, the 'matter_norm' is assigned new value
    self.matter_norm=self.get_sigma8()

    [kmin_camb,kmax_camb,self.it_mt_camb]=self.Delta_pk_lin_CAMB()
    self.k_range=[1.1e-4,800]
    self.numk=1e3

    self.it_k_sigma,self.it_C,self.it_neff=self.prepare_abcdefg()
    self.allfxb=self.fxb()
    self.it_2d_NLPower=self.test_pkNL(makeplot=1)




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



  def Delta_pk_simple(self,k):
    #not validated 
    norm=self.matter_norm
    Omega_m=self.Omega_m
    Omega_b=self.Omega_b
    h=self.h
    k0=self.k0
    ns=self.ns
    As=self.As

    Gamma_s=Omega_m*h*np.exp(-Omega_b*(1.0+np.sqrt(2.0*h)/Omega_m))
    q=k/Gamma_s
    ratio=1.0+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4
    T=np.log(1.0+2.34*q)/(2.34*q)*np.power(ratio,-1.0/4.0)
    Pk=np.power(k,ns)*T**2
    Delta=Pk*norm*k**3/2.0/pi**2
    return Delta

  def inte_tophat(self,k,dlnk):
    Delta2=self.Delta_pk_simple(k)
    h=self.h
    R=8.0/h
    w=tophat(k*R)
    return dlnk*w**2*Delta2

  def get_sigma8(self):
    s8=self.sigma8
    kmin,kmax,numk=1e-6,800,5000
    k=np.logspace(np.log10(kmin),np.log10(kmax),numk)
    dlnk=[np.log(k[i+1])-np.log(k[i]) for i in range(numk-1)]
    dlnk.append(0.0)
    dlnk=np.asarray(dlnk)

    #inte=[self.inte_tophat(k[i],dlnk[i]) for i in range(numk)]
    inte=self.inte_tophat(k,dlnk)
    matter_norm=s8**2/np.sum(inte)
    print 'Matter power norm=',matter_norm
    return matter_norm

  def Delta_pk_lin_CAMB(self):
    data=np.loadtxt('../test_matterpower.dat')
    k=data[:,0]
    Delta2=self.Delta_pk_simple(k)   

    it_mt_camb=InterpolatedUnivariateSpline(k,Delta2,k=self.interp_order)
    print 'CAMB kmin=',k[0],' kmax=',k[-1]
    return [k[0],k[-1],it_mt_camb]
    
  def test_Delta_pk(self,makeplot=1):
    self.Delta_pk_lin_CAMB()
    k=np.logspace(np.log10(8e-5),np.log10(80.0),100)

    fac=k**3/2.0/pi**2
    pkcamb=self.it_mt_camb(k)/fac
    pkfit=self.Delta_pk_simple(k)/fac


    if makeplot==1:
      fig = plt.figure(figsize=(10,9))
      plt.clf()
      plt.xscale('log')
      plt.yscale('log')
      #plt.xlim(1e-3,1e2)
      #plt.ylim(1e-6,1e3)
      plt.plot(k, pkcamb, 'r', label='camb')
      plt.plot(k, pkfit, 'b', label='fit')

      #plt.plot(p1h[:,0],p1h[:,1],label='paper 1h')
      #plt.plot(p2h[:,0],p2h[:,1],label='paper 2h')

      plt.legend(loc="upper left", ncol=1, shadow=True,  fancybox=True, frameon=False,prop={'size':15});
      plt.savefig('test_pk_camb_fit.pdf')


  def Delta_pk_lin(self,k):
    return self.it_mt_camb(k)

  def func_sigma2_inte(self,k,kmin,kmax,numk,R_G,z):    
    D=self.it_growth(z2a(z))
    inte=D*self.Delta_pk_lin(k)*np.exp(-(k*R_G)**2)
    return inte

 
  #the variance integration function
  def sigma2_inte(self,kmin,kmax,numk,R_G,z):
    k=np.logspace(np.log10(kmin),np.log10(kmax),numk)
    inte=0.0
    for i in range(len(k)-1):
      dk=np.log(k[i+1])-np.log(k[i])
      inte+=dk*self.func_sigma2_inte(k[i],kmin,kmax,numk,R_G,z)
    return R_G,inte
 
  '''
  def sigma2_inte(self,kmin,kmax,numk,R_G):
    y=quad(self.func_sigma2_inte,kmin,kmax,args=(kmin,kmax,numk,R_G))
    return y[0]
  '''

  def func_k_sigma_inte(self,ksigma,kmin,kmax,numk,z):
    R_G=1.0/ksigma
    x,y=self.sigma2_inte(kmin,kmax,numk,R_G,z)
    #y=quad(self.func_sigma2_inte,kmin,kmax,args=(kmin,kmax,numk,R_G))
    #return self.sigma2_inte(kmin,kmax,numk,R_G)-1.0
    return y-1.0

  def get_k_sigma_inte(self,kmin,kmax,numk,z):
    sol = fsolve(self.func_k_sigma_inte, 1e-3, args=(kmin,kmax,numk,z))
    return sol[0]




  def dlnsigma2_dlnR2(self,kmin,kmax,numk,z,ksigma):

    dk=1e-1
    R_G_0=1.0/ksigma
    dlnR=np.log(R_G_0)/1000.0

    R_G_1=np.exp(np.log(R_G_0)+dlnR/2.0)
    R_G_2=np.exp(np.log(R_G_0)-dlnR/2.0)

    x0,y0=self.sigma2_inte(kmin,kmax,numk,R_G_0,z)
    x1,y1=self.sigma2_inte(kmin,kmax,numk,R_G_1,z)
    x2,y2=self.sigma2_inte(kmin,kmax,numk,R_G_2,z)
    x0,y0=np.log(x0),np.log(y0)
    x1,y1=np.log(x1),np.log(y1)
    x2,y2=np.log(x2),np.log(y2)

    #deriv=(np.log(y1)--np.log(y2))/(np.log(R_G_1**2)-np.log(R_G_2**2))
    deriv=(y1-2.0*y0+y2)/dlnR**2
    return -deriv


  #neff
  def dlnsigma2_dlnR(self,kmin,kmax,numk,z,ksigma):

    R_G_0=1.0/ksigma
    dlnR=np.log(R_G_0)/1000.0

    R_G_1=np.exp(np.log(R_G_0)+dlnR/2.0)
    R_G_2=np.exp(np.log(R_G_0)-dlnR/2.0)
    x1,y1=self.sigma2_inte(kmin,kmax,numk,R_G_1,z)
    x2,y2=self.sigma2_inte(kmin,kmax,numk,R_G_2,z)
    x1,y1=np.log(x1),np.log(y1)
    x2,y2=np.log(x2),np.log(y2)
    deriv=(y1-y2)/dlnR
    return -deriv-3.0


  def prepare_abcdefg(self):
    [zmin,zmax]=self.z_eff
    numz=10
    z=np.logspace(np.log10(zmin),np.log10(zmax),numz)

    data=[]
    for i in range(len(z)):
      print 'nonlinear->prepare_abcdefg',z[i],i

      k_sigma=self.get_k_sigma_inte(self.k_range[0],self.k_range[1],self.numk,z[i])
      C=self.dlnsigma2_dlnR2(self.k_range[0],self.k_range[1],self.numk,z[i],k_sigma)
      neff=self.dlnsigma2_dlnR(self.k_range[0],self.k_range[1],self.numk,z[i],k_sigma)
      data.append([z[i],k_sigma,C,neff])
    data=np.asarray(data)
    print 'prepared k_sigma, C, neff'

    it_k_sigma=InterpolatedUnivariateSpline(data[:,0],data[:,1],k=self.interp_order)
    it_C=InterpolatedUnivariateSpline(data[:,0],data[:,2],k=self.interp_order)
    it_neff=InterpolatedUnivariateSpline(data[:,0],data[:,3],k=self.interp_order)
    return it_k_sigma,it_C,it_neff

  def fxb(self):    
    omega=self.Omega_m
    fa=np.power(omega,-0.0307)
    fb=np.power(omega,-0.0585)
    fc=np.power(omega,0.0743)
    return [fa,fb,fc]


  def abcdefg(self,z):
    k_sigma=self.it_k_sigma(z)
    C=self.it_C(z)
    neff=self.it_neff(z)
    n=neff    
    a=1.4861+1.8369*n+1.6762*n**2+0.7940*n**3+0.1670*n**4-0.6206*C 
    b=0.9463+0.9466*n+0.3084*n**2-0.9400*C
    c=-0.2807+0.6669*n+0.3214*n**2-0.0793*C
    r=0.8649+0.2989*n+0.1631*C
    alpha=1.3884+0.37*n-0.1452*n**2
    beta=0.8291+0.9854*n+0.3401*n**2
    mu=-3.5442+0.1908*n
    nu=0.9589+1.2857*n

    '''
    a=1.5222+2.8553*n+2.3706*n**2+0.9903*n**3+0.2250*n**4-0.6038*C
    b=-0.5642+0.5864*n+0.5716*n**2-1.5474*C
    c=0.3698+2.0404*n+0.8161*n**2+0.5869*C
    r=0.1971-0.0843*n+0.846*C
    alpha=abs(6.0835+1.3373*n-0.1959*n**2-5.5274*C)
    beta=2.0379-0.7354*n+0.3157*n**2+1.249*n**3+0.398*n**4-0.1682*C
    mu=0.0
    nu=5.2105+3.6902*n
    '''
    a=np.power(10.0,a)
    b=np.power(10.0,b)
    c=np.power(10.0,c)
    mu=np.power(10.0,mu)
    nu=np.power(10.0,nu)
    return [a,b,c,r,alpha,beta,mu,nu,k_sigma,C,neff]


  def f(self,y):
    return y/4.0+y**2/8.0


  def Delta2_Q(self,k,z):
    [a,b,c,r,alpha,beta,mu,nu,k_sigma,C,neff]=self.abcdefg(z)#self.allcoef
    ks=k_sigma
    y=k/ks
    D=self.it_growth(z2a(z))
    D2lin=self.Delta_pk_lin(k)*D**2
    return D2lin*(np.power(1.0+D2lin,beta)/(1.0+alpha*D2lin))*np.exp(-self.f(y))

  def Delta2p_H(self,k,z):
    [a,b,c,r,alpha,beta,mu,nu,k_sigma,C,neff]=self.abcdefg(z)
    [fa,fb,fc]=self.allfxb
    ks=k_sigma
    y=k/ks
    return a*np.power(y,3.0*fa)/(1.0+b*np.power(y,fb)+np.power(c*fc*y,3.0-r))

  def Delta2_H(self,k,z):
    [a,b,c,r,alpha,beta,mu,nu,k_sigma,C,neff]=self.abcdefg(z)
    ks=k_sigma
    y=k/ks
    return self.Delta2p_H(k,z)/(1.0+mu/y+nu/y**2)



  def test_pkNL(self,makeplot=1):
    [kmin,kmax]=self.k_range
    [zmin,zmax]=self.z_eff
    numk,numz=100.0,100.0
    k=np.logspace(np.log10(kmin),np.log10(kmax),numk)
    z=np.logspace(np.log10(zmin),np.log10(zmax),numz)
    Z,K=np.meshgrid(z,k)

    (nx,ny)=np.shape(Z)
    dd=np.zeros((nx,ny))
    dd_1h=np.zeros((nx,ny))
    dd_2h=np.zeros((nx,ny))
    for i in range(nx):
      for j in range(ny):
        dq=self.Delta2_Q(K[i][j],Z[i][j])
        dh=self.Delta2_H(K[i][j],Z[i][j])
        dd[i][j]=dq+dh
        dd_2h[i][j]=dq
        dd_1h[i][j]=dh

    it_2d_NLPower = RectBivariateSpline(k,z,dd)
    it_2d_NLPower_1h = RectBivariateSpline(k,z,dd_1h)
    it_2d_NLPower_2h = RectBivariateSpline(k,z,dd_2h)



    if makeplot==1:

      kk=np.logspace(np.log10(0.1),np.log10(10),100)
      pp1,pp1_1h,pp1_2h=it_2d_NLPower(kk,0),it_2d_NLPower_1h(kk,0),it_2d_NLPower_2h(kk,0)
      '''
      pp2=it_2d_NLPower(kk,0)
      pp3=it_2d_NLPower(kk,0.5)
      pp4=it_2d_NLPower(kk,1)
      pp5=it_2d_NLPower(kk,3)
      '''
      #p1h=np.loadtxt('../digi_paper_data/halo_fit_1h')
      #p2h=np.loadtxt('../digi_paper_data/halo_fit_2h')

      s1h=np.loadtxt('../digi_paper_data/smith_fig14_SCDM_z3_1h')
      s2h=np.loadtxt('../digi_paper_data/smith_fig14_SCDM_z3_2h')

    
      fig = plt.figure(figsize=(10,9))
      plt.clf()
      plt.xscale('log')
      plt.yscale('log')
      plt.xlim(1e-3,1e2)
      plt.ylim(1e-6,1e3)
      #plt.plot(k, dq, 'r', label='2h')
      #plt.plot(k, dh, 'b', label='1h')
      plt.plot(kk, pp1)
      plt.plot(kk, pp1_1h)
      plt.plot(kk, pp1_2h)
      '''
      plt.plot(kk, pp2)
      plt.plot(kk, pp3)
      plt.plot(kk, pp4)
      plt.plot(kk, pp5)
      '''

      plt.plot(s1h[:,0]*self.h,s1h[:,1],'b',label='paper 1h',linestyle='--')
      plt.plot(s2h[:,0]*self.h,s2h[:,1],'r',label='paper 2h',linestyle='--')

      plt.legend(loc="upper left", ncol=1, shadow=True,  fancybox=True, frameon=False,prop={'size':15});
      plt.savefig('test_PK.pdf')
   
    return it_2d_NLPower

