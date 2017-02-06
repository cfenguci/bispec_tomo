import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import pi, cos, sin
from matplotlib.patches import Ellipse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math

from scipy.integrate import odeint
from scipy.integrate import quad

LIGHT_SPEED=299792458.0 #m/s
_G_=6.67428e-11 #m^3*kg^{-1}*s^{-2}
_H2second_=0.3236e-17
Mpc=3.09e22 #[m]

def z2a(z):
  return 1.0/(1.0+z)

def a2z(a):
  return 1.0/a-1.0

def get_rho_crit_0(h):
  h_with_unit=h*_H2second_
  rho_0=3.0*h_with_unit*h_with_unit/8.0/pi/_G_ #//kg/m^3

  #rho_0*=(Mpc/h)**3
  print 'critical density today=',rho_0  
  return rho_0


def func_w(z,w0,wa):
  a=1.0/(1.0+z)
  return w0+(1.0-a)*wa


def get_all_range(k,x,y,method):
  x1=[]
  fx=InterpolatedUnivariateSpline(x[:],y[:],kind=method)
  for i in range(len(k)):
    if k[i]>=x[0] and k[i]<=x[len(x)-1]:
      #print z[i],fx(z[i])
      v=fx(k[i])
    else:
      #print z[i],0
      v=0.0

    x1.append(v)
  return np.asarray(x1)

def integrate(f, a, b, N,p):
    x = np.linspace(a, b, N)
    fx = f(x,p)
    area = np.sum(fx)*(b-a)/N
    return area


def tophat(x):
  return 3.0/x**3*(sin(x)-x*cos(x))


def get_dx(x):
  num=len(x)
  dx=[x[i+1]-x[i] for i in range(num-1)]
  dx.append(0.0)
  return np.asarray(dx)



class cosmo_history():
  def __init__(self,params):
    self.name='cosmo history'

    '''
    self.h=0.72
    self.Gamma=0#1.0/3.0*self.H0
    self.w0=-1.0#-2.0/3.0#-0.98
    self.wa=0.0#-1.0/3.0
    self.Omega_m=0.25    
    self.sigma8=0.8
    '''
    self.h=params['h']
    self.Gamma=params['Gamma']
    self.w0=params['w0']
    self.wa=params['wa']
    self.Omega_m=params['omegam']   
    self.sigma8=params['sigma8']

    

    self.H0=self.h*100.0*1e3 #[m/s/(Mpc)]
    self.Gamma*=self.H0
    self.Omega_phi=1.0-self.Omega_m
    self.Omega_b=0.0441
    self.As=2.1e-9
    self.ns=1.0
    self.k0=0.05 #Mpc^{-1}

    #weak lensing
    self.z0=0.64
    self.beta=3.0/2.0
    self.p0=self.z0/self.beta*math.gamma(3.0/self.beta)
    self.p0=1.0/self.p0
    self.z_med=0.9
    self.ngal=4.7e8
    self.sigma_shot=0.3
    self.fsky=1.0/2.0

    self.rho_crit_0=get_rho_crit_0(self.h)

    self.z_min=0.0 
    self.z_max=1e3
    self.numz=1e4
    self.z = np.linspace(self.z_min,self.z_max, self.numz)


    self.invz = np.linspace(self.z_max,self.z_min, self.numz)

    zoffset=1e-4
    self.a_min=1.0/(1.0+self.z_max-zoffset)
    self.a_max=1.0/(1.0+self.z_min+zoffset)
    self.numa=1e4
    self.a = np.logspace(np.log10(self.a_min),np.log10(self.a_max), self.numa)
    print 1.0/self.a-1.0

    self.get_evol(makeplot=0)

    self.interp_method='linear'
    self.interp_order=1
    self.it_rho_m_z,self.it_rho_phi_z,self.it_H_z,self.it_chi,self.it_dchidz=self.prepare_interpolator()
    #kg/m^3,        kg/m^3,           m/s/Mpc,    Mpc,        Mpc

    self.a_growth,self.growth=self.get_Dplus(makeplot=1)
    self.z_growth=1.0/self.a_growth-1.0
    self.z_eff=[self.z_growth[-1],self.z_growth[0]]
    print 'effective z=',self.z_eff

    self.z_tomo=[[self.z_eff[0],self.z_med],[self.z_med,self.z_eff[1]]]
    self.lmin=100
    self.lmax=2000
    self.Delta_ell=100
    self.LoopZBandCl,self.LoopZBandBl,self.Combi_ell=self.prepare_FisherMat(self.lmin,self.lmax,self.Delta_ell)

    self.shot_tomo=self.prepare_ngal_tomo()
    self.it_G=self.prepare_G_tomo(makeplot=1)
    self.it_W=self.prepare_W_tomo(makeplot=1)


  def func_evol(self,y,z):
    a=1.0/(1.0+z)
    Gamma,w0,wa,rho_crit_0,H0=self.Gamma,self.w0,self.wa,self.rho_crit_0,self.H0

    vw=func_w(z,w0,wa)    

    [rho_m,rho_phi,H,chi]=y

    decay=Gamma

    rho_crit_z=rho_crit_0*(H/H0)**2
    Omega=rho_m/rho_crit_z

    f1=3.0*a*rho_m+a*decay/H*rho_m
    f2=3.0*a*(1.0+vw)*rho_phi-a*decay/H*rho_m

    unit=rho_crit_0/H0**2/3.0
    f3=a/2.0/H*rho_m+a/2.0/H*(1.0+vw)*rho_phi
    f3/=unit

    f6=LIGHT_SPEED/H #[Mpc]
    dydz=[f1,f2,f3,f6]

    return dydz


  def func_growth(self,y,a):
    [D_plus,xi]=y

    Gamma,w0,wa,rho_crit_0,H0=self.Gamma,self.w0,self.wa,self.rho_crit_0,self.H0
    z,vw,H,rho_m,rho_phi,Omega,dHdz=self.z_slice(a)

    f1=xi
    f2=-(3.0-1.0/a/H*dHdz)/a*xi+3.0/2.0*Omega*D_plus/a**2

    dydz=[f1,f2]
    return dydz

  def z_slice(self,a):
    z=1.0/a-1.0
    Gamma,w0,wa,rho_crit_0,H0=self.Gamma,self.w0,self.wa,self.rho_crit_0,self.H0
    vw=func_w(z,w0,wa) 

    H=self.it_H_z(z)
    rho_m=self.it_rho_m_z(z)
    rho_phi=self.it_rho_phi_z(z)

    rho_crit_z=rho_crit_0*(H/H0)**2
    Omega=rho_m/rho_crit_z
    unit=rho_crit_0/H0**2/3.0
    dHdz=a/2.0/H*rho_m+a/2.0/H*(1.0+vw)*rho_phi
    dHdz/=unit

    return z,vw,H,rho_m,rho_phi,Omega,dHdz

  def get_init_growth(self):
    z=self.z_max
    a=1.0/(1.0+z)

    Gamma,w0,wa,rho_crit_0,H0=self.Gamma,self.w0,self.wa,self.rho_crit_0,self.H0
    z1,vw,H,rho_m,rho_phi,Omega,dHdz=self.z_slice(a)

    c0=1.0
    c1=2.0+3.0-1.0/a/H*dHdz
    c2=3.0/2.0*Omega

    DD=np.sqrt(c1**2-4.0*c0*c2)
    alpha=(-c1+DD)/2.0/c0
    #print 'scaling=',alpha
    return alpha

  def get_evol(self,makeplot=1):
    Gamma=self.Gamma
    w0=self.w0
    wa=self.wa
    y0 = [self.rho_crit_0*self.Omega_m,self.rho_crit_0*self.Omega_phi,self.H0,0.0]

    sol = odeint(self.func_evol,y0,self.z)
    self.rho_m_z=sol[:,0]
    self.rho_phi_z=sol[:,1]
    self.H_z=sol[:,2]
    self.chi=sol[:,3]
    if makeplot:
      z=self.z
      rho_m_z_analytical=self.rho_crit_0*self.Omega_m*(1.0+z)**3
      fig = plt.figure(figsize=(10,9))
      plt.clf()
      plt.xscale('log')
      plt.yscale('log')
      plt.plot(z, sol[:, 0], 'b', label=r'$\rho_m$')
      plt.plot(z, rho_m_z_analytical, 'r', label=r'$\rho_m\,\rm{analytical}$')
      plt.plot(z, sol[:, 1], 'g', label=r'$\rho_{\phi}$')
      
      #plt.plot(z, RD, 'r', label=r'$D_{+}$')
      plt.legend(loc="upper left", ncol=1, shadow=True,  fancybox=True, frameon=False,prop={'size':15});
      plt.savefig('test_evol.pdf')


      H_analytic=self.H0*np.sqrt(self.Omega_m*(1.0+self.z)**3+self.Omega_phi)

      fig = plt.figure(figsize=(10,9))
      plt.clf()
      #plt.xscale('log')
      #plt.yscale('log')
      #plt.plot(z, H_analytic, 'b', label=r'$analytical\,H(z)$')
      #plt.plot(z, self.H_z, 'r', label=r'$H(z)$',lw=3)
      plt.plot(z, self.H_z-H_analytic, 'r', label=r'$H(z)$',lw=3)
      #plt.plot(z, RD, 'r', label=r'$D_{+}$')
      plt.legend(loc="upper left", ncol=1, shadow=True,  fancybox=True, frameon=False,prop={'size':15});
      plt.savefig('test_evolH.pdf')


  def prepare_interpolator(self):
    it_rho_m_z=InterpolatedUnivariateSpline(self.z,self.rho_m_z,k=self.interp_order)
    it_rho_phi_z=InterpolatedUnivariateSpline(self.z,self.rho_phi_z,k=self.interp_order)
    it_H_z=InterpolatedUnivariateSpline(self.z,self.H_z,k=self.interp_order)
    it_chi=InterpolatedUnivariateSpline(self.z,self.chi,k=self.interp_order)

    dchidz=LIGHT_SPEED/self.H_z #[Mpc]
    it_dchidz=InterpolatedUnivariateSpline(self.z,dchidz,k=self.interp_order)

    return it_rho_m_z,it_rho_phi_z,it_H_z,it_chi,it_dchidz

  def redshift_distri(self,z):
    z0=self.z0
    p0=self.p0
    beta=self.beta
    return p0*(z/z0)**2*np.exp(-np.power(z/z0,beta))


  def get_Dplus(self,makeplot=1):
    Gamma=self.Gamma
    w0=self.w0
    wa=self.wa    

    a0=self.a_min
    alpha=self.get_init_growth()

    y0 = [np.power(a0,alpha),alpha*np.power(a0,alpha-1.0)]
    sol = odeint(self.func_growth,y0,self.a)

    a=self.a

    amp=sol[-1,0]
    sol[:,0]/=amp
    print 'amplitude=',amp

    if makeplot:
      data_fig=np.loadtxt('../fig2_lcdm')
      data_1=np.loadtxt('../fig2_a')
      data_2=np.loadtxt('../fig2_b')
      data_3=np.loadtxt('../fig2_c')
      fig = plt.figure(figsize=(10,9))
      plt.clf()
      #plt.xscale('log')
      #plt.yscale('log')
      plt.plot(a, sol[:, 0], label=r'$D_{+}$',color='orange',lw=3)
      plt.plot(data_fig[:,0], data_fig[:,1], 'm', label='paper 0')
      plt.plot(data_1[:,0], data_1[:,1], 'g', label='paper 1')
      plt.plot(data_2[:,0], data_2[:,1], 'b', label='paper 2')
      plt.plot(data_3[:,0], data_3[:,1], 'r', label='paper 3')

      plt.legend(loc="upper left", ncol=1, shadow=True,  fancybox=True, frameon=False,prop={'size':15});
      plt.savefig('test_D.pdf')

    return a,sol[:, 0]

  def get_Omega(self,z):
    H0=self.H0
    H=self.it_H_z(z)
    rho_crit_0=self.rho_crit_0
    rho_m=self.it_rho_m_z(z)
    rho_phi=self.it_rho_phi_z(z)
    rho_crit_z=rho_crit_0*(H/H0)**2
    Omega=rho_m/rho_crit_z
    return Omega

  def func_G(self,z0,z,dz):
    chi=self.it_chi(z0)
    chip=self.it_chi(z)
    pz=self.redshift_distri(z)
    v=pz*(1.0-chi/chip)*dz
    return v  

  def get_G(self,id_zband,z_run):
    numz=5000
    #z=np.linspace(zmin,zmax,numz)
    [zmin_tomo,zmax_tomo]=self.z_tomo[id_zband]
    zmin_Real=max(z_run,zmin_tomo)
    #print 'Window in ',zmin_Real,zmax_tomo
    v=0.0
    if zmin_Real<zmax_tomo:
      z=np.logspace(np.log10(zmin_Real),np.log10(zmax_tomo),numz)
      dz=get_dx(z)
      num=len(z)
      inte1=self.func_G(z_run,z,dz)
      v=np.sum(np.asarray(inte1))
    return v


  def prepare_G(self,id_zband,makeplot=1):
    [zmin,zmax]=self.z_eff

    #zmax=0.9
    numz=200
    z=np.logspace(np.log10(zmin),np.log10(zmax),numz)
    data=np.asarray([[z[i],self.get_G(id_zband,z[i])] for i in range(len(z))])  
    #print data
    it_G=InterpolatedUnivariateSpline(data[:,0],data[:,1],k=self.interp_order)

    if makeplot==1:
      fig = plt.figure(figsize=(10,9))
      plt.clf()
      #plt.yscale('log')
      plt.xlim(0,2.0)
      plt.ylim(-1,1.2)
      plt.plot(data[:,0],data[:,1], 'b', label='w')
      plt.legend(loc="upper left", ncol=1, shadow=True,  fancybox=True, frameon=False,prop={'size':15})
      plt.savefig('test_G_%.2f-%.2f.pdf'%(self.z_tomo[id_zband][0],self.z_tomo[id_zband][1]))
    return it_G

  def prepare_G_tomo(self,makeplot=1):
    it_G_tomo=[]
    nbins_tomo,m=np.shape(self.z_tomo)
    for i in range(nbins_tomo):
      it=self.prepare_G(i,makeplot)
      it_G_tomo.append(it)
    return np.asarray(it_G_tomo)

  def func_W_tomo(self,id_zband,z,zmax):
    a=z2a(z)
    G=self.it_G[id_zband](z)
    chi=self.it_chi(z)

    Gamma,w0,wa,rho_crit_0,H0=self.Gamma,self.w0,self.wa,self.rho_crit_0,self.H0

    H=self.it_H_z(z)
    Omega=self.get_Omega(z)
    #Omega=self.Omega_m*H0**2/a**3/H**2
    return 3.0/2.0/LIGHT_SPEED**2*a**2*H**2*Omega*G*chi

  def prepare_W(self,id_zband,makeplot=1):
    [zmin,zmax]=self.z_eff
    numz=5000
    #z=np.linspace(zmin,zmax,numz)
    z=np.logspace(np.log10(zmin),np.log10(zmax),numz)


    data=[]
    for i in range(len(z)):
      chi=self.it_chi(z[i])
      W=self.func_W_tomo(id_zband,z[i],zmax)
      #print chi,W/chi
      data.append([z[i],chi,W])

    data=np.asarray(data)
    it_W=InterpolatedUnivariateSpline(data[:,0],data[:,2],k=self.interp_order)

    if makeplot==1:

      pz=self.redshift_distri(z)
      fig = plt.figure(figsize=(10,9))
      plt.clf()
      plt.xlim(0,3)
      plt.plot(z,pz, 'r', label='paper')  
      plt.legend(loc="upper left", ncol=1, shadow=True,  fancybox=True, frameon=False,prop={'size':15})
      plt.savefig('test_pz.pdf')

      digi=np.loadtxt('../digi_paper_data/fig3_w')
      fig = plt.figure(figsize=(10,9))
      plt.clf()
      plt.xscale('log')

      plt.plot(data[:,1]*self.h,data[:,2]/data[:,1]/self.h**2, 'b', label='w')
      plt.plot(digi[:,0],digi[:,1], 'r', label='paper')  
      plt.legend(loc="upper left", ncol=1, shadow=True,  fancybox=True, frameon=False,prop={'size':15})
      plt.savefig('test_W_%.2f-%.2f.pdf'%(self.z_tomo[id_zband][0],self.z_tomo[id_zband][1]))
    return it_W

  def prepare_W_tomo(self,makeplot=1):
    it_W_tomo=[]
    nbins_tomo,m=np.shape(self.z_tomo)
    for i in range(nbins_tomo):
      it=self.prepare_W(i,makeplot)
      it_W_tomo.append(it)
    return np.asarray(it_W_tomo)


  def func_ngal(self,z,dz):
    pz=self.redshift_distri(z)
    v=pz*dz
    return v

  def get_ngal(self,id_zband):
    numz=1000
    [zmin_tomo,zmax_tomo]=self.z_tomo[id_zband]
    z=np.logspace(np.log10(zmin_tomo),np.log10(zmax_tomo),numz)
    dz=get_dx(z)
    num=len(z)
    inte=self.func_ngal(z,dz)
    return np.sum(inte)

  def prepare_ngal_tomo(self):
    nbins_tomo,m=np.shape(self.z_tomo)
    ngal_tomo=np.asarray([self.get_ngal(i) for i in range(nbins_tomo)])
    print 'Galaxy number tomo', ngal_tomo

    sigma=self.sigma_shot
    n=self.ngal
    shot_tomo=sigma**2/n*ngal_tomo

    return np.asarray(shot_tomo)



  def check_triangle(self,ell):
    [l1,l2,l3]=ell
    flag=np.zeros(3)
    if abs(l1-l2)<=l3 and l3<=(l1+l2):
      flag[0]=1
    if abs(l1-l3)<=l2 and l2<=(l1+l3):
      flag[1]=1
    if abs(l2-l3)<=l1 and l1<=(l2+l3):
      flag[2]=1
    return flag[0]*flag[1]*flag[2]

  def prepare_FisherMat(self,lmin,lmax,dl):
    nbins_tomo,m=np.shape(self.z_tomo)
    id_zband_cl=[[i,j] for i in range(nbins_tomo) for j in xrange(i,nbins_tomo)]
    id_zband_Bl=[[i,j,k] for i in range(nbins_tomo) for j in xrange(0,nbins_tomo) for k in xrange(0,nbins_tomo)]
    combi_ell=[]
    for l1 in xrange(lmin,lmax,dl):
      for l2 in xrange(l1,lmax,dl):
        for l3 in xrange(l2,lmax,dl):
          ell=[l1,l2,l3]
          if self.check_triangle(ell)==1:
            combi_ell.append(ell)

    print id_zband_cl
    print id_zband_Bl
    #print combi_ell

    return np.asarray(id_zband_cl),np.asarray(id_zband_Bl),np.asarray(combi_ell)


