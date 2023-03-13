import numpy as np
import matplotlib.pyplot as plt

# PySPH base and carray imports
from pysph.base.utils import get_particle_array as gpa
from pysph.solver.application import Application
from pysph.sph.scheme import GasDScheme, SchemeChooser
from pysph.sph.gas_dynamics.psph import PSPHScheme
from pysph.sph.gas_dynamics.tsph import TSPHScheme

# PySPH tools
from pysph.tools import uniform_distribution as ud

from custom_application import CustomApplication

# Numerical constants
dim = 2
gamma = 5.0/3.0
gamma1 = gamma - 1.0

# scheme constants
alpha1 = 10.0
alpha2 = 1.0
beta = 2.0
kernel_factor = 1.5 
 
class SedovPointExplosion(CustomApplication):
    def __init__(self,gamma,DoDomain,mirror_x,mirror_y,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params) -> None:
        #self.adaptive = adaptive
        #self.cfl = cfl
        self.pfreq = pfreq
        self.tf = tf
        self.dt = dt
        super().__init__(gamma,DoDomain,mirror_x,mirror_y,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params)

class CustomBlast(SedovPointExplosion):
    def __init__(self,dx,xmin,xmax,ymin,ymax,gamma,kf,DoDomain,mirror_x,mirror_y,xcntr,ycntr,r_init,gaussian,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params) -> None:
        self.dx = dx
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.kernel_factor = kf
        self.xcntr = xcntr
        self.ycntr = ycntr
        self.r_init = r_init
        self.gaussian = gaussian
        super().__init__(gamma,DoDomain,mirror_x,mirror_y,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params)
    
    def create_particles(self):
        data = ud.uniform_distribution_cubic2D(self.dx, self.xmin, self.xmax, self.ymin, self.ymax)
        x = data[0]
        y = data[1]
        dx = data[2]
        dy = data[3]
        volume = dx*dy

        rho = np.ones_like(x)
        p = np.zeros_like(x) # + 1e-5
        e = np.zeros_like(x) + 1e-9  # 2.5e-5
        m = volume * rho
        h = kernel_factor * (m/rho)**(1/dim)

        dist = np.sqrt((x-self.xcntr)**2 + (y-self.ycntr)**2)
        #num_enclosed = np.where(dist<self.r_init,1,0)
        #sum_enclosed = np.sum(num_enclosed.ravel())

        EnergyDensity = 100.0
        Density = 4.0
        e = np.where(dist<self.r_init,EnergyDensity,1e-9)
        rho = np.where(dist<self.r_init,Density,rho)
        p = np.where(e>1e-8,(gamma1*rho)*e,0)
        

        fluid = gpa(name='fluid', x=x, y=y, rho=rho, p=p, e=e, h=h, m=m)
        self.scheme.setup_properties([fluid])

        fluid.h[:] = kernel_factor * (fluid.m/fluid.rho)**(1./dim)

        print("Sedov's point explosion with %d particles"
              % (fluid.get_number_of_particles()))

        return [fluid,]

