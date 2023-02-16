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

# solution parameters
# confirmed stable (r_init: 0.1, dt: 5e-5)
#dt = 2.5e-5
#tf = 0.1

# scheme constants
alpha1 = 10.0
alpha2 = 1.0
beta = 2.0
kernel_factor = 1.5 # 1.2

def gaussian_2d(x,y,mu_x,mu_y,sigma):
    xy = (x-mu_x)**2 + (y-mu_y)**2
    return (np.exp(-0.5*(xy)/(sigma**2)))
 
class SedovPointExplosion(CustomApplication):
    def __init__(self,gamma,DoDomain,mirror_x,mirror_y,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params) -> None:
        #self.adaptive = adaptive
        #self.cfl = cfl
        #self.pfreq = pfreq
        #self.tf = tf
        #self.dt = dt
        super().__init__(gamma,DoDomain,mirror_x,mirror_y,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params)

    def create_particles(self):
        #fpath = os.path.join(os.path.dirname(__file__), 'ndspmhd-sedov-initial-conditions.npz')
        data = np.load('examples/gas_dynamics/ndspmhd-sedov-initial-conditions.npz')
        x = data['x']
        y = data['y']
        rho = data['rho']
        p = data['p']
        e = data['e'] + 1e-9
        h = data['h']
        m = data['m']

        fluid = gpa(name='fluid', x=x, y=y, rho=rho, p=p, e=e, h=h, m=m)
        self.scheme.setup_properties([fluid])

        # set the initial smoothing length proportional to the particle
        # volume
        fluid.h[:] = kernel_factor * (fluid.m/fluid.rho)**(1./dim)

        print("Sedov's point explosion with %d particles"
              % (fluid.get_number_of_particles()))

        return [fluid,]

class CustomSedov(SedovPointExplosion):
    def __init__(self,dx,xmin,xmax,ymin,ymax,gamma,DoDomain,mirror_x,mirror_y,xcntr,ycntr,r_init,gaussian,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params) -> None:
        self.dx = dx
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
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
        p = np.zeros_like(x) + 1e-5
        e = np.zeros_like(x) + 2.5e-5 # 1e-9
        m = volume * rho
        h = kernel_factor * (m/rho)**(1/dim)

        # Introduce a disturbance of energy E=1.0 and set the initial smoothing length
        # proportional to the particle volume
        if(self.gaussian):
            e = 736.565 * gaussian_2d(x=x,y=y,mu_x=self.xcntr,mu_y=self.ycntr,sigma=0.0147) + 1e-9 # 789.459
            p = 526.178 * gaussian_2d(x=x,y=y,mu_x=self.xcntr,mu_y=self.ycntr,sigma=0.0147) # 526.178 lower 490.924

        else:
            dist = np.sqrt((x-self.xcntr)**2 + (y-self.ycntr)**2)
            num_enclosed = np.where(dist<self.r_init,1,0)
            sum_enclosed = np.sum(num_enclosed.ravel())

            Energy = 1.0
            e = np.where(dist<self.r_init,Energy/(sum_enclosed*volume),1e-9) # 1.0/sum_enclosed
            p = np.where(e>1e-8,(gamma1*rho)*e,0)
        

        fluid = gpa(name='fluid', x=x, y=y, rho=rho, p=p, e=e, h=h, m=m)
        self.scheme.setup_properties([fluid])

        fluid.h[:] = kernel_factor * (fluid.m/fluid.rho)**(1./dim)

        print("Sedov's point explosion with %d particles"
              % (fluid.get_number_of_particles()))

        return [fluid,]
 
