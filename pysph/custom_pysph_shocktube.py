import matplotlib.pyplot as plt
import numpy as np

from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array as gpa
from pysph.solver.application import Application

from pysph.sph.scheme import (GasDScheme, ADKEScheme, GSPHScheme,
                              SchemeChooser, add_bool_argument)
from pysph.sph.wc.crksph import CRKSPHScheme
from pysph.sph.gas_dynamics.psph import PSPHScheme
from pysph.sph.gas_dynamics.tsph import TSPHScheme

# PySPH tools
from pysph.tools import uniform_distribution as ud

# Import CustomApplication
from custom_application import CustomApplication

# Numerical constants
dim = 2
#gamma = 1.4
#gamma1 = gamma - 1.0

# solution parameters
#dt = 7.5e-6 # 7.5e-6
#tf = 0.012 # 0.005

# domain sizeC
#xmin = 0.
#xmax = 1
#dx = 0.002 # set dx in class declaration
#ny = 25
#ymin = 0
#ymax = 0.05 # Make sure that ymax == ny * dx !
x0 = 0.5  # initial discontuinity

# scheme constants
alpha1 = 1.0
alpha2 = 1.0
beta = 2.0
kernel_factor = 1.5
#h0 = kernel_factor * dx

# Dummy object
#class Object(object):
#    pass

class ShockTube2D(CustomApplication):
    def __init__(self,dx,xmin,xmax,ymin,ymax,gamma,DoDomain,mirror_x,mirror_y,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params) -> None:
        self.dx = dx
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        #self.adaptive = adaptive
        #self.cfl = cfl
        #self.pfreq = pfreq
        #self.tf = tf
        #self.dt = dt
        #self.options = Object()
        #self.options.scheme = scheme
        super().__init__(gamma,DoDomain,mirror_x,mirror_y,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params)

    def initialize(self):
        #self.xmin = xmin
        #self.xmax = xmax
        #self.ymin = ymin
        #self.ymax = ymax

        # moved 'dx-dependent' variables inside initialize() function to allow for setting dx at runtime
        #self.ymax = ny * self.dx
        h0 = kernel_factor * self.dx

        self.hdx = 1.7
        self.x0 = x0
        self.pl = 1000
        self.pr = 0.01
        self.rhol = 1.0
        self.rhor = 1.0
        self.ul = 0.
        self.ur = 0.
        self.vl = 0.
        self.vr = 0.
        # attach global variables to object
        #self.gamma = gamma
        #self.gamma1 = gamma1

    def add_user_options(self, group):
        add_bool_argument(group, 'smooth-ic', dest='smooth_ic', default=False,
                          help="Smooth the initial condition.")

    def consume_user_options(self):
        self.smooth_ic = self.options.smooth_ic

    def create_particles(self):
        #global dx
        mod = ((self.ymax-self.ymin)%self.dx)
        if (mod>1e-6):
            tmp_ymin = self.ymin + (mod/2)
            tmp_ymax = self.ymax - (mod/2)
        else:
            tmp_ymin = self.ymin
            tmp_ymax = self.ymax
        data = ud.uniform_distribution_cubic2D(self.dx, self.xmin, self.xmax, tmp_ymin, tmp_ymax)

        x = data[0]
        y = data[1]
        dx = data[2]
        dy = data[3]

        # volume estimate
        volume = dx * dy

        # indices on either side of the initial discontinuity
        right_indices = np.where(x > x0)[0]

        # density is uniform
        rho = np.ones_like(x) * self.rhol
        rho[right_indices] = self.rhor

        # pl = 100.0, pr = 0.1
        if self.smooth_ic:
            deltax = 1.5 * dx
            p = (self.pl - self.pr) / (1 + np.exp((x - x0) / deltax)) + \
                self.pr
        else:
            p = np.ones_like(x) * self.pl
            p[right_indices] = self.pr

        # const h and mass
        h = np.ones_like(x) * self.hdx * self.dx
        m = np.ones_like(x) * volume * rho

        # ul = ur = 0
        u = np.ones_like(x) * self.ul
        u[right_indices] = self.ur

        # vl = vr = 0
        v = np.ones_like(x) * self.vl
        v[right_indices] = self.vr

        # thermal energy from the ideal gas EOS
        e = p / (self.gamma1 * rho)

        fluid = gpa(name='fluid', x=x, y=y, rho=rho, p=p, e=e, h=h, m=m,
                    h0=h.copy(), u=u, v=v)
        self.scheme.setup_properties([fluid])

        print("2D Shocktube with %d particles" %
              (fluid.get_number_of_particles()))

        return [fluid, ]

class CustomShockTube2D(ShockTube2D):
    def __init__(self,dx,xmin,xmax,ymin,ymax,gamma,DoDomain,mirror_x,mirror_y,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params) -> None:
        super().__init__(dx,xmin,xmax,ymin,ymax,gamma,DoDomain,mirror_x,mirror_y,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params)

    ###
    ### PLOTTING FUNCTIONS
    ###
    
