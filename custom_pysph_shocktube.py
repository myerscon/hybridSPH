import matplotlib.pyplot as plt
import numpy

from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array as gpa
from pysph.solver.application import Application

from pysph.sph.scheme import (GasDScheme, ADKEScheme, GSPHScheme,
                              SchemeChooser, add_bool_argument)
from pysph.sph.wc.crksph import CRKSPHScheme
from pysph.sph.gas_dynamics.psph import PSPHScheme
from pysph.sph.gas_dynamics.tsph import TSPHScheme
from pysph.sph.gas_dynamics.magma2 import MAGMA2Scheme

# PySPH tools
from pysph.tools import uniform_distribution as ud

# Numerical constants
dim = 2
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 7.5e-6
tf = 0.0005 # 0.005

# domain sizeC
xmin = 0.
xmax = 1
dx = 0.002
ny = 25
ymin = 0
ymax = ny * dx
x0 = 0.5  # initial discontuinity

# scheme constants
alpha1 = 1.0
alpha2 = 1.0
beta = 2.0
kernel_factor = 1.5
h0 = kernel_factor * dx

class ShockTube2D(Application):
    def initialize(self):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.dx = dx
        self.hdx = 1.7
        self.x0 = x0
        self.ny = ny
        self.pl = 1000
        self.pr = 0.01
        self.rhol = 1.0
        self.rhor = 1.0
        self.ul = 0.
        self.ur = 0.
        self.vl = 0.
        self.vr = 0.

    def add_user_options(self, group):
        add_bool_argument(group, 'smooth-ic', dest='smooth_ic', default=False,
                          help="Smooth the initial condition.")

    def consume_user_options(self):
        self.smooth_ic = self.options.smooth_ic

    # Domain manager controls boundaries
    def create_domain(self):
        return DomainManager(
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            mirror_in_x=True, mirror_in_y=True)

    def create_particles(self):
        global dx
        data = ud.uniform_distribution_cubic2D(dx, xmin, xmax, ymin, ymax)

        x = data[0]
        y = data[1]
        dx = data[2]
        dy = data[3]

        # volume estimate
        volume = dx * dy

        # indices on either side of the initial discontinuity
        right_indices = numpy.where(x > x0)[0]

        # density is uniform
        rho = numpy.ones_like(x) * self.rhol
        rho[right_indices] = self.rhor

        # pl = 100.0, pr = 0.1
        if self.smooth_ic:
            deltax = 1.5 * dx
            p = (self.pl - self.pr) / (1 + numpy.exp((x - x0) / deltax)) + \
                self.pr
        else:
            p = numpy.ones_like(x) * self.pl
            p[right_indices] = self.pr

        # const h and mass
        h = numpy.ones_like(x) * self.hdx * self.dx
        m = numpy.ones_like(x) * volume * rho

        # ul = ur = 0
        u = numpy.ones_like(x) * self.ul
        u[right_indices] = self.ur

        # vl = vr = 0
        v = numpy.ones_like(x) * self.vl
        v[right_indices] = self.vr

        # thermal energy from the ideal gas EOS
        e = p / (gamma1 * rho)

        fluid = gpa(name='fluid', x=x, y=y, rho=rho, p=p, e=e, h=h, m=m,
                    h0=h.copy(), u=u, v=v)
        self.scheme.setup_properties([fluid])

        print("2D Shocktube with %d particles" %
              (fluid.get_number_of_particles()))

        return [fluid, ]

    def create_scheme(self):
        self.dt = dt
        self.tf = tf

        adke = ADKEScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            alpha=1, beta=1, k=1.0, eps=0.8, g1=0.5, g2=0.5,
            has_ghosts=True)

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=kernel_factor, alpha1=alpha1, alpha2=alpha2,
            beta=beta, max_density_iterations=1000,
            density_iteration_tolerance=1e-4, has_ghosts=True
        )

        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=1.5,
            g1=0.25, g2=0.5, rsolver=2, interpolation=1, monotonicity=2,
            interface_zero=True, hybrid=False, blend_alpha=2.0,
            niter=40, tol=1e-6, has_ghosts=True
        )

        crksph = CRKSPHScheme(
            fluids=['fluid'], dim=dim, rho0=0, c0=0, nu=0, h0=0, p0=0,
            gamma=gamma, cl=2, has_ghosts=True
        )

        psph = PSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            hfact=kernel_factor
        )

        tsph = TSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            hfact=kernel_factor
        )

        magma2 = MAGMA2Scheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            ndes=40, has_ghosts=True
        )

        s = SchemeChooser(
            default='adke', adke=adke, mpm=mpm, gsph=gsph, crksph=crksph,
            psph=psph, tsph=tsph, magma2=magma2
        )
        return s

    def configure_scheme(self):
        s = self.scheme
        if self.options.scheme == 'mpm':
            s.configure(kernel_factor=kernel_factor)
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=True, pfreq=50)
        elif self.options.scheme == 'adke':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)
        elif self.options.scheme == 'gsph':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)
        elif self.options.scheme == 'crksph':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)
        elif self.options.scheme in ['tsph', 'psph']:
            s.configure(hfact=kernel_factor)
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)
        elif self.options.scheme == 'magma2':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)

class CustomShockTube2D(ShockTube2D):
    def step(self,nsteps=1):
        """ Moves the pysph simulation forward in time by 'nsteps' steps.
        Args:
            self
            nsteps (int): The number of steps to advance the pysph simulation in time
        Returns:
            None
        """
        for i in range(nsteps):
            self.solver.integrator.nnps.update()
            self.solver.integrator.step(self.solver.t, self.solver.dt)
            self.solver.integrator.nnps.update_domain()

    def remove_flagged_particles(self):
        """ Removes all particles in the domain where the particle value particle_type_id is 0.
        Args:
            self
        Returns:
            None
        """
        self.particles[0].tag = numpy.where(self.particles[0].particle_type_id < 1, 1, 0)
        self.particles[0].remove_tagged_particles(tag=1) 

    def remove_flagged_particles2(self):
        """ Alternate version: Removes all particles in the domain where the particle value particle_type_id is 0.
        Args:
            self
        Returns:
            None
        """
        # Slightly slower in initial tests
        remove_array = numpy.extract(self.particles[0].particle_type_id < 1, self.particles[0].orig_idx)
        self.particles[0].remove_particles(remove_array)

    def particle_sort(self):
        """ Sorts existing particles based on orig_idx values
        Args:
            self: Must add particle property 'id' after pysph simulation initialization
        Returns:
            None
        """
        self.particles[0].id = numpy.argsort(self.particles[0].orig_idx)

    def add_particles(self, coords):
        """ Add particles to the simulation. Takes an array of x and y variables. Other variables should be populated later.
        Args:
            self
            coords (size 2 array of float arrays): [x, y], where x and y are 1d arrays of all new particle positions
        Returns:
            None
        """
        new_x = coords[0]
        new_y = coords[1]
        self.particles[0].add_particles(x=new_x,y=new_y)

    ###
    ### PLOTTING FUNCTIONS
    ###
    
    def plot_2D(self,cbar_range=None):
        """ Scatter plot with four subplots corresponding to p, u, v, and rho in a vertical format. Useful for visualizing shocktube results.
        Args:
            self
            cbar_range (double array): array of 8 values corresponding to max and min plotting values of p, u, v, and rho.
        Returns:
            None
        """
        x_vals = self.particles[0].x
        y_vals = self.particles[0].y
        rho_vals = self.particles[0].rho
        u_vals = self.particles[0].u
        v_vals = self.particles[0].v
        p_vals = self.particles[0].p

        if (cbar_range):
            [p_min,p_max,u_min,u_max,v_min,v_max,rho_min,rho_max] = cbar_range
        else:
            p_min = min(p_vals)
            p_max = max(p_vals)
            u_min = min(u_vals)
            u_max = max(u_vals)
            if (abs(u_min)>u_max):
                u_max = abs(u_min)
            else:
                u_min = - u_max
            v_min = min(v_vals)
            v_max = max(v_vals)
            if (abs(v_min)>v_max):
                v_max = abs(v_min)
            else:
                v_min = - v_max
            rho_min = min(rho_vals)
            rho_max = max(rho_vals)

        fig, axs = plt.subplots(4,1,sharex=True)

        my_plot0 = axs[0].scatter(x_vals,y_vals,c=p_vals,cmap="YlOrRd",s=1,vmin=p_min,vmax=p_max)
        ax_cb0 = fig.add_axes([1.0, 0.1, .02, 0.8])
        cb0 = plt.colorbar(my_plot0,ax=axs[0],cax=ax_cb0)

        my_plot1 = axs[1].scatter(x_vals,y_vals,c=u_vals,cmap="PRGn",s=1,vmin=u_min,vmax=u_max)
        ax_cb1 = fig.add_axes([1.1, 0.1, .02, 0.8])
        cb1 = plt.colorbar(my_plot1,ax=axs[1],cax=ax_cb1)

        my_plot2 = axs[2].scatter(x_vals,y_vals,c=v_vals,cmap="BrBG",s=1,vmin=v_min,vmax=v_max)
        ax_cb2 = fig.add_axes([1.2, 0.1, .02, 0.8])
        cb2 = plt.colorbar(my_plot2,ax=axs[2],cax=ax_cb2)

        my_plot3 = axs[3].scatter(x_vals,y_vals,c=rho_vals,cmap="Blues",s=1,vmin=rho_min,vmax=rho_max)
        ax_cb3 = fig.add_axes([1.3, 0.1, .02, 0.8])
        cb3 = plt.colorbar(my_plot3,ax=axs[3],cax=ax_cb3)

        axs[0].set_xlim([0.,1.])

        axs[0].set_aspect('equal')
        axs[0].set_title('Pressure')
        axs[1].set_aspect('equal')
        axs[1].set_title('X Velocity')
        axs[2].set_aspect('equal')
        axs[2].set_title('Y Velocity')
        axs[3].set_aspect('equal')
        axs[3].set_title('Density')
        plt.show()

    def plot_type_id(self):
        """ Scatter plot of pysph particle type ID corresponding to active (2), boundary (1), or void (0) particle.
        Args:
            self
        Returns:
            None
        """
        x_vals = self.particles[0].x
        y_vals = self.particles[0].y
        id_vals = self.particles[0].particle_type_id

        fig, axs = plt.subplots(1,1,sharex=True)
        my_plot0 = axs.scatter(x_vals, y_vals, c=id_vals, cmap='coolwarm')
        axs.set_aspect('equal')
        axs.set_title('Particle Type')
        axs.set_xlim([0,1])
        axs.set_ylim([0,0.05])
        plt.show()