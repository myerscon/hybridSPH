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

# Create a CustomApplication(Application) class to contain common object functions and attributes across PySPH simulations

x0 = 0.5  # initial discontuinity

# scheme constants
dim = 2
#alpha1 = 1.0
#alpha2 = 1.0
#beta = 2.0
kernel_factor = 1.5

# Temporary Domain Boolean
DoDomain = True

class CustomApplication(Application):
    def __init__(self,gamma,DoDomain,mirror_x,mirror_y,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params) -> None:
        self.gamma = gamma
        self.gamma1 = gamma - 1
        self.DoDomain = DoDomain
        self.mirror_in_x = mirror_x
        self.mirror_in_y = mirror_y
        self.adaptive = adaptive
        self.cfl = cfl
        self.pfreq = pfreq
        self.tf = tf
        self.dt = dt
        self.scheme_selection = scheme
        self.sps = scheme_params
        super().__init__()

    # Built in (for each problem):

    def consume_user_options(self):
        self.scheme = self.SchemeChooser

    def create_scheme(self):

        adke = ADKEScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=self.gamma,
            alpha=self.sps[0], beta=self.sps[1], k=self.sps[4], eps=self.sps[5], g1=self.sps[2], g2=self.sps[3],
            has_ghosts=True)

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=self.gamma,
            kernel_factor=self.sps[0], update_alpha1=True, update_alpha2=True, alpha1=self.sps[2], alpha2=self.sps[3],
            beta=self.sps[1], max_density_iterations=1000,
            density_iteration_tolerance=1e-4, has_ghosts=True
        )

        # Changing niter=40
        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=self.gamma,
            kernel_factor=self.sps[0],
            g1=self.sps[2], g2=self.sps[3], rsolver=2, interpolation=self.sps[4], monotonicity=self.sps[5],
            interface_zero=True, hybrid=False, blend_alpha=2.0,
            niter=4000, tol=1e-6, has_ghosts=True
        )

        crksph = CRKSPHScheme(
            fluids=['fluid'], dim=dim, rho0=0, c0=0, nu=0, h0=0, p0=0,
            gamma=self.gamma, cl=2, has_ghosts=True
        )

        psph = PSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=self.gamma,
            hfact=kernel_factor
        )

        tsph = TSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=self.gamma,
            hfact=kernel_factor
        )

        mpm_sedov = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=self.gamma,
            kernel_factor=self.sps[0], alpha1=self.sps[2], alpha2=self.sps[3],
            beta=self.sps[1], adaptive_h_scheme="gsph",
            update_alpha1=True, update_alpha2=True
        )

        s = SchemeChooser(
            default=self.scheme_selection, adke=adke, mpm=mpm, gsph=gsph, crksph=crksph,
            psph=psph, tsph=tsph, mpm_sedov=mpm_sedov
        )

        # Added
        self.SchemeChooser = s
        return s

    def configure_scheme(self):
        s = self.scheme
        if self.options.scheme == 'mpm':
            s.configure(kernel_factor=self.sps[0])
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=self.adaptive, cfl=self.cfl, pfreq=self.pfreq)
        elif self.options.scheme == 'adke':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=self.adaptive, cfl=self.cfl, pfreq=self.pfreq)
        elif self.options.scheme == 'gsph':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=self.adaptive, cfl=self.cfl, pfreq=self.pfreq)
        elif self.options.scheme == 'crksph':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=self.adaptive, cfl=self.cfl, pfreq=self.pfreq)
        elif self.options.scheme in ['tsph', 'psph']:
            s.configure(hfact=kernel_factor)
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=self.adaptive, cfl=self.cfl, pfreq=self.pfreq)
        elif self.options.scheme == 'magma2':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=self.adaptive, cfl=self.cfl, pfreq=self.pfreq)
        elif self.options.scheme == 'mpm_sedov':
            s.configure(kernel_factor=self.sps[0])
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=self.adaptive, cfl=self.cfl, pfreq=self.pfreq)

    def create_domain(self):
        if (self.DoDomain):
            return DomainManager(
                xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax,n_layers=4, # n_layers set to 4 for riemann
                mirror_in_x=self.mirror_in_x, mirror_in_y=self.mirror_in_y,periodic_in_x=False, periodic_in_y=False)
                # periodic boundary conditions don't seem to work...

    # Custom Functions

    def step(self,nsteps=1):
        """ Moves the pysph simulation forward in time by 'nsteps' steps.
        Args:
            self
            nsteps (int): The number of steps to advance the pysph simulation in time
        Returns:
            None
        """
        for i in range(nsteps):
            # Assuming no pre- or post- step callbacks
            #self.solver.integrator.nnps.update()
            self.solver.integrator.step(self.solver.t, self.solver.dt)
            #self.solver.integrator.nnps.update_domain()
            self.solver.t += self.solver.dt
            self.solver.count += 1

            # limit dt growth to doubling each timestep
            #old_dt = self.solver.dt
            #self.solver.dt = self.solver._get_timestep()
            #if (self.solver.dt>2*old_dt):
            #    self.solver.dt = 2*old_dt

            self.solver.update_particle_time()
            if self.solver.execute_commands is not None:
                if self.solver.count % self.solver.command_interval == 0:
                    self.solver.execute_commands(self.solver)

            # update dt for next timestep if adaptive
            #if(self.adaptive):
            #    self.particles[0].dt_cfl = self.particles[0].h/self.particles[0].cs
            #    timestep=self.solver.integrator.compute_time_step(self.dt,self.cfl)
            #    if(timestep is not None):
            #        self.solver.dt=timestep

    def remove_flagged_particles(self):
        """ Removes all particles in the domain where the particle value particle_type_id is 0.
        Args:
            self
        Returns:
            None
        """
        self.particles[0].tag = np.where(self.particles[0].particle_type_id < 1, 1, 0)
        self.particles[0].remove_tagged_particles(tag=1)

    def particle_sort(self):
        """ Assigns each particle a unique value.
        Args:
            self: Must add particle property 'id' after pysph simulation initialization
        Returns:
            None
        """
        #self.particles[0].id = np.argsort(self.particles[0].orig_idx)
        self.particles[0].align_particles()
        self.particles[0].id = np.arange(len(self.particles[0].x))

    def add_particles(self, coords):
        """ Add particles to the simulation. Takes an array of x and y variables. Other variables should be populated later.
        Args:
            self
            coords (size 2 array of float arrays): [x, y], where x and y are 1d arrays of all new particle positions
        Returns:
            None
        """
        num_particles = len(coords[0])
        new_x = coords[0]
        new_y = coords[1]
        self.particles[0].add_particles(**{'x':new_x,'y':new_y})

    def plot_vertical(self,cbar_range=None,xlims=[0.,1.],ylims=[0.,0.05]):
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

        fig, axs = plt.subplots(4,1,sharex=True,sharey=True)

        my_plot0 = axs[0].scatter(x_vals,y_vals,c=p_vals,cmap="YlOrRd",s=10*self.dx,vmin=p_min,vmax=p_max)
        ax_cb0 = fig.add_axes([1.0, 0.1, .02, 0.8])
        cb0 = plt.colorbar(my_plot0,ax=axs[0],cax=ax_cb0)

        my_plot1 = axs[1].scatter(x_vals,y_vals,c=u_vals,cmap="PRGn",s=10*self.dx,vmin=u_min,vmax=u_max)
        ax_cb1 = fig.add_axes([1.1, 0.1, .02, 0.8])
        cb1 = plt.colorbar(my_plot1,ax=axs[1],cax=ax_cb1)

        my_plot2 = axs[2].scatter(x_vals,y_vals,c=v_vals,cmap="BrBG",s=10*self.dx,vmin=v_min,vmax=v_max)
        ax_cb2 = fig.add_axes([1.2, 0.1, .02, 0.8])
        cb2 = plt.colorbar(my_plot2,ax=axs[2],cax=ax_cb2)

        my_plot3 = axs[3].scatter(x_vals,y_vals,c=rho_vals,cmap="Blues",s=10*self.dx,vmin=rho_min,vmax=rho_max)
        ax_cb3 = fig.add_axes([1.3, 0.1, .02, 0.8])
        cb3 = plt.colorbar(my_plot3,ax=axs[3],cax=ax_cb3)

        axs[0].set_xlim(xlims)
        axs[0].set_ylim(ylims)

        axs[0].set_aspect('equal')
        axs[0].set_title('Pressure')
        axs[1].set_aspect('equal')
        axs[1].set_title('X Velocity')
        axs[2].set_aspect('equal')
        axs[2].set_title('Y Velocity')
        axs[3].set_aspect('equal')
        axs[3].set_title('Density')
        plt.show()

    def plot_single(self,cbar_range=None,xlims=[0.,1.],ylims=[0.,1.]):
         """ Plot single variable
         Args:
             self
             cbar_range (double array): array of 8 values corresponding to max and min plotting values of p, u, v, and rho.
         Returns:
             None
         """
         x_vals = self.particles[0].x
         y_vals = self.particles[0].y
         rho_vals = self.particles[0].p
         if (cbar_range):
             [rho_min,rho_max] = cbar_range
         else:
             rho_min = min(rho_vals)
             rho_max = max(rho_vals)
         fig, axs = plt.subplots(1,1,sharex=True,sharey=True)
         my_plot1 = axs.scatter(x_vals,y_vals,c=rho_vals,cmap="YlOrRd",s=10*self.dx,vmin=rho_min,vmax=rho_max)
         ax_cb1 = fig.add_axes([1.0, 0.1, .02, 0.8])
         cb1 = plt.colorbar(my_plot1,ax=axs,cax=ax_cb1)
         axs.set_xlim(xlims)
         axs.set_ylim(ylims)
         axs.set_aspect('equal')
         axs.set_title('Density')
         plt.show()
    

    def plot_square(self,cbar_range=None,xlims=[0.,1.],ylims=[0.,1.]):
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

        fig, axs = plt.subplots(2,2,sharex=True,sharey=True)

        my_plot0 = axs[0,0].scatter(x_vals,y_vals,c=p_vals,cmap="YlOrRd", s=10*self.dx,vmin=p_min,vmax=p_max)
        ax_cb0 = fig.add_axes([1.0, 0.1, .02, 0.8])
        cb0 = plt.colorbar(my_plot0,ax=axs[0,0],cax=ax_cb0)

        my_plot1 = axs[0,1].scatter(x_vals,y_vals,c=rho_vals,cmap="Blues",s=10*self.dx,vmin=rho_min,vmax=rho_max)
        ax_cb1 = fig.add_axes([1.1, 0.1, .02, 0.8])
        cb1 = plt.colorbar(my_plot1,ax=axs[0,1],cax=ax_cb1)

        my_plot2 = axs[1,0].scatter(x_vals,y_vals,c=v_vals,cmap="BrBG",s=10*self.dx,vmin=v_min,vmax=v_max)
        ax_cb2 = fig.add_axes([1.2, 0.1, .02, 0.8])
        cb2 = plt.colorbar(my_plot2,ax=axs[1,0],cax=ax_cb2)

        my_plot3 = axs[1,1].scatter(x_vals,y_vals,c=u_vals,cmap="PRGn",s=10*self.dx,vmin=u_min,vmax=u_max)
        ax_cb3 = fig.add_axes([1.3, 0.1, .02, 0.8])
        cb3 = plt.colorbar(my_plot3,ax=axs[1,1],cax=ax_cb3)

        axs[0,0].set_xlim(xlims)
        axs[0,0].set_ylim(ylims)
        axs[0,1].set_xlim(xlims)
        axs[0,1].set_ylim(ylims)
        axs[1,0].set_xlim(xlims)
        axs[1,0].set_ylim(ylims)
        axs[1,1].set_xlim(xlims)
        axs[1,1].set_ylim(ylims)

        axs[0,0].set_aspect('equal')
        axs[0,0].set_title('Pressure')
        axs[0,1].set_aspect('equal')
        axs[0,1].set_title('Density')
        axs[1,0].set_aspect('equal')
        axs[1,0].set_title('Y Velocity')
        axs[1,1].set_aspect('equal')
        axs[1,1].set_title('X Velocity')
        plt.show()

    def plot_type_id(self,xlims,ylims):
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
        my_plot0 = axs.scatter(x_vals, y_vals, c=id_vals, cmap='viridis', s=100*self.dx, vmin=0, vmax=2) # cmap='coolwarm', s=10*self.dx
        axs.set_aspect('equal')
        #axs.set_title('Particle Type')
        axs.set_xlim(xlims)
        axs.set_ylim(ylims)
        #plt.colorbar(my_plot0)
        plt.show()

    ### DEBUGGING PLOTS

    def plot_vertical_full(self,cbar_range=None,xlims=[0.,1.],ylims=[0.,0.05]):
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
        e_vals = self.particles[0].e
        h_vals = self.particles[0].h
        m_vals = self.particles[0].m

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
        e_min = min(e_vals)
        e_max = max(e_vals)
        h_min = min(h_vals)
        h_max = max(h_vals)
        m_min = min(m_vals)
        m_max = max(m_vals)

        fig, axs = plt.subplots(7,1,sharex=True,sharey=True)

        my_plot0 = axs[0].scatter(x_vals,y_vals,c=p_vals,cmap="YlOrRd",s=100*self.dx,vmin=p_min,vmax=p_max)
        ax_cb0 = fig.add_axes([1.0, 0.1, .02, 0.8])
        cb0 = plt.colorbar(my_plot0,ax=axs[0],cax=ax_cb0)

        my_plot1 = axs[1].scatter(x_vals,y_vals,c=u_vals,cmap="PRGn",s=100*self.dx,vmin=u_min,vmax=u_max)
        ax_cb1 = fig.add_axes([1.1, 0.1, .02, 0.8])
        cb1 = plt.colorbar(my_plot1,ax=axs[1],cax=ax_cb1)

        my_plot2 = axs[2].scatter(x_vals,y_vals,c=v_vals,cmap="BrBG",s=100*self.dx,vmin=v_min,vmax=v_max)
        ax_cb2 = fig.add_axes([1.2, 0.1, .02, 0.8])
        cb2 = plt.colorbar(my_plot2,ax=axs[2],cax=ax_cb2)

        my_plot3 = axs[3].scatter(x_vals,y_vals,c=rho_vals,cmap="Blues",s=100*self.dx,vmin=rho_min,vmax=rho_max)
        ax_cb3 = fig.add_axes([1.3, 0.1, .02, 0.8])
        cb3 = plt.colorbar(my_plot3,ax=axs[3],cax=ax_cb3)

        my_plot4 = axs[4].scatter(x_vals,y_vals,c=e_vals,cmap="Oranges",s=100*self.dx,vmin=e_min,vmax=e_max)
        ax_cb4 = fig.add_axes([1.4, 0.1, .02, 0.8])
        cb4 = plt.colorbar(my_plot4,ax=axs[4],cax=ax_cb4)

        my_plot5 = axs[5].scatter(x_vals,y_vals,c=h_vals,cmap="Purples",s=100*self.dx,vmin=h_min,vmax=h_max)
        ax_cb5 = fig.add_axes([1.5, 0.1, .02, 0.8])
        cb5 = plt.colorbar(my_plot5,ax=axs[5],cax=ax_cb5)

        my_plot6 = axs[6].scatter(x_vals,y_vals,c=m_vals,cmap="Greys",s=100*self.dx,vmin=m_min,vmax=m_max)
        ax_cb6 = fig.add_axes([1.6, 0.1, .02, 0.8])
        cb6 = plt.colorbar(my_plot6,ax=axs[6],cax=ax_cb6)

        axs[0].set_xlim(xlims)
        axs[0].set_ylim(ylims)

        axs[0].set_aspect('equal')
        axs[0].set_title('Pressure')
        axs[1].set_aspect('equal')
        axs[1].set_title('X Velocity')
        axs[2].set_aspect('equal')
        axs[2].set_title('Y Velocity')
        axs[3].set_aspect('equal')
        axs[3].set_title('Density')
        axs[4].set_aspect('equal')
        axs[4].set_title('Energy')
        axs[5].set_aspect('equal')
        axs[5].set_title('Smoothing Length')
        axs[6].set_aspect('equal')
        axs[6].set_title('Mass')
        plt.show()

    def plot_square_full(self,cbar_range=None,xlims=[0.,1.],ylims=[0.,1.]):
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
        e_vals = self.particles[0].e
        h_vals = self.particles[0].h
        m_vals = self.particles[0].m

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
        e_min = min(e_vals)
        e_max = max(e_vals)
        h_min = min(h_vals)
        h_max = max(h_vals)
        m_min = min(m_vals)
        m_max = max(m_vals)

        fig, axs = plt.subplots(4,2,sharex=True,sharey=True)

        my_plot0 = axs[0,0].scatter(x_vals,y_vals,c=p_vals,cmap="YlOrRd", s=10*self.dx,vmin=p_min,vmax=p_max)
        ax_cb0 = fig.add_axes([1.0, 0.1, .02, 0.8])
        cb0 = plt.colorbar(my_plot0,ax=axs[0,0],cax=ax_cb0)

        my_plot1 = axs[0,1].scatter(x_vals,y_vals,c=rho_vals,cmap="Blues",s=10*self.dx,vmin=rho_min,vmax=rho_max)
        ax_cb1 = fig.add_axes([1.1, 0.1, .02, 0.8])
        cb1 = plt.colorbar(my_plot1,ax=axs[0,1],cax=ax_cb1)

        my_plot2 = axs[1,0].scatter(x_vals,y_vals,c=v_vals,cmap="BrBG",s=10*self.dx,vmin=v_min,vmax=v_max)
        ax_cb2 = fig.add_axes([1.2, 0.1, .02, 0.8])
        cb2 = plt.colorbar(my_plot2,ax=axs[1,0],cax=ax_cb2)

        my_plot3 = axs[1,1].scatter(x_vals,y_vals,c=u_vals,cmap="PRGn",s=10*self.dx,vmin=u_min,vmax=u_max)
        ax_cb3 = fig.add_axes([1.3, 0.1, .02, 0.8])
        cb3 = plt.colorbar(my_plot3,ax=axs[1,1],cax=ax_cb3)

        my_plot4 = axs[2,0].scatter(x_vals,y_vals,c=e_vals,cmap="Oranges",s=10*self.dx,vmin=e_min,vmax=e_max)
        ax_cb4 = fig.add_axes([1.4, 0.1, .02, 0.8])
        cb4 = plt.colorbar(my_plot4,ax=axs[2,0],cax=ax_cb4)

        my_plot5 = axs[2,1].scatter(x_vals,y_vals,c=h_vals,cmap="Purples",s=10*self.dx,vmin=h_min,vmax=h_max)
        ax_cb5 = fig.add_axes([1.5, 0.1, .02, 0.8])
        cb5 = plt.colorbar(my_plot5,ax=axs[2,1],cax=ax_cb5)

        my_plot6 = axs[3,0].scatter(x_vals,y_vals,c=m_vals,cmap="Greys",s=10*self.dx,vmin=m_min,vmax=m_max)
        ax_cb6 = fig.add_axes([1.6, 0.1, .02, 0.8])
        cb6 = plt.colorbar(my_plot6,ax=axs[3,0],cax=ax_cb6)

        axs[3,1].axis('off')

        axs[0,0].set_xlim(xlims)
        axs[0,0].set_ylim(ylims)
        axs[0,1].set_xlim(xlims)
        axs[0,1].set_ylim(ylims)
        axs[1,0].set_xlim(xlims)
        axs[1,0].set_ylim(ylims)
        axs[1,1].set_xlim(xlims)
        axs[1,1].set_ylim(ylims)
        axs[2,0].set_xlim(xlims)
        axs[2,0].set_ylim(ylims)
        axs[2,1].set_xlim(xlims)
        axs[2,1].set_ylim(ylims)
        axs[3,0].set_xlim(xlims)
        axs[3,0].set_ylim(ylims)

        axs[0,0].set_aspect('equal')
        axs[0,0].set_title('Pressure')
        axs[0,1].set_aspect('equal')
        axs[0,1].set_title('Density')
        axs[1,0].set_aspect('equal')
        axs[1,0].set_title('Y Velocity')
        axs[1,1].set_aspect('equal')
        axs[1,1].set_title('X Velocity')
        axs[2,0].set_title('Energy')
        axs[2,0].set_aspect('equal')
        axs[2,1].set_title('Smoothing Length')
        axs[2,1].set_aspect('equal')
        axs[3,0].set_title('Mass')
        axs[3,0].set_aspect('equal')
        plt.show()

# Not included in master class: create_particles(), initialize()
# Maybe need to include: add_user_options(), consume_user_options()