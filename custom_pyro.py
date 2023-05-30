import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Verdana') # Changes plotting font # family='Times New Roman'
from scipy.ndimage.filters import uniform_filter
from pyro import Pyro

class CustomPyro(Pyro):
    def primitive_update(self):
        """ Calculates primitive variables from object's conserved variables, stores in self.sim.cc_data.prim_array
        Args:
            self
        Returns:
            None
        """
        ivars = self.solver.Variables(self.sim.cc_data)
        gamma = self.sim.cc_data.get_aux("gamma")
        q = self.solver.cons_to_prim(self.sim.cc_data.data, gamma, ivars, self.sim.cc_data.grid)

        self.sim.cc_data.prim_array = q
        
    def conservative_update(self):
        """ Calculates conserved variables from object's primitive variables, stores in self.sim.cc_data.data
        Args:
            self
        Returns:
            None
        """
        ivars = self.solver.Variables(self.sim.cc_data)
        gamma = self.sim.cc_data.get_aux("gamma")
        U = self.solver.prim_to_cons(self.sim.cc_data.prim_array, gamma, ivars, self.sim.cc_data.grid)

        self.sim.cc_data.data = U

    def pyro_step(self, adaptive=True, dt=None):
        """ Updated pyro_sim object through one timestep. Either fixed or adaptive timesteps can be used.
        
        Args:
            adaptive (boolean): Boolean variable indicating whether to use an adaptive timestep
            dt (float): Timestep to use in case of a fixed timestep
        Returns:
            None
        """
        self.sim.cc_data.fill_BC_all()
        if (adaptive):
            self.sim.compute_timestep()
        else:
            self.sim.dt=dt
        self.sim.evolve()
        
    def compute_gradients(self):
        """ Calculates primitive variable gradients, stores in self.sim.cc_data.prim_gradients
        Args:
            self
        Returns:
            None
        """
        dx = self.sim.cc_data.grid.dx
        dy = self.sim.cc_data.grid.dy

        drho_vals_pre = np.gradient(self.sim.cc_data.prim_array[:,:,0],dx,dy,edge_order=1)
        du_vals_pre = np.gradient(self.sim.cc_data.prim_array[:,:,1],dx,dy,edge_order=1)
        dv_vals_pre = np.gradient(self.sim.cc_data.prim_array[:,:,2],dx,dy,edge_order=1)
        dp_vals_pre = np.gradient(self.sim.cc_data.prim_array[:,:,3],dx,dy,edge_order=1)

        drho_vals = np.sqrt(drho_vals_pre[0]**2+drho_vals_pre[1]**2)
        du_vals = np.sqrt(du_vals_pre[0]**2+du_vals_pre[1]**2)
        dv_vals = np.sqrt(dv_vals_pre[0]**2+dv_vals_pre[1]**2)
        dp_vals = np.sqrt(dp_vals_pre[0]**2+dp_vals_pre[1]**2)

        self.sim.cc_data.prim_gradients = np.stack([drho_vals,du_vals,dv_vals,dp_vals])

    def flag_gradients(self,threshold,code=0):
        """ Calculates binary array flagging gradients above a specified value
        Args:
            self
            threshold: gradient flagging threshold array in order: rho, u, v, p
            code: controls which gradients to can through (0,1,2)
                -0: All values (rho, u, v, p)
                -1: rho and p
                -2: rho only
        Returns:
            None
        """
        if ((code!=0) and (code!=1) and (code!=2)):
            raise Exception("Invalid flag_gradient code (Should be either 0, 1, or 2)!")

        primitive_gradients = self.sim.cc_data.prim_gradients
        gradient_flag = np.zeros_like(primitive_gradients)
        #gradient_final = np.zeros_like(primitive_gradients[0])
        gradient_sum = np.zeros_like(primitive_gradients[0])

        if(code==0):
            for i in range(4):
                gradient_flag[i] = np.where(primitive_gradients[i]>threshold[i],1,0)
        elif(code==1):
            for i in range(0,4,3):
                gradient_flag[i] = np.where(primitive_gradients[i]>threshold[i],1,0)
        elif(code==2):
            gradient_flag[0] = np.where(primitive_gradients[0]>threshold[0],1,0)

        # np.add.reduce(gradient_flag[0],gradient_flag[1],gradient_flag[2],gradient_flag[3])
        for i in range(4):
            gradient_sum += gradient_flag[i]

        self.sim.cc_data.gradient_flag = np.where(gradient_sum>0,1,0)
        # set gradient final to 2x gradient flag in case pysph_zoning isn't called
        self.sim.cc_data.gradient_final = 2.0*self.sim.cc_data.gradient_flag

    def pysph_zoning(self, active_width, boundary_width):
        """ Adds buffer zone to gradient_flag zones for active SPH particles, creates SPH boundary regions.
            gradient_final values of 2, 1, and 0, correspond to pysph active, boundary, and void zones respectively.
        Args:
            self
            active_width (int): number of cells around flagged cells to be marked as active SPH regions
            boundary_width (int): number of cells around active SPH cells to be marked as boundary SPH regions
        Returns:
            None
        """
        array = 1000 * self.sim.cc_data.gradient_flag.copy()
        active_n = (2 * active_width) + 1
        boundary_n = (2 * (active_width + boundary_width)) + 1

        array2 = uniform_filter(array,size=active_n,mode='constant',cval=0.0)
        array2 = np.where(array2>0.,1.,0.)
        array3 = uniform_filter(array,size=boundary_n,mode='constant',cval=0.0)
        array3 = np.where(array3>0.,1.,0.)

        self.sim.cc_data.gradient_final = array2 + array3

    ###
    ### PLOTTING FUNCTIONS
    ###
        
    def plot_2D_vertical(self,cbar_range=None,xlims=[0.,1.],ylims=[0.,0.05]):
        """ Plots primitive variables pressure, density, and x- and y- velocity
        Args:
            self
        Returns:
            None
        """
        x_vals = self.sim.cc_data.grid.x
        y_vals = self.sim.cc_data.grid.y
        rho_vals = np.transpose(self.sim.cc_data.prim_array[:,:,0])
        u_vals = np.transpose(self.sim.cc_data.prim_array[:,:,1])
        v_vals = np.transpose(self.sim.cc_data.prim_array[:,:,2])
        p_vals = np.transpose(self.sim.cc_data.prim_array[:,:,3])

        if (cbar_range):
            [p_min,p_max,u_min,u_max,v_min,v_max,rho_min,rho_max] = cbar_range
        else:
            p_min = min(p_vals.ravel())
            p_max = max(p_vals.ravel())
            u_min = min(u_vals.ravel())
            u_max = max(u_vals.ravel())
            if (abs(u_min)>u_max):
                u_max = abs(u_min)
            else:
                u_min = - u_max
            v_min = min(v_vals.ravel())
            v_max = max(v_vals.ravel())
            if (abs(v_min)>v_max):
                v_max = abs(v_min)
            else:
                v_min = - v_max
            rho_min = min(rho_vals.ravel())
            rho_max = max(rho_vals.ravel())

        fig, axs = plt.subplots(4,1,sharex=True)

        my_plot0 = axs[0].pcolormesh(x_vals, y_vals, p_vals, cmap='YlOrRd', vmin=p_min, vmax=p_max)
        ax_cb0 = fig.add_axes([1.0, 0.1, .02, 0.8])
        cb0 = plt.colorbar(my_plot0,ax=axs[0],cax=ax_cb0)

        my_plot1 = axs[1].pcolormesh(x_vals, y_vals, u_vals, cmap='PRGn', vmin=u_min, vmax=u_max)
        ax_cb1 = fig.add_axes([1.1, 0.1, .02, 0.8])
        cb1 = plt.colorbar(my_plot1,ax=axs[1],cax=ax_cb1)

        my_plot2 = axs[2].pcolormesh(x_vals, y_vals, v_vals, cmap='BrBG', vmin=v_min, vmax=v_max)
        ax_cb2 = fig.add_axes([1.2, 0.1, .02, 0.8])
        cb2 = plt.colorbar(my_plot2,ax=axs[2],cax=ax_cb2)

        my_plot3 = axs[3].pcolormesh(x_vals, y_vals, rho_vals, cmap='Blues', vmin=rho_min, vmax=rho_max)
        ax_cb3 = fig.add_axes([1.32, 0.1, .02, 0.8]) # [1.3, 0.1, .02, 0.8]
        cb3 = plt.colorbar(my_plot3,ax=axs[3],cax=ax_cb3)

        axs[0].set_xlim(xlims)
        axs[0].set_ylim(ylims)
        axs[1].set_xlim(xlims)
        axs[1].set_ylim(ylims)
        axs[2].set_xlim(xlims)
        axs[2].set_ylim(ylims)
        axs[3].set_xlim(xlims)
        axs[3].set_ylim(ylims)

        axs[0].set_aspect('equal')
        axs[0].set_title('Pressure')
        axs[1].set_aspect('equal')
        axs[1].set_title('X Velocity')
        axs[2].set_aspect('equal')
        axs[2].set_title('Y Velocity')
        axs[3].set_aspect('equal')
        axs[3].set_title('Density')
        plt.show()
        
    def plot_gradients_2D_vertical(self,cbar_range=None):
        """ Plots primitive variable gradients for each variable
        Args:
            self
        Returns:
            None
        """
        x_vals = self.sim.cc_data.grid.x
        y_vals = self.sim.cc_data.grid.y
        rho_vals = np.transpose(self.sim.cc_data.prim_gradients[0,:,:])
        u_vals = np.transpose(self.sim.cc_data.prim_gradients[1,:,:])
        v_vals = np.transpose(self.sim.cc_data.prim_gradients[2,:,:])
        p_vals = np.transpose(self.sim.cc_data.prim_gradients[3,:,:])

        if (cbar_range):
            [p_min,p_max,u_min,u_max,v_min,v_max,rho_min,rho_max] = cbar_range
        else:
            p_min = min(p_vals.ravel())
            p_max = max(p_vals.ravel())
            u_min = min(u_vals.ravel())
            u_max = max(u_vals.ravel())
            if (abs(u_min)>u_max):
                u_max = abs(u_min)
            else:
                u_min = - u_max
            v_min = min(v_vals.ravel())
            v_max = max(v_vals.ravel())
            if (abs(v_min)>v_max):
                v_max = abs(v_min)
            else:
                v_min = - v_max
            rho_min = min(rho_vals.ravel())
            rho_max = max(rho_vals.ravel())

        fig, axs = plt.subplots(4,1,sharex=True) 

        my_plot0 = axs[0].pcolormesh(x_vals, y_vals, p_vals, cmap='YlOrRd', vmin=p_min, vmax=p_max)
        ax_cb0 = fig.add_axes([1.0, 0.1, .02, 0.8])
        cb0 = plt.colorbar(my_plot0,ax=axs[0],cax=ax_cb0) 

        my_plot1 = axs[1].pcolormesh(x_vals, y_vals, u_vals, cmap='PRGn', vmin=u_min, vmax=u_max)
        ax_cb1 = fig.add_axes([1.1, 0.1, .02, 0.8])
        cb1 = plt.colorbar(my_plot1,ax=axs[1],cax=ax_cb1)

        my_plot2 = axs[2].pcolormesh(x_vals, y_vals, v_vals, cmap='BrBG', vmin=v_min, vmax=v_max)
        ax_cb2 = fig.add_axes([1.2, 0.1, .02, 0.8])
        cb2 = plt.colorbar(my_plot2,ax=axs[2],cax=ax_cb2)

        my_plot3 = axs[3].pcolormesh(x_vals, y_vals, rho_vals, cmap='Blues', vmin=rho_min, vmax=rho_max)
        ax_cb3 = fig.add_axes([1.3, 0.1, .02, 0.8])
        cb3 = plt.colorbar(my_plot3,ax=axs[3],cax=ax_cb3)

        axs[0].set_xlim([0.,1.])

        axs[0].set_aspect('equal')
        axs[0].set_title('Pressure Gradient')
        axs[1].set_aspect('equal')
        axs[1].set_title('X Velocity Gradient')
        axs[2].set_aspect('equal')
        axs[2].set_title('Y Velocity Gradient')
        axs[3].set_aspect('equal')
        axs[3].set_title('Density Gradient')
        plt.show()

    def plot_2D_mesh_flag(self,cbar_range=None,xlims=[0.,1.],ylims=[0.,0.05]):
        """ Plots gradient flag
        Args:
            self
        Returns:
            None
        """
        x_vals = self.sim.cc_data.grid.x
        y_vals = self.sim.cc_data.grid.y
        flag = self.sim.cc_data.gradient_final

        fig, axs = plt.subplots(1,1,sharex=True)

        my_plot0 = axs.pcolormesh(np.transpose(flag), cmap='coolwarm')
        axs.set_aspect('equal')
        axs.set_title('Gradient Flag')
        axs.set_xlim(xlims)
        axs.set_ylim(ylims)
        plt.show()

    def plot_2D_scatter_flag(self,cbar_range=None,xlims=[0.,1.],ylims=[0.,0.05]):
        """ Plots gradient flag
        Args:
            self
        Returns:
            None
        """
        x_vals = self.sim.cc_data.grid.x2d
        y_vals = self.sim.cc_data.grid.y2d
        flag = self.sim.cc_data.gradient_final

        fig, axs = plt.subplots(1,1,sharex=True)

        my_plot0 = axs.scatter(x_vals, y_vals, c=flag, cmap='coolwarm', marker=",", s=100)
        axs.set_aspect('equal')
        axs.set_title('Gradient Flag')
        axs.set_xlim(xlims)
        axs.set_ylim(ylims)
        plt.show()

    def plot_single(self,cbar_range=None,xlims=[0.,1.],ylims=[0.,1.]):
        """ Plot single variable
        Args:
            self
        Returns:
            None
        """
        x_vals = self.sim.cc_data.grid.x
        y_vals = self.sim.cc_data.grid.y
        rho_vals = np.transpose(self.sim.cc_data.prim_array[:,:,3])
        if (cbar_range):
            [rho_min,rho_max] = cbar_range
        else:
            rho_min = min(rho_vals.ravel())
            rho_max = max(rho_vals.ravel())
        fig, axs = plt.subplots(1,1,sharex=True)
        my_plot1 = axs.pcolormesh(x_vals, y_vals, rho_vals, cmap='YlOrRd', vmin=rho_min, vmax=rho_max)
        ax_cb1 = fig.add_axes([0.85, 0.1, .02, 0.8])
        cb1 = plt.colorbar(my_plot1,ax=axs,cax=ax_cb1)
        axs.set_aspect('equal')
        axs.set_xlim(xlims)
        axs.set_ylim(ylims)
        #axs.set_title('Pressure')
        plt.show()

        
    def plot_2D_square(self,cbar_range=None,xlims=[0.,1.],ylims=[0.,0.05]):
        """ Plots primitive variables pressure, density, and x- and y- velocity
        Args:
            self
        Returns:
            None
        """
        x_vals = self.sim.cc_data.grid.x
        y_vals = self.sim.cc_data.grid.y
        rho_vals = np.transpose(self.sim.cc_data.prim_array[:,:,0])
        u_vals = np.transpose(self.sim.cc_data.prim_array[:,:,1])
        v_vals = np.transpose(self.sim.cc_data.prim_array[:,:,2])
        p_vals = np.transpose(self.sim.cc_data.prim_array[:,:,3])

        if (cbar_range):
            [p_min,p_max,u_min,u_max,v_min,v_max,rho_min,rho_max] = cbar_range
        else:
            p_min = min(p_vals.ravel())
            p_max = max(p_vals.ravel())
            u_min = min(u_vals.ravel())
            u_max = max(u_vals.ravel())
            if (abs(u_min)>u_max):
                u_max = abs(u_min)
            else:
                u_min = - u_max
            v_min = min(v_vals.ravel())
            v_max = max(v_vals.ravel())
            if (abs(v_min)>v_max):
                v_max = abs(v_min)
            else:
                v_min = - v_max
            rho_min = min(rho_vals.ravel())
            rho_max = max(rho_vals.ravel())

        fig, axs = plt.subplots(2,2,sharex=True)

        my_plot0 = axs[0,0].pcolormesh(x_vals, y_vals, p_vals, cmap='YlOrRd', vmin=p_min, vmax=p_max)
        ax_cb0 = fig.add_axes([1.0, 0.1, .02, 0.8])
        cb0 = plt.colorbar(my_plot0,ax=axs[0,0],cax=ax_cb0) 

        my_plot1 = axs[1,0].pcolormesh(x_vals, y_vals, u_vals, cmap='PRGn', vmin=u_min, vmax=u_max)
        ax_cb1 = fig.add_axes([1.1, 0.1, .02, 0.8])
        cb1 = plt.colorbar(my_plot1,ax=axs[1,0],cax=ax_cb1)

        my_plot2 = axs[1,1].pcolormesh(x_vals, y_vals, v_vals, cmap='BrBG', vmin=v_min, vmax=v_max)
        ax_cb2 = fig.add_axes([1.2, 0.1, .02, 0.8])
        cb2 = plt.colorbar(my_plot2,ax=axs[1,1],cax=ax_cb2)

        my_plot3 = axs[0,1].pcolormesh(x_vals, y_vals, rho_vals, cmap='Blues', vmin=rho_min, vmax=rho_max)
        ax_cb3 = fig.add_axes([1.3, 0.1, .02, 0.8])
        cb3 = plt.colorbar(my_plot3,ax=axs[0,1],cax=ax_cb3)

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
        axs[1,0].set_aspect('equal')
        axs[1,0].set_title('X Velocity')
        axs[1,1].set_aspect('equal')
        axs[1,1].set_title('Y Velocity')
        axs[0,1].set_aspect('equal')
        axs[0,1].set_title('Density')
        axs[0,0].set_xlim(xlims)
        axs[0,0].set_xlim(ylims)
        plt.show()
        
    def plot_gradients_2D_square(self,cbar_range=None):
        """ Plots primitive variable gradients for each variable
        Args:
            self
        Returns:
            None
        """
        x_vals = self.sim.cc_data.grid.x
        y_vals = self.sim.cc_data.grid.y
        rho_vals = np.transpose(self.sim.cc_data.prim_gradients[0,:,:])
        u_vals = np.transpose(self.sim.cc_data.prim_gradients[1,:,:])
        v_vals = np.transpose(self.sim.cc_data.prim_gradients[2,:,:])
        p_vals = np.transpose(self.sim.cc_data.prim_gradients[3,:,:])

        if (cbar_range):
            [p_min,p_max,u_min,u_max,v_min,v_max,rho_min,rho_max] = cbar_range
        else:
            p_min = min(p_vals.ravel())
            p_max = max(p_vals.ravel())
            u_min = min(u_vals.ravel())
            u_max = max(u_vals.ravel())
            if (abs(u_min)>u_max):
                u_max = abs(u_min)
            else:
                u_min = - u_max
            v_min = min(v_vals.ravel())
            v_max = max(v_vals.ravel())
            if (abs(v_min)>v_max):
                v_max = abs(v_min)
            else:
                v_min = - v_max
            rho_min = min(rho_vals.ravel())
            rho_max = max(rho_vals.ravel())

        fig, axs = plt.subplots(2,2,sharex=True) 

        my_plot0 = axs[0,0].pcolormesh(x_vals, y_vals, p_vals, cmap='YlOrRd', vmin=p_min, vmax=p_max)
        ax_cb0 = fig.add_axes([1.0, 0.1, .02, 0.8])
        cb0 = plt.colorbar(my_plot0,ax=axs[0,0],cax=ax_cb0) 

        my_plot1 = axs[1,0].pcolormesh(x_vals, y_vals, u_vals, cmap='PRGn', vmin=u_min, vmax=u_max)
        ax_cb1 = fig.add_axes([1.1, 0.1, .02, 0.8])
        cb1 = plt.colorbar(my_plot1,ax=axs[1,0],cax=ax_cb1)

        my_plot2 = axs[1,1].pcolormesh(x_vals, y_vals, v_vals, cmap='BrBG', vmin=v_min, vmax=v_max)
        ax_cb2 = fig.add_axes([1.2, 0.1, .02, 0.8])
        cb2 = plt.colorbar(my_plot2,ax=axs[1,1],cax=ax_cb2)

        my_plot3 = axs[0,1].pcolormesh(x_vals, y_vals, rho_vals, cmap='Blues', vmin=rho_min, vmax=rho_max)
        ax_cb3 = fig.add_axes([1.3, 0.1, .02, 0.8])
        cb3 = plt.colorbar(my_plot3,ax=axs[0,1],cax=ax_cb3)

        axs[0,0].set_xlim([0.,1.])

        axs[0,0].set_aspect('equal')
        axs[0,0].set_title('Pressure Gradient')
        axs[1,0].set_aspect('equal')
        axs[1,0].set_title('X Velocity Gradient')
        axs[1,1].set_aspect('equal')
        axs[1,1].set_title('Y Velocity Gradient')
        axs[0,1].set_aspect('equal')
        axs[0,1].set_title('Density Gradient')
        plt.show()

