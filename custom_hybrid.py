import numpy as np
import matplotlib.pyplot as plt
import pickle

from custom_pyro import CustomPyro
from custom_pysph_shocktube import CustomShockTube2D
from custom_pysph_sedov import CustomSedov
from custom_pysph_blast import CustomBlast
from mesh.patch import Grid2d, CellCenterData2d
from mesh.boundary import BC

class Hybrid_sim():

    ###
    ### INITIALIZATION FUNCTIONS
    ###
    
    def initialize_pyro(self,solver,problem_name,param_file,other_commands):
        self.solver = solver
        self.problem_name = problem_name
        self.param_file = param_file
        self.other_commands = other_commands
        self.pyro_sim = CustomPyro(self.solver)
        self.pyro_sim.initialize_problem(self.problem_name,self.param_file,
                                         other_commands=self.other_commands)
        self.pyro_sim.primitive_update()
        
    def initialize_pysph_shocktube(self,dx,xmin,xmax,ymin,ymax,gamma,kf,DoDomain,mirror_x,mirror_y,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params):
        self.pysph_sim = CustomShockTube2D(dx,xmin,xmax,ymin,ymax,gamma,kf,DoDomain,mirror_x,mirror_y,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params)
        self.pysph_sim.setup()
        self.pysph_sim.particles[0].add_property(name='particle_type_id',type='double',default=2.0)
        self.pysph_sim.particles[0].add_property(name='id',type='int',default=-1)
        self.pysph_sim.particles[0].add_property(name='dt_cfl',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='e1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='h1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='m1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='p1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='rho1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='u1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='v1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='x1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='y1',type='double',default=0.0)

    def initialize_pysph_sedov(self,dx,xmin,xmax,ymin,ymax,gamma,kf,DoDomain,mirror_x,mirror_y,xcntr,ycntr,r_init,gaussian,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params):
        self.pysph_sim = CustomSedov(dx,xmin,xmax,ymin,ymax,gamma,kf,DoDomain,mirror_x,mirror_y,xcntr,ycntr,r_init,gaussian,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params)
        self.pysph_sim.setup()
        self.pysph_sim.particles[0].add_property(name='particle_type_id',type='double',default=2.0)
        self.pysph_sim.particles[0].add_property(name='id',type='int',default=-1)
        self.pysph_sim.particles[0].add_property(name='dt_cfl',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='e1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='h1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='m1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='p1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='rho1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='u1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='v1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='x1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='y1',type='double',default=0.0)

    def initialize_pysph_blast(self,dx,xmin,xmax,ymin,ymax,gamma,kf,DoDomain,mirror_x,mirror_y,xcntr,ycntr,r_init,gaussian,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params):
        self.pysph_sim = CustomSedov(dx,xmin,xmax,ymin,ymax,gamma,kf,DoDomain,mirror_x,mirror_y,xcntr,ycntr,r_init,gaussian,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params)
        self.pysph_sim.setup()
        self.pysph_sim.particles[0].add_property(name='particle_type_id',type='double',default=2.0)
        self.pysph_sim.particles[0].add_property(name='id',type='int',default=-1)
        self.pysph_sim.particles[0].add_property(name='dt_cfl',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='e1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='h1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='m1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='p1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='rho1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='u1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='v1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='x1',type='double',default=0.0)
        self.pysph_sim.particles[0].add_property(name='y1',type='double',default=0.0)
        
    def initialize_particle_map(self,shrink_factor):
        """ Initializes the SPH particle population map based on the underlying Pyro grid and shrink factor.
        Stores the resulting ArrayIndexer object under self.injection_map.
        Args:
            self
            shrink_factor (int): factor to shrink SPH map relative to Pyro grid. Must be a factor of 2.
        Returns:
            None
        """
        assert np.log2(shrink_factor).is_integer(), "shrink_factor must be a factor of 2."
        
        ##2/27# Extract Pyro grid parameters
        #xmin = self.pyro_sim.sim.cc_data.grid.xmin
        #xmax = self.pyro_sim.sim.cc_data.grid.xmax
        #ymin = self.pyro_sim.sim.cc_data.grid.ymin
        #ymax = self.pyro_sim.sim.cc_data.grid.ymax
        #nx = shrink_factor*self.pyro_sim.sim.cc_data.grid.nx
        #ny = shrink_factor*self.pyro_sim.sim.cc_data.grid.ny
        #ng = shrink_factor*self.pyro_sim.sim.cc_data.grid.ng
        #new_grid = Grid2d(nx=nx,ny=ny,ng=ng,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
        new_grid = self.pyro_sim.sim.cc_data.grid.fine_like(shrink_factor)
        self.injection_map = CellCenterData2d(new_grid)
        self.injection_map.grid = new_grid
        self.injection_map.shrink_factor = shrink_factor
        # Boundaries not used, creating outflow boundaries just in case
        bc = BC(xlb="outflow", xrb="outflow",ylb="outflow", yrb="outflow")
        self.injection_map.register_var("particle_map",bc)
        self.injection_map.register_var("gradient_flag_mirror",bc)
        self.injection_map.register_var("gradient_flag_old",bc)

        # Additional Data for PySPH Wall Boundaries:
        self.injection_map.set_aux("x-density",np.zeros(len(self.injection_map.grid.x)))
        self.injection_map.set_aux("x-pressure",np.zeros(len(self.injection_map.grid.x)))
        self.injection_map.set_aux("x-energy",np.zeros(len(self.injection_map.grid.x)))
        self.injection_map.set_aux("x-velocity",np.zeros(len(self.injection_map.grid.x)))
        #self.injection_map.set_aux("y-velocity",np.zeros(len(self.injection_map.grid.x)))

        self.injection_map.create()
        self.injection_map.data[:,:,0] = 1.0

        # Store injection_map unit volume
        self.injection_map.unit_volume = (self.pyro_sim.sim.cc_data.grid.dx * self.pyro_sim.sim.cc_data.grid.dy)/(shrink_factor*shrink_factor)

    ###
    ### ROUTINE FUNCTIONS
    ###
        
    def extend_flag_to_particle_map(self):
        """ Projects the pyro gradient_final array to self.injection_map.data[:,:,1] (particle_flag_mirror)
        Args:
            self
        Returns:
            None
        """
        # Set gradient_flag_old to current gradient_flag_mirror
        self.injection_map.data[:,:,2] = self.injection_map.data[:,:,1]
        self.injection_map.zero("gradient_flag_mirror")
        # Compute new gradient_flag_mirror
        ghosts = self.pyro_sim.sim.cc_data.grid.ng
        shrink = self.injection_map.shrink_factor
        x_len = len(self.injection_map.data[:,0,0]) #
        y_len = len(self.injection_map.data[0,:,0]) #
        for i in range(shrink):
            for j in range(shrink):
                #2/27#self.injection_map.data[i:x_len:shrink,j:y_len:shrink,1] += self.pyro_sim.sim.cc_data.gradient_final[:,:] # expanded to include ghosts
                self.injection_map.data[ghosts+i:-ghosts:shrink,ghosts+j:-ghosts:shrink,1] += self.pyro_sim.sim.cc_data.gradient_final[ghosts:-ghosts,ghosts:-ghosts]
        # injection map centralization fix
        #self.injection_map.data[:,:,1] = np.roll((np.roll(self.injection_map.data[:,:,1],shift=int(shrink/2),axis=0)),shift=-1,axis=1)
        
    def update_particle_map(self):
        """ Calculates the difference between gradient_flag_mirror and gradient_flag_old to update particle_map
        Args:
            self
        Returns:
            None
        """
        # Returns a particle_flag value of 1 when gradient_flag goes from 0 to 1 (new SPH boundary region)
        binary_flag_mirror = np.where(self.injection_map.data[:,:,1]>0,1,0)
        difference = binary_flag_mirror - self.injection_map.data[:,:,2]
        self.injection_map.data[:,:,0] = np.where(difference==1,1,0)
        ## 2/27 update: exclude edges of injection_map
        #self.injection_map.data[:8,:,0] = 0 # np.zeros(len(self.injection_map.data[0,:,0]))
        ##self.injection_map.data[1,:,0] = 0 # np.zeros(len(self.injection_map.data[1,:,0]))
        #self.injection_map.data[-8:,:,0] = 0 # np.zeros(len(self.injection_map.data[-1,:,0]))
        ##self.injection_map.data[-2,:,0] = 0 # np.zeros(len(self.injection_map.data[-2,:,0]))
        #self.injection_map.data[:,:8,0] = 0 # np.zeros(len(self.injection_map.data[:,0,0]))
        ##self.injection_map.data[:,1,0] = 0 # np.zeros(len(self.injection_map.data[:,1,0]))
        #self.injection_map.data[:,-8:,0] = 0 # np.zeros(len(self.injection_map.data[:,-1,0]))
        ##self.injection_map.data[:,-2,0] = 0 # np.zeros(len(self.injection_map.data[:,-2,0]))
        
    def inject_particles(self):
        """ Injects new SPH particles in regions where the injection map is equal to 1.
        Args:
            None
        Returns:
            None
        """
        grid = self.injection_map.grid
        x_vals = np.extract(self.injection_map.data[:,:,0]>0,grid.x2d-((self.injection_map.shrink_factor/2)*self.injection_map.grid.dx)) # 2
        y_vals = np.extract(self.injection_map.data[:,:,0]>0,grid.y2d-((self.injection_map.shrink_factor/2)*self.injection_map.grid.dy)) # 2d sims
        self.pysph_sim.add_particles([x_vals,y_vals])
        
    def flag_pysph_particles(self):
        """ Flags pysph particles based on value of pyro_sim.sim.cc_data.gradient_final at each cell. 
            Particle values (2, 1, or 0) reflect gradient_final values corresponding to active, boundary,
            or void (marked for deletion) particles.
        Args:
            self
        Returns:
            None
        """
        #shrink = self.injection_map.shrink_factor
        x_bins = np.digitize(self.pysph_sim.particles[0].x, self.injection_map.grid.x) # xr # +(self.injection_map.grid.dx*shrink/2)
        y_bins = np.digitize(self.pysph_sim.particles[0].y, self.injection_map.grid.y) # yr # +(self.injection_map.grid.dy*shrink/2)
        self.pysph_sim.particles[0].particle_type_id = self.injection_map.data[x_bins+1,y_bins+1,1] # x_bins,y_bins,1

    def flag_pysph_particles_old(self):
        """ Flags pysph particles based on value of pyro_sim.sim.cc_data.gradient_final at each cell. 
            Particle values (2, 1, or 0) reflect gradient_final values corresponding to active, boundary,
            or void (marked for deletion) particles.
        Args:
            self
        Returns:
            None
        """
        x_bins = np.digitize(self.pysph_sim.particles[0].x, self.pyro_sim.sim.cc_data.grid.x) # xr
        ## addition for larger mesh error. Send to zero which is in pyro boundary cells
        #x_bins = np.where(x_bins<self.pyro_sim.sim.cc_data.grid.nx+2*self.pyro_sim.sim.cc_data.grid.ng-1,x_bins,0) #
        y_bins = np.digitize(self.pysph_sim.particles[0].y, self.pyro_sim.sim.cc_data.grid.y) # yr
        #y_bins = np.where(y_bins<self.pyro_sim.sim.cc_data.grid.ny+2*self.pyro_sim.sim.cc_data.grid.ng-1,y_bins,0) #
        self.pysph_sim.particles[0].particle_type_id = self.pyro_sim.sim.cc_data.gradient_final[x_bins,y_bins] # x_bins,y_bins

    def flag_pysph_wall_particles(self,x_walls=None,y_walls=None):
        """ Sets type id of particles outside the bounds of x_walls or y_walls to 1.1 (code for boundary particles).
        Note: call self.pysph_sim.particle_sort() before
        Args:
            self
            x_walls [double,double]: lower and upper bounds for x wall boundary
            y_walls [double,double]: lower and upper bounds for x wall boundary
        Returns:
            None
        """
        if (x_walls is not None):
            indexes_lower = np.extract(self.pysph_sim.particles[0].x<x_walls[0],self.pysph_sim.particles[0].id)
            indexes_higher = np.extract(self.pysph_sim.particles[0].x>x_walls[1],self.pysph_sim.particles[0].id)
            #indexes = np.concatenate((indexes_lower,indexes_higher))
            #self.pysph_sim.particles[0].particle_type_id[indexes] = np.where(self.pysph_sim.particles[0].particle_type_id>0,1.1,0)
            self.pysph_sim.particles[0].particle_type_id[indexes_lower] = np.where(self.pysph_sim.particles[0].particle_type_id[indexes_lower]>0,1.1,0)
            self.pysph_sim.particles[0].particle_type_id[indexes_higher] = np.where(self.pysph_sim.particles[0].particle_type_id[indexes_higher]>0,1.1,0)
        if (y_walls is not None):
            indexes_lower = np.extract(self.pysph_sim.particles[0].y<y_walls[0],self.pysph_sim.particles[0].id)
            indexes_higher = np.extract(self.pysph_sim.particles[0].y>y_walls[1],self.pysph_sim.particles[0].id)
            #indexes = np.concatenate((indexes_lower,indexes_higher))
            #self.pysph_sim.particles[0].particle_type_id[indexes] = np.where(self.pysph_sim.particles[0].particle_type_id>0,1.1,0)
            self.pysph_sim.particles[0].particle_type_id[indexes_lower] = np.where(self.pysph_sim.particles[0].particle_type_id[indexes_lower]>0,1.1,0)
            self.pysph_sim.particles[0].particle_type_id[indexes_higher] = np.where(self.pysph_sim.particles[0].particle_type_id[indexes_higher]>0,1.1,0)

    ###
    ### PROBLEM SPECIFIC FUNCTIONS
    ###

    def shocktube_vertical_injection(self,boundary_width):
        """ Creates vertically uniform injection for quasi1D shocktube problem.
        Call after extend_flag_to_particle_map() and before update_particle_map()
        Args:
            self
            boundary_width: boundary width value used for flagging
        Returns:
            None
        """
        # Fill gaps first
        # Adding filter to convert boundary cells surrounded by active cells to active cells
        middle_index = len(self.pyro_sim.sim.cc_data.gradient_final[0])//2 # find y-center of gradient final
        middle_slice = self.pyro_sim.sim.cc_data.gradient_final[:,middle_index]
        sliced_array = np.split(middle_slice,np.where(np.diff(middle_slice) != 0)[0]+1)
        # Loop over sliced array, ignoring outer 2 slices on each end (pyro regions and outer boundary regions)
        for i in range(len(sliced_array)-4):
            if (sliced_array[i+2][0] == 1):
                if (len(sliced_array[i+2])<2*boundary_width+1):
                    sliced_array[i+2] *= 2
        # Substitute new middle slice back into gradient_final
        self.pyro_sim.sim.cc_data.gradient_final[:,middle_index] = np.concatenate(sliced_array)

        # For now, create vertical uniformity

        x_len = self.injection_map.grid.nx+2*self.injection_map.grid.ng
        y_len = self.injection_map.grid.ny+2*self.injection_map.grid.ng
        tmp_matrix = np.ones((x_len,y_len))
        for i in range(len(tmp_matrix[0])):
            tmp_matrix[:,i] = np.amax(self.injection_map.data[:,:,1],axis=1)
        self.injection_map.data[:,:,1] = tmp_matrix

        # copy for self.pyro_sim.sim.cc_data.gradient_final while using flag_pysph_particles_old

        x_len = self.pyro_sim.sim.cc_data.grid.nx+2*self.pyro_sim.sim.cc_data.grid.ng
        y_len = self.pyro_sim.sim.cc_data.grid.ny+2*self.pyro_sim.sim.cc_data.grid.ng
        tmp_matrix = np.ones((x_len,y_len))
        for i in range(len(tmp_matrix[0])):
            tmp_matrix[:,i] = np.amax(self.pyro_sim.sim.cc_data.gradient_final[:,:],axis=1)
        self.pyro_sim.sim.cc_data.gradient_final[:,:] = tmp_matrix

    ###
    ### BOUNDARY FUNCTIONS
    ###

    ##### ADKE #####

    def set_boundary_values_adke(self):
        """ Sets old values to current values for all particles
        Args:
            self
        Returns:
            None
        """
        self.pysph_sim.particles[0].copy_over_properties({'e':'e1','h':'h1','m':'m1','p':'p1','rho':'rho1','u':'u1','v':'v1','x':'x1','y':'y1'})

    def get_boundary_values_adke(self):
        """ Sets select values to old values for particles with particle_type_id == 1 or 1.1
        Args:
            self - Should call self.particle_sort() prior to calling
        Returns:
            None
        """
        boundary_ids = np.extract(self.pysph_sim.particles[0].particle_type_id<2.0,self.pysph_sim.particles[0].id)
        self.pysph_sim.particles[0].set_to_zero(['aw','w','w0','z','z0'])
        self.pysph_sim.particles[0].ae[boundary_ids] = 0.0
        self.pysph_sim.particles[0].ah[boundary_ids] = 0.0
        self.pysph_sim.particles[0].am[boundary_ids] = 0.0
        self.pysph_sim.particles[0].arho[boundary_ids] = 0.0
        self.pysph_sim.particles[0].au[boundary_ids] = 0.0
        self.pysph_sim.particles[0].av[boundary_ids] = 0.0
        self.pysph_sim.particles[0].cs[boundary_ids] = 0.0
        self.pysph_sim.particles[0].div[boundary_ids] = 0.0
        self.pysph_sim.particles[0].logrho[boundary_ids] = 0.0

        self.pysph_sim.particles[0].e[boundary_ids] = self.pysph_sim.particles[0].e1[boundary_ids]
        self.pysph_sim.particles[0].h[boundary_ids] = self.pysph_sim.particles[0].h1[boundary_ids]
        self.pysph_sim.particles[0].m[boundary_ids] = self.pysph_sim.particles[0].m1[boundary_ids]
        self.pysph_sim.particles[0].p[boundary_ids] = self.pysph_sim.particles[0].p1[boundary_ids]
        self.pysph_sim.particles[0].rho[boundary_ids] = self.pysph_sim.particles[0].rho1[boundary_ids]
        self.pysph_sim.particles[0].u[boundary_ids] = self.pysph_sim.particles[0].u1[boundary_ids]
        self.pysph_sim.particles[0].v[boundary_ids] = self.pysph_sim.particles[0].v1[boundary_ids]
        self.pysph_sim.particles[0].x[boundary_ids] = self.pysph_sim.particles[0].x1[boundary_ids]
        self.pysph_sim.particles[0].y[boundary_ids] = self.pysph_sim.particles[0].y1[boundary_ids]

    ##### MPM & GPSH #####

    def set_boundary_values(self):
        """ Sets old values to current values for all particles
        Args:
            self
        Returns:
            None
        """
        self.pysph_sim.particles[0].copy_over_properties({'e':'e1','h':'h1','m':'m1','p':'p1','rho':'rho1','u':'u1','v':'v1','x':'x1','y':'y1'})

    def get_boundary_values(self):
        """ Sets select values to old values for particles with particle_type_id == 1 or 1.1
        Args:
            self - Should call self.particle_sort() prior to calling
        Returns:
            None
        """
        boundary_ids = np.extract(self.pysph_sim.particles[0].particle_type_id<2.0,self.pysph_sim.particles[0].id)
        self.pysph_sim.particles[0].set_to_zero(['aw','w','w0','z','z0'])
        self.pysph_sim.particles[0].aalpha1[boundary_ids] = 0.0
        self.pysph_sim.particles[0].aalpha2[boundary_ids] = 0.0
        self.pysph_sim.particles[0].ae[boundary_ids] = 0.0
        self.pysph_sim.particles[0].ah[boundary_ids] = 0.0
        self.pysph_sim.particles[0].alpha1[boundary_ids] = 0.0
        self.pysph_sim.particles[0].alpha10[boundary_ids] = 0.0
        self.pysph_sim.particles[0].alpha2[boundary_ids] = 0.0
        self.pysph_sim.particles[0].alpha20[boundary_ids] = 0.0
        self.pysph_sim.particles[0].am[boundary_ids] = 0.0
        self.pysph_sim.particles[0].arho[boundary_ids] = 0.0
        self.pysph_sim.particles[0].au[boundary_ids] = 0.0
        self.pysph_sim.particles[0].av[boundary_ids] = 0.0
        self.pysph_sim.particles[0].cs[boundary_ids] = 0.0
        self.pysph_sim.particles[0].del2e[boundary_ids] = 0.0
        self.pysph_sim.particles[0].div[boundary_ids] = 0.0
        self.pysph_sim.particles[0].grhox[boundary_ids] = 0.0
        self.pysph_sim.particles[0].grhoy[boundary_ids] = 0.0
        self.pysph_sim.particles[0].grhoz[boundary_ids] = 0.0
        self.pysph_sim.particles[0].omega[boundary_ids] = 0.0

        self.pysph_sim.particles[0].e[boundary_ids] = self.pysph_sim.particles[0].e1[boundary_ids]
        self.pysph_sim.particles[0].h[boundary_ids] = self.pysph_sim.particles[0].h1[boundary_ids]
        self.pysph_sim.particles[0].m[boundary_ids] = self.pysph_sim.particles[0].m1[boundary_ids]
        self.pysph_sim.particles[0].p[boundary_ids] = self.pysph_sim.particles[0].p1[boundary_ids]
        self.pysph_sim.particles[0].rho[boundary_ids] = self.pysph_sim.particles[0].rho1[boundary_ids]
        self.pysph_sim.particles[0].u[boundary_ids] = self.pysph_sim.particles[0].u1[boundary_ids]
        self.pysph_sim.particles[0].v[boundary_ids] = self.pysph_sim.particles[0].v1[boundary_ids]
        self.pysph_sim.particles[0].x[boundary_ids] = self.pysph_sim.particles[0].x1[boundary_ids]
        self.pysph_sim.particles[0].y[boundary_ids] = self.pysph_sim.particles[0].y1[boundary_ids]

    ##### WALL #####

    def set_x_wall_values(self):
        """ Sets x_wall_boundary values for shocktube problems
        Note: Call after particle_sort()
        Args:
            self
        Returns:
            None
        """
        #self.pysph_sim.particle_sort()
        physical_ids = np.extract(self.pysph_sim.particles[0].particle_type_id < 1.1, # == 2.0
                            self.pysph_sim.particles[0].id)
        inside_ids = np.extract(self.pysph_sim.particles[0].particle_type_id > 1.1, # == 1.0
                            self.pysph_sim.particles[0].id)
        real_ids = np.concatenate((physical_ids,inside_ids))
        w = np.histogram(self.pysph_sim.particles[0].x[real_ids],self.injection_map.grid.x)[0]
        w_rho = np.histogram(self.pysph_sim.particles[0].x[real_ids],self.injection_map.grid.x,
                        weights=self.pysph_sim.particles[0].rho[real_ids])[0]
        w_p = np.histogram(self.pysph_sim.particles[0].x[real_ids],self.injection_map.grid.x,
                        weights=self.pysph_sim.particles[0].p[real_ids])[0]
        w_e = np.histogram(self.pysph_sim.particles[0].x[real_ids],self.injection_map.grid.x,
                        weights=self.pysph_sim.particles[0].e[real_ids])[0]
        w_u = np.histogram(self.pysph_sim.particles[0].x[real_ids],self.injection_map.grid.x,
                        weights=self.pysph_sim.particles[0].u[real_ids])[0]
        #w_v = np.histogram(self.pysph_sim.particles[0].x[real_ids],self.injection_map.grid.x,
        #                weights=self.pysph_sim.particles[0].u[real_ids])[0]
        # To avoid Div0, included conditional w=0->1
        inv_w = np.where(w>0,w,1.0)
        self.injection_map.aux['x-density']=w_rho/inv_w
        self.injection_map.aux['x-pressure']=w_p/inv_w
        self.injection_map.aux['x-energy']=w_e/inv_w
        self.injection_map.aux['x-velocity']=w_u/inv_w
        #self.injection_map.aux['y-velocity']=w_v/inv_w

    def get_x_wall_values(self):
        """ Retrieves x_wall_boundary values for particles with type id of 1.1 (wall)
        Args:
            self
        Returns:
            None
        """
        boundary_ids = np.extract(self.pysph_sim.particles[0].particle_type_id == 1.1,
                                self.pysph_sim.particles[0].id)
        x_bins = np.digitize(self.pysph_sim.particles[0].x[boundary_ids],self.injection_map.grid.x) # xr
        self.pysph_sim.particles[0].rho[boundary_ids] = self.injection_map.aux['x-density'][x_bins]
        self.pysph_sim.particles[0].p[boundary_ids] = self.injection_map.aux['x-pressure'][x_bins]
        self.pysph_sim.particles[0].e[boundary_ids] = self.injection_map.aux['x-energy'][x_bins]
        self.pysph_sim.particles[0].u[boundary_ids] = self.injection_map.aux['x-velocity'][x_bins]
        #self.pysph_sim.particles[0].v[boundary_ids] = self.injection_map.aux['y-velocity'][x_bins]

    ###
    ### TRANSFER FUNCTIONS
    ###
        
    def pyro_to_pysph(self,mode=None):
        """ Transfers pyro primitive variables to pysph particles via 2d linear interpolation
        Args:
            self: Must run self.pysph_sim.particle_sort() prior to initialize/set value: id
            mode (str): 'all'; extrapolate values to all pysph particles
                        None (default); extrapolate only to pysph particles marked as boundaries
        Returns:
            None
        """
        # Note: Using extract_particles creates a new array, increasing function time from ~2ms to ~10ms min.
        if (mode == 'all'):
            extracted_ids = self.pysph_sim.particles[0].id
        #elif (mode == 'boundary'):
        #    extracted_ids = np.extract(self.pysph_sim.particles[0].particle_type_id == 1.1,self.pysph_sim.particles[0].id)
        else:
            # 2/28: modifying to include wall boundary particles
            #extracted_ids = np.extract(self.pysph_sim.particles[0].particle_type_id == 1,self.pysph_sim.particles[0].id)
            extracted_ids0 = np.extract(self.pysph_sim.particles[0].particle_type_id > 0,self.pysph_sim.particles[0].id)
            extracted_ids = np.extract(self.pysph_sim.particles[0].particle_type_id[extracted_ids0] < 2,self.pysph_sim.particles[0].id)
            # attempted fix
            extracted_ids = extracted_ids[np.argsort(extracted_ids)]
        particles_x = self.pysph_sim.particles[0].x[extracted_ids]
        particles_y = self.pysph_sim.particles[0].y[extracted_ids]
        
        x_bins = np.digitize(particles_x, self.pyro_sim.sim.cc_data.grid.x) # xr
        y_bins = np.digitize(particles_y, self.pyro_sim.sim.cc_data.grid.y) # yr
        
        inv_dx = 1/self.pyro_sim.sim.cc_data.grid.dx
        inv_dy = 1/self.pyro_sim.sim.cc_data.grid.dy
        
        x_1_i = (self.pyro_sim.sim.cc_data.grid.x[x_bins] - particles_x)*inv_dx
        x_i_0 = (particles_x - self.pyro_sim.sim.cc_data.grid.x[x_bins-1])*inv_dx
        y_1_i = (self.pyro_sim.sim.cc_data.grid.y[y_bins] - particles_y)*inv_dy
        y_i_0 = (particles_y - self.pyro_sim.sim.cc_data.grid.y[y_bins-1])*inv_dy
        
        self.pysph_sim.particles[0].rho[extracted_ids] = (x_1_i*y_1_i*self.pyro_sim.sim.cc_data.prim_array[x_bins-1,y_bins-1,0]
                                                 + x_i_0*y_1_i*self.pyro_sim.sim.cc_data.prim_array[x_bins,y_bins-1,0]
                                                 + x_1_i*y_i_0*self.pyro_sim.sim.cc_data.prim_array[x_bins-1,y_bins,0]
                                                 + x_i_0*y_i_0*self.pyro_sim.sim.cc_data.prim_array[x_bins-1,y_bins,0])
        self.pysph_sim.particles[0].u[extracted_ids] = (x_1_i*y_1_i*self.pyro_sim.sim.cc_data.prim_array[x_bins-1,y_bins-1,1]
                                                 + x_i_0*y_1_i*self.pyro_sim.sim.cc_data.prim_array[x_bins,y_bins-1,1]
                                                 + x_1_i*y_i_0*self.pyro_sim.sim.cc_data.prim_array[x_bins-1,y_bins,1]
                                                 + x_i_0*y_i_0*self.pyro_sim.sim.cc_data.prim_array[x_bins-1,y_bins,1])
        self.pysph_sim.particles[0].v[extracted_ids] = (x_1_i*y_1_i*self.pyro_sim.sim.cc_data.prim_array[x_bins-1,y_bins-1,2]
                                                 + x_i_0*y_1_i*self.pyro_sim.sim.cc_data.prim_array[x_bins,y_bins-1,2]
                                                 + x_1_i*y_i_0*self.pyro_sim.sim.cc_data.prim_array[x_bins-1,y_bins,2]
                                                 + x_i_0*y_i_0*self.pyro_sim.sim.cc_data.prim_array[x_bins-1,y_bins,2])
        self.pysph_sim.particles[0].p[extracted_ids] = (x_1_i*y_1_i*self.pyro_sim.sim.cc_data.prim_array[x_bins-1,y_bins-1,3]
                                                 + x_i_0*y_1_i*self.pyro_sim.sim.cc_data.prim_array[x_bins,y_bins-1,3]
                                                 + x_1_i*y_i_0*self.pyro_sim.sim.cc_data.prim_array[x_bins-1,y_bins,3]
                                                 + x_i_0*y_i_0*self.pyro_sim.sim.cc_data.prim_array[x_bins-1,y_bins,3])
        #self.pysph_sim.particles[0].e[extracted_ids] = (x_1_i*y_1_i*self.pyro_sim.sim.cc_data.data[x_bins-1,y_bins-1,1]
        #                                         + x_i_0*y_1_i*self.pyro_sim.sim.cc_data.data[x_bins,y_bins-1,1]
        #                                         + x_1_i*y_i_0*self.pyro_sim.sim.cc_data.data[x_bins-1,y_bins,1]
        #                                         + x_i_0*y_i_0*self.pyro_sim.sim.cc_data.data[x_bins-1,y_bins,1])
        # 3/1: e = p / (self.gamma1 * rho)
        self.pysph_sim.particles[0].e[extracted_ids] = self.pysph_sim.particles[0].p[extracted_ids]/(self.pysph_sim.gamma1*self.pysph_sim.particles[0].rho[extracted_ids])
        self.pysph_sim.particles[0].m[extracted_ids] = self.injection_map.unit_volume * self.pysph_sim.particles[0].rho[extracted_ids]
        #self.pysph_sim.particles[0].h[extracted_ids] = np.sqrt(self.pysph_sim.particles[0].m[extracted_ids]/self.pysph_sim.particles[0].rho[extracted_ids])
        self.pysph_sim.particles[0].h[extracted_ids] = self.pysph_sim.kernel_factor * self.pysph_sim.dx

    def pysph_to_pyro(self,mode=None):
        """ Transfers pysph field data to pyro primitive variables via 2d linear interpolation
        Args:
            self
            mode (str): 'all'; extrapolate values to all pysph particles
                        None (default); extrapolate only to pysph particles marked as boundaries
        Returns:
            None
        Note: Test a version combining w,rho,u,v,p and using higher dimensional np.histogramdd method
        """
        #indicies = np.extract(self.pyro_sim.sim.cc_data.gradient_final>1,)
        ghosts = self.pyro_sim.sim.cc_data.grid.ng
        ghosts1 = ghosts - 1
        
        x_bins = np.digitize(self.pysph_sim.particles[0].x, self.pyro_sim.sim.cc_data.grid.x) # xr
        ## addition for larger mesh error. Send to zero which is in pyro boundary cells
        #x_bins = np.where(x_bins<self.pyro_sim.sim.cc_data.grid.nx+2*self.pyro_sim.sim.cc_data.grid.ng-1,x_bins,0) #
        y_bins = np.digitize(self.pysph_sim.particles[0].y, self.pyro_sim.sim.cc_data.grid.y) # yr
        #y_bins = np.where(y_bins<self.pyro_sim.sim.cc_data.grid.ny+2*self.pyro_sim.sim.cc_data.grid.ng-1,y_bins,0) #

        inv_dx = 1/self.pyro_sim.sim.cc_data.grid.dx
        inv_dy = 1/self.pyro_sim.sim.cc_data.grid.dy

        x_1_i = (self.pyro_sim.sim.cc_data.grid.x[x_bins] - self.pysph_sim.particles[0].x)*inv_dx
        x_i_0 = (self.pysph_sim.particles[0].x - self.pyro_sim.sim.cc_data.grid.x[x_bins-1])*inv_dx
        y_1_i = (self.pyro_sim.sim.cc_data.grid.y[y_bins] - self.pysph_sim.particles[0].y)*inv_dy
        y_i_0 = (self.pysph_sim.particles[0].y - self.pyro_sim.sim.cc_data.grid.y[y_bins-1])*inv_dy

        w00 = x_1_i*y_1_i
        w01 = x_i_0*y_1_i
        w10 = x_1_i*y_i_0
        w11 = x_i_0*y_i_0
        w_rho00 = x_1_i*y_1_i*self.pysph_sim.particles[0].rho
        w_rho01 = x_i_0*y_1_i*self.pysph_sim.particles[0].rho
        w_rho10 = x_1_i*y_i_0*self.pysph_sim.particles[0].rho
        w_rho11 = x_i_0*y_i_0*self.pysph_sim.particles[0].rho
        w_u00 = x_1_i*y_1_i*self.pysph_sim.particles[0].u
        w_u01 = x_i_0*y_1_i*self.pysph_sim.particles[0].u
        w_u10 = x_1_i*y_i_0*self.pysph_sim.particles[0].u
        w_u11 = x_i_0*y_i_0*self.pysph_sim.particles[0].u
        w_v00 = x_1_i*y_1_i*self.pysph_sim.particles[0].v
        w_v01 = x_i_0*y_1_i*self.pysph_sim.particles[0].v
        w_v10 = x_1_i*y_i_0*self.pysph_sim.particles[0].v
        w_v11 = x_i_0*y_i_0*self.pysph_sim.particles[0].v
        w_p00 = x_1_i*y_1_i*self.pysph_sim.particles[0].p
        w_p01 = x_i_0*y_1_i*self.pysph_sim.particles[0].p
        w_p10 = x_1_i*y_i_0*self.pysph_sim.particles[0].p
        w_p11 = x_i_0*y_i_0*self.pysph_sim.particles[0].p
        
        xp = self.pysph_sim.particles[0].x
        yp = self.pysph_sim.particles[0].y
        x_cc = self.pyro_sim.sim.cc_data.grid.x
        y_cc = self.pyro_sim.sim.cc_data.grid.y
        
        hist_w_00 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w00)
        hist_w_01 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w01)
        hist_w_10 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w10)
        hist_w_11 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w11)
        hist_w_rho_00 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w_rho00)
        hist_w_rho_01 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w_rho01)
        hist_w_rho_10 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w_rho10)
        hist_w_rho_11 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w_rho11)
        hist_w_u_00 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w_u00)
        hist_w_u_01 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w_u01)
        hist_w_u_10 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w_u10)
        hist_w_u_11 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w_u11)
        hist_w_v_00 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w_v00)
        hist_w_v_01 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w_v01)
        hist_w_v_10 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w_v10)
        hist_w_v_11 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w_v11)
        hist_w_p_00 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w_p00)
        hist_w_p_01 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w_p01)
        hist_w_p_10 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w_p10)
        hist_w_p_11 = np.histogram2d(x=xp,y=yp,bins=[x_cc,y_cc],weights=w_p11)
        
        hist_w = hist_w_00[0][1:,1:] + hist_w_01[0][:-1,1:] + hist_w_10[0][1:,:-1] + hist_w_11[0][:-1,:-1]
        hist_rho = hist_w_rho_00[0][1:,1:] + hist_w_rho_01[0][:-1,1:] + hist_w_rho_10[0][1:,:-1] + hist_w_rho_11[0][:-1,:-1]
        hist_u = hist_w_u_00[0][1:,1:] + hist_w_u_01[0][:-1,1:] + hist_w_u_10[0][1:,:-1] + hist_w_u_11[0][:-1,:-1]
        hist_v = hist_w_v_00[0][1:,1:] + hist_w_v_01[0][:-1,1:] + hist_w_v_10[0][1:,:-1] + hist_w_v_11[0][:-1,:-1]
        hist_p = hist_w_p_00[0][1:,1:] + hist_w_p_01[0][:-1,1:] + hist_w_p_10[0][1:,:-1] + hist_w_p_11[0][:-1,:-1]
        # Div/0 fix; Check where hist_w>0, meaning where particles exist, and otherwise values will do to zero
        hist_w = np.where(hist_w>0,hist_w,1)
        rho = hist_rho[ghosts1:-ghosts1,ghosts1:-ghosts1]/hist_w[ghosts1:-ghosts1,ghosts1:-ghosts1]
        u = hist_u[ghosts1:-ghosts1,ghosts1:-ghosts1]/hist_w[ghosts1:-ghosts1,ghosts1:-ghosts1]
        v = hist_v[ghosts1:-ghosts1,ghosts1:-ghosts1]/hist_w[ghosts1:-ghosts1,ghosts1:-ghosts1]
        p = hist_p[ghosts1:-ghosts1,ghosts1:-ghosts1]/hist_w[ghosts1:-ghosts1,ghosts1:-ghosts1]
        
        if (mode == 'all'):
            rho_sub = rho
            u_sub = u
            v_sub = v
            p_sub = p
        else:
            rho_sub = np.where(self.pyro_sim.sim.cc_data.gradient_final[ghosts:-ghosts,ghosts:-ghosts]>1,rho,self.pyro_sim.sim.cc_data.prim_array[ghosts:-ghosts,ghosts:-ghosts,0])
            u_sub = np.where(self.pyro_sim.sim.cc_data.gradient_final[ghosts:-ghosts,ghosts:-ghosts]>1,u,self.pyro_sim.sim.cc_data.prim_array[ghosts:-ghosts,ghosts:-ghosts,1])
            v_sub = np.where(self.pyro_sim.sim.cc_data.gradient_final[ghosts:-ghosts,ghosts:-ghosts]>1,v,self.pyro_sim.sim.cc_data.prim_array[ghosts:-ghosts,ghosts:-ghosts,2])
            p_sub = np.where(self.pyro_sim.sim.cc_data.gradient_final[ghosts:-ghosts,ghosts:-ghosts]>1,p,self.pyro_sim.sim.cc_data.prim_array[ghosts:-ghosts,ghosts:-ghosts,3])
        
        self.pyro_sim.sim.cc_data.prim_array[ghosts:-ghosts,ghosts:-ghosts,0] = rho_sub
        self.pyro_sim.sim.cc_data.prim_array[ghosts:-ghosts,ghosts:-ghosts,1] = u_sub
        self.pyro_sim.sim.cc_data.prim_array[ghosts:-ghosts,ghosts:-ghosts,2] = v_sub
        self.pyro_sim.sim.cc_data.prim_array[ghosts:-ghosts,ghosts:-ghosts,3] = p_sub

    ###
    ### PLOTTING FUNCTIONS
    ###
        
    def plot_2D_scatter_map(self,injection_code=1,xlims=[0.,1.],ylims=[0.,0.05]):
        """ Plots gradient flag extrapolated to the injection map
        Args:
            self
            injection_code (int): int corresponding to which injection_map.data values to plot
        Returns:
            None
        """
        x_vals = self.injection_map.grid.x2d
        y_vals = self.injection_map.grid.y2d
        flag_map = self.injection_map.data[:,:,injection_code]

        fig, axs = plt.subplots(1,1,sharex=True)

        my_plot0 = axs.scatter(x_vals, y_vals, c=flag_map, s=10*self.injection_map.grid.dx, cmap='coolwarm')
        axs.set_aspect('equal')
        axs.set_title('Injection Map')
        axs.set_xlim(xlims)
        axs.set_ylim(ylims)
        plt.show()

    ###
    ### IO FUNCTIONS
    ###
        
    def save_pyro_data(self,filepath):
        """ Saves pyro primitive variables as numpy arrays, grid and simulation data as pickles.
        Args:
            self
            filepath (string): file path to saved destination data
        Returns:
            None
        """
        rho_array = self.pyro_sim.sim.cc_data.prim_array[:,:,0].v()
        u_array = self.pyro_sim.sim.cc_data.prim_array[:,:,1].v()
        v_array = self.pyro_sim.sim.cc_data.prim_array[:,:,2].v()
        p_array = self.pyro_sim.sim.cc_data.prim_array[:,:,3].v()
        np.save(filepath + 'desnity.npy',rho_array)
        np.save(filepath + 'u.npy',u_array)
        np.save(filepath + 'v.npy',v_array)
        np.save(filepath + 'pressure.npy',p_array)

        grid = vars(self.pyro_sim.sim.cc_data.grid)
        f = open(filepath + 'grid.pkl','wb')
        pickle.dump(grid,f)
        f.close()

        simulation_data = {'solver':self.solver,'problem_name':self.problem_name,'param_file':self.param_file,
                          'other_commands':self.other_commands,'runtime_parameters':self.pyro_sim.sim.rp,
                          'steps':self.pyro_sim.sim.n,'time':self.pyro_sim.sim.cc_data.t}
        f = open(filepath + 'simulation_data.pkl','wb')
        pickle.dump(simulation_data,f)
        f.close()

    def load_pyro_data_to_object(self,filepath):
        """ Loads pyro primitive variables and grid/simulation data and stores it in the current object
        Args:
            self
            filepath (string): file path to saved destination data
        Returns:
            list containing rho, u, v, p arrays and grid, simulation data dictionaries
        """
        rho_array = np.load(filepath + 'desnity.npy')
        u_array = np.load(filepath + 'u.npy')
        v_array = np.load(filepath + 'v.npy')
        p_array = np.load(filepath + 'pressure.npy')

        with open(filepath + 'grid.pkl', 'rb') as handle:
            grid = pickle.load(handle)
        with open(filepath + 'simulation_data.pkl', 'rb') as handle:
            simulation_data = pickle.load(handle)

        ghosts = grid['ng']

        self.pyro_sim.sim.cc_data.prim_array[ghosts:-ghosts,ghosts:-ghosts,0] = rho_array
        self.pyro_sim.sim.cc_data.prim_array[ghosts:-ghosts,ghosts:-ghosts,1] = u_array
        self.pyro_sim.sim.cc_data.prim_array[ghosts:-ghosts,ghosts:-ghosts,2] = v_array
        self.pyro_sim.sim.cc_data.prim_array[ghosts:-ghosts,ghosts:-ghosts,3] = p_array

        #self.pyro_sim.sim.cc_data.grid = grid
        self.pyro_sim.sim.n = simulation_data['steps']
        self.pyro_sim.sim.cc_data.t = simulation_data['time']
        
    def load_pyro_data(self,filepath):
        """ Loads pyro primitive variables and grid/simulation data, returns
        Args:
            filepath (string): file path to saved destination data
        Returns:
            list containing rho, u, v, p arrays and grid, simulation data dictionaries
        """
        rho_array = np.load(filepath + 'desnity.npy')
        u_array = np.load(filepath + 'u.npy')
        v_array = np.load(filepath + 'v.npy')
        p_array = np.load(filepath + 'pressure.npy')

        with open(filepath + 'grid.pkl', 'rb') as handle:
            grid = pickle.load(handle)
        with open(filepath + 'simulation_data.pkl', 'rb') as handle:
            simulation_data = pickle.load(handle)

        return [rho_array,u_array,v_array,p_array,grid,simulation_data]
    
    def save_time(self,filepath,runtime):
        """ Saves runtime variable to a text file; Saves simulation runtime.
        Args:
            self
            filepath (string): file path to saved data destination
            runtime (float): number recorded as simulation runtime
        Returns:
            None
        """
        f = open(filepath + 'runtime.txt', 'w')
        f.write('Simulation Runtime (seconds): ' + str(runtime))
        f.close()

    def save_pysph_data(self,filepath):
        """ Saves Pysph primitive variables as numpy arrays.
        Args:
            self
            filepath (string): file path to saved destination data
        Returns:
            None
        """
        x_array = self.pysph_sim.particles[0].x
        y_array = self.pysph_sim.particles[0].y
        rho_array = self.pysph_sim.particles[0].rho
        u_array = self.pysph_sim.particles[0].u
        v_array = self.pysph_sim.particles[0].v
        p_array = self.pysph_sim.particles[0].p
        
        np.save(filepath + 'x.npy',x_array)
        np.save(filepath + 'y.npy',y_array)
        np.save(filepath + 'rho.npy',rho_array)
        np.save(filepath + 'u.npy',u_array)
        np.save(filepath + 'v.npy',v_array)
        np.save(filepath + 'p.npy',p_array)

    def load_pysph_data_to_object(self,filepath):
        """ Loads pysph primitive variables and adds particles with values to pysph_sim object.
        Args:
            self
            filepath (string): file path to saved destination data
        Returns:
            list containing x, y, rho, u, v, p arrays
        """
        x_array = np.load(filepath + 'x.npy')
        y_array = np.load(filepath + 'y.npy')
        rho_array = np.load(filepath + 'rho.npy')
        u_array = np.load(filepath + 'u.npy')
        v_array = np.load(filepath + 'v.npy')
        p_array = np.load(filepath + 'p.npy')
        
        # Delete any existing particles
        if (len(self.pysph_sim.particles[0].x)>0):
            self.pysph_sim.particles[0].particle_type_id = np.zeros(len(self.pysph_sim.particles[0].x))
            self.pysph_sim.remove_flagged_particles()
            
        # Initialize new particles
        self.pysph_sim.particles[0].add_particles(**{'x':x_array,'y':y_array,'rho':rho_array,'u':u_array,'v':v_array,'p':p_array})

    def load_pysph_data(self,filepath):
        """ Loads pysph primitive variables and returns as a dictionary.
        Args:
            self
            filepath (string): file path to saved destination data
        Returns:
            list containing x, y, rho, u, v, p arrays
        """
        x_array = np.load(filepath + 'x.npy')
        y_array = np.load(filepath + 'y.npy')
        rho_array = np.load(filepath + 'rho.npy')
        u_array = np.load(filepath + 'u.npy')
        v_array = np.load(filepath + 'v.npy')
        p_array = np.load(filepath + 'p.npy')
        
        return {'x':x_array,'y':y_array,'rho':rho_array,'u':u_array,'v':v_array,'p':p_array}
