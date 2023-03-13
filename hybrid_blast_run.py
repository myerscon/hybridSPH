import numpy as np
import time
from custom_hybrid import Hybrid_sim

filepath = 'saved_data/blast/'
file_name = 'hybrid_blast_01_25/' #

solver = "compressible"
problem_name = "sedov"
param_file = "inputs.sedov"
other_commands = ["driver.max_steps=2400","driver.tmax=0.025","vis.dovis=0",
                  "mesh.nx=25","mesh.ny=25"] #

dx = 0.01 # 

# PySPH Scheme Parameters
adke_params = [1.0,1.0,0.5,0.5,1.0,0.8] # 
mpm_params = [1.5,2.0,10.0,1.0,None,None] # 
gsph_params = [1.5,None,0.1,0.5,1,2]
params = [adke_params,mpm_params,gsph_params]

hybrid_blast = Hybrid_sim()
hybrid_blast.initialize_pyro(solver,problem_name,param_file,other_commands)
hybrid_blast.initialize_pysph_blast(dx=dx,xmin=0.0,xmax=1.00,ymin=0.0,ymax=1.0,gamma=1.4,kf=1.5,xcntr=0.5,ycntr=0.5,
                                    r_init=0.1,gaussian=False,DoDomain=False,mirror_x=False,mirror_y=False,adaptive=False,
                                    cfl=0.3,pfreq=10000,tf=0.025,dt=1e-4,scheme='gsph',scheme_params=params[2]) #

hybrid_blast.initialize_particle_map(4) #
hybrid_blast.pyro_sim.primitive_update()
hybrid_blast.pysph_sim.particle_sort()

count = 0 
sub_count = 0

# hybrid parameters
active_width = 2 # 4
boundary_width = 2 # 4
thresholds = [24,24,24,24] # 

hybrid_blast.pysph_sim.particles[0].e = np.where(hybrid_blast.pysph_sim.particles[0].p>0,400.0,1e-9)
hybrid_blast.pysph_sim.particles[0].rho = np.where(hybrid_blast.pysph_sim.particles[0].p>0,4.0,1.0)
hybrid_blast.pysph_sim.particles[0].p = 0.4*hybrid_blast.pysph_sim.particles[0].rho*hybrid_blast.pysph_sim.particles[0].e

# cylinder center
x_cntr = 0.5
y_cntr = 0.5
r_init = 0.1
EnergyDensity = 100.0
Density = 4.0

# Make Values Uniform
hybrid_blast.pyro_sim.sim.cc_data.data[:,:,1] = 1e-9 # 2.5e-5
hybrid_blast.pyro_sim.sim.cc_data.prim_array[:,:,3] = 0 # 1e-5

# Calculate Cell Volume
unit_volume = hybrid_blast.pyro_sim.sim.cc_data.grid.dx * hybrid_blast.pyro_sim.sim.cc_data.grid.dy

# Constant energy and density within volume, pressure calculated after
dist = np.sqrt((hybrid_blast.pyro_sim.sim.cc_data.grid.x2d-x_cntr)**2+(hybrid_blast.pyro_sim.sim.cc_data.grid.y2d-y_cntr)**2)
hybrid_blast.pyro_sim.sim.cc_data.data[:,:,1] = np.where(dist<r_init,EnergyDensity,1e-9)
hybrid_blast.pyro_sim.sim.cc_data.data[:,:,0] = np.where(dist<r_init,Density,1.0)
hybrid_blast.pyro_sim.primitive_update()
# temp
hybrid_blast.pyro_sim.sim.cc_data.prim_array[:,:,3] = 0.4 * hybrid_blast.pyro_sim.sim.cc_data.data[:,:,1] * hybrid_blast.pyro_sim.sim.cc_data.data[:,:,0]
hybrid_blast.pyro_sim.conservative_update()

start = time.time()
pyro_dt = 1e-3
for j in range(25): # 
    hybrid_blast.pyro_sim.pyro_step(adaptive=False, dt=pyro_dt) # 
    hybrid_blast.pyro_sim.primitive_update()
    hybrid_blast.pyro_sim.compute_gradients()
    hybrid_blast.pyro_sim.flag_gradients(threshold=thresholds)
    hybrid_blast.pyro_sim.pysph_zoning(active_width,boundary_width)
    hybrid_blast.extend_flag_to_particle_map()
    hybrid_blast.update_particle_map()
    if (count == 0):
        hybrid_blast.injection_map.data[:,:,0] = 0
    count += 1
    hybrid_blast.inject_particles()
    hybrid_blast.pysph_sim.particles[0].align_particles() ##
    hybrid_blast.pysph_sim.particle_sort()
    hybrid_blast.flag_pysph_particles_old() # _old
    hybrid_blast.pysph_sim.remove_flagged_particles()
    hybrid_blast.pysph_sim.particles[0].align_particles() ##
    hybrid_blast.pysph_sim.particle_sort()
    if (j>0):
        hybrid_blast.pyro_to_pysph()
    hybrid_blast.set_boundary_values() # _adke
    while (hybrid_blast.pysph_sim.solver.t < hybrid_blast.pyro_sim.sim.cc_data.t):
        if ((hybrid_blast.pysph_sim.solver.t+hybrid_blast.pysph_sim.solver.dt)>hybrid_blast.pyro_sim.sim.cc_data.t):
            hybrid_blast.pysph_sim.solver.dt = hybrid_blast.pyro_sim.sim.cc_data.t - hybrid_blast.pysph_sim.solver.t
        elif(sub_count > 5):
            hybrid_blast.pysph_sim.particles[0].dt_cfl = np.where(hybrid_blast.pysph_sim.particles[0].cs>0,hybrid_blast.pysph_sim.particles[0].h/hybrid_blast.pysph_sim.particles[0].cs,1.0)
            hybrid_blast.pysph_sim.solver.dt = hybrid_blast.pysph_sim.cfl*hybrid_blast.pysph_sim.particles[0].dt_cfl.min()
        hybrid_blast.pysph_sim.step(1)
        sub_count += 1
        hybrid_blast.pysph_sim.particle_sort()
        hybrid_blast.get_boundary_values() # _adke
        #print("Finished PySPH step " + str(sub_count) + ". t=" + str(round(hybrid_blast.pysph_sim.solver.t,6)) + ", dt = " + str(hybrid_blast.pysph_sim.solver.dt) + " ")
    hybrid_blast.pysph_to_pyro()
    hybrid_blast.pyro_sim.conservative_update()
    hybrid_blast.pyro_sim.sim.cc_data.fill_BC_all()
    #print("COMPLETED LARGE LOOP NUMBER " + str(j+1) + ". Pyro t=" + str(count*pyro_dt) + " ")
end = time.time()

hybrid_blast.save_time(filepath + file_name,(end-start))
hybrid_blast.save_pysph_data(filepath + file_name + 'pysph_data/')
hybrid_blast.save_pyro_data(filepath + file_name + 'pyro_data/')