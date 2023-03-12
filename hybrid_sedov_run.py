import numpy as np
import time
from custom_hybrid import Hybrid_sim

filepath = 'saved_data/sedov/'
file_name = 'hybrid_sedov_01_25/' #

solver = "compressible"
problem_name = "sedov"
param_file = "inputs.sedov"
other_commands = ["driver.max_steps=2400","driver.tmax=0.05","vis.dovis=0",
                  "mesh.nx=25","mesh.ny=25"] #

dx = 0.01 # 

# PySPH Scheme Parameters
adke_params = [1.0,1.0,0.5,0.5,1.0,0.8]
mpm_params = [1.5,2.0,10.0,1.0,None,None]
gsph_params = [1.5,None,0.1,0.5,1,2]
params = [adke_params,mpm_params,gsph_params]

hybrid_sedov = Hybrid_sim()
hybrid_sedov.initialize_pyro(solver,problem_name,param_file,other_commands)
hybrid_sedov.initialize_pysph_sedov(dx=dx,xmin=0.0,xmax=1.00,ymin=0.0,ymax=1.0,gamma=1.4,kf=1.5,xcntr=0.5,ycntr=0.5,
                                    r_init=dx,gaussian=False,DoDomain=False,mirror_x=False,mirror_y=False,adaptive=False,
                                    cfl=0.3,pfreq=10000,tf=0.05,dt=1e-6,scheme='gsph',scheme_params=params[2]) #

hybrid_sedov.initialize_particle_map(4) #
hybrid_sedov.pyro_sim.primitive_update()
hybrid_sedov.pysph_sim.particle_sort()

count = 0 
sub_count = 0

# hybrid parameters
active_width = 1 # 4
boundary_width = 1 # 4
thresholds = [12,12,12,12] # [12,12,12,12] # [24,24,24,24]

# Make Values Uniform
hybrid_sedov.pyro_sim.sim.cc_data.data[:,:,1] = 1e-9 # 2.5e-5
hybrid_sedov.pyro_sim.sim.cc_data.prim_array[:,:,3] = 0 # 1e-5

# Calculate Cell Volume
unit_volume = hybrid_sedov.pyro_sim.sim.cc_data.grid.dx * hybrid_sedov.pyro_sim.sim.cc_data.grid.dy

# Inject energy (either odd or even number of cells)
if (len(hybrid_sedov.pyro_sim.sim.cc_data.data[:,:,1])%2==0):
    lower_index = len(hybrid_sedov.pyro_sim.sim.cc_data.data)//2 - 1
    hybrid_sedov.pyro_sim.sim.cc_data.data[lower_index:lower_index+2,lower_index:lower_index+2,1] = 0.25/unit_volume
    # Calculate Pressure (e = p / (gamma-1)); gamma = 1.4
    hybrid_sedov.pyro_sim.sim.cc_data.prim_array[:,:,3] = 0.4 * hybrid_sedov.pyro_sim.sim.cc_data.data[:,:,1]
else:
    index = len(hybrid_sedov.pyro_sim.sim.cc_data.data)//2
    hybrid_sedov.pyro_sim.sim.cc_data.data[index,index,1] = 1.0/unit_volume
    hybrid_sedov.pyro_sim.sim.cc_data.prim_array[:,:,3] = 0.4 * hybrid_sedov.pyro_sim.sim.cc_data.data[:,:,1]

start = time.time()
pyro_dt = 1e-3
for j in range(50): # 
    hybrid_sedov.pyro_sim.pyro_step(adaptive=False, dt=pyro_dt)
    hybrid_sedov.pyro_sim.primitive_update()
    hybrid_sedov.pyro_sim.compute_gradients()
    hybrid_sedov.pyro_sim.flag_gradients(threshold=thresholds)
    hybrid_sedov.pyro_sim.pysph_zoning(active_width,boundary_width)
    hybrid_sedov.extend_flag_to_particle_map()
    hybrid_sedov.update_particle_map()
    if (count == 0):
        hybrid_sedov.injection_map.data[:,:,0] = 0
    count += 1
    hybrid_sedov.inject_particles()
    hybrid_sedov.pysph_sim.particles[0].align_particles() ##
    hybrid_sedov.pysph_sim.particle_sort()
    hybrid_sedov.flag_pysph_particles_old() # _old
    hybrid_sedov.pysph_sim.remove_flagged_particles()
    hybrid_sedov.pysph_sim.particles[0].align_particles() ##
    hybrid_sedov.pysph_sim.particle_sort()
    if (j>0):
        hybrid_sedov.pyro_to_pysph() ##
    hybrid_sedov.set_boundary_values() # _adke
    while (hybrid_sedov.pysph_sim.solver.t < hybrid_sedov.pyro_sim.sim.cc_data.t):
        if ((hybrid_sedov.pysph_sim.solver.t+hybrid_sedov.pysph_sim.solver.dt)>hybrid_sedov.pyro_sim.sim.cc_data.t):
            hybrid_sedov.pysph_sim.solver.dt = hybrid_sedov.pyro_sim.sim.cc_data.t - hybrid_sedov.pysph_sim.solver.t
        elif(sub_count > 5):
            hybrid_sedov.pysph_sim.particles[0].dt_cfl = np.where(hybrid_sedov.pysph_sim.particles[0].cs>0,hybrid_sedov.pysph_sim.particles[0].h/hybrid_sedov.pysph_sim.particles[0].cs,1e-4)
            hybrid_sedov.pysph_sim.solver.dt = hybrid_sedov.pysph_sim.cfl*hybrid_sedov.pysph_sim.particles[0].dt_cfl.min()
        hybrid_sedov.pysph_sim.step(1)
        sub_count += 1
        hybrid_sedov.pysph_sim.particle_sort()
        hybrid_sedov.get_boundary_values() # _adke
        #print("Finished PySPH step " + str(sub_count) + ". t=" + str(round(hybrid_sedov.pysph_sim.solver.t,6)) + ", dt = " + str(hybrid_sedov.pysph_sim.solver.dt) + " ")
    hybrid_sedov.pysph_to_pyro()
    hybrid_sedov.pyro_sim.conservative_update()
    hybrid_sedov.pyro_sim.sim.cc_data.fill_BC_all()
    #print("COMPLETED LARGE LOOP NUMBER " + str(j+1) + ". Pyro t=" + str(count*pyro_dt) + " ")
end = time.time()

hybrid_shocktube.save_time(filepath + file_name,(end-start))
hybrid_shocktube.save_pysph_data(filepath + file_name + 'pysph_data/')
hybrid_shocktube.save_pyro_data(filepath + file_name + 'pyro_data/')