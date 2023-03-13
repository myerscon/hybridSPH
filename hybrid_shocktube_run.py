import numpy as np
import time
from custom_hybrid import Hybrid_sim

filepath = 'saved_data/shocktube/'
file_name = 'hybrid_shocktube_001_250/' #

# Simulation Parameters (Pyro/PySPH)
solver = "compressible"
problem_name = "sod"
param_file = "inputs.sod.x"
other_commands = ["driver.max_steps=2400","driver.tmax=0.006","vis.dovis=0",
                  "mesh.nx=250","mesh.ny=12","mesh.xmax=1.0","mesh.ymin=0.0","mesh.ymax=0.048", # 
                  "sod.p_left=1000.0","sod.p_right=0.01","sod.dens_left=1.0","sod.dens_right=1.0"]
dx = 0.001 # 

# PySPH Scheme Parameters
adke_params = [1.0,1.0,0.5,0.5,1.0,0.8]
mpm_params = [1.5,2.0,1.0,0.1,None,None]
gsph_params = [1.5,None,0.25,0.5,1,2]
params = [adke_params,mpm_params,gsph_params]

hybrid_shocktube = Hybrid_sim()
hybrid_shocktube.initialize_pyro(solver,problem_name,param_file,other_commands)
hybrid_shocktube.initialize_pysph_shocktube(dx=dx,xmin=0.0,xmax=1.0,ymin=-0.0045,ymax=0.0525,gamma=1.4,kf=1.7,
                                            DoDomain=False,mirror_x=False,mirror_y=False,adaptive=False,cfl=0.5,
                                            pfreq=10000,tf=0.006,dt=2e-5,scheme='gsph',scheme_params=params[2]) #

hybrid_shocktube.initialize_particle_map(4)
hybrid_shocktube.pyro_sim.primitive_update()
hybrid_shocktube.pysph_sim.particle_sort()

count = 0 # 
sub_count = 0 # for debugging

# hybrid parameters
active_width = 8
boundary_width = 8
thresholds = [4000,4000,4000,4000]
y_walls = [0,0.048]

start = time.time()
pyro_dt = 2e-4
for j in range(int(0.006/2e-4)):
    hybrid_shocktube.pyro_sim.pyro_step(adaptive=False, dt=pyro_dt) # 
    hybrid_shocktube.pyro_sim.primitive_update()
    hybrid_shocktube.pyro_sim.compute_gradients()
    hybrid_shocktube.pyro_sim.flag_gradients(threshold=thresholds)
    hybrid_shocktube.pyro_sim.pysph_zoning(active_width,boundary_width)
    hybrid_shocktube.extend_flag_to_particle_map()
    hybrid_shocktube.shocktube_vertical_injection(boundary_width=boundary_width)
    hybrid_shocktube.update_particle_map()
    if (count == 0):
        hybrid_shocktube.injection_map.data[:,:,0] = 0
    count += 1
    hybrid_shocktube.inject_particles()
    hybrid_shocktube.pysph_sim.particles[0].align_particles() ##
    hybrid_shocktube.pysph_sim.particle_sort()
    hybrid_shocktube.flag_pysph_particles_old() # _old
    hybrid_shocktube.flag_pysph_wall_particles(y_walls=y_walls)
    hybrid_shocktube.pysph_sim.remove_flagged_particles()
    hybrid_shocktube.pysph_sim.particles[0].align_particles() ##
    hybrid_shocktube.pysph_sim.particle_sort()
    hybrid_shocktube.pyro_to_pysph()
    hybrid_shocktube.set_boundary_values()
    while (hybrid_shocktube.pysph_sim.solver.t < hybrid_shocktube.pyro_sim.sim.cc_data.t):
        if ((hybrid_shocktube.pysph_sim.solver.t+hybrid_shocktube.pysph_sim.solver.dt)>hybrid_shocktube.pyro_sim.sim.cc_data.t):
            hybrid_shocktube.pysph_sim.solver.dt = hybrid_shocktube.pyro_sim.sim.cc_data.t - hybrid_shocktube.pysph_sim.solver.t
        elif(sub_count > 5):
            hybrid_shocktube.pysph_sim.particles[0].dt_cfl = np.where(hybrid_shocktube.pysph_sim.particles[0].cs>0,hybrid_shocktube.pysph_sim.particles[0].h/hybrid_shocktube.pysph_sim.particles[0].cs,1.0)
            hybrid_shocktube.pysph_sim.solver.dt = hybrid_shocktube.pysph_sim.cfl*hybrid_shocktube.pysph_sim.particles[0].dt_cfl.min()
        hybrid_shocktube.pysph_sim.step(1)
        sub_count += 1
        hybrid_shocktube.pysph_sim.particle_sort()
        hybrid_shocktube.get_boundary_values()
        hybrid_shocktube.set_x_wall_values()
        hybrid_shocktube.get_x_wall_values()
        #print("Finished PySPH step " + str(sub_count) + ". t=" + str(round(hybrid_shocktube.pysph_sim.solver.t,6)) + ", dt = " + str(hybrid_shocktube.pysph_sim.solver.dt) + " ")
    hybrid_shocktube.pysph_to_pyro()
    hybrid_shocktube.pyro_sim.conservative_update()
    hybrid_shocktube.pyro_sim.sim.cc_data.fill_BC_all()
    #print("COMPLETED LARGE LOOP NUMBER " + str(j+1) + ". Pyro t=" + str(count*pyro_dt) + " ")
end = time.time()

hybrid_shocktube.save_time(filepath + file_name,(end-start))
hybrid_shocktube.save_pysph_data(filepath + file_name + 'pysph_data/')
hybrid_shocktube.save_pyro_data(filepath + file_name + 'pyro_data/')