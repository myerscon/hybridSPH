import numpy as np
import time
from custom_hybrid import Hybrid_sim

filepath = 'saved_data/blast/pyro_blast_100/' #

solver = "compressible"
problem_name = "sedov"
param_file = "inputs.sedov"
other_commands = ["driver.max_steps=2400","driver.tmax=0.025","vis.dovis=0",
                  "mesh.nx=100","mesh.ny=100", #
                  "mesh.xmin=-1.0","mesh.xmax=1.0","mesh.ymin=-1.0","mesh.ymax=1.0"] 

pyro_blast = Hybrid_sim()
pyro_blast.initialize_pyro(solver,problem_name,param_file,other_commands)

# cylinder center
x_cntr = 0.0
y_cntr = 0.0
r_init = 0.1
EnergyDensity = 100.0
Density = 1.001 # 10.0

# Make Values Uniform
pyro_blast.pyro_sim.sim.cc_data.data[:,:,1] = 1e-9 # 2.5e-5
pyro_blast.pyro_sim.sim.cc_data.prim_array[:,:,3] = 0.01 # 1e-5

# Calculate Cell Volume
unit_volume = pyro_blast.pyro_sim.sim.cc_data.grid.dx * pyro_blast.pyro_sim.sim.cc_data.grid.dy

# Constant energy and density within volume, pressure calculated after
dist = np.sqrt((pyro_blast.pyro_sim.sim.cc_data.grid.x2d-x_cntr)**2+(pyro_blast.pyro_sim.sim.cc_data.grid.y2d-y_cntr)**2)
pyro_blast.pyro_sim.sim.cc_data.data[:,:,0] = np.where(dist<r_init,Density,1.0)
pyro_blast.pyro_sim.primitive_update()
pyro_blast.pyro_sim.sim.cc_data.prim_array[:,:,3] = np.where(dist<r_init,1000,0.01) # 1000
pyro_blast.pyro_sim.conservative_update()

# pyro run
start = time.time()
count = 0
while (not pyro_blast.pyro_sim.sim.finished()):
    pyro_blast.pyro_sim.pyro_step()
    #print("Finished step " + str(count) + "...") # debugging only
    count += 1
    if (count > pyro_blast.pyro_sim.sim.max_steps):
        break
end = time.time()
pyro_blast.pyro_sim.primitive_update()
pyro_blast.save_time(filepath,(end-start))
pyro_blast.save_pyro_data(filepath)
