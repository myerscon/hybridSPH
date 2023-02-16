import time
import numpy as np
from custom_hybrid import Hybrid_sim

filepath = 'saved_data/sedov/pyro_sedov_128/'

solver = "compressible"
problem_name = "sedov"
param_file = "inputs.sedov"
other_commands = ["vis.dovis=0","mesh.nx=128","mesh.ny=128"]

pyro_sedov = Hybrid_sim()
pyro_sedov.initialize_pyro(solver,problem_name,param_file,other_commands)

# Make Values Uniform
pyro_sedov.pyro_sim.sim.cc_data.data[:,:,1] = 2.5e-5
pyro_sedov.pyro_sim.sim.cc_data.prim_array[:,:,3] = 1e-5

# Calculate Cell Volume
unit_volume = pyro_sedov.pyro_sim.sim.cc_data.grid.dx * pyro_sedov.pyro_sim.sim.cc_data.grid.dy

"""
# Inject energy (either odd or even number of cells)
if (len(pyro_sedov.pyro_sim.sim.cc_data.data[:,:,1])%2==0):
    lower_index = len(pyro_sedov.pyro_sim.sim.cc_data.data)//2 - 1
    pyro_sedov.pyro_sim.sim.cc_data.data[lower_index:lower_index+2,lower_index:lower_index+2,1] = 0.25/unit_volume
    # Calculate Pressure (e = p / (gamma-1)); gamma = 1.4
    pyro_sedov.pyro_sim.sim.cc_data.prim_array[:,:,3] = 0.4 * pyro_sedov.pyro_sim.sim.cc_data.data[:,:,1]
else:
    index = len(pyro_sedov.pyro_sim.sim.cc_data.data)//2
    pyro_sedov.pyro_sim.sim.cc_data.data[index,index,1] = 1.0/unit_volume
    pyro_sedov.pyro_sim.sim.cc_data.prim_array[:,:,3] = 0.4 * pyro_sedov.pyro_sim.sim.cc_data.data[:,:,1]
"""

def gaussian_2d(x,y,mu_x,mu_y,sigma):
    xy = (x-mu_x)**2 + (y-mu_y)**2
    return (np.exp(-0.5*(xy)/(sigma**2)))

xcntr = 0.5
ycntr = 0.5
x = pyro_sedov.pyro_sim.sim.cc_data.grid.x2d
y = pyro_sedov.pyro_sim.sim.cc_data.grid.y2d

pyro_sedov.pyro_sim.sim.cc_data.data[:,:,1] = 736.565 * gaussian_2d(x=x,y=y,mu_x=xcntr,mu_y=ycntr,sigma=0.0147) + 1e-9 # 789.459
pyro_sedov.pyro_sim.sim.cc_data.prim_array[:,:,3] = 526.178 * gaussian_2d(x=x,y=y,mu_x=xcntr,mu_y=ycntr,sigma=0.0147) # 526.178 lower 490.924

start = time.time()
count = 0
while (not pyro_sedov.pyro_sim.sim.finished()):
    pyro_sedov.pyro_sim.pyro_step()
    #print("Finished step " + str(count) + "...") # debugging only
    count += 1
    if (count > pyro_sedov.pyro_sim.sim.max_steps):
        break
end = time.time()
pyro_sedov.save_time(filepath,(end-start))

pyro_sedov.pyro_sim.primitive_update()
pyro_sedov.save_pyro_data(filepath)