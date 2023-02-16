import time
from custom_hybrid import Hybrid_sim

filepath = 'saved_data/shocktube/pyro_shocktube_1000/'

solver = "compressible"
problem_name = "sod"
param_file = "inputs.sod.x"
other_commands = ["driver.max_steps=2400","driver.tmax=0.012","vis.dovis=0",
                  "mesh.nx=1000","mesh.ny=50","mesh.xmax=1.0","mesh.ymin=0.0","mesh.ymax=0.05",
                  "sod.p_left=1000.0","sod.p_right=0.01","sod.dens_left=1.0","sod.dens_right=1.0"]

pyro_sim1 = Hybrid_sim()
pyro_sim1.initialize_pyro(solver,problem_name,param_file,other_commands)

#pyro_sim1.pyro_sim.primitive_update() # debugging only
#pyro_sim1.pyro_sim.plot_2D_vertical([0,1000,-30,30,-5,5,0,8]) # debugging only

start = time.time()
count = 0
while (not pyro_sim1.pyro_sim.sim.finished()):
    pyro_sim1.pyro_sim.pyro_step() # adaptive=False,dt=(dt)
    print("Finished step " + str(count) + "...") # debugging only
    count += 1
    if (count > pyro_sim1.pyro_sim.sim.max_steps):
        break
end = time.time()
pyro_sim1.save_time(filepath,(end-start))
#print(end-start) # debugging only

pyro_sim1.pyro_sim.primitive_update()
#pyro_sim1.pyro_sim.plot_2D_vertical([0,1000,-30,30,-5,5,0,8]) # debugging only

pyro_sim1.save_pyro_data(filepath)