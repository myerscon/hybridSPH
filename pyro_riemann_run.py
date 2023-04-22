import numpy as np
import time
from custom_hybrid import Hybrid_sim

filepath = 'saved_data/riemann/pyro_riemann_100/' #

solver = "compressible"
problem_name = "quad"
param_file = "inputs.quad"
other_commands = ["driver.tmax=0.25","quadrant.cx=0.5","quadrant.cy=0.5",
                 "mesh.nx=100","mesh.ny=100",
                 "quadrant.p1=1.1","quadrant.rho1=1.1","quadrant.u1=0.0","quadrant.v1=0.0",
                 "quadrant.p2=0.35","quadrant.rho2=0.5065","quadrant.u2=0.8939","quadrant.v2=0.0",
                 "quadrant.p3=1.1","quadrant.rho3=1.1","quadrant.u3=0.8939","quadrant.v3=0.8939",
                 "quadrant.p4=0.35","quadrant.rho4=0.5065","quadrant.u4=0.0","quadrant.v4=0.8939"]

pyro_riemann = Hybrid_sim()
pyro_riemann.initialize_pyro(solver,problem_name,param_file,other_commands)

start = time.time()
count = 0
while (not pyro_riemann.pyro_sim.sim.finished()):
    pyro_riemann.pyro_sim.pyro_step()
    #print("Finished step " + str(count) + "...") # debugging only
    count += 1
    if (count > pyro_riemann.pyro_sim.sim.max_steps):
        break
    #if (count % 10 == 0):
    #    pyro_riemann.pyro_sim.primitive_update()
    #    pyro_riemann.pyro_sim.plot_2D_square(xlims=[0,1],ylims=[0,1])
    #    print("t=" + str(pyro_riemann.pyro_sim.sim.cc_data.t))
end = time.time()


pyro_riemann.pyro_sim.primitive_update()
pyro_riemann.save_time(filepath,(end-start))
pyro_riemann.save_pyro_data(filepath)
