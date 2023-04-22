import numpy as np
import time
from custom_hybrid import Hybrid_sim

dx = 0.01
dx_str = '01'
filepath = 'saved_data/riemann/'
max_steps = 10000

adke_params = [1.0,1.0,0.5,0.5,1.0,0.8]
mpm_params = [1.5,2.0,10.0,1.0,None,None]
gsph_params = [1.5,None,0.25,0.5,1,2]
params = [adke_params,mpm_params,gsph_params]

pysph_riemann = Hybrid_sim()
pysph_riemann.initialize_pysph_riemann(dx=dx,xmin=0,xmax=1.0,ymin=0,ymax=1.0,gamma=1.4,kf=1.5,xcntr=0.5,ycntr=0.5,
                                    r_init=None,gaussian=False,DoDomain=True,mirror_x=True,mirror_y=True,adaptive=False,
                                    cfl=0.5,pfreq=50000,tf=0.25,dt=1e-4,scheme='gsph',scheme_params=params[2]) #

count = 0
start = time.time()
pysph_riemann.set_boundary_values()
while (pysph_riemann.pysph_sim.solver.t<pysph_riemann.pysph_sim.solver.tf):
    if ((pysph_riemann.pysph_sim.solver.t+pysph_riemann.pysph_sim.solver.dt)>pysph_riemann.pysph_sim.solver.tf):
        pysph_riemann.pysph_sim.solver.dt = pysph_riemann.pysph_sim.solver.tf - pysph_riemann.pysph_sim.solver.t
    pysph_riemann.pysph_sim.step(1)
    count += 1
    pysph_riemann.pysph_sim.particle_sort()
    pysph_riemann.get_boundary_values()
    pysph_riemann.pysph_sim.boundary_set()
    pysph_riemann.pysph_sim.particles[0].dt_cfl = pysph_riemann.pysph_sim.particles[0].h/pysph_riemann.pysph_sim.particles[0].cs
    pysph_riemann.pysph_sim.solver.dt=pysph_riemann.pysph_sim.cfl*pysph_riemann.pysph_sim.particles[0].dt_cfl.min()
    if (count > max_steps):
        break
end = time.time()

pysph_riemann.save_time(filepath + 'pysph_riemann_' + dx_str + '/',(end-start))
pysph_riemann.save_pysph_data(filepath + 'pysph_riemann_' + dx_str + '/')