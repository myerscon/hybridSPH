import time
from custom_hybrid import Hybrid_sim

dx = 0.001 
dx_str = '001_mpm'
filepath = 'saved_data/shocktube/'
fixed = False
max_steps = 10000

# PySPH Scheme Parameters
adke_params = [1.0,1.0,0.5,0.5,1.0,0.8]
mpm_params = [1.5,2.0,1.0,0.1,None,None]
gsph_params = [1.5,None,0.25,0.5,1,2]
params = [adke_params,mpm_params,gsph_params]

pysph_sim = Hybrid_sim()
pysph_sim.initialize_pysph_shocktube(dx=dx,xmin=0.0,xmax=1.0,ymin=0.0,ymax=0.05,gamma=1.4,DoDomain=True,
                                            mirror_x=True,mirror_y=True,adaptive=False,cfl=0.3,pfreq=10000,
                                            tf=0.012,dt=2e-5,scheme='mpm',scheme_params=params[1])
# Remember: ymax->0.048 for dx>=0.004

if (fixed):
    start = time.time()
    pysph_sim.pysph_sim.solver.solve(show_progress=False)
    end = time.time()
else:
    count = 0
    start = time.time()
    while (pysph_sim.pysph_sim.solver.t<pysph_sim.pysph_sim.solver.tf):
        # Near tf, ensure dt is such that final time is tf
        if ((pysph_sim.pysph_sim.solver.t+pysph_sim.pysph_sim.solver.dt)>pysph_sim.pysph_sim.solver.tf):
            pysph_sim.pysph_sim.solver.dt = pysph_sim.pysph_sim.solver.tf - pysph_sim.pysph_sim.solver.t
        pysph_sim.pysph_sim.step(1)
        count += 1
        if ((count)%10==0):
            pysph_sim.pysph_sim.particles[0].dt_cfl = pysph_sim.pysph_sim.particles[0].h/pysph_sim.pysph_sim.particles[0].cs
            pysph_sim.pysph_sim.solver.dt=pysph_sim.pysph_sim.solver.integrator.compute_time_step(pysph_sim.pysph_sim.dt,pysph_sim.pysph_sim.cfl)
        if (count > max_steps):
            break
    end = time.time()

pysph_sim.save_time(filepath + 'pysph_shocktube_' + dx_str + '/',(end-start))
pysph_sim.save_pysph_data(filepath + 'pysph_shocktube_' + dx_str + '/')
