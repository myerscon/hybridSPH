import numpy as np
import time
from custom_hybrid import Hybrid_sim

dx = 0.01
dx_str = '01'
filepath = 'saved_data/blast/'
max_steps = 10000

adke_params = [1.0,1.0,0.5,0.5,1.0,0.8] 
mpm_params = [1.5,2.0,10.0,1.0,None,None] 
gsph_params = [1.5,None,0.1,0.5,1,2]
params = [adke_params,mpm_params,gsph_params]

hybrid_blast = Hybrid_sim()
hybrid_blast.initialize_pyro(solver,problem_name,param_file,other_commands)
hybrid_blast.initialize_pysph_blast(dx=dx,xmin=0.0,xmax=1.00,ymin=0.0,ymax=1.0,gamma=1.4,kf=1.5,xcntr=0.5,ycntr=0.5,
                                    r_init=0.1,gaussian=False,DoDomain=False,mirror_x=False,mirror_y=False,adaptive=False,
                                    cfl=0.3,pfreq=10000,tf=0.025,dt=1e-4,scheme='gsph',scheme_params=params[2]) #

count = 0
start = time.time()
while (hybrid_blast.pysph_sim.solver.t<hybrid_blast.pysph_sim.solver.tf):
    # Near tf, ensure dt is such that final time is tf
    if ((hybrid_blast.pysph_sim.solver.t+hybrid_blast.pysph_sim.solver.dt)>hybrid_blast.pysph_sim.solver.tf):
        hybrid_blast.pysph_sim.solver.dt = hybrid_blast.pysph_sim.solver.tf - hybrid_blast.pysph_sim.solver.t
    hybrid_blast.pysph_sim.step(1)
    count += 1
    if (count>10): 
        hybrid_blast.pysph_sim.particles[0].dt_cfl = hybrid_blast.pysph_sim.particles[0].h/hybrid_blast.pysph_sim.particles[0].cs
        hybrid_blast.pysph_sim.solver.dt=hybrid_blast.pysph_sim.cfl*hybrid_blast.pysph_sim.particles[0].dt_cfl.min()
        #if (count%10==0):
        #    print("Finished step " + str(count) + ". t=" + str(round(hybrid_blast.pysph_sim.solver.t,6)) + ", dt = " + str(hybrid_blast.pysph_sim.solver.dt) + " ")
       if (count > max_steps):
        break
end = time.time()

hybrid_blast.save_time(filepath + 'pysph_blast_' + dx_str + '/',(end-start))
hybrid_blast.save_pysph_data(filepath + 'pysph_blast_' + dx_str + '/')