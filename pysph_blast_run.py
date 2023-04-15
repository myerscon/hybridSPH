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

pysph_blast = Hybrid_sim()
pysph_blast.initialize_pysph_blast(dx=dx,xmin=-1.0,xmax=1.00,ymin=-1.0,ymax=1.0,gamma=1.4,kf=1.5,xcntr=0.0,ycntr=0.0,
                                    r_init=0.1,gaussian=False,DoDomain=False,mirror_x=False,mirror_y=False,adaptive=False,
                                    cfl=0.5,pfreq=10000,tf=0.025,dt=2e-5,scheme='gsph',scheme_params=params[2]) #

pysph_blast.pysph_sim.particles[0].rho = np.where(pysph_blast.pysph_sim.particles[0].p>0,1.001,1.0) 
pysph_blast.pysph_sim.particles[0].p = np.where(pysph_blast.pysph_sim.particles[0].rho>1.0,1000,0.01) 
pysph_blast.pysph_sim.particles[0].e = np.where(pysph_blast.pysph_sim.particles[0].rho>1.0,2.5e3,6.68e-2) # 

count = 0
start = time.time()
while (pysph_blast.pysph_sim.solver.t<pysph_blast.pysph_sim.solver.tf):
    # Near tf, ensure dt is such that final time is tf
    if ((pysph_blast.pysph_sim.solver.t+pysph_blast.pysph_sim.solver.dt)>pysph_blast.pysph_sim.solver.tf):
        pysph_blast.pysph_sim.solver.dt = pysph_blast.pysph_sim.solver.tf - pysph_blast.pysph_sim.solver.t
    pysph_blast.pysph_sim.step(1)
    count += 1
    if (count>10): 
        pysph_blast.pysph_sim.particles[0].dt_cfl = pysph_blast.pysph_sim.particles[0].h/pysph_blast.pysph_sim.particles[0].cs
        pysph_blast.pysph_sim.solver.dt=pysph_blast.pysph_sim.cfl*pysph_blast.pysph_sim.particles[0].dt_cfl.min()
        #if (count%10==0):
        #    print("Finished step " + str(count) + ". t=" + str(round(pysph_blast.pysph_sim.solver.t,6)) + ", dt = " + str(pysph_blast.pysph_sim.solver.dt) + " ")
        if (count > max_steps):
            break
end = time.time()

pysph_blast.save_time(filepath + 'pysph_blast_' + dx_str + '/',(end-start))
pysph_blast.save_pysph_data(filepath + 'pysph_blast_' + dx_str + '/')