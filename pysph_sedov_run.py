import time
from custom_hybrid import Hybrid_sim

dx = 0.01
dx_str = '01_gsph'
filepath = 'saved_data/sedov/'
fixed = False
max_steps = 10000

adke_params = [1.0,1.0,0.5,0.5,1.0,0.8] # [1.0,1.0,1.0,1.0,1.0,0.5]
mpm_params = [1.5,2.0,10.0,1.0,None,None] # alpha2->0?
gsph_params = [1.5,None,0.1,0.5,1,2]
params = [adke_params,mpm_params,gsph_params]

pysph_sedov = Hybrid_sim()
pysph_sedov.initialize_pysph_sedov(dx=dx,xmin=0.0,xmax=1.00,ymin=0.0,ymax=1.0,gamma=1.4,xcntr=0.5,ycntr=0.5,
                                    r_init=dx,gaussian=False,DoDomain=False,mirror_x=False,mirror_y=False,adaptive=False,
                                    cfl=0.3,pfreq=10000,tf=0.1,dt=1e-6,scheme='gsph',scheme_params=params[2])
current_scheme = pysph_sedov.pysph_sim.scheme_selection

if (fixed==True):
    count = 0
    start = time.time()
    while (pysph_sedov.pysph_sim.solver.t<pysph_sedov.pysph_sim.solver.tf):
        # Near tf, ensure dt is such that final time is tf
        if ((pysph_sedov.pysph_sim.solver.t+pysph_sedov.pysph_sim.solver.dt)>pysph_sedov.pysph_sim.solver.tf):
            pysph_sedov.pysph_sim.solver.dt = pysph_sedov.pysph_sim.solver.tf - pysph_sedov.pysph_sim.solver.t
        pysph_sedov.pysph_sim.step(1)
        count += 1
        if (count % 10 == 0):
            print("Finished step " + str(count) + ". t=" + str(round(pysph_sedov.pysph_sim.solver.t,6)) + ", dt = " + str(pysph_sedov.pysph_sim.solver.dt) + " ")
        if (count > max_steps):
            break
    end = time.time()
else:
    count = 0
    start = time.time()
    while (pysph_sedov.pysph_sim.solver.t<pysph_sedov.pysph_sim.solver.tf):
        # Near tf, ensure dt is such that final time is tf
        if ((pysph_sedov.pysph_sim.solver.t+pysph_sedov.pysph_sim.solver.dt)>pysph_sedov.pysph_sim.solver.tf):
            pysph_sedov.pysph_sim.solver.dt = pysph_sedov.pysph_sim.solver.tf - pysph_sedov.pysph_sim.solver.t
        pysph_sedov.pysph_sim.step(1)
        count += 1
        if (current_scheme == 'gsph'):
            if (count>10): # count%10==0
                pysph_sedov.pysph_sim.particles[0].dt_cfl = pysph_sedov.pysph_sim.particles[0].h/pysph_sedov.pysph_sim.particles[0].cs
                pysph_sedov.pysph_sim.solver.dt=pysph_sedov.pysph_sim.cfl*pysph_sedov.pysph_sim.particles[0].dt_cfl.min()
                if (count%10==0):
                    print("Finished step " + str(count) + ". t=" + str(round(pysph_sedov.pysph_sim.solver.t,6)) + ", dt = " + str(pysph_sedov.pysph_sim.solver.dt) + " ")
        else:
            if ((count>100)and(count%10==0)): # count%10==0
                pysph_sedov.pysph_sim.particles[0].dt_cfl = pysph_sedov.pysph_sim.particles[0].h/pysph_sedov.pysph_sim.particles[0].cs
                pysph_sedov.pysph_sim.solver.dt=min(2*pysph_sedov.pysph_sim.solver.dt,pysph_sedov.pysph_sim.cfl*pysph_sedov.pysph_sim.particles[0].dt_cfl.min())
                print("Finished step " + str(count) + ". t=" + str(round(pysph_sedov.pysph_sim.solver.t,6)) + ", dt = " + str(pysph_sedov.pysph_sim.solver.dt) + " ")
        if (count > max_steps):
            break
    end = time.time()

print(end-start)

pysph_sedov.save_time(filepath + 'pysph_sedov_' + dx_str + '/',(end-start))
pysph_sedov.save_pysph_data(filepath + 'pysph_sedov_' + dx_str + '/')