import time
from custom_hybrid import Hybrid_sim

dx = 0.02
dx_str = '02'
filepath = 'saved_data/sedov/'

pysph_sedov = Hybrid_sim()
pysph_sedov.initialize_pysph_sedov(dx=dx,xmin=0.0,xmax=1.00,ymin=0.0,ymax=1.0,xcntr=0.5,ycntr=0.5,
                                   r_init=0.005, adaptive=False, cfl=0.3, pfreq=10000,tf=0.1,dt=1e-4)

start = time.time()
pysph_sedov.pysph_sim.solver.solve(show_progress=False)
end = time.time()

pysph_sedov.save_time(filepath + 'pysph_sedov_' + dx_str + '/',(end-start))
pysph_sedov.save_pysph_data(filepath + 'pysph_sedov_' + dx_str + '/')