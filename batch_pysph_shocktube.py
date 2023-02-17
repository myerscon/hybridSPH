# Batch script for submitting pysph jobs
import os
import subprocess
import time

# method of creating directories:
#path = "test_directory"
#print(os.path.exists(path))
#if not (os.path.exists(path)):
#    os.makedirs(path)
#print(os.path.exists(path))

# Input Parameters
dx_list = ['001','002','004','008']
dt_list = ['2e-5','4e-5','8e-5','16e-5']
ymax_list = ['0.05','0.05','0.048','0.048']
scheme_list = ['adke','mpm','gsph','adke_fixed','mpm_fixed','gsph_fixed']
#gsph_mono_list = [] # for iterating over gsph monotonicity 
fixed_list = [False,False,False,True,True,True]

parentdir = 'saved_data/shocktube/'

for j in range(0,3,1): # len(scheme_list)
    for i in range(1): # len(dx_list)
        currentdir = 'pysph_shocktube_' + dx_list[i] + '_' + scheme_list[j]
        fulldir = parentdir + currentdir

        if not (os.path.exists(fulldir)):
            os.makedirs(fulldir)

        # Modify pysph_shocktube_batch.py
        file1 = open('pysph_shocktube_batch.py','r')
        lines = file1.readlines()
        file1.close()

        lines[3]='dx = 0.' + dx_list[i] + '\n'
        lines[4]="dx_str = '" + str(dx_list[i]) + "_" + str(scheme_list[j]) + "'\n"
        lines[6]="fixed = " + str(fixed_list[j]) + "\n"
#        lines[12]="gsph_params = [1.5,None,0.25,0.5,1," + 2 + "]\n"
        lines[16]="pysph_sim.initialize_pysph_shocktube(dx=dx,xmin=0.0,xmax=1.0,ymin=0.0,ymax=" + ymax_list[i] + ",gamma=1.4,DoDomain=True,\n"
        lines[18]="                                            tf=0.012,dt=" + dt_list[i] + ",scheme='" + str(scheme_list[j%3]) + "',scheme_params=params[" + str(j%3) + "])\n"

        file1 = open('pysph_shocktube_batch.py','w')
        file1.writelines(lines)
        file1.close()

        # copy to new python file
        subprocess.call(["cp","pysph_shocktube_batch.py","pysph_shocktube_batch" + str(i) + str(j) + ".py"])

        # Modify run_job.sh
        file2 = open('run_job.sh','r')
        lines2 = file2.readlines()
        file2.close()

        lines2[1]="#SBATCH -J " + currentdir + "    # name of my job\n"
        lines2[4]="#SBATCH -o " + fulldir + "/sim.out        # name of output file for batch script\n"
        lines2[5]="#SBATCH -e " + fulldir + "/sim.err        # name of error file for batch script\n"
        lines2[19]="python3 pysph_shocktube_batch" + str(i) + str(j) + ".py"

        file2 = open('run_job.sh','w')
        file2.writelines(lines2)
        file2.close()

        subprocess.run(["sbatch","run_job.sh"])
        print("Submitted job " + currentdir + ".")
        time.sleep(3)
