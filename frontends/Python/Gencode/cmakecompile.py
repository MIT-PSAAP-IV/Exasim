import os
import subprocess

def cmakecompile(pde):
    
    print("Compile C++ Exasim code using cmake...")

    cdir = os.getcwd()
    os.makedirs(pde['buildpath'], exist_ok=True)
    os.chdir(pde['buildpath'])

    if os.path.exists("exasimfe"):
        os.remove("exasimfe")

    sourcepath = os.path.join(pde['exasimpath'], "examples")

    if pde['mpiprocs'] == 1:
        if pde['platform'] == "gpu":
            comstr = ["cmake", "-S", sourcepath, "-B", ".", "-D", "EXASIM_MPI=OFF", "-D", "EXASIM_CUDA=ON"]
        elif pde['platform'] == "hip":
            comstr = ["cmake", "-S", sourcepath, "-B", ".", "-D", "CMAKE_CXX_COMPILER=hipcc", "-D", "EXASIM_MPI=OFF", "-D", "EXASIM_HIP=ON"]
        else:
            comstr = ["cmake", "-S", sourcepath, "-B", ".", "-D", "EXASIM_MPI=OFF"]
    else:
        if pde['platform'] == "gpu":
            comstr = ["cmake", "-S", sourcepath, "-B", ".", "-D", "EXASIM_MPI=ON", "-D", "EXASIM_CUDA=ON"]
        elif pde['platform'] == "hip":
            comstr = ["cmake", "-S", sourcepath, "-B", ".", "-D", "CMAKE_CXX_COMPILER=hipcc", "-D", "EXASIM_MPI=ON", "-D", "EXASIM_HIP=ON"]
        else:
            comstr = ["cmake", "-S", sourcepath, "-B", ".", "-D", "EXASIM_MPI=ON"]

    subprocess.run(comstr, check=True)
    subprocess.run(["cmake", "--build", ".", "--target", "exasimfe"], check=True)

    os.chdir(cdir)
    
    return comstr

    # print("Run C++ Exasim code ...")
    # pdenum = " " + str(numpde) + " "
    # mpirun = pde['mpirun']
    # DataPath = pde['buildpath']    

    # if pde['platform'] == "cpu":
    #     if pde['mpiprocs'] == 1:
    #         cmd = f"./cpuEXASIM {pdenum} {DataPath}/datain/ {DataPath}/dataout/out"
    #     else:
    #         cmd = f"{mpirun} -np {pde['mpiprocs']} ./cpumpiEXASIM {pdenum} {DataPath}/datain/ {DataPath}/dataout/out"
    # elif pde['platform'] == "gpu":
    #     if pde['mpiprocs'] == 1:
    #         cmd = f"./gpuEXASIM {pdenum} {DataPath}/datain/ {DataPath}/dataout/out"
    #     else:
    #         cmd = f"{mpirun} -np {pde['mpiprocs']} ./gpumpiEXASIM {pdenum} {DataPath}/datain/ {DataPath}/dataout/out"

    # start_time = time.time()    
    # subprocess.run(cmd, shell=True)    
    # end_time = time.time()
    # print(f"Elapsed time: {end_time - start_time} seconds")

    # os.chdir(cdir)
    
    # return cmd
