# HPC_project_2022
This is the repository for the HPC project: NextDBScan SYCL.
Credit to https://github.com/ernire for the original project: NextDBScan.
NextDBScan is an implementation of the DBScan clustering algorithm that can run with CUDA, OMP and MPI.
The goal of the HPC Project is to port the NextDBScan CUDA version to a SYCL version for compatibility and extensibility purposes.
The translation phase is supported by the Intel OneAPI DPC++ Compatibility Tool, that automatically translates the CUDA code to SYCL code.
Not all CUDA code, and in particular thrust code can be translated automatically. It may need some manual intervention to correct errors and complete translation (as in this case). 
The original file is named magma_exa_cu.h. The output of DPCT is located in "dpct_output/magma_exa_sycl.h".
The final version of the SYCL file is called magma_exa_sycl.h and it's located in nextdbscan-exa-master. It is a reviewed and fixed translation for the DPCT output.

Since there have been some modification, I don't ensure that the command "make cu" can actually compile the CUDA code. To mitigate this, a working version of the nextdb-scan with CUDA back-end is already available as "nextdbscan-exa-cu".
The SYCL back-end can be enabled with the compilation of the second sycl command: "make sycl-2". This code targets an NVIDIA GPU with sm_86 cuda capabilities (RTX 3070ti).
All the available inputs are inside the "input" folder.
The command to execute the code has to look like this: "./a.out -m 1 -e 0.01 -t 1 -o input/three_blobs.csv", where a.out is the name of the executable (a.out refers to the SYCL back-end), m is the minimum number of points needed to form a cluster, e is the distance constant that is used as a search radius to find clusters and t is the thread parameter. This thread parameter is only useful in MPI and OpenMP implementations, since both CUDA and SYCL use maximum Compute Units available on the platform. The o parameters specifies the input file. We only use .csv files.

The other repository for this project is https://github.com/LucaSiviero/HPC. In this repository there is the commit history and all the advancements made in the code. This new repository was only created because there was an issue with my local repository, but this is the only repository to check for the final code submission.
