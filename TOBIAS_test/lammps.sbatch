#!/bin/tcsh  -f 
#SBATCH -p dr-endel # normal                        # Queue  -- normal [development]
#SBATCH -J scnC05.N                     # Job Name
#SBATCH -o JOB_aiMD.%J                   # Name of the output and error file (%J expands to jobID)

#SBATCH --mail-type=ALL                  # {BEGIN|END|FAIL|ALL}    Specify when user notifications are to be sent
#SBATCH --mail-user=pkroll@uta.edu       # Specify the email address to use for notifications

#SBATCH -n 40                            # Total # of mpi tasks  [here: 44 possible]

limit stacksize unlimited
source /usr/nas/ohpc_hpcrnp.uta.edu/pub/intel/bin/compilervars.csh intel64
setenv FI_PROVIDER_PATH /usr/nas/ohpc_hpcrnp.uta.edu/pub/intel/compilers_and_libraries_2019/linux/mpi/intel64/libfabric/lib/prov
setenv PATH $PATH\:/usr/nas/ohpc_hpcrnp.uta.edu/pub/intel/compilers_and_libraries/linux/mpi/intel64/bin

################################
set progvers = lmp_mpi 
set ncore    = 40
# ################################

############################## enter_parameters #####################


cp data.alpha_Si3N4  model.input 

set in_lmp = in.Si3N4
set str    = Si3N4

mpirun -np   ${ncore} ~/bin/${progvers} < ${in_lmp} >>& out.${str}


