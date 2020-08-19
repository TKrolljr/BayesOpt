from mpi4py import MPI
from lammps import lammps

lammps = lammps()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print("hello",rank,"\n")
#x = ForceField_Tersoff("ffield.original")
#x.data[0][0][6] = "7.000"
#x.data[0][0][9] = "3.000"
#x.writeForceField("ffield.tersoff")
lammps.file("CompFile.lmp")


etersoff = lammps.extract_compute ("etersoff", 0, 0)
print(etersoff,"\n")
