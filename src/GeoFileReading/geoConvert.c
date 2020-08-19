/**
This code and dependencies are adopted from 
GARFFIELD: Genetic-Algorithm Reactive Force FIELD parameter optimizer.

   Copyright (2012-2014) California Institute of Technology
   Andres Jaramillo-Botero (ajaramil@caltech.edu)
   http://www.wag.caltech.edu/home/ajaramil/GARFfield.html
   
Additions and glue (like this file) by Tobias Kroll
Aug 6, 2020
*/

/*This module takes in a geo file (bgf or xyz format) and converts it to lammps-readable
structure files.


*/

/*
argv[1] = the geo file to be read and converted
argv[2] = the force field file applicable to the geo file
*/

#include "structures.h"
#include "tersoff_ffield.h"

#include "stdio.h"
#include "mpi.h"
#include <stdlib.h>

int debug_level = 1;
int pqeq_flag = 0;
int initial_write = 1;

enum {REAX, REAXC, EFF, CG, MORSE, COMB, TERSOFF, TERSOFF_MOD, ZHOU_EAM};

int main(int argc, char **argv)
{
	MPI_Init (&argc, &argv);
	double **ffid_pointer;
	tersoff_interaction *tersoffffdata;
	char** dfiles;
	char** atypes;
	int* cellflag;
	char** fname;
	ffieldtype* ff_struc_pointer;
	int num;
	int forcefield2optimize = TERSOFF;
	char* tempFfieldPath;
	tempFfieldPath = "throwaway.ffield";
	int rank = 0;
	

	char* geoFile = argv[1];
	char* ffieldFile = argv[2];
	
	
	char copyCMD[256];
	char geoFile2[256];
	
	
	sprintf(copyCMD,"cp %s %s.0",geoFile,geoFile);
	system(copyCMD);
	ff_struc_pointer = (ffieldtype *) scalloc (1, sizeof (ffieldtype),"fftype");

    ff_struc_pointer->tersoffff = (tersoff_interaction *) scalloc (1, sizeof (tersoff_interaction),"ffdata");
    tersoffffdata = (tersoff_interaction *) scalloc (1, sizeof (tersoff_interaction),"tersoffdata");
    int ndim = Read_Force_Field_TERSOFF (ffieldFile, tersoffffdata, &ffid_pointer, rank);
    Write_Force_Field_TERSOFF (tempFfieldPath, tersoffffdata, ffid_pointer, rank);

    *ff_struc_pointer->tersoffff = *tersoffffdata;
    
	geo2data(geoFile,&dfiles,&atypes,&cellflag,&fname,ff_struc_pointer,&num,forcefield2optimize,rank);
	remove(tempFfieldPath);
}