#!/usr/bin/awk -f
# This program reads Tersoff parameters for elements from a single line
# and makes the tersoff parameter file suitable for LAMMPS for all the
# elements. The input parameter file for single elements has to be of
# the format specified here. 
# Line 1 : Comment / Statement / descriptor
# Line 2 : Number of elements
# Line 3 : Element 1
# Line 4 : Parameters for Element 1 in order of: m,gamma,lambda3,c,d,h,n,beta,lambda2,B,R,D,lambda1,A 
# ......
# Line 2n+2 : Parameters for Element n
# Line 2n+2 .. to end set of Xo (chi) values for all unique pairs of atoms.
# syntax: $>echo | awk -f tersmak.awk
# Make sure Terspar.dat is in valid format as described above
{
 getline Comments < ARGV[1]
 getline Numel < ARGV[1]
 for (eles=1;eles<=Numel;eles++)
 {
  getline Elenam[eles]< ARGV[1]
  getline Elepar[eles] < ARGV[1]
  split(Elepar[eles],pars);
  m[eles]=pars[1];
  gamma[eles]=pars[2];
  lambda3[eles]=pars[3];
  c[eles]=pars[4];
  d[eles]=pars[5];
  h[eles]=pars[6];
  n[eles]=pars[7];
  beta[eles]=pars[8];
  lambda2[eles]=pars[9];
  B[eles]=pars[10];
  R[eles]=pars[11];
  D[eles]=pars[12];
  lambda1[eles]=pars[13];
  A[eles]=pars[14];
 }
 getline Comment < ARGV[1] 
 for (ii=1;ii<=Numel;ii++)
 {
  for (jj=1;jj<=ii;jj++)
  {
   getline chival < ARGV[1]
   split(chival,chi);
   chiX[ii,jj]=chi[3];
   chiX[jj,ii]=chi[3];
  }
 }
}
END{
 print"### Parameters m gamma lambda3 c d h n beta lambda2 B R D lambda1 A"
 for (el1=1;el1<=Numel;el1++)
 {
  print "# # # Element:" Elenam[el1] " : " Elepar[el1]
  for (el2=1;el2<=Numel;el2++)
  {
   for (el3=1;el3<=Numel;el3++)
   {
    elcomb=Elenam[el1]" "Elenam[el2]" "Elenam[el3]
    m[elcomb]=m[el1];
    gamma[elcomb]=gamma[el1];
    lambda3[elcomb]=lambda3[el1];
    c[elcomb]=c[el1];
    d[elcomb]=d[el1];
    h[elcomb]=h[el1];
    n[elcomb]=n[el1];
    beta[elcomb]=beta[el1];
    lambda2[elcomb]=(lambda2[el1]+lambda2[el2])/2.0;
    lambda1[elcomb]=(lambda1[el1]+lambda1[el2])/2.0;
    B[elcomb]=chiX[el1,el2]*sqrt(B[el1]*B[el2]);
    A[elcomb]=sqrt(A[el1]*A[el2]);
    D[elcomb]=(sqrt((R[el1]+D[el1])*(R[el2]+D[el2]))-sqrt((R[el1]-D[el1])*(R[el2]-D[el2])))/2
    R[elcomb]=(sqrt((R[el1]+D[el1])*(R[el2]+D[el2]))+sqrt((R[el1]-D[el1])*(R[el2]-D[el2])))/2
    printf("%s %.2f %.2f %.4f ",elcomb,m[elcomb],gamma[elcomb],lambda3[elcomb])
    printf("%9.4e %9.4e %12.4e ",c[elcomb],d[elcomb],h[elcomb])
    printf("%6.3f %10.4e ",n[elcomb],beta[elcomb])
    printf("%8.4f %10.4e",lambda2[elcomb],B[elcomb])
    printf("%8.4f %8.4f",R[elcomb],D[elcomb])
    printf("%8.4f %10.4e\n",lambda1[elcomb],A[elcomb])
   }
  }
 }
} 
