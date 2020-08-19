/* ----------------------------------------------------------------------
   GARFFIELD: Genetic-Algorithm Reactive Force FIELD parameter optimizer.

   Copyright (2012-2014) California Institute of Technology
   Andres Jaramillo-Botero (ajaramil@caltech.edu)
   http://www.wag.caltech.edu/home/ajaramil/GARFfield.html
------------------------------------------------------------------------- */

#ifndef REAXC_TOOL_BOX_H
#define REAXC_TOOL_BOX_H

void *scalloc (int, int, char *);
void safe_free (void *, char *);
void *smalloc (long, char *);
int tokenize_string (char *, char ***);

#endif
