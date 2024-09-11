// photoionization_UVB.h
#ifndef PHOTOIONIZATION_UVB
#define PHOTOIONIZATION_UVB

// Declaration of recombination functions
void load_UVB_data();
void update_UVB(double redshift);

// Array that contain the UVB values at a given redshift
// This is currently a bit of waste of space but it's small
// enough that it shouldn't matter
// Index is element number | Ion
extern double HM12_UVB_z[27][27];

#endif