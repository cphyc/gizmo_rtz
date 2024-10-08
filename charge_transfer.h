// charge_transfer.h
#ifndef CHARGE_TRANSFER
#define CHARGE_TRANSFER

void load_ct_rates();
double charge_transfer_recombination(int ion, int nelem, double T);
double charge_transfer_ionization(int ion, int nelem, double T);

#endif