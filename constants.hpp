// constants.h
#ifndef CONSTANTS_H
#define CONSTANTS_H
#include "types.hpp"

#ifdef RTZ_FLOAT32
using Real = float;
#else
using Real = double;
#endif

// Boltzmann constant (erg/K)
const Real KB = Real(1.380649E-16); // erg/K

// Speed of light cgs
const Real C_CGS = Real(2.9979246E10); // cm/s

// Planck's constant
const Real H_PLANCK = Real(6.626196E-27); // erg * s

// Conversion from eV to erg
const Real EV_2_ERG = Real(1.602E-12);

#endif // CONSTANTS_H
