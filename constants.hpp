// constants.h
#ifndef CONSTANTS_H
#define CONSTANTS_H
#include "types.hpp"

#ifdef RTZ_FLOAT32_CONSTANTS
using ConstantsReal = float;
#else
using ConstantsReal = Real;
#endif

// Boltzmann constant (erg/K)
const ConstantsReal KB = ConstantsReal(1.380649E-16); // erg/K

// Speed of light cgs
const ConstantsReal C_CGS = ConstantsReal(2.9979246E10); // cm/s

// Planck's constant
const ConstantsReal H_PLANCK = ConstantsReal(6.626196E-27); // erg * s

// Conversion from eV to erg
const ConstantsReal EV_2_ERG = ConstantsReal(1.602E-12);

#endif // CONSTANTS_H
