// molecules.h
#pragma once
#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
double alpha_H2_prim(double T, double xe, double H2_cosmic_ray_ionization_rate, double G0, double xHI, double xHII, double nH){
    // H- channel for H2 formation
    double formation_rate = 0.0;

    // Primordial channel
    double logT = log10(T);
    double Te = T * 8.621738E-5; // K -> eV
    double lnTe = log(Te);

    // ! Creation and destruction channels of H- included with updated rates from Glover et al. 2010
    // ! H + e- -> H- + gamma
    double k1;
    if (T < 6000.0) {
        k1 = pow(10.0, -17.845 + logT * (0.762 + logT * (0.1523 - 0.03274 * logT)));
    } else {
        double logT2 = logT * logT;
        k1 = pow(10.0, -16.420 + logT2 * (0.1998 + logT2 * (-5.447E-3 + 4.0415E-5 * logT2)));
    }

    // ! H- + H -> H2 + e
    double k2 = 4.0E-9*pow(fmax(T,300.0),-0.17);

    // ! H- + H+ -> H + H
    double k5 = 2.4E-6/sqrt(T)*(1.0 + T/20000.);

    // ! H- + CR --> H + e-
    double k_hm_cr = 1.28E-13 * (H2_cosmic_ray_ionization_rate / 1.E-16);

    // ! H- + gamma --> H + e-
    double k_hm_gamma = 5.9E-9 * G0;

    // ! H- + e -> H + e + e
    double k13 = -1.801849334E1 + lnTe * (2.36085220E0 + lnTe * (-2.82744300E-1 + lnTe * (1.62331664E-2 + lnTe * (-3.36501203E-2 + lnTe * (1.17832978E-2 + lnTe * (-1.65619470E-3 + lnTe * (1.06827520E-4 - 2.63128581E-6 * lnTe)))))));
    k13 = exp(fmax(k13,-92.0));

    // ! H- + H --> H + H + e-
    // ! I think this reaction was broken in glover so I took the results from
    // ! https://www.aanda.org/articles/aa/pdf/2016/02/aa27262-15.pdf Table A1
    // ! Harley added the fudge factor for continuity
    double k14 = 1.357772745525155 * 2.5634E-15 * pow(Te, 1.78186); // Note that T must be in eV for this reaction to make sense
    if (T > 1160.0) {
        k14 = -3.388464953E1 + lnTe * (1.13944933E0 + lnTe * (-1.4210135E-1 + lnTe * (8.4644554E-3 + lnTe * (-1.4328641E-3 + lnTe * (2.0122503E-4 + lnTe * (8.6639632E-5 + lnTe * (-2.5850097E-5 + lnTe * (2.4555012E-6 - 8.0683825E-8 * lnTe))))))));
        k14 = exp(k14);
    }

    // ! H- + H+ --> H2+ + e- --> H + H (via recombinative dissociation)
    double k15 = 6.9E-9 * pow(T,-0.35);
    if (T > 8000.0) {
        k15 = 9.6E-7 * pow(T,-0.90);
    }

    // Correct steady state denominator terms scaled to cm^3 s^-1
    double cr_photo_destruction = (k_hm_cr + k_hm_gamma) / (nH + 1e-40);
    double collisional_destruction = k2 * xHI + k5 * xHII + k13 * xe + k14 * xHI + k15 * xHII;

    formation_rate += k1 * k2 * xe * xHI / (collisional_destruction + cr_photo_destruction);

    return formation_rate;
}

KOKKOS_FUNCTION
double alpha_H2_dust(double T, double dust_to_gas_mass_ratio_over_mw){
    // Formation on dust
    double clumping_factor = 1.0;
    double T2 = T / 100.0;
    double formation_rate = dust_to_gas_mass_ratio_over_mw * (3.5E-17) * clumping_factor * sqrt(fmin(T2,1e2));

    return formation_rate;
}

KOKKOS_FUNCTION
double alpha_H2(double T, double dust_to_gas_mass_ratio_over_mw, double xe, double H2_cosmic_ray_ionization_rate, double G0, double xHI, double xHII, double nH){
    // Creation rate of molecular hydrogen
    // We consider both the primordial channel (via H-) as well
    // as formation on dust

    double formation_rate = 0.0;

    // Formation rate on dust. Consider only HI
    formation_rate += alpha_H2_dust(T, dust_to_gas_mass_ratio_over_mw) * xHI * nH;

    // Primordial H- channel
    formation_rate += alpha_H2_prim(T, xe, H2_cosmic_ray_ionization_rate, G0, xHI, xHII, nH) * xHI * nH;

    return formation_rate;
}

KOKKOS_INLINE_FUNCTION
double beta_H2_krome(double T, double nH, double ne, double nH2, double nHe){
    // From Bovino 2016
    // https://www.aanda.org/articles/aa/pdf/2016/06/aa28158-16.pdf

    double destruction_rate = 0.0;

    // H2 + H --> H + H + H (k18)
    destruction_rate += ((6.67E-12 * sqrt(T) * exp(-1.0 * (1.0 + (63593.0/T)))) * nH);

    // H2 + H2 --> H2 + H + H (k19)
    destruction_rate += ((5.996E-30 * pow(T,4.1881) * pow(1.0 + (6.761E-6 * T),-5.6881) * exp(-54657.4/T)) * nH2);

    // H2 + e- --> H + H + e- (k22)
    destruction_rate += (4.38E-10 * pow(T,0.35) * exp(-102000.0/T) * ne);

    // H2 + He --> H + H + He
    destruction_rate += (pow(10.0,-27.029 + (3.801 * log10(T)) - (29487.0/T)) * nHe);

    return destruction_rate;
}

KOKKOS_INLINE_FUNCTION
double comp_Sd(double nHI, double nH2, double dx_SS, double Z) {
    //   ! Returns the self shielding factor for dust
    //   ! see section 2.2 http://iopscience.iop.org/0004-637X/697/1/55/pdf/apj_697_1_55.pdf

    double Sdeff = 2.34e-21;    // dust cross section cm^2 ! Updated for bare-gr-s
    double cNHI = nHI*dx_SS;    // HI column density
    double cNH2 = nH2*dx_SS;    // H2 column density

    double ss_factor = exp(-Sdeff*Z*(cNHI + (2.0*cNH2)));

    return ss_factor;
}

KOKKOS_INLINE_FUNCTION
double comp_SH2(double nH2, double dx_SS) {
    // ! Returns the self shielding factor for dust
    // ! see section 2.2 http://iopscience.iop.org/0004-637X/697/1/55/pdf/apj_697_1_55.pdf

    double cNH2 = nH2*dx_SS;  // H2 column density
    double xfac = cNH2/(5e14);
    double wH2 = 0.2;

    double Sa = (1.0 - wH2)/((1.0 + xfac)*(1.0 + xfac));
    double Sb = wH2/sqrt(1.0 + xfac);
    double Sc = exp(-0.00085*sqrt(1.0 + xfac));

    double ss_factor = Sa + (Sb*Sc);

    return ss_factor;
}

KOKKOS_INLINE_FUNCTION
double alpha_CO(double G0, double xi_cr_H2, double nCII, double nH2, double xO, double nH){

    //! see glover 2012
    double k0 = 5.E-16; //! cm^3 s^-1
    double k1 = 5.E-10; //! Rate coefficient for the formation of CO from O + CHx

    //! Assuming CO formation is modulated by CH2+, https://home.strw.leidenuniv.nl/~ewine/photo/display_ch2+_65e31a07e69e64dbd64d37801983018f.html
    double gammaCHx_cr = 8.88E-15 * (xi_cr_H2 / 1E-16); //! Cosmic rays
    double gammaCHx = (1.41E-10 * G0) + gammaCHx_cr;

    double beta = k1 * xO/(k1*xO + gammaCHx/nH);
    double formation_rate = k0 * nCII * nH2 * beta;

    return fmax(formation_rate,1e-100);
}

KOKKOS_INLINE_FUNCTION
double beta_CO(double G0, double xi_cr_H2){
    //! CO destruction
    double gammaCO = 2.43E-10 * G0;
    double gammaCO_cr = 4.62E-15 * (xi_cr_H2 / 1E-16);

    double destruction_rate = gammaCO + gammaCO_cr;

    return fmax(destruction_rate,1e-100);
}

// sco_table has 52 entries
constexpr double sco_col1[52] = {
    1.000e+00, 1.000e+12, 1.650e+12, 2.995e+12, 5.979e+12, 1.313e+13,
    3.172e+13, 8.429e+13, 2.464e+14, 7.923e+14, 1.670e+15, 2.595e+15,
    4.435e+15, 6.008e+15, 8.952e+15, 1.334e+16, 1.661e+16, 2.274e+16,
    3.115e+16, 4.266e+16, 5.843e+16, 8.002e+16, 1.096e+17, 1.501e+17,
    2.055e+17, 2.815e+17, 4.241e+17, 6.389e+17, 9.625e+17, 1.450e+18,
    2.184e+18, 3.291e+18, 4.124e+18, 5.685e+18, 7.838e+18, 1.080e+19,
    1.285e+19, 1.681e+19, 2.199e+19, 2.538e+19, 3.222e+19, 4.091e+19,
    5.193e+19, 5.893e+19, 7.356e+19, 8.269e+19, 9.246e+19, 1.031e+20,
    1.148e+20, 1.277e+20, 1.419e+20, 1.578e+20
};

constexpr double sco_col2[52] = {
    1.000e+00, 9.990e-01, 9.981e-01, 9.961e-01, 9.912e-01, 9.815e-01,
    9.601e-01, 9.113e-01, 8.094e-01, 6.284e-01, 4.808e-01, 3.889e-01,
    2.827e-01, 2.293e-01, 1.695e-01, 1.224e-01, 1.017e-01, 7.764e-02,
    5.931e-02, 4.546e-02, 3.506e-02, 2.728e-02, 2.143e-02, 1.700e-02,
    1.360e-02, 1.094e-02, 8.273e-03, 6.283e-03, 4.773e-03, 3.611e-03,
    2.704e-03, 1.986e-03, 1.657e-03, 1.258e-03, 9.332e-04, 6.745e-04,
    5.596e-04, 4.123e-04, 2.982e-04, 2.490e-04, 1.827e-04, 1.324e-04,
    9.473e-05, 7.891e-05, 5.668e-05, 4.732e-05, 3.967e-05, 3.327e-05,
    2.788e-05, 2.331e-05, 1.944e-05, 1.619e-05
};

// sh2_table has 43 entries
constexpr double sh2_col1[43] = {
    1.000e+00, 2.666e+13, 3.801e+14, 6.634e+15, 8.829e+16, 9.268e+17,
    1.007e+18, 2.021e+18, 3.036e+18, 4.051e+18, 5.066e+18, 6.082e+18,
    7.097e+18, 8.112e+18, 9.341e+18, 1.014e+19, 2.030e+19, 3.045e+19,
    4.061e+19, 5.076e+19, 6.092e+19, 7.107e+19, 8.123e+19, 9.353e+19,
    1.015e+20, 2.031e+20, 3.047e+20, 4.062e+20, 5.078e+20, 6.094e+20,
    7.109e+20, 8.125e+20, 9.355e+20, 1.016e+21, 2.031e+21, 3.047e+21,
    4.063e+21, 5.078e+21, 6.094e+21, 7.110e+21, 8.125e+21, 9.355e+21,
    1.016e+22
};

constexpr double sh2_col2[43] = {
    1.000e+00, 9.999e-01, 9.893e-01, 9.678e-01, 9.465e-01, 9.137e-01,
    9.121e-01, 8.966e-01, 8.862e-01, 8.781e-01, 8.716e-01, 8.660e-01,
    8.612e-01, 8.569e-01, 8.524e-01, 8.497e-01, 8.262e-01, 8.118e-01,
    8.011e-01, 7.921e-01, 7.841e-01, 7.769e-01, 7.702e-01, 7.626e-01,
    7.579e-01, 7.094e-01, 6.712e-01, 6.378e-01, 6.074e-01, 5.791e-01,
    5.524e-01, 5.271e-01, 4.977e-01, 4.793e-01, 2.837e-01, 1.526e-01,
    7.774e-02, 3.952e-02, 2.093e-02, 1.199e-02, 7.666e-03, 5.333e-03,
    4.666e-03
};


KOKKOS_INLINE_FUNCTION
double comp_SCO(const TabulatedData& tabData, double nco_mol, double nh2, double dx_SS) {
    // initialize to 1.0
    double ss_factor = 1.0;

    // CO self-shielding
    // Pinned to table at lower boundary, extrapolated above upper boundary
    double effcNCO = fmax(nco_mol*dx_SS, tabData.sco_col1(1));
    if (effcNCO >= 1.578e+20) {
        ss_factor *= 1.619e-05;
    } else {
        // Find the lower closest index
        int idxCO = 0;
        for (int i=0; i<51; i++) { // If above upper boundary then extrapolate using the slope 
            // between the last two points, so conveniently we set i to not go to 51
            if (effcNCO >= tabData.sco_col1(i)) idxCO = i;
        }

        double logeffcNCO = log10(effcNCO);
        // Interpolate or extrapolate automatically
        double logsCO = log10(tabData.sco_col2(idxCO))
                        + (log10(tabData.sco_col2(idxCO+1)) - log10(tabData.sco_col2(idxCO)))
                        * (logeffcNCO - log10(tabData.sco_col1(idxCO)))
                        / (log10(tabData.sco_col1(idxCO+1)) - log10(tabData.sco_col1(idxCO)));

        ss_factor *= pow(10.0,logsCO);
    }

    // H2 shielding of CO
    // Pinned to table at lower boundary, extrapolated above upper boundary
    double effcNH2 = fmax(nh2*dx_SS, tabData.sh2_col1(1));
    if (effcNH2 >= 1.016e+22) {
        ss_factor *= 4.666e-03;
    } else {
        // Find the lower closest index
        int idxH2 = 0;
        for (int i=1; i<42; i++) {
            if (effcNH2 >= tabData.sh2_col1(i)) idxH2 = i;
        }

        double logeffcNH2 = log10(effcNH2);
        // Interpolate or extrapolate automatically
        double logsH2 = log10(tabData.sh2_col2(idxH2))
                        + (log10(tabData.sh2_col2(idxH2+1)) - log10(tabData.sh2_col2(idxH2)))
                        * (logeffcNH2 - log10(tabData.sh2_col1(idxH2)))
                        / (log10(tabData.sh2_col1(idxH2+1)) - log10(tabData.sh2_col1(idxH2)));

        ss_factor *= pow(10.0,logsH2);
    }

    return ss_factor;
}
