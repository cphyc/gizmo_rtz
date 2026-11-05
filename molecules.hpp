// molecules.h
#pragma once
#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
Real alpha_H2_prim(Real T, Real xe, Real H2_cosmic_ray_ionization_rate, Real G0, Real xHI, Real xHII, Real nH){
    // H- channel for H2 formation
    Real formation_rate = Real(0.0);

    // Primordial channel
    Real logT = rtz_math::log10(T);
    Real Te = T * Real(8.621738E-5); // K -> eV
    Real lnTe = rtz_math::log(Te);

    // ! Creation and destruction channels of H- included with updated rates from Glover et al. 2010
    // ! H + e- -> H- + gamma
    Real k1;
    if (T < Real(6000.0)) {
        k1 = rtz_math::exp10(-Real(17.845) + logT * (Real(0.762) + logT * (Real(0.1523) - Real(0.03274) * logT)));
    } else {
        Real logT2 = logT * logT;
        k1 = rtz_math::exp10(-Real(16.420) + logT2 * (Real(0.1998) + logT2 * (-Real(5.447E-3) + Real(4.0415E-5) * logT2)));
    }

    // ! H- + H -> H2 + e
    Real k2 = Real(4.0E-9)*rtz_math::pow(rtz_math::fmax(T,Real(300.0)),-Real(0.17));

    // ! H- + H+ -> H + H
    Real k5 = Real(2.4E-6)/rtz_math::sqrt(T)*(Real(1.0) + T/Real(20000.));

    // ! H- + CR --> H + e-
    Real k_hm_cr = Real(1.28E-13) * (H2_cosmic_ray_ionization_rate / Real(1.E-16));

    // ! H- + gamma --> H + e-
    Real k_hm_gamma = Real(5.9E-9) * G0;

    // ! H- + e -> H + e + e
    Real k13 = -Real(1.801849334E1) + lnTe * (Real(2.36085220E0) + lnTe * (-Real(2.82744300E-1) + lnTe * (Real(1.62331664E-2) + lnTe * (-Real(3.36501203E-2) + lnTe * (Real(1.17832978E-2) + lnTe * (-Real(1.65619470E-3) + lnTe * (Real(1.06827520E-4) - Real(2.63128581E-6) * lnTe)))))));
    k13 = rtz_math::exp(rtz_math::fmax(k13,-Real(92.0)));

    // ! H- + H --> H + H + e-
    // ! I think this reaction was broken in glover so I took the results from
    // ! https://www.aanda.org/articles/aa/pdf/2016/02/aa27262-15.pdf Table A1
    // ! Harley added the fudge factor for continuity
    Real k14 = Real(1.357772745525155) * Real(2.5634E-15) * rtz_math::pow(Te, Real(1.78186)); // Note that T must be in eV for this reaction to make sense
    if (T > Real(1160.0)) {
        k14 = -Real(3.388464953E1) + lnTe * (Real(1.13944933E0) + lnTe * (-Real(1.4210135E-1) + lnTe * (Real(8.4644554E-3) + lnTe * (-Real(1.4328641E-3) + lnTe * (Real(2.0122503E-4) + lnTe * (Real(8.6639632E-5) + lnTe * (-Real(2.5850097E-5) + lnTe * (Real(2.4555012E-6) - Real(8.0683825E-8) * lnTe))))))));
        k14 = rtz_math::exp(k14);
    }

    // ! H- + H+ --> H2+ + e- --> H + H (via recombinative dissociation)
    Real k15 = Real(6.9E-9) * rtz_math::pow(T,-Real(0.35));
    if (T > Real(8000.0)) {
        k15 = Real(9.6E-7) * rtz_math::pow(T,-Real(0.90));
    }

    // Correct steady state denominator terms scaled to cm^3 s^-1
    Real cr_photo_destruction = (k_hm_cr + k_hm_gamma) / (nH + Real(1e-40));
    Real collisional_destruction = k2 * xHI + k5 * xHII + k13 * xe + k14 * xHI + k15 * xHII;

    formation_rate += k1 * k2 * xe * xHI / (collisional_destruction + cr_photo_destruction);

    return formation_rate;
}

KOKKOS_FUNCTION
Real alpha_H2_dust(Real T, Real dust_to_gas_mass_ratio_over_mw){
    // Formation on dust
    Real clumping_factor = Real(1.0);
    Real T2 = T / Real(100.0);
    Real formation_rate = dust_to_gas_mass_ratio_over_mw * (Real(3.5E-17)) * clumping_factor * rtz_math::sqrt(rtz_math::fmin(T2,Real(1e2)));

    return formation_rate;
}

KOKKOS_FUNCTION
Real alpha_H2(Real T, Real dust_to_gas_mass_ratio_over_mw, Real xe, Real H2_cosmic_ray_ionization_rate, Real G0, Real xHI, Real xHII, Real nH){
    // Creation rate of molecular hydrogen
    // We consider both the primordial channel (via H-) as well
    // as formation on dust

    Real formation_rate = Real(0.0);

    // Formation rate on dust. Consider only HI
    formation_rate += alpha_H2_dust(T, dust_to_gas_mass_ratio_over_mw) * xHI * nH;

    // Primordial H- channel
    formation_rate += alpha_H2_prim(T, xe, H2_cosmic_ray_ionization_rate, G0, xHI, xHII, nH) * xHI * nH;

    return formation_rate;
}

KOKKOS_INLINE_FUNCTION
Real beta_H2_krome(Real T, Real nH, Real ne, Real nH2, Real nHe){
    // From Bovino 2016
    // https://www.aanda.org/articles/aa/pdf/2016/06/aa28158-16.pdf

    Real destruction_rate = Real(0.0);

    // H2 + H --> H + H + H (k18)
    destruction_rate += ((Real(6.67E-12) * rtz_math::sqrt(T) * rtz_math::exp(-Real(1.0) * (Real(1.0) + (Real(63593.0)/T)))) * nH);

    // H2 + H2 --> H2 + H + H (k19)
    destruction_rate += ((Real(5.996E-30) * rtz_math::pow(T,Real(4.1881)) * rtz_math::pow(Real(1.0) + (Real(6.761E-6) * T),-Real(5.6881)) * rtz_math::exp(-Real(54657.4)/T)) * nH2);

    // H2 + e- --> H + H + e- (k22)
    destruction_rate += (Real(4.38E-10) * rtz_math::pow(T,Real(0.35)) * rtz_math::exp(-Real(102000.0)/T) * ne);

    // H2 + He --> H + H + He
    destruction_rate += (rtz_math::exp10(-Real(27.029) + (Real(3.801) * rtz_math::log10(T)) - (Real(29487.0)/T)) * nHe);

    return destruction_rate;
}

KOKKOS_INLINE_FUNCTION
Real comp_Sd(Real nHI, Real nH2, Real dx_SS, Real Z) {
    //   ! Returns the self shielding factor for dust
    //   ! see section 2.2 http://iopscience.iop.org/0004-637X/697/1/55/pdf/apj_697_1_55.pdf

    Real Sdeff = Real(2.34e-21);    // dust cross section cm^2 ! Updated for bare-gr-s
    Real cNHI = nHI*dx_SS;    // HI column density
    Real cNH2 = nH2*dx_SS;    // H2 column density

    Real ss_factor = rtz_math::exp(-Sdeff*Z*(cNHI + (Real(2.0)*cNH2)));

    return ss_factor;
}

KOKKOS_INLINE_FUNCTION
Real comp_SH2(Real nH2, Real dx_SS) {
    // ! Returns the self shielding factor for dust
    // ! see section 2.2 http://iopscience.iop.org/0004-637X/697/1/55/pdf/apj_697_1_55.pdf

    Real cNH2 = nH2*dx_SS;  // H2 column density
    Real xfac = cNH2/(Real(5e14));
    Real wH2 = Real(0.2);

    Real Sa = (Real(1.0) - wH2)/((Real(1.0) + xfac)*(Real(1.0) + xfac));
    Real Sb = wH2/rtz_math::sqrt(Real(1.0) + xfac);
    Real Sc = rtz_math::exp(-Real(0.00085)*rtz_math::sqrt(Real(1.0) + xfac));

    Real ss_factor = Sa + (Sb*Sc);

    return ss_factor;
}

KOKKOS_INLINE_FUNCTION
Real alpha_CO(Real G0, Real xi_cr_H2, Real nCII, Real nH2, Real xO, Real nH){

    //! see glover 2012
    Real k0 = Real(5.E-16); //! cm^3 s^-1
    Real k1 = Real(5.E-10); //! Rate coefficient for the formation of CO from O + CHx

    //! Assuming CO formation is modulated by CH2+, https://home.strw.leidenuniv.nl/~ewine/photo/display_ch2+_65e31a07e69e64dbd64d37801983018f.html
    Real gammaCHx_cr = Real(8.88E-15) * (xi_cr_H2 / Real(1E-16)); //! Cosmic rays
    Real gammaCHx = (Real(1.41E-10) * G0) + gammaCHx_cr;

    Real beta = k1 * xO/(k1*xO + gammaCHx/nH);
    Real formation_rate = k0 * nCII * nH2 * beta;

    return rtz_math::fmax(formation_rate,Real(1e-100));
}

KOKKOS_INLINE_FUNCTION
Real beta_CO(Real G0, Real xi_cr_H2){
    //! CO destruction
    Real gammaCO = Real(2.43E-10) * G0;
    Real gammaCO_cr = Real(4.62E-15) * (xi_cr_H2 / Real(1E-16));

    Real destruction_rate = gammaCO + gammaCO_cr;

    return rtz_math::fmax(destruction_rate,Real(1e-100));
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
Real comp_SCO(const TabulatedData& tabData, Real nco_mol, Real nh2, Real dx_SS) {
    // initialize to 1.0
    Real ss_factor = Real(1.0);

    // CO self-shielding
    // Pinned to table at lower boundary, extrapolated above upper boundary
    Real effcNCO = rtz_math::fmax(nco_mol*dx_SS, tabData.sco_col1(1));
    if (effcNCO >= Real(1.578e+20)) {
        ss_factor *= Real(1.619e-05);
    } else {
        // Find the lower closest index
        int idxCO = 0;
        for (int i=0; i<51; i++) { // If above upper boundary then extrapolate using the slope
            // between the last two points, so conveniently we set i to not go to 51
            if (effcNCO >= tabData.sco_col1(i)) idxCO = i;
        }

        Real logeffcNCO = rtz_math::log10(effcNCO);
        // Interpolate or extrapolate automatically
        Real logsCO = rtz_math::log10(tabData.sco_col2(idxCO))
                        + (rtz_math::log10(tabData.sco_col2(idxCO+1)) - rtz_math::log10(tabData.sco_col2(idxCO)))
                        * (logeffcNCO - rtz_math::log10(tabData.sco_col1(idxCO)))
                        / (rtz_math::log10(tabData.sco_col1(idxCO+1)) - rtz_math::log10(tabData.sco_col1(idxCO)));

        ss_factor *= rtz_math::exp10(logsCO);
    }

    // H2 shielding of CO
    // Pinned to table at lower boundary, extrapolated above upper boundary
    Real effcNH2 = rtz_math::fmax(nh2*dx_SS, tabData.sh2_col1(1));
    if (effcNH2 >= Real(1.016e+22)) {
        ss_factor *= Real(4.666e-03);
    } else {
        // Find the lower closest index
        int idxH2 = 0;
        for (int i=1; i<42; i++) {
            if (effcNH2 >= tabData.sh2_col1(i)) idxH2 = i;
        }

        Real logeffcNH2 = rtz_math::log10(effcNH2);
        // Interpolate or extrapolate automatically
        Real logsH2 = rtz_math::log10(tabData.sh2_col2(idxH2))
                        + (rtz_math::log10(tabData.sh2_col2(idxH2+1)) - rtz_math::log10(tabData.sh2_col2(idxH2)))
                        * (logeffcNH2 - rtz_math::log10(tabData.sh2_col1(idxH2)))
                        / (rtz_math::log10(tabData.sh2_col1(idxH2+1)) - rtz_math::log10(tabData.sh2_col1(idxH2)));

        ss_factor *= rtz_math::exp10(logsH2);
    }

    return ss_factor;
}
