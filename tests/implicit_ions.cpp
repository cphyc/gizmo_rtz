#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "types.hpp"
#include "implicit_ions.hpp"

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Fixture {
    std::array<Element, MAX_ELEMENTS> elements{};
    CompactIonData ions{};
    std::array<Real, IMPLICIT_ION_WORK> data{};
    Fixture(int stages = 3) {
        std::array<int, MAX_ELEMENTS> active;
        active.fill(-1);
        elements[1].atomic_number = 1; elements[1].n_ions = 2;
        elements[2].atomic_number = 2; elements[2].n_ions = stages;
        active[1] = active[2] = 0;
        ions.init_offsets(elements.data(), active.data());
        ions.n_element[1] = ions.n_element_new[1] = Real(2.5);
        ions.n_element[2] = ions.n_element_new[2] = Real(0.3);
        SparseIonSystem s(data.data());
        s.old[ions.offsets[1]] = Real(0.8);
        s.old[ions.offsets[1]+1] = Real(0.2);
        s.old[ions.offsets[2]] = Real(1);
    }
    std::array<Real, IMPLICIT_ION_WORK> run(Real dt, Real hydrogen_mass = 1, bool serial = false, int stride = 1) {
        Kokkos::View<Element*> e("implicit_elements", MAX_ELEMENTS);
        auto eh = Kokkos::create_mirror_view(e);
        for (int i = 0; i < MAX_ELEMENTS; ++i) eh(i) = elements[i];
        Kokkos::deep_copy(e, eh);
        Kokkos::View<CompactIonData*> n("implicit_ions", 1);
        auto nh = Kokkos::create_mirror_view(n); nh(0) = ions; Kokkos::deep_copy(n, nh);
        const int column = stride - 1;
        Kokkos::View<Real*> w("implicit_work", IMPLICIT_ION_WORK * stride);
        auto wh = Kokkos::create_mirror_view(w);
        Kokkos::deep_copy(wh, Real(-555));
        for (int i = 0; i < IMPLICIT_ION_WORK; ++i) wh(i * stride + column) = data[i];
        Kokkos::deep_copy(w, wh);
        Kokkos::View<int*> ok("implicit_result", 1);
        if (serial) {
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {
                ok(0) = SparseIonSystem(w.data() + column, stride).solve(SerialIonExecutor{}, e.data(), n(0), dt, hydrogen_mass);
            });
        } else {
            using Policy = Kokkos::TeamPolicy<>;
#ifdef KOKKOS_ENABLE_CUDA
            Policy policy(1, 32);
#else
            Policy policy(1, Kokkos::AUTO());
#endif
            Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const Policy::member_type& team) {
                const bool result = SparseIonSystem(w.data() + column, stride).solve(
                    TeamIonExecutor<Policy::member_type>{team}, e.data(), n(0), dt, hydrogen_mass);
                Kokkos::single(Kokkos::PerTeam(team), [&]() { ok(0) = result; });
            });
        }
        auto result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ok);
        require(result(0) == 1, "sparse nonlinear solve failed");
        Kokkos::deep_copy(wh, w);
        std::array<Real, IMPLICIT_ION_WORK> answer{};
        for (int i = 0; i < IMPLICIT_ION_WORK; ++i) answer[i] = wh(i * stride + column);
        for (int i = 0; i < IMPLICIT_ION_WORK; ++i)
            for (int j = 0; j < column; ++j)
                require(wh(i * stride + j) == Real(-555), "strided solve overwrote a neighboring cell");
        return answer;
    }
};

void linear_and_stiff() {
    for (Real dt : {Real(1e-7), Real(0.2), Real(4), Real(1e6), Real(1e12)}) {
        Fixture f(3); SparseIonSystem s(f.data.data());
        const int h = f.ions.offsets[1], b = f.ions.offsets[2];
        s.up[h] = Real(0.7); s.down[h+1] = Real(0.2);
        s.up[b] = Real(0.3); s.up[b+1] = Real(0.7);
        s.down[b+1] = Real(0.2); s.down[b+2] = Real(0.8);
        auto answer = f.run(dt); SparseIonSystem out(answer.data());
        const Real expected = (Real(0.2) + dt * Real(0.7)) / (Real(1) + dt * Real(0.9));
        require(std::abs(out.x[h+1] - expected) < 3e-12, "hydrogen backward Euler analytic mismatch");
        require(std::abs(out.x[h] + out.x[h+1] - 1) < 3e-12, "hydrogen mass not conserved");
        Real mass = 0;
        for (int j = 0; j < 3; ++j) {
            const int k = b+j;
            Real incoming = 0;
            if (j > 0) incoming += s.up[k-1] * out.x[k-1];
            if (j < 2) incoming += s.down[k+1] * out.x[k+1];
            const Real lhs = (1 + dt*(s.up[k]+s.down[k])) * out.x[k];
            const Real rhs = s.old[k] + dt*incoming;
            require(std::abs(lhs-rhs) / (1+std::abs(lhs)+std::abs(rhs)) < 3e-12,
                    "tridiagonal backward Euler residual too large");
            require(out.x[k] >= 0, "negative abundance"); mass += out.x[k];
        }
        require(std::abs(mass-1) < 3e-12, "element mass not conserved");
        auto serial = f.run(dt, 1, true, 3); SparseIonSystem serial_out(serial.data());
        for (int k = h; k < b+3; ++k)
            require(std::abs(out.x[k]-serial_out.x[k]) < 3e-12, "team and serial solvers disagree");
    }
}

void manufactured_charge_exchange() {
    Fixture f(3); SparseIonSystem s(f.data.data());
    const int h = f.ions.offsets[1], b = f.ions.offsets[2];
    const Real dt = Real(0.03), hp = Real(0.35), hi = Real(0.65), nh = Real(2.5), nz = Real(0.3);
    const Real target[3] = {Real(0.5), Real(0.3), Real(0.2)};
    s.ct_up[b] = Real(0.8); s.ct_up[b+1] = Real(0.4);
    s.ct_down[b+1] = Real(0.7); s.ct_down[b+2] = Real(0.9);
    Real hydrogen_rate = 0;
    for (int j = 0; j < 3; ++j) {
        const int k = b+j;
        Real rate = -(s.ct_up[k]*nh*hp+s.ct_down[k]*nh*hi)*target[j];
        if (j > 0) rate += s.ct_up[k-1]*nh*hp*target[j-1];
        if (j < 2) rate += s.ct_down[k+1]*nh*hi*target[j+1];
        s.old[k] = target[j]-dt*rate;
        hydrogen_rate += nz*(s.ct_down[k]*hi-s.ct_up[k]*hp)*target[j];
    }
    s.old[h+1] = hp-dt*hydrogen_rate; s.old[h] = 1-s.old[h+1];
    auto answer = f.run(dt); SparseIonSystem out(answer.data());
    require(std::abs(out.x[h+1]-hp) < 3e-12, "nonlinear hydrogen charge-exchange mismatch");
    Real old_charge = nh*s.old[h+1], new_charge = nh*out.x[h+1];
    for (int j = 0; j < 3; ++j) {
        require(std::abs(out.x[b+j]-target[j]) < 3e-12, "nonlinear partner charge-exchange mismatch");
        old_charge += nz*j*s.old[b+j]; new_charge += nz*j*out.x[b+j];
    }
    require(std::abs(old_charge-new_charge) < 3e-12, "charge exchange fails to conserve electrons");
}

void molecular_hydrogen_mass() {
    Fixture f; SparseIonSystem s(f.data.data());
    const int h = f.ions.offsets[1];
    s.up[h] = Real(0.7); s.down[h+1] = Real(0.2);
    auto answer = f.run(Real(4), Real(0.6)); SparseIonSystem out(answer.data());
    require(std::abs(out.x[h]+out.x[h+1]-Real(0.6)) < 3e-12, "atomic H ignores frozen H2 mass");
}

void stiff_charge_exchange() {
    Fixture f(3); SparseIonSystem s(f.data.data());
    const int h = f.ions.offsets[1], b = f.ions.offsets[2];
    s.ct_up[b] = Real(0.8); s.ct_up[b+1] = Real(0.4);
    s.ct_down[b+1] = Real(0.7); s.ct_down[b+2] = Real(0.9);
    auto answer = f.run(Real(1e12)); SparseIonSystem out(answer.data());
    Real charge = Real(2.5)*out.x[h+1];
    for (int j = 0; j < 3; ++j) charge += Real(0.3)*j*out.x[b+j];
    require(std::abs(charge-Real(0.5)) < 3e-12, "stiff charge exchange loses total charge");
    const Real neutral = out.x[h];
    for (int j = 0; j < 2; ++j) {
        const Real forward = s.ct_up[b+j]*out.x[h+1]*out.x[b+j];
        const Real reverse = s.ct_down[b+j+1]*neutral*out.x[b+j+1];
        require(std::abs(forward-reverse) < 3e-12, "stiff CT does not approach detailed balance");
    }
}
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    int result = 0;
    try {
        linear_and_stiff(); manufactured_charge_exchange(); molecular_hydrogen_mass(); stiff_charge_exchange();
        std::cout << "Implicit ions: analytic, stiff, sparse residual, positivity, conservation, CT and executor checks passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; result = 1; }
    clear_ion_offset_cache();
    Kokkos::finalize();
    return result;
}
