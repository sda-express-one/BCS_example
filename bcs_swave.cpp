// bcs_swave.cpp
// ---------------------------------------------------------------------------
// s-wave BCS superconductor on the 2D square lattice (nearest-neighbour
// tight binding), constant pairing interaction U0 (standard BCS, no
// extended-s form factor), optional Debye cutoff |xi_k| < hbar*omega_D.
//
//   xi_k = -2t (cos kx + cos ky) - mu                 (lattice constant a = 1)
//   gap equation:  1 = U0 (1/N^2) sum_k' tanh(E_k / 2 kB T) / (2 E_k)
//                  E_k = sqrt(xi_k^2 + Delta^2),  k' : |xi_k| < wD (wD=0: all k)
//
// Computes
//   1) the critical temperature Tc and Delta(T) for 0 <= T <= Tc
//   2) the quasiparticle density of states N_s(E)/N_n(E) at T = Tdos:
//        - direct histogram over the k grid
//        - exact N->infinity lattice result, using the analytic square-lattice
//          DOS rho(e) = K(sqrt(1-(e/4t)^2)) / (2 pi^2 t)
//        - textbook constant-DOS BCS / Dynes form Re[z/sqrt(z^2-Delta^2)], z=E+i*gamma
//
// Units: energies in meV, temperatures in K.
//
// Build : make            (g++ -O3 -march=native -fopenmp -std=c++17)
// Usage : ./bcs_swave [key=value ...]
//   N=4000        k-grid size N x N for the gap equation (must be even)
//   t=440         hopping (meV)
//   mu=0          chemical potential (meV); mu=0 is half filling
//   U0=240        constant attractive coupling (meV)
//   wD=0          Debye cutoff hbar*omega_D (meV); 0 = no cutoff (as in the thesis)
//   nT=200        number of temperature steps for Delta(T)
//   Tdos=0        temperature (K) at which the DOS is computed
//   Emax_over_D=10  DOS energy window, in units of Delta(Tdos)
//   nbins=1000    number of DOS bins in [0, Emax]
//   gamma=0       Dynes broadening (meV) for the constant-DOS column
//   Ndos=16000    k-grid size for the DOS histogram (must be even)
//   out=out       output directory
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr double kB = 0.08617333262;  // Boltzmann constant, meV/K
constexpr double PI = 3.14159265358979323846;

struct Params {
    int N = 4000;
    double t = 440.0;
    double mu = 0.0;
    double U0 = 240.0;
    double wD = 0.0;
    int nT = 200;
    double Tdos = 0.0;
    double Emax_over_D = 10.0;
    int nbins = 1000;
    double gamma = 0.0;
    int Ndos = 16000;
    std::string out = "out";
};

Params parse_args(int argc, char** argv) {
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            std::printf(
                "usage: %s [N=4000] [t=440] [mu=0] [U0=240] [wD=0] [nT=200] [Tdos=0]\n"
                "          [Emax_over_D=10] [nbins=1000] [gamma=0] [Ndos=16000] [out=out]\n",
                argv[0]);
            std::exit(0);
        }
        auto pos = a.find('=');
        if (pos == std::string::npos) throw std::runtime_error("bad argument (expected key=value): " + a);
        kv[a.substr(0, pos)] = a.substr(pos + 1);
    }
    Params p;
    for (auto& [k, v] : kv) {
        if (k == "N") p.N = std::stoi(v);
        else if (k == "t") p.t = std::stod(v);
        else if (k == "mu") p.mu = std::stod(v);
        else if (k == "U0") p.U0 = std::stod(v);
        else if (k == "wD") p.wD = std::stod(v);
        else if (k == "nT") p.nT = std::stoi(v);
        else if (k == "Tdos") p.Tdos = std::stod(v);
        else if (k == "Emax_over_D") p.Emax_over_D = std::stod(v);
        else if (k == "nbins") p.nbins = std::stoi(v);
        else if (k == "gamma") p.gamma = std::stod(v);
        else if (k == "Ndos") p.Ndos = std::stoi(v);
        else if (k == "out") p.out = v;
        else throw std::runtime_error("unknown parameter: " + k);
    }
    if (p.N < 4 || p.N % 2) throw std::runtime_error("N must be even and >= 4");
    if (p.Ndos < 4 || p.Ndos % 2) throw std::runtime_error("Ndos must be even and >= 4");
    if (p.nT < 2 || p.nbins < 10) throw std::runtime_error("nT >= 2 and nbins >= 10 required");
    return p;
}

// ---------------------------------------------------------------------------
// k grid reduced by symmetry.
// Periodic grid k_i = -pi + 2 pi i / N, i = 0..N-1 (no double-counted edge).
// cos k_i = cos k_{N-i}, so the distinct cosines are i = 0..N/2 with
// multiplicity m_i = 1 (i = 0, N/2) or 2. xi depends on cos kx + cos ky only,
// symmetric under kx <-> ky, so we sum over pairs a <= b with weight
// m_a m_b (x2 if a != b). Total weight = N^2 exactly. ~N^2/8 points.
// ---------------------------------------------------------------------------
struct Cosines {
    std::vector<double> c;
    std::vector<double> m;
};

Cosines make_cosines(int N) {
    Cosines cs;
    const int h = N / 2;
    cs.c.resize(h + 1);
    cs.m.resize(h + 1);
    for (int i = 0; i <= h; ++i) {
        const double k = -PI + 2.0 * PI * i / N;
        cs.c[i] = std::cos(k);
        cs.m[i] = (i == 0 || i == h) ? 1.0 : 2.0;
    }
    return cs;
}

struct KGrid {
    std::vector<double> xi2;  // xi_k^2 for k inside the pairing shell
    std::vector<double> w;    // symmetry weights
    double norm = 1.0;        // N^2
};

KGrid build_gap_grid(const Params& p) {
    const Cosines cs = make_cosines(p.N);
    const int h = p.N / 2;
    KGrid g;
    g.norm = double(p.N) * double(p.N);
    g.xi2.reserve(size_t(h + 1) * (h + 2) / 2);
    g.w.reserve(size_t(h + 1) * (h + 2) / 2);
    for (int a = 0; a <= h; ++a) {
        for (int b = a; b <= h; ++b) {
            const double xi = -2.0 * p.t * (cs.c[a] + cs.c[b]) - p.mu;
            if (p.wD > 0.0 && std::fabs(xi) >= p.wD) continue;
            g.xi2.push_back(xi * xi);
            g.w.push_back(cs.m[a] * cs.m[b] * (a == b ? 1.0 : 2.0));
        }
    }
    return g;
}

// (1/N^2) sum_k tanh(E/2kT)/(2E)
double gap_sum(const KGrid& g, double D, double T) {
    const double D2 = D * D;
    const size_t n = g.xi2.size();
    const double* xi2 = g.xi2.data();
    const double* w = g.w.data();
    double s = 0.0;
    if (T <= 0.0) {
#pragma omp parallel for reduction(+ : s) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            const double E = std::sqrt(xi2[i] + D2);
            s += (E > 0.0) ? w[i] / (2.0 * E) : HUGE_VAL;
        }
    } else {
        const double beta = 1.0 / (kB * T);
#pragma omp parallel for reduction(+ : s) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            const double E = std::sqrt(xi2[i] + D2);
            const double x = 0.5 * beta * E;
            // tanh(x)/(2E) -> beta/4 for x -> 0
            s += (x < 1e-8) ? w[i] * 0.25 * beta : w[i] * std::tanh(x) / (2.0 * E);
        }
    }
    return s / g.norm;
}

// Brent's root finder (Numerical Recipes zbrent), requires fa*fb <= 0.
template <class F>
double brent(F&& f, double a, double b, double fa, double fb, double xtol, int maxit = 300) {
    if (fa == 0.0) return a;
    if (fb == 0.0) return b;
    if ((fa > 0.0) == (fb > 0.0)) throw std::runtime_error("brent: root not bracketed");
    double c = b, fc = fb, d = b - a, e = d;
    for (int it = 0; it < maxit; ++it) {
        if ((fb > 0.0) == (fc > 0.0)) {
            c = a;
            fc = fa;
            e = d = b - a;
        }
        if (std::fabs(fc) < std::fabs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }
        const double tol1 = 2.0 * DBL_EPSILON * std::fabs(b) + 0.5 * xtol;
        const double xm = 0.5 * (c - b);
        if (std::fabs(xm) <= tol1 || fb == 0.0) return b;
        if (std::fabs(e) >= tol1 && std::fabs(fa) > std::fabs(fb)) {
            const double s = fb / fa;
            double pp, q;
            if (a == c) {
                pp = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                const double qq = fa / fc, r = fb / fc;
                pp = s * (2.0 * xm * qq * (qq - r) - (b - a) * (r - 1.0));
                q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (pp > 0.0) q = -q;
            pp = std::fabs(pp);
            const double min1 = 3.0 * xm * q - std::fabs(tol1 * q);
            const double min2 = std::fabs(e * q);
            if (2.0 * pp < std::min(min1, min2)) {
                e = d;
                d = pp / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        b += (std::fabs(d) > tol1) ? d : (xm >= 0.0 ? tol1 : -tol1);
        fb = f(b);
    }
    std::fprintf(stderr, "warning: brent did not converge\n");
    return b;
}

double solve_Tc(const KGrid& g, const Params& p) {
    auto f = [&](double T) { return p.U0 * gap_sum(g, 0.0, T) - 1.0; };
    double Thi = 1.0, fhi = f(Thi);
    while (fhi > 0.0) {
        Thi *= 2.0;
        if (Thi > 1e6) throw std::runtime_error("Tc > 1e6 K ?! check parameters");
        fhi = f(Thi);
    }
    double Tlo = 0.5 * Thi, flo = f(Tlo);
    while (flo < 0.0) {
        Tlo *= 0.5;
        if (Tlo < 1e-6) throw std::runtime_error("no superconducting solution down to 1e-6 K (U0 too small?)");
        flo = f(Tlo);
    }
    if (Tlo * 2.0 < Thi) Thi = 2.0 * Tlo, fhi = f(Thi);
    return brent(f, Tlo, Thi, flo, fhi, 1e-12 * Thi);
}

// Delta(T) for T < Tc. guess: a value near the expected solution (or <=0).
double solve_Delta(const KGrid& g, const Params& p, double T, double guess) {
    auto f = [&](double D) { return p.U0 * gap_sum(g, D, T) - 1.0; };
    const double lo = 1e-12 * p.t;
    const double flo = f(lo);
    if (flo <= 0.0) return 0.0;  // T >= Tc (or numerically at Tc)
    double hi = (guess > 0.0) ? 1.05 * guess : 1.0;
    double fhi = f(hi);
    while (fhi > 0.0) {
        hi *= 2.0;
        if (hi > 1e3 * p.t) throw std::runtime_error("Delta bracket failed");
        fhi = f(hi);
    }
    return brent(f, lo, hi, flo, fhi, 1e-11 * std::max(hi, 1e-3));
}

// Analytic DOS per site per spin of the nearest-neighbour square lattice band
// e = -2t(cos kx + cos ky):  rho(e) = K(k) / (2 pi^2 t),  k = sqrt(1 - (e/4t)^2)
double rho_band(double e, double t) {
    const double x = e / (4.0 * t);
    if (std::fabs(x) >= 1.0) return 0.0;
    if (x == 0.0) return HUGE_VAL;  // log van Hove singularity
    return std::comp_ellint_1(std::sqrt(1.0 - x * x)) / (2.0 * PI * PI * t);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Params p = parse_args(argc, argv);
        namespace fs = std::filesystem;
        fs::create_directories(p.out);
        const auto t0 = std::chrono::steady_clock::now();
        auto elapsed = [&]() {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        };

        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        std::printf("s-wave BCS on the 2D square lattice (constant U0)\n");
        std::printf("  N=%d t=%g mu=%g U0=%g wD=%g (meV), threads=%d\n", p.N, p.t, p.mu, p.U0, p.wD, nthreads);

        // ---------------- gap equation ----------------
        const KGrid g = build_gap_grid(p);
        std::printf("  k points in pairing shell (reduced): %zu\n", g.xi2.size());
        if (g.xi2.empty()) throw std::runtime_error("no k points inside the pairing shell");

        const double Tc = solve_Tc(g, p);
        const double D0 = solve_Delta(g, p, 0.0, -1.0);
        const double ratio = 2.0 * D0 / (kB * Tc);
        std::printf("  Tc      = %.6f K\n", Tc);
        std::printf("  Delta0  = %.6f meV\n", D0);
        std::printf("  2Delta0/kBTc = %.4f   (weak-coupling BCS: 3.528)\n", ratio);

        // Delta(T): T = Tc (1 - (1-u)^2), denser close to Tc
        std::vector<double> Ts, Ds;
        double guess = D0;
        for (int j = 0; j <= p.nT; ++j) {
            const double u = double(j) / p.nT;
            const double T = Tc * (1.0 - (1.0 - u) * (1.0 - u));
            double D;
            if (j == 0) D = D0;
            else if (j == p.nT) D = 0.0;
            else D = solve_Delta(g, p, T, guess);
            if (D > 0.0) guess = D;
            Ts.push_back(T);
            Ds.push_back(D);
        }
        {
            const std::string fn = p.out + "/gap_vs_T.dat";
            FILE* fp = std::fopen(fn.c_str(), "w");
            if (!fp) throw std::runtime_error("cannot open " + fn);
            std::fprintf(fp, "# s-wave gap vs temperature, N=%d t=%g mu=%g U0=%g wD=%g\n", p.N, p.t, p.mu, p.U0, p.wD);
            std::fprintf(fp, "# Tc = %.10g K, Delta0 = %.10g meV\n", Tc, D0);
            std::fprintf(fp, "# T[K]  T/Tc  Delta[meV]  Delta/Delta0\n");
            for (size_t i = 0; i < Ts.size(); ++i)
                std::fprintf(fp, "%.10e %.10e %.10e %.10e\n", Ts[i], Ts[i] / Tc, Ds[i], Ds[i] / D0);
            std::fclose(fp);
        }
        std::printf("  Delta(T) written (%zu points)            [%.2f s]\n", Ts.size(), elapsed());

        // ---------------- density of states ----------------
        double Dd = 0.0;
        if (p.Tdos <= 0.0) Dd = D0;
        else if (p.Tdos < Tc) Dd = solve_Delta(g, p, p.Tdos, D0);
        bool dos_ok = Dd > 0.0;
        if (!dos_ok) std::fprintf(stderr, "warning: Tdos >= Tc, Delta = 0, DOS not computed\n");

        if (dos_ok) {
            const double Emax = p.Emax_over_D * Dd;
            const int nb = p.nbins;
            const double dE = Emax / nb;
            std::vector<double> hs(nb, 0.0), hn(nb, 0.0);

            // single pass over the (symmetry-reduced) Ndos x Ndos grid
            const Cosines cs = make_cosines(p.Ndos);
            const int h = p.Ndos / 2;
#pragma omp parallel
            {
                std::vector<double> ls(nb, 0.0), ln(nb, 0.0);
#pragma omp for schedule(dynamic, 16)
                for (int a = 0; a <= h; ++a) {
                    for (int b = a; b <= h; ++b) {
                        const double xi = -2.0 * p.t * (cs.c[a] + cs.c[b]) - p.mu;
                        const double w = cs.m[a] * cs.m[b] * (a == b ? 1.0 : 2.0);
                        const double axi = std::fabs(xi);
                        const double gap = (p.wD <= 0.0 || axi < p.wD) ? Dd : 0.0;
                        const double E = std::sqrt(xi * xi + gap * gap);
                        const long is = long(E / dE);
                        if (is < nb) ls[is] += w;
                        const long in = long(axi / dE);
                        if (in < nb) ln[in] += w;
                    }
                }
#pragma omp critical
                for (int i = 0; i < nb; ++i) {
                    hs[i] += ls[i];
                    hn[i] += ln[i];
                }
            }

            // exact lattice result: n(x) = rho(x+mu) + rho(-x+mu) (both branches of xi)
            auto nboth = [&](double x) { return rho_band(x + p.mu, p.t) + rho_band(-x + p.mu, p.t); };
            auto Ns_exact = [&](double E) {
                double s = 0.0;
                if (E > Dd) {
                    const double xi = std::sqrt(E * E - Dd * Dd);
                    if (p.wD <= 0.0 || xi < p.wD) s += nboth(xi) * E / xi;
                }
                if (p.wD > 0.0 && E >= p.wD) s += nboth(E);  // unpaired states outside the shell
                return s;
            };

            const std::string fn = p.out + "/dos.dat";
            FILE* fp = std::fopen(fn.c_str(), "w");
            if (!fp) throw std::runtime_error("cannot open " + fn);
            std::fprintf(fp, "# quasiparticle DOS, s-wave, Tdos=%g K, Delta=%.10g meV, Ndos=%d, nbins=%d, gamma=%g\n",
                         p.Tdos, Dd, p.Ndos, nb, p.gamma);
            std::fprintf(fp, "# normalised to the normal-state DOS at the SAME energy, N_n(E) (not N_n(E_F):\n");
            std::fprintf(fp, "#   at mu=0 the 2D square lattice has a log van Hove divergence at E_F)\n");
            std::fprintf(fp, "# E/Delta  E[meV]  Ns/Nn_histogram  Ns/Nn_exact_lattice  Ns/Nn_constDOS_BCS(Dynes)  Ns_hist[1/meV/site/spin]\n");
            const double norm = double(p.Ndos) * double(p.Ndos) * dE;
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                for (int k = 0; k < nb; ++k) {
                    const int i = (sgn < 0) ? nb - 1 - k : k;
                    const double E = (i + 0.5) * dE;
                    const double rh = hn[i] > 0.0 ? hs[i] / hn[i] : 0.0;
                    const double nn = nboth(E);
                    const double rx = nn > 0.0 ? Ns_exact(E) / nn : 0.0;
                    const std::complex<double> z(E, p.gamma);
                    const double rd = std::real(z / std::sqrt(z * z - Dd * Dd));
                    std::fprintf(fp, "%.8e %.8e %.8e %.8e %.8e %.8e\n", sgn * E / Dd, sgn * E, rh, rx, rd,
                                 hs[i] / norm);
                }
            }
            std::fclose(fp);
            std::printf("  DOS written (Ndos=%d, %d bins, dE=%.4g meV)  [%.2f s]\n", p.Ndos, nb, dE, elapsed());
        }

        // ---------------- summary ----------------
        {
            const std::string fn = p.out + "/summary.txt";
            FILE* fp = std::fopen(fn.c_str(), "w");
            if (!fp) throw std::runtime_error("cannot open " + fn);
            std::fprintf(fp, "N = %d\nt = %.10g\nmu = %.10g\nU0 = %.10g\nwD = %.10g\n", p.N, p.t, p.mu, p.U0, p.wD);
            std::fprintf(fp, "Tc = %.10g\nDelta0 = %.10g\nratio_2D0_kTc = %.10g\n", Tc, D0, ratio);
            std::fprintf(fp, "Tdos = %.10g\nDelta_dos = %.10g\nNdos = %d\nnbins = %d\ngamma = %.10g\n", p.Tdos, Dd,
                         p.Ndos, p.nbins, p.gamma);
            // weak-coupling BCS estimate (meaningful with a cutoff and mu away from the van Hove point)
            if (p.wD > 0.0) {
                const double N0 = rho_band(p.mu, p.t);
                if (std::isfinite(N0) && N0 > 0.0) {
                    const double lam = N0 * p.U0;
                    const double TcB = 1.1339 * p.wD / kB * std::exp(-1.0 / lam);
                    const double D0B = p.wD / std::sinh(1.0 / lam);
                    std::fprintf(fp, "N0 = %.10g\nlambda = %.10g\nTc_BCS = %.10g\nDelta0_BCS = %.10g\n", N0, lam, TcB, D0B);
                    std::printf("  weak-coupling BCS (N0 = rho(mu)): lambda=%.4f Tc=%.4f K Delta0=%.4f meV\n", lam, TcB, D0B);
                }
            }
            std::fclose(fp);
        }
        std::printf("  done, output in %s/                       [%.2f s]\n", p.out.c_str(), elapsed());
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
