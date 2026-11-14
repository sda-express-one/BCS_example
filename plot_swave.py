#!/usr/bin/env python
"""Plot the output of bcs_swave (s-wave BCS on the 2D square lattice).

Usage (conda env00):
    conda run -n env00 python plot_swave.py [output_dir]

Reads <dir>/summary.txt, <dir>/gap_vs_T.dat, <dir>/dos.dat and writes
gap_vs_T, gap_universal and dos figures (PNG + PDF) into the same directory,
plus an untitled PDF-only set (*_notitle.pdf).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KB = 0.08617333262  # meV/K
BCS_RATIO = np.pi / np.exp(np.euler_gamma)  # Delta0 / kB Tc = 1.764 (weak coupling)


def read_summary(path):
    d = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = (s.strip() for s in line.split("=", 1))
            d[k] = float(v)
    return d


def bcs_universal(tred):
    """Weak-coupling continuum BCS: Delta(T)/Delta0 vs T/Tc.

    Solves  ln(Delta0/Delta) = 2 * int_0^inf dxi f(E)/E,  E = sqrt(xi^2 + Delta^2),
    in units Delta0 = 1, kB Tc = 1/1.764.
    """
    out = np.zeros_like(tred)
    for i, tr in enumerate(tred):
        if tr <= 0:
            out[i] = 1.0
            continue
        if tr >= 1:
            out[i] = 0.0
            continue
        kT = tr / BCS_RATIO
        xi = np.linspace(0.0, 60.0 * kT, 20001)

        def g(y):
            E = np.sqrt(xi**2 + y**2)
            f = 1.0 / (np.exp(np.minimum(E / kT, 700.0)) + 1.0)
            return np.log(1.0 / y) - 2.0 * np.trapezoid(f / E, xi)

        lo, hi = 1e-10, 1.0  # g(lo) > 0, g(1) <= 0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if g(mid) > 0:
                lo = mid
            else:
                hi = mid
        out[i] = 0.5 * (lo + hi)
    return out


# output style: set by make_plots()
CFG = {"titles": True, "exts": ("png", "pdf"), "suffix": ""}


def set_title(ax, text):
    if CFG["titles"]:
        ax.set_title(text)


def save(fig, outdir, name):
    name = name + CFG["suffix"]
    for ext in CFG["exts"]:
        fig.savefig(outdir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outdir / name}." + "/.".join(CFG["exts"]))


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "out"
    # standard set: with titles, PNG + PDF
    CFG.update(titles=True, exts=("png", "pdf"), suffix="")
    make_plots(outdir)
    # additional set: no titles, PDF only (e.g. for LaTeX / slides with their own captions)
    CFG.update(titles=False, exts=("pdf",), suffix="_notitle")
    make_plots(outdir)


def make_plots(outdir):
    s = read_summary(outdir / "summary.txt")
    Tc, D0, ratio = s["Tc"], s["Delta0"], s["ratio_2D0_kTc"]
    params = f"$t={s['t']:g}$, $\\mu={s['mu']:g}$, $U_0={s['U0']:g}$ meV" + (
        f", $\\hbar\\omega_D={s['wD']:g}$ meV" if s["wD"] > 0 else ", no cutoff"
    )

    # ---------- 1) Delta(T) ----------
    T, tred, D, dred = np.loadtxt(outdir / "gap_vs_T.dat", unpack=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(T, D, color="C0", lw=2, label=r"$\Delta_s(T)$, numerical")
    ax.axvline(Tc, color="gray", ls="--", lw=1)
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"$\Delta_s(T)$ (meV)")
    ax.set_xlim(0, 1.05 * Tc)
    ax.set_ylim(0, 1.1 * D0)
    set_title(ax, r"s-wave gap vs temperature")
    ax.text(
        0.03, 0.05,
        f"$T_c$ = {Tc:.2f} K\n$\\Delta_0$ = {D0:.3f} meV\n$2\\Delta_0/k_BT_c$ = {ratio:.3f}\n{params}",
        transform=ax.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="0.7"),
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    save(fig, outdir, "gap_vs_T")

    # ---------- 2) universal curve ----------
    tt = np.linspace(0, 1, 121)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(tred, dred, color="C0", lw=2, label="lattice, numerical")
    ax.plot(tt, bcs_universal(tt), "k--", lw=1.3, label="BCS weak coupling (continuum)")
    ax.set_xlabel(r"$T/T_c$")
    ax.set_ylabel(r"$\Delta(T)/\Delta_0$")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    set_title(ax, "reduced gap: comparison with BCS")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    save(fig, outdir, "gap_universal")

    # ---------- 3) DOS ----------
    dos_file = outdir / "dos.dat"
    if not dos_file.exists():
        print("  no dos.dat, skipping DOS plot")
        return
    ered, E, rh, rx, rd, _ = np.loadtxt(dos_file, unpack=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.fill_between(ered, 0, rh, color="C0", alpha=0.18, lw=0)
    ax.plot(ered, rh, ".", ms=2.5, color="C0", label=r"$k$-grid histogram")
    ax.plot(ered, rx, "-", lw=1.4, color="C3", label=r"exact lattice ($N\to\infty$)")
    lab = "const. DOS BCS" + (f" (Dynes $\\Gamma$={s['gamma']:g} meV)" if s["gamma"] > 0 else "")
    ax.plot(ered, rd, "--", lw=1.0, color="k", label=lab)
    ax.axhline(1.0, color="gray", lw=0.7)
    top = 1.15 * np.nanmax(rh)
    ax.set_ylim(0, top)
    ax.set_xlim(ered.min(), ered.max())
    ax.set_xlabel(r"$E/\Delta_s$")
    ax.set_ylabel(r"$N_s(E)/N_n(E)$")
    Tdos = s["Tdos"]
    set_title(ax, f"quasiparticle DOS, s-wave, $T$ = {Tdos:g} K, $\\Delta$ = {s['Delta_dos']:.3f} meV")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    save(fig, outdir, "dos")


if __name__ == "__main__":
    main()
