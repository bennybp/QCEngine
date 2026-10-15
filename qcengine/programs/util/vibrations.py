"""Harmonic vibrational analysis in the layout of the proposed QCSchema ``Vibrations`` model.

Harnesses that compute a Hessian run ``harmonic_analysis`` on it and report the
result as the ``VIBRATIONS`` QCVariable (QCFractal stores it as the
``vibrations`` record property). Using one analysis for every program gives
directly comparable results. It follows the definitions, units and conventions
of psi4's ``qcdb.vib.harmonic_analysis`` (an independent implementation, so
psi4 is not needed); psi4's irrep labels (``gamma``) are not reproduced.

- every per-mode array has one entry per mode, for all 3*natom modes, sorted
  from most negative to most positive force constant;
- ``x``, ``q`` and ``w`` are (3*natom, 3*natom) with one column per mode;
- ``omega`` is in atomic units (E_h/hbar) and ``omega_in_wavenumbers`` in cm^-1,
  negative for imaginary modes;
- ``trv`` is "v" (vibration), "tr" (projected translation/rotation) or "-"
  (unclassified zero mode).
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np
import qcelemental as qcel

# Gradient RMS [Eh/a0] below which rotations are projected out (psi4.frequency() criterion)
STATIONARY_GRADIENT_RMS = 1.0e-2

# Optional per-mode fields of the Vibrations model, in model order
_PER_MODE_FIELDS = ("degeneracy", "trv", "gamma", "mu", "k", "dq0", "qtp0", "xtp0", "theta_vib", "infrared_intensity")


def build_vibrations(
    omega_in_wavenumbers: Sequence[float],
    x: np.ndarray,
    *,
    masses: Optional[Sequence[float]] = None,
    q: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    **per_mode: Optional[Sequence[Any]],
) -> Dict[str, Any]:
    """Assemble a JSON-serializable ``Vibrations`` dictionary.

    Parameters
    ----------
    omega_in_wavenumbers
        Frequency of each mode [cm^-1]; negative for imaginary modes.
    x
        (3*natom, nmode) normalized un-mass-weighted normal modes [a0], one column per mode.
    masses
        (natom,) atomic masses [u]. If ``q`` and ``w`` are not given, they are
        derived from ``x`` with these masses, using psi4's definitions
        (``q`` normalized mass-weighted, ``w = q / sqrt(m)``, ``x = w / |w|``).
    q, w
        (3*natom, nmode) normalized mass-weighted [a0 u^1/2] and un-mass-weighted [a0] modes.
    **per_mode
        Any of the optional per-mode fields: degeneracy, trv, gamma, mu, k,
        dq0, qtp0, xtp0, theta_vib, infrared_intensity. ``None`` entries in
        ``gamma`` become "-".
    """
    unknown = set(per_mode) - set(_PER_MODE_FIELDS)
    if unknown:
        raise TypeError(f"Unknown Vibrations fields: {sorted(unknown)}")

    omega_cm = np.asarray(omega_in_wavenumbers, dtype=float)
    nmode = len(omega_cm)
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != nmode:
        raise ValueError(f"x has shape {x.shape}; expected (3*natom, {nmode}).")

    if q is None and w is None and masses is not None:
        sqrtm = np.sqrt(np.repeat(np.asarray(masses, dtype=float), 3))
        q = sqrtm[:, None] * x
        q = q / np.linalg.norm(q, axis=0)
        w = q / sqrtm[:, None]

    result = {
        "omega": (omega_cm / qcel.constants.hartree2wavenumbers).tolist(),
        "omega_in_wavenumbers": omega_cm.tolist(),
        "x": x.tolist(),
    }
    for key, arr in (("q", q), ("w", w)):
        if arr is not None:
            arr = np.asarray(arr, dtype=float)
            if arr.shape != x.shape:
                raise ValueError(f"{key} has shape {arr.shape}; expected {x.shape}.")
            result[key] = arr.tolist()

    for key in _PER_MODE_FIELDS:
        values = per_mode.get(key)
        if values is None:
            continue
        if len(values) != nmode:
            raise ValueError(f"Got {len(values)} values of {key} for {nmode} modes.")
        if key == "degeneracy":
            result[key] = [int(v) for v in values]
        elif key == "trv":
            result[key] = [str(v).lower() for v in values]
        elif key == "gamma":
            result[key] = ["-" if v is None else str(v) for v in values]
        else:
            result[key] = [float(v) for v in values]
    return result


def projection_flags(gradient: Optional[np.ndarray], external_potential: bool = False) -> Dict[str, bool]:
    """Which rigid motions to project out, following psi4.frequency().

    Translations are projected unless an external potential/field breaks
    translational invariance. Rotations are projected only in addition, and
    only at a stationary point (gradient RMS below ``STATIONARY_GRADIENT_RMS``);
    without a gradient, rotations are not projected.
    """
    project_trans = not external_potential
    stationary = gradient is not None and float(np.sqrt(np.mean(np.square(gradient)))) < STATIONARY_GRADIENT_RMS
    return {"project_trans": project_trans, "project_rot": project_trans and stationary}


def _rigid_motion_space(geometry: np.ndarray, masses: np.ndarray, trans: bool, rot: bool) -> np.ndarray:
    """Orthonormal mass-weighted translations and/or rotations, (nrt, 3*natom)."""
    natom = len(masses)
    sqrtm = np.sqrt(masses)
    vectors = []
    if trans:
        for axis in range(3):
            t = np.zeros((natom, 3))
            t[:, axis] = sqrtm
            vectors.append(t.ravel())
    if rot:
        com = masses @ geometry / masses.sum()
        for axis in np.eye(3):
            vectors.append((sqrtm[:, None] * np.cross(geometry - com, axis)).ravel())
    if not vectors:
        return np.zeros((0, 3 * natom))
    # drop linear dependencies (linear molecules, atoms); tolerance ~ max geometric deviation
    u, sv, _ = np.linalg.svd(np.array(vectors).T, full_matrices=False)
    return u[:, sv > 1.0e-2].T


def _in_space(vector: np.ndarray, space: np.ndarray, tol: float = 1.0e-4) -> bool:
    """Whether a unit vector lies in the span of orthonormal rows ``space``."""
    if space.shape[0] == 0:
        return False
    return np.linalg.norm(vector - space.T @ (space @ vector)) < tol


def harmonic_analysis(
    hessian: np.ndarray,
    geometry: np.ndarray,
    masses: Sequence[float],
    *,
    dipole_derivatives: Optional[np.ndarray] = None,
    project_trans: bool = True,
    project_rot: bool = True,
) -> Dict[str, Any]:
    """Harmonic vibrational analysis of a Cartesian Hessian.

    Parameters
    ----------
    hessian
        (3*natom, 3*natom) non-mass-weighted Hessian [Eh/a0^2].
    geometry
        (natom, 3) Cartesian coordinates [a0], in the frame of ``hessian``.
    masses
        (natom,) atomic masses [u].
    dipole_derivatives
        (3*natom, 3) derivatives of the dipole moment [e a0/a0], row per
        Cartesian coordinate, column per dipole component; gives IR intensities.
    project_trans, project_rot
        Project translations / rotations out of the mass-weighted Hessian
        (see ``projection_flags``).

    Returns
    -------
    dict
        ``Vibrations`` fields (see ``build_vibrations``) for all 3*natom modes.
    """
    c = qcel.constants
    masses = np.asarray(masses, dtype=float)
    geometry = np.asarray(geometry, dtype=float).reshape(-1, 3)
    natom = len(masses)
    hessian = np.asarray(hessian, dtype=float).reshape(3 * natom, 3 * natom)

    sqrtm = np.repeat(np.sqrt(masses), 3)
    mwhess = hessian / np.outer(sqrtm, sqrtm)
    rigid = _rigid_motion_space(geometry, masses, project_trans, project_rot)
    projector = np.identity(3 * natom) - rigid.T @ rigid
    force_constant_au, q = np.linalg.eigh(projector @ mwhess @ projector)

    # sign convention: the largest-magnitude element of each mode (first, within 1e-2) is positive
    for i in range(q.shape[1]):
        col = q[:, i]
        imax = np.argmax(np.abs(col) > np.abs(col).max() - 1.0e-2)
        q[:, i] = col * np.sign(col[imax])

    # frequency [cm^-1]: sqrt(Eh / (a0^2 u)) / (2 pi c)
    uconv_cm_1 = np.sqrt(c.na * c.hartree2J * 1.0e19) / (2 * np.pi * c.c * c.bohr2angstroms)
    frequency = np.lib.scimath.sqrt(force_constant_au) * uconv_cm_1  # complex; imaginary modes on the imaginary axis
    omega_cm = np.where(frequency.imag > frequency.real, -frequency.imag, frequency.real)

    _, inverse, counts = np.unique(np.around(frequency, 1), return_inverse=True, return_counts=True)

    trv = []
    for i in range(3 * natom):
        if _in_space(q[:, i], rigid):
            trv.append("tr")
        elif abs(frequency[i]) < 1.0e-3:
            trv.append("-")
        else:
            trv.append("v")

    w = q / sqrtm[:, None]
    mu = 1.0 / np.sum(w**2, axis=0)
    x = np.sqrt(mu) * w

    uconv_mdyne_a = (0.1 * (2 * np.pi * c.c) ** 2) / c.na
    k = mu * (frequency * frequency).real * uconv_mdyne_a

    # turning points and RMS deviation for v=0 (zero for imaginary/zero modes)
    uconv_s = np.sqrt((c.c * (2 * np.pi * c.bohr2angstroms) ** 2) / (c.h * c.na * 1.0e21))
    with np.errstate(divide="ignore"):
        qtp0 = 1.0 / (np.sqrt(frequency.real) * uconv_s)
        xtp0 = 1.0 / (np.sqrt(frequency.real * mu) * uconv_s)
    qtp0[np.isinf(qtp0)] = 0.0
    xtp0[np.isinf(xtp0)] = 0.0
    theta_vib = frequency.real * 100 * c.h * c.c / c.kb

    infrared_intensity = None
    if dipole_derivatives is not None:
        uconv_kmmol = (
            c.get("Avogadro constant")
            * np.pi
            * 1.0e-3
            * c.get("electron mass in u")
            * c.get("fine-structure constant") ** 2
            * c.get("atomic unit of length")
            / 3
        )
        dmu_dq = np.asarray(dipole_derivatives, dtype=float).reshape(3 * natom, 3).T @ w
        infrared_intensity = np.sum(dmu_dq**2, axis=0) * uconv_kmmol

    return build_vibrations(
        omega_cm,
        x,
        q=q,
        w=w,
        degeneracy=counts[inverse],
        trv=trv,
        mu=mu,
        k=k,
        dq0=qtp0 / np.sqrt(2.0),
        qtp0=qtp0,
        xtp0=xtp0,
        theta_vib=theta_vib,
        infrared_intensity=infrared_intensity,
    )
