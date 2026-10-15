"""Harmonic vibrational analysis in the layout of the proposed QCSchema ``Vibrations`` model.

Harnesses that compute a Hessian report the program's own harmonic analysis as
the ``VIBRATIONS`` QCVariable (QCFractal stores it as the ``vibrations`` record
property). The fields mirror the ``Vibrations`` model proposed for QCSchema
result properties and psi4's ``qcdb.vib.harmonic_analysis``:

- every per-mode array has one entry per mode; ``x``, ``q`` and ``w`` are
  (3*natom, nmode) with one column per mode, as in psi4;
- ``omega`` is in atomic units (E_h/hbar) and ``omega_in_wavenumbers`` in cm^-1,
  negative for imaginary modes;
- ``trv`` is "v" (vibration), "t", "r", "tr" (translation/rotation) or "-"
  (unclassified); ``gamma`` is the irrep or "-".

Programs that only report vibrations (e.g. Gaussian) give nmode = 3*natom - 6
(or - 5); psi4 reports all 3*natom modes, classified by ``trv``.
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np
import qcelemental as qcel

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
