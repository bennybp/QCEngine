"""Parse Gaussian log output: energies from the log text and archive entry, geometry and gradients via cclib."""

import io
import re
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

import numpy as np
import qcelemental as qcel
from qcelemental.models import Molecule

from ..util import PreservingDict
from ..util.vibrations import build_vibrations

# rq-0743e952
_NORMAL_TERMINATION_RE = re.compile(r"Normal termination of Gaussian")
_VERSION_RE = re.compile(r"Gaussian\s+(\d+),\s+Revision\s+([\w.]+),")


# Archive entry ("1\\1\\GINC-...\\@") printed at the end of a normal termination.
# Its first section names the job type and the method Gaussian actually ran
# (e.g. "RCCSD(T)-FC"); the "Version=" section lists energies of every level
# computed at the reference geometry, rounded to ~7 decimals.
_ARCHIVE_START_RE = re.compile(r"1\\1\\GINC")
_FORCES_MARKER = "Forces (Hartrees/Bohr)"
_HF_SCF_LABELS = {"RHF", "UHF", "ROHF"}
_SCF_DONE_RE = re.compile(r"SCF Done:\s+E\((\S+)\)\s+=\s+(\S+)")
_DISPERSION_RE = re.compile(r"Dispersion energy=\s+(\S+)\s+Hartrees")
_DOUBLE_HYBRID_RE = re.compile(r"E2\((\S+)\)\s+=\s+(\S+)\s+E\(\1\)\s+=\s+(\S+)")
_ALPHA_BETA_RE = re.compile(r"(\d+)\s+alpha electrons\s+(\d+)\s+beta electrons")
_PUNCH_NUMBER_RE = re.compile(r"^-?\d*\.\d+D[+-]\d+$")
_AMPLITUDES_RE = r"Wavefunction amplitudes converged\. +E\(Corr\)= +(\S+)"

# Post-SCF levels on an HF reference: archive label -> (qcvar stem, full-precision log pattern).
# A job prints at most one of CISD/CCD/CCSD/QCISD, so they share the amplitudes line.
_CORRELATED_LEVELS = {
    "MP2": ("MP2", r"EUMP2 *= *(\S+)"),
    "MP3": ("MP3", r"EUMP3 *= *(\S+)"),
    "MP4DQ": ("MP4(DQ)", r"UMP4\(DQ\) *= *(\S+)"),
    "MP4SDQ": ("MP4(SDQ)", r"UMP4\(SDQ\) *= *(\S+)"),
    "MP4SDTQ": ("MP4", r"UMP4\(SDTQ\) *= *(\S+)"),
    "CISD": ("CISD", _AMPLITUDES_RE),
    "CCD": ("CCD", _AMPLITUDES_RE),
    "CCSD": ("CCSD", _AMPLITUDES_RE),
    "QCISD": ("QCISD", _AMPLITUDES_RE),
    "CCSD(T)": ("CCSD(T)", r"(?<!Q)CCSD\(T\) *= *(\S+)"),
    "QCISD(T)": ("QCISD(T)", r"QCISD\(T\) *= *(\S+)"),
}

# Archive values are rounded to 6-7 decimals.
_ARCHIVE_ATOL = 2.0e-6


class UnsupportedMethodError(Exception):
    """Gaussian ran a method whose energies the harvester cannot extract."""


def _parse_archive(log_content: str) -> Tuple[str, Dict[str, str]]:
    """Return (method, {key: value}) from the last archive entry in the log.

    Values are the raw strings of the "Version=" section, e.g. "HF" -> "-76.0105"
    or "Dipole" -> "0.,0.,-0.6358857".
    """
    starts = [m.start() for m in _ARCHIVE_START_RE.finditer(log_content)]
    if not starts:
        raise ValueError("No archive entry found in the Gaussian log.")
    text = log_content[starts[-1] :]
    end = text.find("\\\\@")
    text = re.sub(r"\n ", "", text[: end if end != -1 else len(text)])
    sections = text.split("\\\\")
    fields = sections[0].split("\\")
    if len(fields) < 5:
        raise ValueError("Malformed archive entry in the Gaussian log.")

    items = {}
    for section in sections:
        if section.startswith("Version="):
            for item in section.split("\\"):
                key, _, value = item.partition("=")
                items[key] = value
    return fields[4], items


def _parse_punch(punch_content: str, natoms: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Parse the derivatives that Punch=Derivatives writes to fort.7 after the archive copy.

    Returns the gradient (natoms, 3) and, for frequency jobs, the full Hessian
    (3*natoms, 3*natoms) from its row-wise lower triangle, in the input
    orientation. Jobs without derivatives leave only the archive copy: (None, None).
    """
    # The archive copy may wrap anywhere (even inside its closing "\\@"), so take
    # the run of Fortran-format numbers at the end of the file.
    tokens = punch_content.split()
    start = len(tokens)
    while start > 0 and _PUNCH_NUMBER_RE.match(tokens[start - 1]):
        start -= 1
    values = np.array([_fortran_float(t) for t in tokens[start:]])
    ndim = 3 * natoms
    ntri = ndim * (ndim + 1) // 2
    if len(values) == 0:
        return None, None
    if len(values) == ndim:
        return values.reshape(natoms, 3), None
    if len(values) != ndim + ntri:
        raise ValueError(f"Unexpected number of values ({len(values)}) in the Gaussian punch file.")
    hess = np.zeros((ndim, ndim))
    hess[np.tril_indices(ndim)] = values[ndim:]
    hess = hess + np.tril(hess, -1).T
    return values[:ndim].reshape(natoms, 3), hess


def _method_family(archive_method: str) -> str:
    """Reduce an archive method label (e.g. "UCCSD(T)-FC") to its level ("CCSD(T)").

    Labels that are not HF or a known post-SCF level (DFT functionals,
    semiempirical methods) are returned unchanged.
    """
    name = re.sub(r"-(FC\d*|Full)$", "", archive_method)
    for prefix in ("RO", "R", "U"):
        if name.startswith(prefix) and (name[len(prefix) :] == "HF" or name[len(prefix) :] in _CORRELATED_LEVELS):
            return name[len(prefix) :]
    return name


def _decimal(text: str) -> Decimal:
    return Decimal(text.replace("D", "E").replace("d", "e"))


def _check_archive(label: str, value: Decimal, archive: Dict[str, Decimal]) -> None:
    if abs(float(value - archive[label])) > _ARCHIVE_ATOL:
        raise ValueError(
            f"Gaussian {label} energy {value} from the log disagrees with the archive entry ({archive[label]})."
        )


def harvest_energies(
    log_content: str,
    functional_label: Optional[str] = None,
    dispersion_level: Optional[str] = None,
) -> PreservingDict:
    """Extract energies at the reference geometry from a Gaussian log.

    What was computed is taken from the archive entry, so the result follows
    what Gaussian actually ran rather than the requested method string.
    Full-precision values come from the log body, restricted to the part
    before the first gradient printout (later SCF/post-SCF blocks belong to
    the displaced geometries of a numerical Hessian), and are cross-checked
    against the archive values.

    Raises
    ------
    UnsupportedMethodError
        If Gaussian ran a post-SCF method the harvester does not know.
    ValueError
        If the log is incomplete or inconsistent.
    """
    method, items = _parse_archive(log_content)
    archive = {k: Decimal(v) for k, v in items.items() if k == "HF" or k in _CORRELATED_LEVELS}
    if "HF" not in archive:
        raise ValueError("No HF= energy found in the Gaussian archive entry.")
    family = _method_family(method)

    ref_end = _ARCHIVE_START_RE.search(log_content).start()
    forces_at = log_content.find(_FORCES_MARKER)
    if forces_at != -1:
        ref_end = min(ref_end, forces_at)
    ref_text = log_content[:ref_end]

    # Last SCF before the reference cut: Stable=Opt reruns the SCF at the same geometry.
    scf_matches = _SCF_DONE_RE.findall(ref_text)
    if not scf_matches:
        raise ValueError("No 'SCF Done' line found in the Gaussian log.")
    scf_label, scf_text = scf_matches[-1]
    scf = _decimal(scf_text)
    _check_archive("HF", scf, archive)

    # Gaussian includes empirical dispersion in the SCF Done energy.
    disp_matches = _DISPERSION_RE.findall(ref_text)
    disp = _decimal(disp_matches[-1]) if disp_matches else None

    qcvars = PreservingDict()
    qcvars["SCF TOTAL ENERGY"] = scf
    qcvars["CURRENT REFERENCE ENERGY"] = scf
    if disp is not None:
        qcvars["DISPERSION CORRECTION ENERGY"] = disp

    if scf_label in _HF_SCF_LABELS:
        qcvars["HF TOTAL ENERGY"] = scf if disp is None else scf - disp
        current = scf
        for label, (stem, pattern) in _CORRELATED_LEVELS.items():
            if label not in archive:
                continue
            matches = re.findall(pattern, ref_text)
            if not matches:
                if label == family:
                    raise ValueError(f"Could not find the full-precision {label} energy in the Gaussian log.")
                continue
            value = _decimal(matches[0])
            _check_archive(label, value, archive)
            qcvars[f"{stem} TOTAL ENERGY"] = value
            qcvars[f"{stem} CORRELATION ENERGY"] = value - scf
            if label == family:
                current = value
        if family != "HF" and family not in _CORRELATED_LEVELS:
            raise UnsupportedMethodError(
                f"Gaussian ran method '{method}', whose energy the Gaussian harness cannot extract. "
                f"Supported post-HF levels: {sorted(_CORRELATED_LEVELS)}."
            )
    else:
        # DFT or another non-HF SCF (e.g. semiempirical)
        if any(label in archive for label in _CORRELATED_LEVELS if label != "MP2"):
            raise UnsupportedMethodError(
                f"Gaussian ran method '{method}' with post-SCF levels the Gaussian harness cannot extract."
            )
        current = scf
        if "MP2" in archive:
            # double hybrid: the archive MP2 entry is the double-hybrid total
            dh = _DOUBLE_HYBRID_RE.findall(ref_text)
            if not dh:
                raise ValueError(f"Could not find the double-hybrid energy for '{method}' in the Gaussian log.")
            _, e2_text, total_text = dh[0]
            current = _decimal(total_text)
            _check_archive("MP2", current, archive)
            qcvars["DOUBLE-HYBRID CORRECTION ENERGY"] = _decimal(e2_text)
        qcvars["DFT TOTAL ENERGY"] = current
        qcvars["DFT FUNCTIONAL TOTAL ENERGY"] = scf if disp is None else scf - disp

    qcvars["CURRENT ENERGY"] = current

    if disp is not None and functional_label is not None and dispersion_level is not None:
        tag = f"{functional_label}-{dispersion_level}"
        qcvars[f"{tag} DISPERSION CORRECTION ENERGY"] = disp
        qcvars[f"{tag} TOTAL ENERGY"] = current
        qcvars[f"{functional_label} FUNCTIONAL TOTAL ENERGY"] = scf - disp

    return qcvars


def _fortran_float(s: str) -> float:
    """Convert a Fortran-style D-exponent string (e.g. '-0.76D+02') to float."""
    return float(s.replace("D", "E").replace("d", "e"))


def _parse_force_constants(log_content: str, ndim: int) -> Optional[np.ndarray]:
    """Parse the lower-triangle 'Force constants in Cartesian coordinates' block
    from a Gaussian Freq log into a full symmetric (ndim, ndim) matrix.

    Returns None if the block is not found.
    """
    marker = "Force constants in Cartesian coordinates:"
    idx = log_content.find(marker)
    if idx == -1:
        return None

    # Extract the block: starts after marker, ends at "Leave Link"
    block_start = idx + len(marker)
    block_end = log_content.find("Leave Link", block_start)
    if block_end == -1:
        block_end = len(log_content)
    block = log_content[block_start:block_end]

    hess = np.zeros((ndim, ndim))
    # Parse lower-triangle entries: row lines have format "  <row>  <val1> <val2> ..."
    # Column header lines have format "       <col1>  <col2>  ..."
    col_offset = 0
    for line in block.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        # Column header line: all tokens are integers
        try:
            cols = [int(t) for t in tokens]
            col_offset = cols[0] - 1  # 1-based → 0-based
            continue
        except ValueError:
            pass
        # Data line: first token is row index, rest are Fortran D-exponents
        try:
            row = int(tokens[0]) - 1  # 1-based → 0-based
            vals = [_fortran_float(t) for t in tokens[1:]]
        except (ValueError, IndexError):
            continue
        for j, val in enumerate(vals):
            c = col_offset + j
            hess[row, c] = val
            hess[c, row] = val  # symmetric

    return hess


# rq-d9434480
def harvest(
    in_mol: Molecule,
    method: str,
    log_content: str,
    punch_content: Optional[str] = None,
) -> Tuple[PreservingDict, Optional[np.ndarray], Optional[np.ndarray], Molecule, Optional[Dict[str, Any]]]:
    """Parse a Gaussian log file and return QC data for building an AtomicResult.

    Parameters
    ----------
    in_mol:
        The input molecule used for geometry alignment and NRE cross-check.
    method:
        Lowercase QCSchema method string (e.g. "hf", "mp2", "b3lyp").
    log_content:
        Full text of gaussian.log.
    punch_content:
        Text of fort.7 from Punch=Derivatives, if present; preferred source of the Hessian.

    Returns
    -------
    (qcvars, grad, hess, out_mol, vib)
        qcvars:  PreservingDict of QC variables (energies, charges, etc.).
        grad:    Gradient array shape (3*natoms,) in Hartree/Bohr, or None.
        hess:    Hessian matrix shape (3*natoms, 3*natoms) in Hartree/Bohr², or None.
        out_mol: Output molecule aligned to in_mol.
        vib:     Gaussian's harmonic analysis as a Vibrations dict (programs/util/vibrations.py), or None.

    Raises
    ------
    ValueError
        If no coordinate information can be extracted from the log.
    ValueError
        If the parsed geometry's NRE deviates from in_mol's by more than 1e-3 Hartree.
    """
    # rq-bb0a5d96 — parse with cclib; exceptions propagate to caller (rq-bb775969)
    import cclib

    ccdata = cclib.io.ccopen(io.StringIO(log_content)).parse()

    qcvars = PreservingDict()

    # rq-1d5da737 — a recognised dispersion suffix on the method gives the
    # formal label used to tag the functional-specific dispersion qcvars.
    from .germinate import recognize_dispersion

    functional, _emp_disp_value, dispersion_level, _ = recognize_dispersion(method)
    functional_label = functional.upper() if dispersion_level is not None else None

    # --- Geometry and output molecule ---
    # rq-8d6fc024 rq-a76c3d7b rq-b7c7b8cb
    if not hasattr(ccdata, "atomcoords") or len(ccdata.atomcoords) == 0:
        raise ValueError("No coordinate information could be extracted from the Gaussian log.")

    # atomcoords in Ångström → convert to Bohr for QCElemental
    angstrom_to_bohr = qcel.constants.conversion_factor("angstrom", "bohr")
    coords_bohr = ccdata.atomcoords[-1] * angstrom_to_bohr  # shape (natoms, 3)

    # rq-f1d712bb rq-66067019 rq-7411efa0 — cclib cannot distinguish ghost atoms
    # from real ones in Gaussian output (Gaussian marks ghosts via the "Atomic Type"
    # column = 1000, which cclib does not expose). Trust in_mol.real as the sole
    # source of ghost designation; sanity-check only that the parsed atom count
    # matches in_mol when ghosts are present.
    in_real = list(in_mol.real)
    has_ghosts = not all(in_real)
    if has_ghosts:
        if ccdata.natom != len(in_mol.symbols):
            raise ValueError(
                f"Gaussian log atom count ({ccdata.natom}) does not match the "
                f"input molecule's atom count ({len(in_mol.symbols)})."
            )
        symbols = list(in_mol.symbols)
        real_flags = in_real
    else:
        symbols = [qcel.periodictable.to_symbol(z) for z in ccdata.atomnos]
        real_flags = None

    # rq-f1d712bb — preserve ghost designation on calc_mol so its NRE excludes ghosts
    calc_mol_kwargs = dict(
        symbols=symbols,
        geometry=coords_bohr.flatten(),
        fix_com=True,
        fix_orientation=True,
    )
    if real_flags is not None:
        calc_mol_kwargs["real"] = real_flags
    calc_mol = Molecule(**calc_mol_kwargs)

    natoms = ccdata.natom
    qcvars["N ATOMS"] = str(natoms)

    if hasattr(ccdata, "nbasis") and ccdata.nbasis is not None:
        qcvars["N BASIS FUNCTIONS"] = str(ccdata.nbasis)
    if hasattr(ccdata, "nmo") and ccdata.nmo is not None:
        qcvars["N MOLECULAR ORBITALS"] = str(ccdata.nmo)
    # cclib's homos has a single entry for ROHF, so read the counts from the log
    alpha_beta = _ALPHA_BETA_RE.search(log_content)
    if alpha_beta:
        qcvars["N ALPHA ELECTRONS"] = alpha_beta.group(1)
        qcvars["N BETA ELECTRONS"] = alpha_beta.group(2)

    # rq-0753cd03
    qcvars["NUCLEAR REPULSION ENERGY"] = str(round(calc_mol.nuclear_repulsion_energy(), 10))

    # rq-90604835 rq-a69d0898 rq-e5a98119 — cross-check NRE against input molecule.
    # Molecule.nuclear_repulsion_energy() excludes ghost atoms (zero Z), so when
    # calc_mol.real mirrors in_mol.real both NREs sum over real atoms only.
    if abs(calc_mol.nuclear_repulsion_energy() - in_mol.nuclear_repulsion_energy()) > 1.0e-3:
        raise ValueError(
            f"Gaussian output geometry (NRE: {calc_mol.nuclear_repulsion_energy():.6f}) "
            f"is inconsistent with the input molecule (NRE: {in_mol.nuclear_repulsion_energy():.6f})."
        )

    # Frame considerations (same pattern as GAMESS harvester)
    # rq-8d6fc024 rq-f8ce079b
    align_kwargs = dict(atoms_map=False, mols_align=True, run_mirror=True, verbose=0)
    if has_ghosts:
        align_kwargs["generic_ghosts"] = True
    if in_mol.fix_com and in_mol.fix_orientation:
        # Impose input frame if important as signalled by fix_*=True
        return_mol = in_mol
        _, data = calc_mol.align(in_mol, **align_kwargs)
        mill = data["mill"]
    else:
        return_mol, _ = in_mol.align(calc_mol, **align_kwargs)
        mill = qcel.molutil.compute_scramble(
            len(in_mol.symbols), do_resort=False, do_shift=False, do_rotate=False, do_mirror=False
        )  # identity AlignmentMill

    # --- Energies ---
    # rq-37f40e92 rq-929a262a rq-c5165f1f rq-1d5da737
    qcvars.update(harvest_energies(log_content, functional_label, dispersion_level))

    # --- Gradient (forces → negate for gradient) ---
    # rq-36736115 rq-3b0adf96
    grad: Optional[np.ndarray] = None
    if hasattr(ccdata, "grads") and ccdata.grads is not None and len(ccdata.grads) > 0:
        # ccdata.grads shape: (nsteps, natoms, 3) in Hartree/Bohr (forces = -dE/dR)
        # Gradient = -forces; align_gradient expects (natom, 3)
        calc_grad = -ccdata.grads[-1]  # shape (natoms, 3)
        grad = mill.align_gradient(calc_grad).flatten()

    # --- Hessian ---
    # rq-f2e32162 rq-344f3432
    hess: Optional[np.ndarray] = None
    punch_hess = _parse_punch(punch_content, natoms)[1] if punch_content else None
    if hasattr(ccdata, "hessian") and ccdata.hessian is not None:
        # cclib hessian would be in standard orientation → transform with mill
        hess = mill.align_hessian(np.array(ccdata.hessian))
    else:
        # cclib's Gaussian parser does not populate ccdata.hessian (the full
        # 3N×3N Cartesian Hessian matrix). It does populate ccdata.vibfconsts
        # (per-normal-mode reduced force constants, shape (nmodes,)) from the
        # "Frc consts --" lines, but that's not the matrix we need. Fall back
        # to direct log parsing of "Force constants in Cartesian coordinates"
        # (printed in the INPUT orientation).
        parsed_hess = punch_hess if punch_hess is not None else _parse_force_constants(log_content, 3 * natoms)
        if parsed_hess is not None:
            if in_mol.fix_com and in_mol.fix_orientation:
                # return_mol = in_mol (input frame), hessian already in input frame
                hess = parsed_hess
            else:
                # return_mol is in calc/standard frame; need to transform
                # hessian from input→standard. The mill from calc_mol.align(in_mol)
                # transforms calc→input, so we need input→calc.
                # rq-f8ce079b
                _, data = in_mol.align(calc_mol, **align_kwargs)
                input_to_calc_mill = data["mill"]
                hess = input_to_calc_mill.align_hessian(parsed_hess)

    # --- Harmonic analysis (Freq jobs) ---
    # Gaussian's own analysis; modes are printed in the same (standard) frame as
    # cclib's atomcoords, so they rotate into the returned frame like the gradient.
    vib: Optional[Dict[str, Any]] = None
    # Gaussian prints vibrations only (no translations/rotations).
    if getattr(ccdata, "vibfreqs", None) is not None and getattr(ccdata, "vibdisps", None) is not None:
        modes = [mill.align_gradient(np.asarray(d)).ravel() for d in ccdata.vibdisps]
        x = np.array([m / np.linalg.norm(m) for m in modes]).T
        # Gaussian's own masses, used to derive the mass-weighted modes
        masses = getattr(ccdata, "atommasses", None)
        vib = build_vibrations(
            ccdata.vibfreqs,
            x,
            masses=masses if masses is not None else in_mol.masses,
            trv=["v"] * len(ccdata.vibfreqs),
            gamma=getattr(ccdata, "vibsyms", None),
            mu=getattr(ccdata, "vibrmasses", None),
            k=getattr(ccdata, "vibfconsts", None),
            infrared_intensity=getattr(ccdata, "vibirs", None),
        )

    # --- Properties: dipole and Mulliken charges ---
    # rq-c7fd08b5 rq-943856cc rq-53f8ed12 rq-4c4a1542 rq-ffae3fa6
    # The archive dipole is in a.u. and the input orientation. For post-SCF
    # gradients it may come from a correlated density, so only SCF methods report it.
    _, archive_items = _parse_archive(log_content)
    if "Dipole" in archive_items and "SCF TOTAL ENERGY" in qcvars and "MP2 TOTAL ENERGY" not in qcvars:
        dipole = np.array([float(x) for x in archive_items["Dipole"].split(",")])
        if not (in_mol.fix_com and in_mol.fix_orientation):
            _, data = in_mol.align(calc_mol, **align_kwargs)
            dipole = data["mill"].align_vector(dipole)
        if "HF TOTAL ENERGY" in qcvars:
            qcvars["HF DIPOLE"] = dipole
        qcvars["SCF DIPOLE"] = dipole
        qcvars["CURRENT DIPOLE"] = dipole

    if hasattr(ccdata, "atomcharges") and ccdata.atomcharges:
        mulliken = ccdata.atomcharges.get("mulliken")
        if mulliken is not None:
            qcvars["MULLIKEN CHARGES"] = list(mulliken)

    return qcvars, grad, hess, return_mol, vib


def is_normal_termination(log_content: str) -> bool:
    """Return True if the log contains a normal termination marker.

    rq-90141f40 rq-12c41d5e rq-c20746b1
    """
    return bool(_NORMAL_TERMINATION_RE.search(log_content))

