"""Parse Gaussian log output: energies from the log text and archive entry, geometry and gradients via cclib."""

import io
import re
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import numpy as np
import qcelemental as qcel
from qcelemental.models import Molecule

from ..util import PreservingDict

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


# Archive geometry vs. input geometry [Angstrom]; the archive has ~10 decimals,
# and Gaussian's own symmetrization moves atoms by < 1e-8 Angstrom
_GEOMETRY_ATOL = 1.0e-5

_STANDARD_ORIENTATION_RE = re.compile(r"Standard orientation:\s*\n\s*-+\n.*?\n.*?\n\s*-+\n(.*?)\n\s*-+", re.S)
_ORIENTATION_TABLE_RE = re.compile(r"((?:Input|Standard) orientation:\s*\n\s*-+\n.*?\n.*?\n\s*-+\n)(.*?)(\n\s*-+)", re.S)

# Gaussian's fixed-width fields for the last three columns of per-atom tables:
# orientation tables (F12.6) and forces (F15.9). Large values fill the field,
# so neighboring fields can run together without a space.
ORIENTATION_FIELD = 12
FORCES_FIELD = 15

_FORCES_RE = re.compile(r"Forces \(Hartrees/Bohr\)\s*\n.*?\n\s*-+\n(.*?)\n\s*-+", re.S)


def _fixed_fields(line: str, width: int) -> List[str]:
    """The last three fixed-width fields of a table row."""
    line = line.rstrip()
    return [line[len(line) - (3 - i) * width : len(line) - (2 - i) * width] for i in range(3)]


def _atom_table(block: str, natom: int, width: int) -> np.ndarray:
    """Last three columns of a Gaussian per-atom table (geometry or forces), (natom, 3).

    Reads them as fixed-width fields of ``width`` characters, so values that fill
    their field and run together (e.g. far-translated molecules) still parse.
    """
    rows = [_fixed_fields(line, width) for line in block.strip("\n").splitlines()]
    if len(rows) != natom:
        raise ValueError(f"Expected {natom} atoms in a Gaussian table, found {len(rows)}.")
    return np.array(rows, dtype=float)


def _center_of_nuclear_charge(mol: Molecule) -> np.ndarray:
    """Gaussian's dipole origin [a0]: center of the nuclear charges of the real atoms (full Z, also with ECPs)."""
    charges = np.array([qcel.periodictable.to_Z(s) if real else 0 for s, real in zip(mol.symbols, mol.real)], dtype=float)
    return charges @ mol.geometry.reshape(-1, 3) / charges.sum()


def archive_geometry(log_content: str) -> np.ndarray:
    """Cartesian geometry [Angstrom] of the last archive entry, (natom, 3).

    It is in the input orientation: the input geometry for single points, the
    final geometry for optimizations.
    """
    start = log_content.rfind("1\\1\\GINC")
    if start == -1:
        raise ValueError("No archive entry found in the Gaussian log.")
    sections = re.sub(r"\n ", "", log_content[start:]).split("\\\\")
    if len(sections) < 4:
        raise ValueError("Malformed archive entry in the Gaussian log.")
    # "charge,mult\Label,x,y,z\..." (labels may carry ",0" freeze flags, -Bq, (Iso=...))
    atoms = sections[3].split("\\")[1:]
    return np.array([atom.split(",")[-3:] for atom in atoms], dtype=float)


_ROTCONST_LINE_RE = re.compile(r"^ Rotational constants \(GHZ\):.*$", re.M)


def _sanitize_for_cclib(log_content: str) -> str:
    """Make overflowed fields that cclib cannot parse readable.

    Gaussian prints rotational constants that overflow their format (nearly
    linear molecules) as asterisks, and the coordinate columns of orientation
    tables run together for large coordinates; the harness uses neither.
    """
    text = _ROTCONST_LINE_RE.sub(lambda m: re.sub(r"\*{2,}", "inf", m.group(0)), log_content)

    def separate(m: re.Match) -> str:
        rows = []
        for line in m.group(2).split("\n"):
            head = line.rstrip()[: -3 * ORIENTATION_FIELD]
            rows.append(" ".join([head] + [f.strip() for f in _fixed_fields(line, ORIENTATION_FIELD)]))
        return m.group(1) + "\n".join(rows) + m.group(3)

    return _ORIENTATION_TABLE_RE.sub(separate, text)


class UnsupportedMethodError(Exception):
    """Gaussian ran a method whose energies the harvester cannot extract."""


class GeometryChangedError(Exception):
    """Gaussian computed at a geometry that differs from the input molecule."""


def _parse_archive(log_content: str) -> Tuple[str, str, Dict[str, str]]:
    """Return (method, basis, {key: value}) from the last archive entry in the log.

    Semiempirical jobs have basis "ZDO".

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
    if len(fields) < 6:
        raise ValueError("Malformed archive entry in the Gaussian log.")

    items = {}
    for section in sections:
        if section.startswith("Version="):
            for item in section.split("\\"):
                key, _, value = item.partition("=")
                items[key] = value
    return fields[4], fields[5], items


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
    ref_text: Optional[str] = None,
    check_archive: bool = True,
) -> PreservingDict:
    """Extract energies at the reference geometry from a Gaussian log.

    What was computed is taken from the archive entry, so the result follows
    what Gaussian actually ran rather than the requested method string.
    Full-precision values come from the log body, restricted to the part
    before the first gradient printout (later SCF/post-SCF blocks belong to
    the displaced geometries of a numerical Hessian), and are cross-checked
    against the archive values.

    ``ref_text`` replaces that reference section (e.g. one optimization step);
    ``check_archive=False`` skips the comparison with the archive values, which
    belong to the final geometry.

    Raises
    ------
    UnsupportedMethodError
        If Gaussian ran a post-SCF method the harvester does not know.
    ValueError
        If the log is incomplete or inconsistent.
    """
    method, archive_basis, items = _parse_archive(log_content)
    archive = {k: Decimal(v) for k, v in items.items() if k == "HF" or k in _CORRELATED_LEVELS}
    if "HF" not in archive:
        raise ValueError("No HF= energy found in the Gaussian archive entry.")
    family = _method_family(method)

    if ref_text is None:
        ref_end = _ARCHIVE_START_RE.search(log_content).start()
        forces_at = log_content.find(_FORCES_MARKER)
        if forces_at != -1:
            ref_end = min(ref_end, forces_at)
        ref_text = log_content[:ref_end]

    def archive_check(label: str, value: Decimal) -> None:
        if check_archive:
            _check_archive(label, value, archive)

    # Last SCF before the reference cut: Stable=Opt reruns the SCF at the same geometry.
    scf_matches = _SCF_DONE_RE.findall(ref_text)
    if not scf_matches:
        raise ValueError("No 'SCF Done' line found in the Gaussian log.")
    scf_label, scf_text = scf_matches[-1]
    scf = _decimal(scf_text)
    archive_check("HF", scf)

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
            archive_check(label, value)
            qcvars[f"{stem} TOTAL ENERGY"] = value
            qcvars[f"{stem} CORRELATION ENERGY"] = value - scf
            if label == family:
                current = value
        if family != "HF" and family not in _CORRELATED_LEVELS:
            raise UnsupportedMethodError(
                f"Gaussian ran method '{method}', whose energy the Gaussian harness cannot extract. "
                f"Supported post-HF levels: {sorted(_CORRELATED_LEVELS)}."
            )
    elif archive_basis == "ZDO":
        # semiempirical (e.g. PM6): no HF or DFT qcvars
        if any(label in archive for label in _CORRELATED_LEVELS):
            raise UnsupportedMethodError(
                f"Gaussian ran method '{method}' with post-SCF levels the Gaussian harness cannot extract."
            )
        current = scf
    else:
        # DFT
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
            archive_check("MP2", current)
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
) -> Tuple[PreservingDict, Optional[np.ndarray], Optional[np.ndarray], Molecule, Optional[np.ndarray]]:
    """Parse a Gaussian log file and return QC data for building an AtomicResult.

    Parameters
    ----------
    in_mol:
        The input molecule; returned unchanged after checking it against Gaussian's geometry.
    method:
        Lowercase QCSchema method string (e.g. "hf", "mp2", "b3lyp").
    log_content:
        Full text of gaussian.log.
    punch_content:
        Text of fort.7 from Punch=Derivatives, if present; preferred source of the Hessian.

    Returns
    -------
    (qcvars, grad, hess, out_mol, dipder)
        qcvars:  PreservingDict of QC variables (energies, charges, etc.).
        grad:    Gradient array shape (3*natoms,) in Hartree/Bohr, or None.
        hess:    Hessian matrix shape (3*natoms, 3*natoms) in Hartree/Bohr², or None.
        out_mol: The input molecule (all quantities are in its frame).
        dipder:  Dipole derivatives (3*natoms, 3) [e a0/a0] (row per coordinate), or None.

    Raises
    ------
    ValueError
        If the log has no archive entry, or Gaussian's geometry differs from in_mol.
    """
    # rq-bb0a5d96 — parse with cclib; exceptions propagate to caller (rq-bb775969)
    import cclib

    ccdata = cclib.io.ccopen(io.StringIO(_sanitize_for_cclib(log_content))).parse()

    qcvars = PreservingDict()

    # rq-1d5da737 — a recognised dispersion suffix on the method gives the
    # formal label used to tag the functional-specific dispersion qcvars.
    from .germinate import recognize_dispersion

    functional, _emp_disp_value, dispersion_level, _ = recognize_dispersion(method)
    functional_label = functional.upper() if dispersion_level is not None else None

    # --- Geometry and output molecule ---
    # rq-8d6fc024 rq-a76c3d7b rq-b7c7b8cb rq-f1d712bb
    # Everything returned below (forces, punch/log Hessian, archive dipole and
    # dipole derivatives) is printed in Gaussian's input orientation, i.e. the
    # input molecule's frame and atom order, with or without symmetry. So the
    # input molecule is returned unchanged and nothing is rotated. (cclib's
    # coordinates and gradients are in the standard orientation when symmetry is
    # on and are not used.) The archive geometry (~10 decimals) must match it.
    natoms = len(in_mol.symbols)
    bohr_to_angstrom = qcel.constants.conversion_factor("bohr", "angstrom")
    gaussian_geometry = archive_geometry(log_content)
    if gaussian_geometry.shape != (natoms, 3):
        raise ValueError(
            f"Gaussian's atom count ({len(gaussian_geometry)}) does not match the input molecule's ({natoms})."
        )
    deviation = np.abs(gaussian_geometry - in_mol.geometry.reshape(-1, 3) * bohr_to_angstrom).max()
    if deviation > _GEOMETRY_ATOL:
        raise ValueError(
            f"Gaussian's geometry differs from the input molecule by up to {deviation:.2e} Angstrom "
            "(symmetrized or reoriented?)."
        )
    # With symmetry on, Gaussian computes in its standard orientation, possibly with a
    # symmetrized geometry (e.g. Symmetry=Loose) that the archive does not show. Compare
    # interatomic distances of the first (reference) standard orientation with the input.
    standard = _STANDARD_ORIENTATION_RE.search(log_content)
    if standard:
        calc = _atom_table(standard.group(1), natoms, ORIENTATION_FIELD)
        ref = in_mol.geometry.reshape(-1, 3) * bohr_to_angstrom
        dist = lambda g: np.linalg.norm(g[:, None, :] - g[None, :, :], axis=-1)  # noqa: E731
        deviation = np.abs(dist(calc) - dist(ref)).max()
        if deviation > _GEOMETRY_ATOL:
            raise GeometryChangedError(
                f"Gaussian computed at a geometry whose interatomic distances differ from the input molecule's "
                f"by up to {deviation:.2e} Angstrom (symmetrized). Use NoSymm or tighter Symmetry options."
            )
    return_mol = in_mol

    qcvars["N ATOMS"] = str(natoms)
    if getattr(ccdata, "nbasis", None) is not None:
        qcvars["N BASIS FUNCTIONS"] = str(ccdata.nbasis)
    if getattr(ccdata, "nmo", None) is not None:
        qcvars["N MOLECULAR ORBITALS"] = str(ccdata.nmo)
    # cclib's homos has a single entry for ROHF, so read the counts from the log
    alpha_beta = _ALPHA_BETA_RE.search(log_content)
    if alpha_beta:
        qcvars["N ALPHA ELECTRONS"] = alpha_beta.group(1)
        qcvars["N BETA ELECTRONS"] = alpha_beta.group(2)

    # rq-0753cd03 — ghost atoms (real=False) do not contribute
    qcvars["NUCLEAR REPULSION ENERGY"] = str(round(in_mol.nuclear_repulsion_energy(), 10))

    # --- Energies ---
    # rq-37f40e92 rq-929a262a rq-c5165f1f rq-1d5da737
    qcvars.update(harvest_energies(log_content, functional_label, dispersion_level))

    # --- Gradient (forces → negate for gradient) ---
    # rq-36736115 rq-3b0adf96 — the last forces block is the reference geometry,
    # also for numerical Hessians (displaced geometries come first)
    grad: Optional[np.ndarray] = None
    forces_blocks = _FORCES_RE.findall(log_content)
    if forces_blocks:
        grad = -_atom_table(forces_blocks[-1], natoms, FORCES_FIELD).flatten()

    # --- Hessian ---
    # rq-f2e32162 rq-344f3432 rq-f8ce079b — punch file (10 significant figures),
    # else the log's "Force constants in Cartesian coordinates" (6)
    hess: Optional[np.ndarray] = None
    if punch_content:
        hess = _parse_punch(punch_content, natoms)[1]
    if hess is None:
        hess = _parse_force_constants(log_content, 3 * natoms)

    # --- Properties: dipole and Mulliken charges ---
    # rq-c7fd08b5 rq-943856cc rq-53f8ed12 rq-4c4a1542 rq-ffae3fa6
    # The archive dipole is in a.u. For post-SCF gradients it may come from a
    # correlated density, so only SCF methods report it.
    _, _, archive_items = _parse_archive(log_content)
    if "Dipole" in archive_items and "SCF TOTAL ENERGY" in qcvars and "MP2 TOTAL ENERGY" not in qcvars:
        dipole = np.array([float(x) for x in archive_items["Dipole"].split(",")])
        # Gaussian's origin is the center of nuclear charge; QCSchema's is the
        # coordinate origin: mu(0) = mu(P) + Q P (matters for charged molecules)
        charge = in_mol.molecular_charge
        if charge != 0:
            dipole = dipole + charge * _center_of_nuclear_charge(in_mol)
        if "HF TOTAL ENERGY" in qcvars:
            qcvars["HF DIPOLE"] = dipole
        qcvars["SCF DIPOLE"] = dipole
        qcvars["CURRENT DIPOLE"] = dipole

    # Dipole derivatives (frequency jobs): row per Cartesian coordinate, column per dipole component
    dipder: Optional[np.ndarray] = None
    if "DipoleDeriv" in archive_items:
        dipder = np.array([float(v) for v in archive_items["DipoleDeriv"].split(",")]).reshape(3 * natoms, 3)

    if hasattr(ccdata, "atomcharges") and ccdata.atomcharges:
        mulliken = ccdata.atomcharges.get("mulliken")
        if mulliken is not None:
            qcvars["MULLIKEN CHARGES"] = list(mulliken)

    return qcvars, grad, hess, return_mol, dipder


def is_normal_termination(log_content: str) -> bool:
    """Return True if the log contains a normal termination marker.

    rq-90141f40 rq-12c41d5e rq-c20746b1
    """
    return bool(_NORMAL_TERMINATION_RE.search(log_content))

