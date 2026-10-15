"""Parse each step of a Gaussian Opt (NoSymm) log."""

import re
from typing import List, NamedTuple

import numpy as np
import qcelemental as qcel
from qcelemental.models.v2 import Molecule

from qcengine.programs.gaussian.harvester import (
    _ALPHA_BETA_RE,
    _FORCES_RE,
    FORCES_FIELD,
    ORIENTATION_FIELD,
    _atom_table,
    _parse_archive,
    archive_geometry,
    harvest_energies,
)
from qcengine.programs.gaussian.germinate import recognize_dispersion
from qcengine.programs.util import PreservingDict

_GEOMETRY_RE = re.compile(r"Input orientation:\s*\n\s*-+\n.*?\n.*?\n\s*-+\n(.*?)\n\s*-+", re.S)

# Final archive geometry vs. the last printed (6-decimal) geometry [Angstrom]
_FINAL_GEOMETRY_ATOL = 2.0e-6


class OptimizationStep(NamedTuple):
    molecule: Molecule
    qcvars: PreservingDict
    gradient: np.ndarray  # (natom, 3) [Eh/a0]


def harvest_optimization(in_mol: Molecule, method: str, log_content: str) -> List[OptimizationStep]:
    """Molecule, energies and gradient of every step of a Gaussian Opt NoSymm run.

    With NoSymm, Gaussian prints geometries and forces in the input orientation.
    Each step is the text from one "Input orientation" block to the next and
    holds that geometry's SCF (and post-SCF) energies and forces; the geometry
    reprinted after convergence repeats the last step. Printed geometries have
    6 decimals; the final step uses the archive geometry (10 decimals).
    Energies of the final step are cross-checked against the archive.
    """
    natom = len(in_mol.symbols)
    functional, _, dispersion_level, _ = recognize_dispersion(method)
    functional_label = functional.upper() if dispersion_level is not None else None

    starts = [m.start() for m in re.finditer(r"Input orientation:", log_content)] + [len(log_content)]
    segments = [log_content[a:b] for a, b in zip(starts[:-1], starts[1:])]
    segments = [seg for seg in segments if "SCF Done" in seg and "Forces (Hartrees/Bohr)" in seg]
    if not segments:
        raise ValueError("No completed optimization steps found in the Gaussian log.")

    ang2bohr = qcel.constants.conversion_factor("angstrom", "bohr")
    # New geometries: keep the input's identity and metadata, but not the fields
    # that describe the input geometry itself
    base = in_mol.model_dump(exclude={"id", "provenance"})
    if base.get("identifiers"):
        base["identifiers"] = {
            k: v for k, v in base["identifiers"].items() if k not in ("molecule_hash", "pubchem_conformerid")
        }

    _parse_archive(log_content)  # validates the archive entry
    final_geometry = archive_geometry(log_content)

    alpha_beta = _ALPHA_BETA_RE.search(log_content)

    steps = []
    for i, seg in enumerate(segments):
        last = i == len(segments) - 1
        geometry = _atom_table(_GEOMETRY_RE.search(seg).group(1), natom, ORIENTATION_FIELD)
        if last:
            if np.abs(final_geometry - geometry).max() > _FINAL_GEOMETRY_ATOL:
                raise ValueError("Gaussian archive geometry does not match the last optimization step.")
            geometry = final_geometry
        forces = _atom_table(_FORCES_RE.search(seg).group(1), natom, FORCES_FIELD)

        ref_text = seg[: seg.find("Forces (Hartrees/Bohr)")]
        qcvars = harvest_energies(
            log_content,
            functional_label,
            dispersion_level,
            ref_text=ref_text,
            check_archive=last,
        )

        molecule = Molecule(**{**base, "geometry": (geometry * ang2bohr).ravel()})
        qcvars["NUCLEAR REPULSION ENERGY"] = str(round(molecule.nuclear_repulsion_energy(), 10))
        qcvars["N ATOMS"] = natom
        if alpha_beta:
            qcvars["N ALPHA ELECTRONS"] = alpha_beta.group(1)
            qcvars["N BETA ELECTRONS"] = alpha_beta.group(2)
        steps.append(OptimizationStep(molecule, qcvars, -forces))
    return steps
