"""Tests for the Gaussian harness: germinate, keywords, harvester, and runner."""

import importlib.util
import re
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import qcelemental as qcel
from qcelemental.models.v2 import AtomicInput, AtomicSpecification, BasisSet, Molecule

from qcengine.exceptions import ConvergenceError, InputError, ResourceError, UnknownError
from qcengine.programs.gaussian.basis import _cartesian_flags, build_bse_basis, parse_bse_basis
from qcengine.programs.gaussian.germinate import muster_modelchem
from qcengine.programs.gaussian.harvester import (
    UnsupportedMethodError,
    _parse_punch,
    harvest,
    harvest_energies,
    is_normal_termination,
)
from qcengine.programs.gaussian.keywords import build_com_file, build_route_line
from qcengine.programs.gaussian import runner as gaussian_runner
from qcengine.programs.gaussian.runner import GaussianHarness
from qcengine.testing import uusing as using

_cclib_available = importlib.util.find_spec("cclib") is not None
requires_cclib = pytest.mark.skipif(not _cclib_available, reason="cclib not installed")

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def water():
    """Simple water molecule for use in tests."""
    return Molecule.from_data(
        """
        0 1
        O  0.000000000  0.000000000  0.117176320
        H  0.000000000  0.757335953 -0.468705281
        H  0.000000000 -0.757335953 -0.468705281
        units angstrom
        """
    )


@pytest.fixture
def water_energy_input(water):
    """Minimal AtomicInput for a closed-shell HF/STO-3G energy on water."""
    return AtomicInput(molecule=water, specification=AtomicSpecification(program="gaussian", driver="energy", model={"method": "HF", "basis": "STO-3G"}))


@pytest.fixture
def water_energy_input_all_files(water):
    """AtomicInput with native_files protocol set to 'all' for testing file output."""
    return AtomicInput(
        molecule=water,
        specification=AtomicSpecification(
            program="gaussian",
            driver="energy",
            model={"method": "HF", "basis": "STO-3G"},
            protocols={"native_files": "all"},
        ),
    )


@pytest.fixture
def harness():
    return GaussianHarness()


# ---------------------------------------------------------------------------
# Mock ccdata helper
# ---------------------------------------------------------------------------


def _make_ccdata(
    scfenergies=None,
    mpenergies=None,
    ccenergies=None,
    atomcoords=None,
    atomnos=None,
    grads=None,
    hessian=None,
    moments=None,
    atomcharges=None,
):
    """Build a minimal mock ccdata object."""
    # Water geometry in Angstrom
    if atomcoords is None:
        atomcoords = np.array(
            [[[0.0, 0.0, 0.117176], [0.0, 0.757336, -0.468705], [0.0, -0.757336, -0.468705]]]
        )
    if atomnos is None:
        atomnos = np.array([8, 1, 1])
    if scfenergies is None:
        scfenergies = np.array([-2041.3])

    obj = MagicMock()
    obj.scfenergies = scfenergies
    obj.mpenergies = mpenergies
    obj.ccenergies = ccenergies
    obj.atomcoords = atomcoords
    obj.atomnos = atomnos
    obj.grads = grads
    obj.hessian = hessian
    obj.moments = moments
    obj.atomcharges = atomcharges if atomcharges is not None else {}
    # Make hasattr work correctly
    if grads is None:
        del obj.grads
    if hessian is None:
        del obj.hessian
    if moments is None:
        del obj.moments
    return obj


def _make_ccdata_with_attrs(**kwargs):
    """Like _make_ccdata but use spec=None so we can control hasattr."""
    ccdata = type("CCData", (), {})()
    defaults = dict(
        scfenergies=np.array([-2041.3]),
        atomcoords=np.array(
            [[[0.0, 0.0, 0.117176], [0.0, 0.757336, -0.468705], [0.0, -0.757336, -0.468705]]]
        ),
        atomnos=np.array([8, 1, 1]),
        mpenergies=None,
        ccenergies=None,
        grads=None,
        hessian=None,
        moments=None,
        atomcharges={},
    )
    defaults.update(kwargs)
    for k, v in defaults.items():
        setattr(ccdata, k, v)
    # cclib's Gaussian parser sets `natom` as an int. Mirror that here so the
    # harvester's `ccdata.natom` reads work without each test having to set it.
    if not hasattr(ccdata, "natom"):
        setattr(ccdata, "natom", len(ccdata.atomnos))
    return ccdata


def _patch_cclib(ccdata):
    """Return a context manager that patches cclib to return ccdata."""
    mock_ccopen = MagicMock()
    mock_ccopen.return_value.parse.return_value = ccdata
    return patch("cclib.io.ccopen", mock_ccopen)


_WATER_ANGSTROM = [[0.0, 0.0, 0.117176320], [0.0, 0.757335953, -0.468705281], [0.0, -0.757335953, -0.468705281]]


def _g16_log(scf="-76.0105049953", scf_label="RHF", method=None, body="", archive="", pre="", basis="STO-3G", mol=None, forces=None):
    """Minimal Gaussian log: SCF Done line, extra body lines, forces, archive entry, normal termination.

    ``archive`` is appended to the archive's Version= section after ``HF=``,
    e.g. ``"\\MP2=-76.2284380"``; ``method`` is the archive method label
    (defaults to ``scf_label``). The archive geometry is that of ``mol``
    (default: the water fixture); ``forces`` (natom, 3) adds a forces block.
    """
    method = method or scf_label
    if mol is None:
        symbols, geom = ["O", "H", "H"], np.array(_WATER_ANGSTROM)
    else:
        symbols, geom = mol.symbols, mol.geometry.reshape(-1, 3) * qcel.constants.conversion_factor("bohr", "angstrom")
    atoms = "\\".join(f"{sym},{x:.10f},{y:.10f},{z:.10f}" for sym, (x, y, z) in zip(symbols, geom))
    forces_block = ""
    if forces is not None:
        rows = "\n".join(f"  {i + 1:5d}  {1:8d}  {fx:15.9f}{fy:15.9f}{fz:15.9f}" for i, (fx, fy, fz) in enumerate(forces))
        forces_block = (
            " Center     Atomic                   Forces (Hartrees/Bohr)\n"
            " Number     Number              X              Y              Z\n"
            " -------------------------------------------------------------------\n"
            f"{rows}\n"
            " -------------------------------------------------------------------\n"
        )
    return (
        f"{pre}"
        f" SCF Done:  E({scf_label}) =  {scf}     A.U. after   10 cycles\n"
        f"{body}"
        f"{forces_block}"
        f" 1\\1\\GINC-HOST\\SP\\{method}\\{basis}\\H2O1\\USER\\13-Oct-2026\\0\\\\#P\\\\t\\\\0,1\\{atoms}\\\\"
        f"Version=ES64L-G16RevC.01\\HF={float(scf):.7f}{archive}\\RMSD=1.000e-09\\PG=C02V\\\\@\n"
        " Normal termination of Gaussian 16 at Tue Oct 13 12:00:00 2026.\n"
    )


# ---------------------------------------------------------------------------
# Tests: germinate.muster_modelchem — method mapping
# ---------------------------------------------------------------------------


# rq-0199bb02
def test_muster_modelchem_hf():
    result = muster_modelchem("hf", "energy", 1)
    assert result["method_string"] == "HF"
    assert result["job_type"] == ""
    assert result["extra_keywords"] == {}


# rq-7049aaa5
def test_muster_modelchem_scf_alias():
    result = muster_modelchem("scf", "energy", 1)
    assert result["method_string"] == "HF"


# rq-546faafb
def test_muster_modelchem_mp2():
    result = muster_modelchem("mp2", "energy", 1)
    assert result["method_string"] == "MP2"


# rq-2052a2fa
def test_muster_modelchem_ccsd():
    result = muster_modelchem("ccsd", "energy", 1)
    assert result["method_string"] == "CCSD"


# rq-92492893
def test_muster_modelchem_ccsd_t():
    result = muster_modelchem("ccsd(t)", "energy", 1)
    assert result["method_string"] == "CCSD(T)"


# rq-851d782c
def test_muster_modelchem_passthrough():
    result = muster_modelchem("B3LYP", "energy", 1)
    assert result["method_string"] == "B3LYP"


# rq-079b1e22
def test_muster_modelchem_case_insensitive():
    result = muster_modelchem("CCSD(T)", "energy", 1)
    assert result["method_string"] == "CCSD(T)"


# ---------------------------------------------------------------------------
# Tests: germinate.muster_modelchem — driver mapping
# ---------------------------------------------------------------------------


# rq-c266b9bb
def test_muster_modelchem_energy_job_type():
    result = muster_modelchem("hf", "energy", 1)
    assert result["job_type"] == ""


# rq-f1c116db
def test_muster_modelchem_gradient_job_type():
    result = muster_modelchem("hf", "gradient", 1)
    assert result["job_type"] == "Force=NoStep"


# rq-82942b90
def test_muster_modelchem_hessian_job_type():
    result = muster_modelchem("hf", "hessian", 1)
    assert result["job_type"] == "Freq"
    assert result["extra_keywords"] == {"Punch": "Derivatives"}


# rq-f512b60b
def test_muster_modelchem_properties_driver():
    result = muster_modelchem("hf", "properties", 1)
    assert result["job_type"] == ""
    assert result["extra_keywords"].get("Pop") == "Full"


# rq-e008c6fe
def test_muster_modelchem_unsupported_driver():
    with pytest.raises(InputError):
        muster_modelchem("hf", "optimization", 1)


# ---------------------------------------------------------------------------
# Tests: germinate.muster_modelchem — open-shell handling
# ---------------------------------------------------------------------------


# rq-972b6470
def test_muster_modelchem_singlet_no_u_prefix():
    result = muster_modelchem("hf", "energy", 1)
    assert result["method_string"] == "HF"


# rq-6d73bb9e
def test_muster_modelchem_doublet_auto_u_prefix():
    result = muster_modelchem("hf", "energy", 2)
    assert result["method_string"] == "UHF"


# rq-c946c285
def test_muster_modelchem_triplet_dft_auto_u_prefix():
    result = muster_modelchem("B3LYP", "energy", 3)
    assert result["method_string"] == "UB3LYP"


# rq-25c4ad49
def test_muster_modelchem_no_double_u_prefix():
    result = muster_modelchem("UHF", "energy", 2)
    assert result["method_string"] == "UHF"


# rq-fe13245b
def test_muster_modelchem_rohf_no_u_prefix():
    result = muster_modelchem("ROHF", "energy", 2)
    assert result["method_string"] == "ROHF"


# ---------------------------------------------------------------------------
# Tests: keywords.build_route_line
# ---------------------------------------------------------------------------


# rq-af032242
def test_build_route_line_hf_energy():
    line = build_route_line("HF", "STO-3G", "", {}, {})
    assert line == "#P HF/STO-3G"


# rq-9acd1ce5
def test_build_route_line_gradient():
    line = build_route_line("HF", "STO-3G", "Force=NoStep", {}, {})
    assert line.startswith("#P HF/STO-3G")
    assert "Force=NoStep" in line


# rq-cd0d6637
def test_build_route_line_properties_pop_full():
    line = build_route_line("HF", "STO-3G", "", {"Pop": "Full"}, {})
    assert "Pop=Full" in line


# rq-7152455e
def test_build_route_line_user_keyword_appended():
    line = build_route_line("HF", "STO-3G", "", {}, {"SCF": "Tight"})
    assert "SCF=Tight" in line


# rq-e7897f09
def test_build_route_line_reserved_keywords_excluded():
    line = build_route_line("HF", "STO-3G", "", {}, {"memory": "8GB", "nprocs": "4"})
    assert "memory" not in line
    assert "nprocs" not in line


# ---------------------------------------------------------------------------
# Tests: keywords.build_com_file
# ---------------------------------------------------------------------------


# rq-d7e8161d
def test_build_com_file_link0_section():
    com = build_com_file(
        {"NProcShared": 4, "Mem": "8GB"},
        "#P HF/STO-3G",
        "title",
        0,
        1,
        "H  0.0  0.0  0.0\nH  0.0  0.0  0.74",
    )
    assert com.startswith("%NProcShared=4\n%Mem=8GB")


# rq-342a2ca3
def test_build_com_file_charge_multiplicity_line():
    com = build_com_file({"NProcShared": 1, "Mem": "1GB"}, "#P HF/STO-3G", "t", 1, 2, "H 0 0 0")
    assert "1 2" in com


# rq-b819d8e0
def test_build_com_file_coordinate_precision():
    atom_block = "O   0.0000000000   0.1171763200   0.0000000000"
    com = build_com_file({"NProcShared": 1, "Mem": "1GB"}, "#P HF/STO-3G", "t", 0, 1, atom_block)
    # Check there are at least 10 decimal digits in the atom line
    assert re.search(r"\d+\.\d{10}", com)


# rq-23d8b2c6
def test_build_com_file_trailing_blank_line():
    com = build_com_file({"NProcShared": 1, "Mem": "1GB"}, "#P HF/STO-3G", "t", 0, 1, "H 0 0 0")
    assert com.endswith("\n\n")


# rq-db92093a
def test_build_input_memory_has_floor(water_energy_input, harness):
    from qcengine.config import TaskConfig

    cfg = TaskConfig(scratch_directory=None, scratch_messy=False, ncores=1, nnodes=1, memory=0.05, retries=0, mpiexec_command=None)
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/usr/local/g16/g16"):
        job = harness.build_input(water_energy_input, cfg)
    assert "%Mem=100MB" in job["infiles"]["gaussian.com"]


# rq-4ad16d95
def test_build_input_memory_leaves_headroom(water_energy_input, harness):
    from qcengine.config import TaskConfig

    cfg = TaskConfig(scratch_directory=None, scratch_messy=False, ncores=4, nnodes=1, memory=8.0, retries=0, mpiexec_command=None)
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/usr/local/g16/g16"):
        job = harness.build_input(water_energy_input, cfg)
    # 90% of 8 GiB, in MB
    assert "%Mem=7372MB" in job["infiles"]["gaussian.com"]


# rq-0aa99076
def test_build_input_basisset_object_raises(water, harness):
    from qcengine.config import TaskConfig

    bs = BasisSet(name="sto-3g", center_data={}, atom_map=[])
    inp = AtomicInput(molecule=water, specification=AtomicSpecification(program="gaussian", driver="energy", model={"method": "HF", "basis": bs}))
    cfg = TaskConfig(scratch_directory=None, scratch_messy=False, ncores=1, nnodes=1, memory=1.0, retries=0, mpiexec_command=None)
    with pytest.raises(InputError):
        harness.build_input(inp, cfg)


# ---------------------------------------------------------------------------
# Tests: runner.GaussianHarness.found()
# ---------------------------------------------------------------------------


# rq-4cf7c460
def test_found_true_when_g16_and_cclib_available(harness):
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/usr/local/g16/g16"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            assert harness.found() is True


# rq-d3b8d8c3
def test_found_falls_back_to_g09(harness):
    def fake_find():
        # Simulates g16 absent, g09 present
        from unittest.mock import call
        return "/usr/local/g09/g09"

    with patch("qcengine.programs.gaussian.runner._find_gaussian", side_effect=fake_find):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            assert harness.found() is True


# rq-8e21b7d7
def test_found_false_when_no_gaussian(harness):
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value=None):
        assert harness.found() is False


# rq-b5e72c6d
def test_found_false_when_cclib_missing(harness):
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/usr/local/g16/g16"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=False):
            assert harness.found() is False


# rq-bd98d417
def test_found_raises_when_gaussian_missing_raise_error(harness):
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value=None):
        with pytest.raises(ModuleNotFoundError):
            harness.found(raise_error=True)


# rq-242ef1d8
def test_found_raises_when_cclib_missing_raise_error(harness):
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/usr/local/g16/g16"):
        with patch(
            "qcengine.programs.gaussian.runner.which_import",
            side_effect=ModuleNotFoundError("cclib not found"),
        ):
            with pytest.raises(ModuleNotFoundError):
                harness.found(raise_error=True)


# ---------------------------------------------------------------------------
# Tests: runner.GaussianHarness.get_version()
# ---------------------------------------------------------------------------

_VERSION_BANNER_16 = (
    " Gaussian 16, Revision C.02,\n"
    " M. J. Frisch, et al.\n"
    " Copyright (c) 1988-2017, Gaussian, Inc.\n"
)

_VERSION_BANNER_09 = (
    " Gaussian 09, Revision D.01,\n"
    " M. J. Frisch, et al.\n"
)


# rq-186c5060
def test_get_version_g16(harness):
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            with patch(
                "qcengine.programs.gaussian.runner.execute",
                return_value=(True, {"stdout": _VERSION_BANNER_16, "stderr": "", "outfiles": {}}),
            ):
                harness.version_cache.clear()
                version = harness.get_version()
    assert version == "16.C.02"


# rq-850ede4f
def test_get_version_g09(harness):
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g09"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            with patch(
                "qcengine.programs.gaussian.runner.execute",
                return_value=(True, {"stdout": _VERSION_BANNER_09, "stderr": "", "outfiles": {}}),
            ):
                harness.version_cache.clear()
                version = harness.get_version()
    assert version == "09.D.01"


# rq-3e134642
def test_get_version_cached(harness):
    harness.version_cache["/g16"] = "16.C.02"
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            with patch("qcengine.programs.gaussian.runner.execute") as mock_exe:
                version = harness.get_version()
                mock_exe.assert_not_called()
    assert version == "16.C.02"
    harness.version_cache.clear()


# rq-20d9a784
def test_get_version_unparseable_raises(harness):
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            with patch(
                "qcengine.programs.gaussian.runner.execute",
                return_value=(False, {"stdout": "no version here", "stderr": "", "outfiles": {}}),
            ):
                harness.version_cache.clear()
                with pytest.raises(UnknownError):
                    harness.get_version()


# ---------------------------------------------------------------------------
# Tests: runner.GaussianHarness.compute() — error dispatch
# ---------------------------------------------------------------------------

_NORMAL_LOG = _g16_log()


def _mock_compute(harness, water_energy_input, log_text):
    """Helper: patch found, build_input, execute, and get_version for compute tests."""
    job_record = {
        "infiles": {"gaussian.com": "%NProcShared=1\n#P HF/STO-3G\n"},
        "command": ["/g16", "gaussian.com"],
        "scratch_directory": None,
        "scratch_messy": False,
    }
    dexe = {
        "outfiles": {"gaussian.log": log_text},
        "stdout": "",
        "stderr": "",
    }
    return job_record, dexe


# rq-c77a2c13
def test_compute_raises_for_basisset_object(harness, water):
    bs = BasisSet(name="sto-3g", center_data={}, atom_map=[])
    inp = AtomicInput(molecule=water, specification=AtomicSpecification(program="gaussian", driver="energy", model={"method": "HF", "basis": bs}))
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            with pytest.raises(InputError):
                harness.compute(inp, MagicMock())


# rq-a4c449da
def test_compute_raises_resource_error_on_memory_failure(harness, water_energy_input):
    log = "galloc: could not allocate memory\nError termination\n"
    job_record, dexe = _mock_compute(harness, water_energy_input, log)
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            with patch.object(GaussianHarness, "build_input", return_value=job_record):
                with patch.object(GaussianHarness, "execute", return_value=(False, dexe)):
                    with pytest.raises(ResourceError):
                        harness.compute(water_energy_input, MagicMock())


# rq-165224fc
def test_compute_raises_input_error_on_ernie(harness, water_energy_input):
    log = "Ernie: is not defined as a Gaussian input file\nError termination\n"
    job_record, dexe = _mock_compute(harness, water_energy_input, log)
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            with patch.object(GaussianHarness, "build_input", return_value=job_record):
                with patch.object(GaussianHarness, "execute", return_value=(False, dexe)):
                    with pytest.raises(InputError):
                        harness.compute(water_energy_input, MagicMock())


# rq-edb73b83
def test_compute_raises_input_error_on_internal_coord(harness, water_energy_input):
    log = "Error in internal coordinate system\nError termination\n"
    job_record, dexe = _mock_compute(harness, water_energy_input, log)
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            with patch.object(GaussianHarness, "build_input", return_value=job_record):
                with patch.object(GaussianHarness, "execute", return_value=(False, dexe)):
                    with pytest.raises(InputError):
                        harness.compute(water_energy_input, MagicMock())


# rq-71aceb93
def test_compute_raises_unknown_error_on_abnormal_termination(harness, water_energy_input):
    log = "Something went wrong.\nError termination via Lnk1e\n"
    job_record, dexe = _mock_compute(harness, water_energy_input, log)
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            with patch.object(GaussianHarness, "build_input", return_value=job_record):
                with patch.object(GaussianHarness, "execute", return_value=(False, dexe)):
                    with pytest.raises(UnknownError):
                        harness.compute(water_energy_input, MagicMock())


# Messages from real Gaussian 16 C.01 failures
@pytest.mark.parametrize(
    "log, error",
    [
        (" Convergence failure -- run terminated.\n Error termination via Lnk1e in /g16/l502.exe\n", ConvergenceError),
        (" Atomic number out of range for 6-31G basis set.\n Error termination via Lnk1e in /g16/l301.exe\n", InputError),
        (" Last state= \"GCL\"\n Error termination via Lnk1e in /opt/g16/l1.exe at Tue Oct 13\n", InputError),
        (" Illegal IType or MSType generated by parse.\n Error termination via Lnk1e in /g16/l1.exe\n", InputError),
        (" The combination of multiplicity 2 and    10 electrons is impossible.\n Error termination\n", InputError),
        (" Small interatomic distances encountered:      2     1\n Problem with the distance matrix.\n", InputError),
        (" Out-of-memory error in routine RdGeom-1 (IEnd=      10500090 MxCore=        131072)\n", ResourceError),
        (" R6DS8: Unable to choose the S8 parameter, IExCor=    0 IXCFnc=  0 ScaHFX=  1.000000 IDFTD=3\n", InputError),
        (" Erroneous write. Write -1 instead of 8.\n Error termination via Lnk1e in /g16/l502.exe\n", UnknownError),
    ],
)
def test_compute_classifies_errors(harness, water_energy_input, log, error):
    job_record, dexe = _mock_compute(harness, water_energy_input, log)
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            with patch.object(GaussianHarness, "build_input", return_value=job_record):
                with patch.object(GaussianHarness, "execute", return_value=(False, dexe)):
                    with pytest.raises(error):
                        harness.compute(water_energy_input, MagicMock())


@requires_cclib
def test_compute_unsupported_method_raises_input_error(harness, water_energy_input):
    log = _g16_log(method="RCID-FC", body=" EUMP2 =    -0.76010504995300D+02\n", archive="\\MP2=-76.0105050")
    job_record, dexe = _mock_compute(harness, water_energy_input, log)
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            with patch.object(GaussianHarness, "build_input", return_value=job_record):
                with patch.object(GaussianHarness, "execute", return_value=(True, dexe)):
                    with patch.object(GaussianHarness, "get_version", return_value="16.C.01"):
                        with _patch_cclib(_make_ccdata_with_attrs()):
                            with pytest.raises(InputError, match="RCID-FC"):
                                harness.compute(water_energy_input, MagicMock())


def test_execute_keeps_gaussian_scratch_in_job_directory(harness):
    job = {"command": ["/g16", "gaussian.com"], "infiles": {}, "scratch_directory": None, "scratch_messy": False}
    with patch("qcengine.programs.gaussian.runner.execute", return_value=(True, {})) as mock_execute:
        harness.execute(job)
    assert mock_execute.call_args.kwargs["environment"]["GAUSS_SCRDIR"] == "."
    assert "fort.7" in mock_execute.call_args.args[2]


# ---------------------------------------------------------------------------
# Tests: runner.GaussianHarness.compute() — success paths
# ---------------------------------------------------------------------------

def _full_compute_patches(harness, water_energy_input, log_text, ccdata, return_result):
    """Return context-manager stack for a successful compute() call."""
    from contextlib import ExitStack

    job_record = {
        "infiles": {"gaussian.com": "%NProcShared=1\n#P HF/STO-3G\n"},
        "command": ["/g16", "gaussian.com"],
        "scratch_directory": None,
        "scratch_messy": False,
    }
    dexe = {
        "outfiles": {"gaussian.log": log_text},
        "stdout": "",
        "stderr": "",
    }

    stack = ExitStack()
    stack.enter_context(patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"))
    stack.enter_context(patch("qcengine.programs.gaussian.runner.which_import", return_value=True))
    stack.enter_context(patch.object(GaussianHarness, "build_input", return_value=job_record))
    stack.enter_context(patch.object(GaussianHarness, "execute", return_value=(True, dexe)))
    stack.enter_context(patch.object(GaussianHarness, "get_version", return_value="16.C.02"))
    stack.enter_context(_patch_cclib(ccdata))
    return stack


# rq-99807237
@requires_cclib
def test_compute_returns_atomic_result_energy(harness, water_energy_input_all_files, water):
    log_text = _NORMAL_LOG
    ccdata = _make_ccdata_with_attrs(scfenergies=np.array([-2041.3]))

    with _full_compute_patches(harness, water_energy_input_all_files, log_text, ccdata, None):
        result = harness.compute(water_energy_input_all_files, MagicMock())

    assert result.success is True
    assert isinstance(result.return_result, float)
    assert "input" in result.native_files  # rq-f13e0d24
    assert "gaussian.log" in result.native_files
    assert result.provenance.creator == "Gaussian"


# rq-e62578e2
@requires_cclib
def test_compute_returns_atomic_result_gradient(harness, water):
    forces = np.array([[0.01, 0.0, -0.02], [0.0, 0.005, 0.01], [0.0, -0.005, 0.01]])
    ccdata = _make_ccdata_with_attrs(
        scfenergies=np.array([-2041.3]),
        grads=forces[np.newaxis, :, :],  # shape (1, 3, 3)
    )
    inp = AtomicInput(molecule=water, specification=AtomicSpecification(program="gaussian", driver="gradient", model={"method": "HF", "basis": "STO-3G"}))
    log_text = _g16_log(forces=forces)

    job_record = {
        "infiles": {"gaussian.com": ""},
        "command": ["/g16", "gaussian.com"],
        "scratch_directory": None,
        "scratch_messy": False,
    }
    dexe = {"outfiles": {"gaussian.log": log_text}, "stdout": "", "stderr": ""}

    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            with patch.object(GaussianHarness, "build_input", return_value=job_record):
                with patch.object(GaussianHarness, "execute", return_value=(True, dexe)):
                    with patch.object(GaussianHarness, "get_version", return_value="16.C.02"):
                        with _patch_cclib(ccdata):
                            result = harness.compute(inp, MagicMock())

    assert result.success is True
    # AtomicResult shapes gradient as (natoms, 3); 3 atoms → shape (3, 3)
    assert np.array(result.return_result).shape == (3, 3)


# rq-4d683dac
@requires_cclib
def test_compute_returns_atomic_result_hessian(harness, water):
    hess_matrix = np.eye(9) * 0.5
    ccdata = _make_ccdata_with_attrs(
        scfenergies=np.array([-2041.3]),
        hessian=hess_matrix,
    )
    inp = AtomicInput(molecule=water, specification=AtomicSpecification(program="gaussian", driver="hessian", model={"method": "HF", "basis": "STO-3G"}))
    log_text = _NORMAL_LOG

    job_record = {
        "infiles": {"gaussian.com": ""},
        "command": ["/g16", "gaussian.com"],
        "scratch_directory": None,
        "scratch_messy": False,
    }
    punch = _punch(list(np.zeros(9)) + list(hess_matrix[np.tril_indices(9)]))
    dexe = {"outfiles": {"gaussian.log": log_text, "fort.7": punch}, "stdout": "", "stderr": ""}

    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            with patch.object(GaussianHarness, "build_input", return_value=job_record):
                with patch.object(GaussianHarness, "execute", return_value=(True, dexe)):
                    with patch.object(GaussianHarness, "get_version", return_value="16.C.02"):
                        with _patch_cclib(ccdata):
                            result = harness.compute(inp, MagicMock())

    assert result.success is True
    # AtomicResult shapes hessian as (3*natoms, 3*natoms); 3 atoms → (9, 9)
    assert np.array(result.return_result).shape == (9, 9)


# rq-1a051bc2
@requires_cclib
def test_compute_returns_atomic_result_properties(harness, water):
    ccdata = _make_ccdata_with_attrs(
        scfenergies=np.array([-2041.3]),
        moments=[None, np.array([0.0, 1.85, 0.0])],
        atomcharges={"mulliken": [-0.5, 0.25, 0.25]},
    )
    inp = AtomicInput(molecule=water, specification=AtomicSpecification(program="gaussian", driver="properties", model={"method": "HF", "basis": "STO-3G"}))
    log_text = _NORMAL_LOG

    job_record = {
        "infiles": {"gaussian.com": ""},
        "command": ["/g16", "gaussian.com"],
        "scratch_directory": None,
        "scratch_messy": False,
    }
    dexe = {"outfiles": {"gaussian.log": log_text}, "stdout": "", "stderr": ""}

    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            with patch.object(GaussianHarness, "build_input", return_value=job_record):
                with patch.object(GaussianHarness, "execute", return_value=(True, dexe)):
                    with patch.object(GaussianHarness, "get_version", return_value="16.C.02"):
                        with _patch_cclib(ccdata):
                            result = harness.compute(inp, MagicMock())

    assert result.success is True
    assert isinstance(result.return_result, float)
    assert "CURRENT ENERGY" in result.extras["qcvars"]


# ---------------------------------------------------------------------------
# Tests: harvester.harvest() — energy parsing
# ---------------------------------------------------------------------------


def _make_water_mol():
    return Molecule.from_data(
        """
        0 1
        O  0.000000000  0.000000000  0.117176320
        H  0.000000000  0.757335953 -0.468705281
        H  0.000000000 -0.757335953 -0.468705281
        units angstrom
        """
    )


# rq-71b27e68
def test_harvest_energies_hf():
    qcvars = harvest_energies(_g16_log())
    assert qcvars["CURRENT ENERGY"] == Decimal("-76.0105049953")
    assert qcvars["HF TOTAL ENERGY"] == qcvars["SCF TOTAL ENERGY"] == qcvars["CURRENT ENERGY"]
    assert "DFT TOTAL ENERGY" not in qcvars


# rq-6bff3228
def test_harvest_energies_dft_has_no_hf_total():
    qcvars = harvest_energies(_g16_log(scf="-76.4087309161", scf_label="RB3LYP"))
    assert qcvars["CURRENT ENERGY"] == Decimal("-76.4087309161")
    assert qcvars["DFT TOTAL ENERGY"] == qcvars["DFT FUNCTIONAL TOTAL ENERGY"] == qcvars["CURRENT ENERGY"]
    assert "HF TOTAL ENERGY" not in qcvars


# rq-578bb785
def test_harvest_energies_mp2():
    log = _g16_log(
        scf="-74.9630231626",
        method="RMP2-FC",
        body=" E2 =    -0.3544594200D-01 EUMP2 =    -0.74998469104638D+02\n",
        archive="\\MP2=-74.9984691",
    )
    qcvars = harvest_energies(log)
    assert qcvars["MP2 TOTAL ENERGY"] == Decimal("-74.998469104638")
    assert qcvars["MP2 CORRELATION ENERGY"] == Decimal("-74.998469104638") - Decimal("-74.9630231626")
    assert qcvars["CURRENT ENERGY"] == qcvars["MP2 TOTAL ENERGY"]


_CCSD_T_BODY = (
    " E2 =    -0.3544594200D-01 EUMP2 =    -0.74998469104637D+02\n"
    " CCSD(T)\n"
    " =======\n"
    " Wavefunction amplitudes converged. E(Corr)=     -75.012383393\n"
    " CCSD(T)= -0.75012450878D+02\n"
)


# rq-1b7ae62e rq-eddef743
def test_harvest_energies_ccsd_t():
    log = _g16_log(
        scf="-74.9630231626",
        method="RCCSD(T)-FC",
        body=_CCSD_T_BODY,
        archive="\\MP2=-74.9984691\\CCSD=-75.0123834\\CCSD(T)=-75.0124509",
    )
    qcvars = harvest_energies(log)
    assert qcvars["CCSD TOTAL ENERGY"] == Decimal("-75.012383393")
    assert qcvars["CCSD(T) TOTAL ENERGY"] == Decimal("-75.012450878")
    assert qcvars["CURRENT ENERGY"] == qcvars["CCSD(T) TOTAL ENERGY"]


@pytest.mark.parametrize(
    "method, body, archive, expected, stem",
    [
        (
            "RMP3-FC",
            " EUMP2 =    -0.74998469104637D+02\n E3=       -0.96243567D-02        EUMP3=      -0.75008093461D+02\n",
            "\\MP2=-74.9984691\\MP3=-75.0080935",
            "-75.008093461",
            "MP3",
        ),
        (
            "RMP4SDTQ-FC",
            " EUMP2 =    -0.74998469104637D+02\n"
            " E4(SDQ)=  -0.28677025D-02        UMP4(SDQ)=  -0.75010961164D+02\n"
            " E4(SDTQ)= -0.29153588D-02        UMP4(SDTQ)= -0.75011008820D+02\n",
            "\\MP2=-74.9984691\\MP4SDQ=-75.0109612\\MP4SDTQ=-75.0110088",
            "-75.011008820",
            "MP4",
        ),
        (
            "RQCISD(T)-FC",
            " EUMP2 =    -0.74998469104637D+02\n"
            " Wavefunction amplitudes converged. E(Corr)=     -75.012400197\n"
            " QCISD(T)= -0.75012457677D+02\n",
            "\\MP2=-74.9984691\\QCISD=-75.0124002\\QCISD(T)=-75.0124577",
            "-75.012457677",
            "QCISD(T)",
        ),
        (
            "RCISD-FC",
            " EUMP2 =    -0.74998469104637D+02\n Wavefunction amplitudes converged. E(Corr)=     -75.011795318\n",
            "\\MP2=-74.9984691\\CISD=-75.0117953",
            "-75.011795318",
            "CISD",
        ),
    ],
)
def test_harvest_energies_post_hf_levels(method, body, archive, expected, stem):
    qcvars = harvest_energies(_g16_log(scf="-74.9630231626", method=method, body=body, archive=archive))
    assert qcvars["CURRENT ENERGY"] == Decimal(expected)
    assert qcvars[f"{stem} TOTAL ENERGY"] == Decimal(expected)


def test_harvest_energies_open_shell_prefix():
    log = _g16_log(
        scf="-54.7749910183",
        scf_label="ROHF",
        method="ROMP2-FC",
        body=" E2 =    -0.3136439645D-01 EUMP2 =    -0.54806355414722D+02\n",
        archive="\\MP2=-54.8063554",
    )
    assert harvest_energies(log)["CURRENT ENERGY"] == Decimal("-54.806355414722")


def test_harvest_energies_double_hybrid():
    log = _g16_log(
        scf="-75.1988324129",
        scf_label="RB2PLYP",
        method="RB2PLYP-FC",
        body=" E2(B2PLYP) =    -0.1204732391D-01 E(B2PLYP) =    -0.75210879736849D+02\n",
        archive="\\MP2=-75.2108797",
    )
    qcvars = harvest_energies(log)
    assert qcvars["CURRENT ENERGY"] == Decimal("-75.210879736849")
    assert qcvars["DFT TOTAL ENERGY"] == qcvars["CURRENT ENERGY"]
    assert qcvars["DOUBLE-HYBRID CORRECTION ENERGY"] == Decimal("-0.01204732391")
    assert qcvars["SCF TOTAL ENERGY"] == Decimal("-75.1988324129")
    assert "MP2 TOTAL ENERGY" not in qcvars


def test_harvest_energies_numerical_hessian_uses_reference_geometry():
    """Energies after the first Forces block belong to displaced geometries."""
    log = _g16_log(
        scf="-74.9630231626",
        method="RCCSD-FC",
        body=(
            " Wavefunction amplitudes converged. E(Corr)=     -75.012383387\n"
            " ***** Axes restored to original set *****\n"
            " Center     Atomic                   Forces (Hartrees/Bohr)\n"
            " SCF Done:  E(RHF) =  -74.9630232616     A.U. after    5 cycles\n"
            " Wavefunction amplitudes converged. E(Corr)=     -75.012487506\n"
        ),
        archive="\\CCSD=-75.0123834",
    )
    qcvars = harvest_energies(log)
    assert qcvars["SCF TOTAL ENERGY"] == Decimal("-74.9630231626")
    assert qcvars["CURRENT ENERGY"] == Decimal("-75.012383387")


def test_harvest_energies_last_scf_before_reference_cut():
    """Stable=Opt reruns the SCF at the same geometry; the last SCF is the result."""
    log = _g16_log(
        scf="-149.517259423",
        scf_label="UHF",
        pre=(
            " SCF Done:  E(UHF) =  -149.461276695     A.U. after    8 cycles\n"
            " SCF Done:  E(UHF) =  -149.484344617     a.u. after    5 cycles\n"
        ),
    )
    assert harvest_energies(log)["CURRENT ENERGY"] == Decimal("-149.517259423")


def test_harvest_energies_semiempirical_not_dft():
    qcvars = harvest_energies(_g16_log(scf="-0.0862439508", scf_label="RPM6", basis="ZDO"))
    assert qcvars["CURRENT ENERGY"] == Decimal("-0.0862439508")
    assert "DFT TOTAL ENERGY" not in qcvars
    assert "HF TOTAL ENERGY" not in qcvars


def test_harvest_energies_unsupported_method_raises():
    log = _g16_log(
        scf="-74.9630231626",
        method="RCID-FC",
        body=" EUMP2 =    -0.74998469104637D+02\n",
        archive="\\MP2=-74.9984691",
    )
    with pytest.raises(UnsupportedMethodError, match="RCID-FC"):
        harvest_energies(log)


def test_harvest_energies_archive_mismatch_raises():
    log = _g16_log(scf="-74.9630231626", archive="", method="RHF").replace("HF=-74.9630232", "HF=-74.9640232")
    with pytest.raises(ValueError, match="disagrees with the archive"):
        harvest_energies(log)


def test_harvest_energies_missing_archive_raises():
    with pytest.raises(ValueError, match="No archive entry"):
        harvest_energies(" SCF Done:  E(RHF) =  -76.0105049953     A.U. after   10 cycles\n")


def test_harvest_energies_missing_scf_raises():
    log = _g16_log().replace(" SCF Done:  E(RHF) =  -76.0105049953     A.U. after   10 cycles\n", "")
    with pytest.raises(ValueError, match="SCF Done"):
        harvest_energies(log)


# ---------------------------------------------------------------------------
# Tests: harvester.harvest() — gradient parsing
# ---------------------------------------------------------------------------


# rq-36736115
@requires_cclib
def test_harvest_gradient_negated():
    """The gradient is the negated last forces block (input orientation)."""
    in_mol = _make_water_mol()
    forces = np.array([[0.01, 0.0, -0.02], [0.0, 0.005, 0.01], [0.0, -0.005, 0.01]])
    log = _g16_log(mol=in_mol, forces=10 * forces) + _g16_log(mol=in_mol, forces=forces)
    with _patch_cclib(_make_ccdata_with_attrs()):
        _, grad, _, _, _ = harvest(in_mol, "hf", log)
    np.testing.assert_allclose(grad, -forces.flatten(), atol=1e-12)


# rq-3b0adf96
@requires_cclib
def test_harvest_no_gradient_when_absent():
    in_mol = _make_water_mol()
    ccdata = _make_ccdata_with_attrs(scfenergies=np.array([-2041.3]))
    # grads is None (no attribute)
    with _patch_cclib(ccdata):
        qcvars, grad, hess, out_mol, _ = harvest(in_mol, "hf", _g16_log(mol=in_mol, ))
    assert grad is None


# ---------------------------------------------------------------------------
# Tests: harvester.harvest() — Hessian parsing
# ---------------------------------------------------------------------------


# rq-f2e32162
@requires_cclib
def test_harvest_hessian():
    in_mol = _make_water_mol()
    hess_matrix = np.eye(9) * 0.5
    punch = _punch(list(np.zeros(9)) + list(hess_matrix[np.tril_indices(9)]))
    with _patch_cclib(_make_ccdata_with_attrs()):
        _, _, hess, _, _ = harvest(in_mol, "hf", _g16_log(mol=in_mol), punch)
    np.testing.assert_allclose(hess, hess_matrix)


# rq-344f3432
@requires_cclib
def test_harvest_no_hessian_when_absent():
    in_mol = _make_water_mol()
    ccdata = _make_ccdata_with_attrs(scfenergies=np.array([-2041.3]))
    with _patch_cclib(ccdata):
        qcvars, grad, hess, out_mol, _ = harvest(in_mol, "hf", _g16_log(mol=in_mol, ))
    assert hess is None


# ---------------------------------------------------------------------------
# Tests: harvester.harvest() — properties parsing
# ---------------------------------------------------------------------------


# rq-4c4a1542
@requires_cclib
def test_harvest_dipole_from_archive():
    in_mol = _make_water_mol()
    with _patch_cclib(_make_ccdata_with_attrs()):
        qcvars, _, _, _, _ = harvest(in_mol, "hf", _g16_log(mol=in_mol, archive="\\Dipole=0.,0.,-0.6358857"))
    np.testing.assert_allclose(qcvars["CURRENT DIPOLE"], [0.0, 0.0, -0.6358857])
    np.testing.assert_allclose(qcvars["HF DIPOLE"], [0.0, 0.0, -0.6358857])


@requires_cclib
def test_harvest_no_dipole_for_post_hf():
    in_mol = _make_water_mol()
    log = _g16_log(
        method="RMP2-FC",
        body=" EUMP2 =    -0.76010504995300D+02\n",
        archive="\\MP2=-76.0105050\\Dipole=0.,0.,-0.6358857",
    )
    with _patch_cclib(_make_ccdata_with_attrs()):
        qcvars, _, _, _, _ = harvest(in_mol, "mp2", log)
    assert "CURRENT DIPOLE" not in qcvars


@requires_cclib
def test_harvest_electron_counts_from_log():
    """cclib's homos has one entry for ROHF; the log line gives both counts."""
    in_mol = _make_water_mol()
    log = _g16_log(pre="    5 alpha electrons        4 beta electrons\n")
    with _patch_cclib(_make_ccdata_with_attrs(homos=np.array([4]))):
        qcvars, _, _, _, _ = harvest(in_mol, "hf", log)
    assert qcvars["N ALPHA ELECTRONS"] == 5
    assert qcvars["N BETA ELECTRONS"] == 4


# rq-ffae3fa6
@requires_cclib
def test_harvest_mulliken_charges():
    in_mol = _make_water_mol()
    charges = [-0.5, 0.25, 0.25]
    ccdata = _make_ccdata_with_attrs(
        scfenergies=np.array([-2041.3]),
        atomcharges={"mulliken": charges},
    )
    with _patch_cclib(ccdata):
        qcvars, grad, hess, out_mol, _ = harvest(in_mol, "hf", _g16_log(mol=in_mol, ))
    assert "MULLIKEN CHARGES" in qcvars
    assert len(qcvars["MULLIKEN CHARGES"]) == 3


# rq-53f8ed12
@requires_cclib
def test_harvest_missing_mulliken_no_error():
    in_mol = _make_water_mol()
    ccdata = _make_ccdata_with_attrs(
        scfenergies=np.array([-2041.3]),
        atomcharges={},
    )
    with _patch_cclib(ccdata):
        qcvars, grad, hess, out_mol, _ = harvest(in_mol, "hf", _g16_log(mol=in_mol, ))
    assert "MULLIKEN CHARGES" not in qcvars


# ---------------------------------------------------------------------------
# Tests: harvester.harvest() — geometry and molecule output
# ---------------------------------------------------------------------------


# rq-a76c3d7b
@requires_cclib
def test_harvest_output_molecule_symbols():
    in_mol = _make_water_mol()
    ccdata = _make_ccdata_with_attrs(scfenergies=np.array([-2041.3]))
    with _patch_cclib(ccdata):
        qcvars, grad, hess, out_mol, _ = harvest(in_mol, "hf", _g16_log(mol=in_mol, ))
    assert list(out_mol.symbols) == ["O", "H", "H"]


# rq-a9194afb
@requires_cclib
def test_harvest_nre_cross_check_passes():
    in_mol = _make_water_mol()
    ccdata = _make_ccdata_with_attrs(scfenergies=np.array([-2041.3]))
    # Should not raise
    with _patch_cclib(ccdata):
        harvest(in_mol, "hf", _g16_log(mol=in_mol, ))


# rq-0b4cf843
@requires_cclib
def test_harvest_geometry_mismatch_raises():
    in_mol = _make_water_mol()
    moved = Molecule(**{**in_mol.model_dump(), "geometry": in_mol.geometry + 1.0e-3})
    with _patch_cclib(_make_ccdata_with_attrs()):
        with pytest.raises(ValueError, match="differs from the input molecule"):
            harvest(in_mol, "hf", _g16_log(mol=moved))


# ---------------------------------------------------------------------------
# Tests: harvester.harvest() — error conditions
# ---------------------------------------------------------------------------


# rq-b7c7b8cb
@requires_cclib
def test_harvest_no_archive_raises():
    in_mol = _make_water_mol()
    log = " SCF Done:  E(RHF) =  -76.0105049953     A.U. after   10 cycles\n Normal termination of Gaussian 16\n"
    with _patch_cclib(_make_ccdata_with_attrs()):
        with pytest.raises(ValueError, match="No archive entry"):
            harvest(in_mol, "hf", log)


# rq-bb775969
@requires_cclib
def test_harvest_cclib_parse_failure_propagates():
    in_mol = _make_water_mol()
    with patch(
        "cclib.io.ccopen",
        side_effect=Exception("cclib parse error"),
    ):
        with pytest.raises(Exception, match="cclib parse error"):
            harvest(in_mol, "hf", "unparseable log")


# ---------------------------------------------------------------------------
# Tests: harvester.is_normal_termination()
# ---------------------------------------------------------------------------


# rq-90141f40
def test_is_normal_termination_true():
    log = "Some output\nNormal termination of Gaussian 16\n"
    assert is_normal_termination(log) is True


# rq-12c41d5e
def test_is_normal_termination_false():
    log = "Some output\nError termination via Lnk1e\n"
    assert is_normal_termination(log) is False


# ---------------------------------------------------------------------------
# Tests: ghost-atom support
# ---------------------------------------------------------------------------


def _make_hene_ghost_mol():
    """He real at origin, Ne ghost at (2.5, 0, 0) Å, with fix_com/fix_orientation."""
    return Molecule.from_data(
        """
        0 1
        He 0.0 0.0 0.0
        @Ne 2.5 0.0 0.0
        nocom
        noreorient
        """
    )


def _default_taskconfig(**overrides):
    from qcengine.config import TaskConfig

    cfg = dict(
        scratch_directory=None,
        scratch_messy=False,
        ncores=1,
        nnodes=1,
        memory=1.0,
        retries=0,
        mpiexec_command=None,
    )
    cfg.update(overrides)
    return TaskConfig(**cfg)


def _spec(driver="energy", method="HF", basis="STO-3G"):
    return AtomicSpecification(
        program="gaussian",
        driver=driver,
        model={"method": method, "basis": basis},
    )


# rq-2229f087
def test_ghost_real_atom_emits_plain_symbol(harness):
    mol = Molecule.from_data(
        {
            "symbols": ["H"],
            "geometry": [0.0, 0.0, 0.0],
            "real": [True],
            "molecular_charge": 0,
            "molecular_multiplicity": 2,
        }
    )
    inp = AtomicInput(molecule=mol, specification=_spec())
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        job = harness.build_input(inp, _default_taskconfig())
    com = job["infiles"]["gaussian.com"]
    atom_lines = [ln for ln in com.splitlines() if ln.lstrip().startswith(("H(", "H-Bq("))]
    assert any(ln.lstrip().startswith("H(Iso=") for ln in atom_lines)
    assert not any(ln.lstrip().startswith("H-Bq") for ln in atom_lines)


# rq-916e4919
def test_ghost_atom_emits_bq_suffix(harness):
    mol = Molecule.from_data(
        {
            "symbols": ["H"],
            "geometry": [0.0, 0.0, 0.0],
            "real": [False],
            "molecular_charge": 0,
            "molecular_multiplicity": 1,
        }
    )
    inp = AtomicInput(molecule=mol, specification=_spec())
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        job = harness.build_input(inp, _default_taskconfig())
    com = job["infiles"]["gaussian.com"]
    assert any(ln.lstrip().startswith("H-Bq(Iso=") for ln in com.splitlines())


# rq-2a015edf
def test_ghost_mixed_real_and_ghost(harness):
    mol = _make_hene_ghost_mol()
    inp = AtomicInput(molecule=mol, specification=_spec())
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        job = harness.build_input(inp, _default_taskconfig())
    lines = [ln for ln in job["infiles"]["gaussian.com"].splitlines() if ln.strip()]
    he_idx = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("He(Iso="))
    ne_idx = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("Ne-Bq(Iso="))
    assert he_idx < ne_idx  # input order preserved


# rq-a100502c
def test_ghost_route_line_unchanged(harness):
    mol = _make_hene_ghost_mol()
    inp = AtomicInput(molecule=mol, specification=_spec(method="HF", basis="6-31G*"))
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        job = harness.build_input(inp, _default_taskconfig())
    lines = job["infiles"]["gaussian.com"].splitlines()
    route_lines = [ln for ln in lines if ln.startswith("#P")]
    assert route_lines == ["#P HF/6-31G*"]


# rq-ce76f384
def test_ghost_all_atoms_ghost(harness):
    mol = Molecule.from_data(
        {
            "symbols": ["H", "H"],
            "geometry": [0.0, 0.0, 0.0, 0.0, 0.0, 1.4],
            "real": [False, False],
            "molecular_charge": 0,
            "molecular_multiplicity": 1,
        }
    )
    inp = AtomicInput(molecule=mol, specification=_spec())
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        job = harness.build_input(inp, _default_taskconfig())
    com = job["infiles"]["gaussian.com"]
    atom_lines = [
        ln for ln in com.splitlines()
        if ln.lstrip().startswith(("H ", "H-Bq"))
    ]
    assert len(atom_lines) == 2
    assert all("-Bq" in ln for ln in atom_lines)


# rq-ceef73bd
def test_ghost_coordinate_precision_preserved(harness):
    mol = Molecule.from_data(
        {
            "symbols": ["H"],
            "geometry": [1.23456789012 / qcel.constants.conversion_factor("bohr", "angstrom"), 0.0, 0.0],
            "real": [False],
            "molecular_charge": 0,
            "molecular_multiplicity": 1,
        }
    )
    inp = AtomicInput(molecule=mol, specification=_spec())
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        job = harness.build_input(inp, _default_taskconfig())
    com = job["infiles"]["gaussian.com"]
    ghost_line = next(ln for ln in com.splitlines() if "H-Bq" in ln)
    assert re.search(r"\d+\.\d{10}", ghost_line)


# rq-b27d4aff
@requires_cclib
def test_ghost_out_mol_real_preserved():
    in_mol = _make_hene_ghost_mol()
    ccdata = _make_ccdata_with_attrs(
        scfenergies=np.array([-77.7]),
        atomcoords=np.array([[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]]),
        atomnos=np.array([2, 10]),
    )
    with _patch_cclib(ccdata):
        _, _, _, out_mol, _ = harvest(in_mol, "hf", _g16_log(mol=in_mol, ))
    assert list(out_mol.real) == [True, False]


# rq-731f01ba
@requires_cclib
def test_ghost_out_mol_symbols_from_in_mol():
    in_mol = _make_hene_ghost_mol()
    ccdata = _make_ccdata_with_attrs(
        scfenergies=np.array([-77.7]),
        atomcoords=np.array([[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]]),
        atomnos=np.array([2, 10]),
    )
    with _patch_cclib(ccdata):
        _, _, _, out_mol, _ = harvest(in_mol, "hf", _g16_log(mol=in_mol, ))
    assert list(out_mol.symbols) == ["He", "Ne"]


# rq-7411efa0
@requires_cclib
def test_ghost_atom_count_sanity_check_passes():
    in_mol = _make_hene_ghost_mol()
    ccdata = _make_ccdata_with_attrs(
        scfenergies=np.array([-77.7]),
        atomcoords=np.array([[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]]),
        atomnos=np.array([2, 10]),
    )
    with _patch_cclib(ccdata):
        # Should not raise: 2 atoms in cclib output == 2 atoms in in_mol
        harvest(in_mol, "hf", _g16_log(mol=in_mol, ))


# rq-66067019
@requires_cclib
def test_ghost_atom_count_mismatch_raises():
    in_mol = _make_hene_ghost_mol()  # 2 atoms
    three = Molecule.from_data("He 0 0 0\n@Ne 2.5 0 0\n@Ne 5 0 0")
    with _patch_cclib(_make_ccdata_with_attrs()):
        with pytest.raises(ValueError, match="atom count"):
            harvest(in_mol, "hf", _g16_log(mol=three))


# rq-a69d0898
@requires_cclib
def test_ghost_nre_check_excludes_ghosts():
    in_mol = _make_hene_ghost_mol()
    ccdata = _make_ccdata_with_attrs(
        scfenergies=np.array([-77.7]),
        atomcoords=np.array([[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]]),
        atomnos=np.array([2, 10]),
    )
    with _patch_cclib(ccdata):
        # NRE = 0 for both (single real atom), no ValueError
        harvest(in_mol, "hf", _g16_log(mol=in_mol, ))


# rq-b57c882b
@requires_cclib
def test_ghost_geometry_mismatch_still_raises():
    in_mol = Molecule.from_data(
        {
            "symbols": ["H", "H", "He"],
            "geometry": [0.0, 0.0, 0.0, 0.0, 0.0, 1.4, 5.0, 0.0, 0.0],
            "real": [True, True, False],
            "fix_com": True,
            "fix_orientation": True,
        }
    )
    moved_ghost = Molecule(**{**in_mol.model_dump(), "geometry": [0.0, 0.0, 0.0, 0.0, 0.0, 1.4, 5.1, 0.0, 0.0]})
    with _patch_cclib(_make_ccdata_with_attrs()):
        with pytest.raises(ValueError, match="differs from the input molecule"):
            harvest(in_mol, "hf", _g16_log(mol=moved_ghost))


# rq-e5a98119
@requires_cclib
def test_ghost_nre_check_unchanged_for_all_real():
    in_mol = _make_water_mol()
    ccdata = _make_ccdata_with_attrs(scfenergies=np.array([-2041.3]))
    with _patch_cclib(ccdata):
        # Existing behaviour: no ghosts, full-mol NRE comparison, no raise
        harvest(in_mol, "hf", _g16_log(mol=in_mol, ))


# rq-d97cc860
@requires_cclib
def test_ghost_gradient_spans_real_and_ghost():
    in_mol = _make_hene_ghost_mol()
    # Forces: nonzero on He, zero on Ne (ghost)
    forces = np.array([[0.001, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with _patch_cclib(_make_ccdata_with_attrs()):
        _, grad, _, _, _ = harvest(in_mol, "hf", _g16_log(mol=in_mol, forces=forces))
    assert grad.shape == (6,)
    np.testing.assert_allclose(grad[3:6], np.zeros(3), atol=1e-12)


# rq-f8ce079b
@requires_cclib
@pytest.mark.parametrize("fixed_frame", [True, False])
def test_harvest_returns_input_molecule_without_alignment(fixed_frame):
    """All harvested quantities are in Gaussian's input orientation, so the input
    molecule is returned unchanged and no alignment (atom-mapping search) is done."""
    from qcelemental.models.v2 import Molecule as V2Molecule

    in_mol = Molecule(**{**_make_hene_ghost_mol().model_dump(), "fix_com": fixed_frame, "fix_orientation": fixed_frame})

    def no_align(*args, **kwargs):
        raise AssertionError("Molecule.align must not be called")

    with _patch_cclib(_make_ccdata_with_attrs()):
        with patch.object(V2Molecule, "align", no_align):
            _, _, _, out_mol, _ = harvest(in_mol, "hf", _g16_log(mol=in_mol))
    assert out_mol is in_mol


# rq-04028610
@requires_cclib
def test_ghost_compute_preserves_ghost_designation(harness):
    in_mol = _make_hene_ghost_mol()
    inp = AtomicInput(molecule=in_mol, specification=_spec(method="HF", basis="STO-3G"))

    ccdata = _make_ccdata_with_attrs(
        scfenergies=np.array([-77.7]),
        atomcoords=np.array([[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]]),
        atomnos=np.array([2, 10]),
    )
    log_text = _g16_log(mol=in_mol)
    job_record = {
        "infiles": {"gaussian.com": "%NProcShared=1\n#P HF/STO-3G\n"},
        "command": ["/g16", "gaussian.com"],
        "scratch_directory": None,
        "scratch_messy": False,
    }
    dexe = {"outfiles": {"gaussian.log": log_text}, "stdout": "", "stderr": ""}

    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        with patch("qcengine.programs.gaussian.runner.which_import", return_value=True):
            with patch.object(GaussianHarness, "build_input", return_value=job_record):
                with patch.object(GaussianHarness, "execute", return_value=(True, dexe)):
                    with patch.object(GaussianHarness, "get_version", return_value="16.C.02"):
                        with _patch_cclib(ccdata):
                            result = harness.compute(inp, MagicMock())

    assert list(result.molecule.real) == [True, False]
    assert list(result.molecule.symbols) == ["He", "Ne"]


# ---------------------------------------------------------------------------
# Tests: tricky-ghost (mp2(full) method + NoSymm)
# ---------------------------------------------------------------------------


# rq-2cb2c5c0
def test_tricky_ghost_mp2_full_method_string():
    """muster_modelchem("mp2(full)") emits Gaussian's "MP2(Full)" route token."""
    result = muster_modelchem("mp2(full)", "energy", 1)
    assert result["method_string"] == "MP2(Full)"


# rq-2cb2c5c0
def test_tricky_ghost_mp2_full_method_string_gradient():
    """muster_modelchem("mp2(full)", "gradient") uses Force=NoStep job type."""
    result = muster_modelchem("mp2(full)", "gradient", 1)
    assert result["method_string"] == "MP2(Full)"
    assert result["job_type"] == "Force=NoStep"


# rq-2cb2c5c0
def test_tricky_ghost_mp2_full_open_shell():
    """Open-shell mp2(full) gets a U prefix like other MP2 variants."""
    result = muster_modelchem("mp2(full)", "energy", 2)
    assert result["method_string"] == "UMP2(Full)"


# rq-2cb2c5c0
def test_tricky_ghost_nosymm_emitted_bare():
    """NoSymm in user_keywords with empty value emits as a bare token."""
    line = build_route_line("MP2(Full)", "6-31g*", "Force=NoStep", {}, {"NoSymm": ""})
    assert line.startswith("#P MP2(Full)/6-31g* Force=NoStep")
    assert " NoSymm" in line
    assert "NoSymm=" not in line  # bare, not "NoSymm="


# rq-2cb2c5c0
@requires_cclib
def test_tricky_ghost_mp2_full_harvested_as_mp2():
    """An MP2(Full) job ("RMP2-Full" in the archive) populates the MP2 qcvars."""
    in_mol = _make_water_mol()
    log = _g16_log(
        scf="-74.9630231626",
        method="RMP2-Full",
        body=" E2 =    -0.3554565274D-01 EUMP2 =    -0.74998568815099D+02\n",
        archive="\\MP2=-74.9985688",
    )
    with _patch_cclib(_make_ccdata_with_attrs()):
        qcvars, _, _, _, _ = harvest(in_mol, "mp2(full)", log)
    assert qcvars["MP2 TOTAL ENERGY"] == Decimal("-74.998568815099")
    assert qcvars["CURRENT ENERGY"] == qcvars["MP2 TOTAL ENERGY"]


# ---------------------------------------------------------------------------
# Tests: atom-label tolerance
# ---------------------------------------------------------------------------


def _build_labeled_h_mol(real_flags=None):
    """4 H atoms with the labels from test_atom_labels."""
    geom_bohr = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 5.0, 5.0, 0.0]
    return Molecule.from_data(
        {
            "symbols": ["H", "H", "H", "H"],
            "atom_labels": ["", "5", "_other", "_4sq"],
            "geometry": geom_bohr,
            "real": real_flags if real_flags is not None else [True, True, True, True],
            "molecular_charge": 0,
            "molecular_multiplicity": 1,
        }
    )


# rq-20063525
def test_atom_labels_single_labeled_h_emits_bare_symbol(harness):
    mol = Molecule.from_data(
        {
            "symbols": ["H"],
            "atom_labels": ["_other"],
            "geometry": [0.0, 0.0, 0.0],
            "real": [True],
            "molecular_charge": 0,
            "molecular_multiplicity": 2,
        }
    )
    inp = AtomicInput(molecule=mol, specification=_spec())
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        job = harness.build_input(inp, _default_taskconfig())
    com = job["infiles"]["gaussian.com"]
    atom_lines = [
        ln for ln in com.splitlines()
        if ln.lstrip().startswith("H(Iso=") or "_other" in ln
    ]
    assert any(ln.lstrip().startswith("H(Iso=") for ln in atom_lines)
    assert not any("_other" in ln for ln in atom_lines)


# rq-da7df6e9
def test_atom_labels_four_labeled_h_emit_bare_symbols(harness):
    mol = _build_labeled_h_mol()
    inp = AtomicInput(molecule=mol, specification=_spec(method="MP2", basis="aug-cc-pvdz"))
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        job = harness.build_input(inp, _default_taskconfig())
    com = job["infiles"]["gaussian.com"]
    h_lines = [ln for ln in com.splitlines() if ln.lstrip().startswith("H(Iso=")]
    assert len(h_lines) == 4
    for forbidden in ("H5", "H_other", "H_4sq"):
        assert forbidden not in com


# rq-0892bdc0 — ghost atom with non-empty label still emits "<sym>-Bq", label dropped
def test_atom_labels_ghost_with_label_emits_bq(harness):
    mol = Molecule.from_data(
        {
            "symbols": ["He", "H"],
            "atom_labels": ["", "ghost_a"],
            "geometry": [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
            "real": [True, False],
            "molecular_charge": 0,
            "molecular_multiplicity": 1,
        }
    )
    inp = AtomicInput(molecule=mol, specification=_spec())
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        job = harness.build_input(inp, _default_taskconfig())
    com = job["infiles"]["gaussian.com"]
    ghost_lines = [ln for ln in com.splitlines() if "H-Bq" in ln]
    assert len(ghost_lines) == 1
    assert "ghost_a" not in com


# rq-d4eaf3c7
@requires_cclib
def test_atom_labels_out_mol_preserves_labels_when_fixed():
    """When fix_com=True and fix_orientation=True, out_mol is in_mol → labels preserved."""
    mol = Molecule.from_data(
        {
            "symbols": ["H", "H", "H", "H"],
            "atom_labels": ["", "5", "_other", "_4sq"],
            "geometry": [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 5.0, 5.0, 0.0],
            "molecular_charge": 0,
            "molecular_multiplicity": 1,
            "fix_com": True,
            "fix_orientation": True,
        }
    )
    ccdata = _make_ccdata_with_attrs(
        scfenergies=np.array([-45.0]),
        atomcoords=np.array(
            [[[0.0, 0.0, 0.0], [2.645886, 0.0, 0.0], [0.0, 2.645886, 0.0], [2.645886, 2.645886, 0.0]]]
        ),
        atomnos=np.array([1, 1, 1, 1]),
    )
    with _patch_cclib(ccdata):
        _, _, _, out_mol, _ = harvest(mol, "hf", _g16_log(mol=mol))
    assert list(out_mol.atom_labels) == ["", "5", "_other", "_4sq"]


# ---------------------------------------------------------------------------
# Tests: empirical dispersion (germinate)
# ---------------------------------------------------------------------------


# rq-0c1b5066
def test_dispersion_germinate_d3():
    """muster_modelchem strips '-d3' and emits EmpiricalDispersion=GD3."""
    result = muster_modelchem("b3lyp-d3", "energy", 1)
    assert result["method_string"] == "B3LYP"
    assert result["extra_keywords"]["EmpiricalDispersion"] == "GD3"
    assert result["dispersion_level"] == "D3"
    assert result["functional_name"] == "B3LYP"


# rq-0c1b5066
def test_dispersion_germinate_d3bj():
    result = muster_modelchem("b3lyp-d3bj", "energy", 1)
    assert result["method_string"] == "B3LYP"
    assert result["extra_keywords"]["EmpiricalDispersion"] == "GD3BJ"
    assert result["dispersion_level"] == "D3(BJ)"


# rq-0c1b5066
def test_dispersion_germinate_d2():
    result = muster_modelchem("b3lyp-d2", "energy", 1)
    assert result["method_string"] == "B3LYP"
    assert result["extra_keywords"]["EmpiricalDispersion"] == "GD2"
    assert result["dispersion_level"] == "D2"


# rq-0c1b5066
def test_dispersion_germinate_d3zero2b_alias():
    result = muster_modelchem("b3lyp-d3zero2b", "energy", 1)
    assert result["dispersion_level"] == "D3"
    assert result["functional_name"] == "B3LYP"


# rq-0c1b5066
def test_dispersion_germinate_bare_d_alias():
    """The bare '-d' alias maps to D2 (per QCEngine's get_dispersion_aliases)."""
    result = muster_modelchem("b3lyp-d", "energy", 1)
    assert result["dispersion_level"] == "D2"


# rq-0c1b5066
def test_dispersion_germinate_no_suffix_unchanged():
    result = muster_modelchem("b3lyp", "energy", 1)
    assert result["method_string"] == "B3LYP"
    assert "EmpiricalDispersion" not in result["extra_keywords"]
    assert "dispersion_level" not in result


# rq-0c1b5066
def test_dispersion_germinate_hyphenated_non_dispersion_preserved():
    """cam-B3LYP must not get falsely stripped (cam isn't an alias)."""
    result = muster_modelchem("cam-b3lyp", "energy", 1)
    assert result["method_string"] == "CAM-B3LYP"
    assert "EmpiricalDispersion" not in result["extra_keywords"]
    assert "dispersion_level" not in result


# rq-0c1b5066
def test_dispersion_germinate_hf_d3():
    result = muster_modelchem("hf-d3", "energy", 1)
    assert result["method_string"] == "HF"
    assert result["extra_keywords"]["EmpiricalDispersion"] == "GD3"
    assert result["functional_name"] == "HF"


# rq-0c1b5066
def test_dispersion_germinate_open_shell():
    result = muster_modelchem("b3lyp-d3", "energy", 2)
    assert result["method_string"] == "UB3LYP"
    assert result["extra_keywords"]["EmpiricalDispersion"] == "GD3"
    assert result["dispersion_level"] == "D3"
    assert result["functional_name"] == "B3LYP"


# rq-0c1b5066
def test_dispersion_germinate_rohf_no_double_prefix():
    result = muster_modelchem("rohf-d3", "energy", 2)
    assert result["method_string"] == "ROHF"
    assert result["extra_keywords"]["EmpiricalDispersion"] == "GD3"


# rq-0c1b5066
@pytest.mark.parametrize("method", ["b3lyp-d3m", "b3lyp-d3mbj", "b3lyp-d4", "b3lyp-d3op", "b3lyp-nl"])
def test_dispersion_germinate_unsupported_levels_raise(method):
    with pytest.raises(InputError, match=r"not natively supported"):
        muster_modelchem(method, "energy", 1)


# rq-0c1b5066 — explicit "longer suffix wins" test
def test_dispersion_germinate_longer_suffix_wins():
    """d3bj must be matched before d3 + stray 'bj'."""
    result = muster_modelchem("b3lyp-d3bj", "energy", 1)
    assert result["dispersion_level"] == "D3(BJ)"
    assert result["functional_name"] == "B3LYP"


# ---------------------------------------------------------------------------
# Tests: empirical dispersion (runner conflict detection)
# ---------------------------------------------------------------------------


# rq-59b816b7
def test_dispersion_conflict_raises(harness, water):
    inp = AtomicInput(
        molecule=water,
        specification=AtomicSpecification(
            program="gaussian",
            driver="energy",
            model={"method": "b3lyp-d3", "basis": "STO-3G"},
            keywords={"EmpiricalDispersion": "GD3BJ"},
        ),
    )
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        with pytest.raises(InputError, match="overspecified"):
            harness.build_input(inp, _default_taskconfig())


# rq-59b816b7
def test_dispersion_conflict_case_insensitive(harness, water):
    inp = AtomicInput(
        molecule=water,
        specification=AtomicSpecification(
            program="gaussian",
            driver="energy",
            model={"method": "b3lyp-d3", "basis": "STO-3G"},
            keywords={"empiricaldispersion": "GD3"},
        ),
    )
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        with pytest.raises(InputError, match="overspecified"):
            harness.build_input(inp, _default_taskconfig())


# rq-59b816b7
def test_dispersion_user_keyword_only_passes_through(harness, water):
    """If the method has no dispersion suffix, a user EmpiricalDispersion kw passes through."""
    inp = AtomicInput(
        molecule=water,
        specification=AtomicSpecification(
            program="gaussian",
            driver="energy",
            model={"method": "B3LYP", "basis": "STO-3G"},
            keywords={"EmpiricalDispersion": "GD3"},
        ),
    )
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        job = harness.build_input(inp, _default_taskconfig())
    assert "EmpiricalDispersion=GD3" in job["infiles"]["gaussian.com"]


# rq-59b816b7
def test_dispersion_method_suffix_only(harness, water):
    """method='b3lyp-d3' produces exactly one EmpiricalDispersion token and bare B3LYP/basis."""
    inp = AtomicInput(
        molecule=water,
        specification=AtomicSpecification(
            program="gaussian",
            driver="energy",
            model={"method": "b3lyp-d3", "basis": "STO-3G"},
        ),
    )
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        job = harness.build_input(inp, _default_taskconfig())
    com = job["infiles"]["gaussian.com"]
    assert com.count("EmpiricalDispersion=GD3") == 1
    # Route line has B3LYP/STO-3G, NOT B3LYP-D3/STO-3G
    route = next(ln for ln in com.splitlines() if ln.startswith("#P"))
    assert "B3LYP/STO-3G" in route
    assert "B3LYP-D3/" not in route


# ---------------------------------------------------------------------------
# Tests: empirical dispersion (harvester)
# ---------------------------------------------------------------------------

# B3LYP-D3(BJ)/6-31G* water: Gaussian's SCF Done energy already includes the
# dispersion term printed on the "Dispersion energy=" line.
_DISP_SCF = "-76.4093047918"
_DISP_D = "-0.0005738757"
_DISP_LOG = _g16_log(
    scf=_DISP_SCF,
    scf_label="RB3LYP",
    pre=f" R6Disp:  Grimme-D3(BJ) Dispersion energy=       {_DISP_D} Hartrees.\n",
)


# rq-1d5da737
def test_dispersion_harvest_not_double_counted():
    qcvars = harvest_energies(_DISP_LOG)
    assert qcvars["CURRENT ENERGY"] == Decimal(_DISP_SCF)
    assert qcvars["SCF TOTAL ENERGY"] == Decimal(_DISP_SCF)
    assert qcvars["DFT TOTAL ENERGY"] == Decimal(_DISP_SCF)
    assert qcvars["DISPERSION CORRECTION ENERGY"] == Decimal(_DISP_D)
    assert qcvars["DFT FUNCTIONAL TOTAL ENERGY"] == Decimal(_DISP_SCF) - Decimal(_DISP_D)


# rq-1d5da737
def test_dispersion_harvest_functional_qcvars_from_suffix():
    qcvars = harvest_energies(_DISP_LOG, functional_label="B3LYP", dispersion_level="D3(BJ)")
    assert qcvars["B3LYP-D3(BJ) DISPERSION CORRECTION ENERGY"] == Decimal(_DISP_D)
    assert qcvars["B3LYP-D3(BJ) TOTAL ENERGY"] == Decimal(_DISP_SCF)
    assert qcvars["B3LYP FUNCTIONAL TOTAL ENERGY"] == Decimal(_DISP_SCF) - Decimal(_DISP_D)


# rq-1d5da737
def test_dispersion_harvest_no_suffix_generic_qcvars_only():
    """Dispersion from a user EmpiricalDispersion keyword: no formal label is known."""
    qcvars = harvest_energies(_DISP_LOG)
    assert not any(k.endswith("DISPERSION CORRECTION ENERGY") and k != "DISPERSION CORRECTION ENERGY" for k in qcvars)


# rq-1d5da737
def test_dispersion_harvest_absent_no_qcvars():
    qcvars = harvest_energies(_g16_log(scf="-76.4087309161", scf_label="RB3LYP"))
    assert "DISPERSION CORRECTION ENERGY" not in qcvars


# rq-1d5da737
@requires_cclib
def test_dispersion_harvest_suffix_reaches_qcvars():
    in_mol = _make_water_mol()
    with _patch_cclib(_make_ccdata_with_attrs()):
        qcvars, _, _, _, _ = harvest(in_mol, "b3lyp-d3bj", _DISP_LOG)
    assert "B3LYP-D3(BJ) TOTAL ENERGY" in qcvars


# rq-1d5da737
@requires_cclib
def test_dispersion_harvest_gradient_unchanged():
    """Gaussian's printed forces are the total (dispersion-inclusive) force."""
    in_mol = _make_water_mol()
    forces = np.array([[0.01, 0.0, -0.02], [0.0, 0.005, 0.01], [0.0, -0.005, 0.01]])
    log = _g16_log(scf=_DISP_SCF, scf_label="RB3LYP", pre=f" R6Disp:  Grimme-D3(BJ) Dispersion energy=       {_DISP_D} Hartrees.\n", forces=forces)
    with _patch_cclib(_make_ccdata_with_attrs()):
        _, grad, _, _, _ = harvest(in_mol, "b3lyp-d3bj", log)
    np.testing.assert_allclose(grad, -forces.flatten(), atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: punch file (Punch=Derivatives)
# ---------------------------------------------------------------------------


def _punch(values):
    return " archive copy\\\\@\n" + "\n".join(f"   {v:.10E}".replace("E", "D") for v in values) + "\n"


def test_parse_punch_gradient_and_hessian():
    grad = np.arange(1, 4, dtype=float) * 0.01
    tri = np.arange(1, 7, dtype=float) * 0.1  # 3x3 lower triangle, row-wise
    g, h = _parse_punch(_punch(list(grad) + list(tri)), 1)
    np.testing.assert_allclose(g, grad.reshape(1, 3))
    np.testing.assert_allclose(h, [[0.1, 0.2, 0.4], [0.2, 0.3, 0.5], [0.4, 0.5, 0.6]])


def test_parse_punch_archive_wrapped_inside_terminator():
    """The archive copy can wrap between the two characters of its closing "\\\\@"."""
    grad = [0.01, 0.02, 0.03]
    punch = "1\\1\\GINC-HOST\\SP\\RHF\\STO-3G\\0,1\\H,0,0.,0.,0.\\\\\n@\n" + _punch(grad).split("\n", 1)[1]
    g, h = _parse_punch(punch, 1)
    np.testing.assert_allclose(g, [grad])
    assert h is None


def test_parse_punch_without_derivatives():
    assert _parse_punch(" archive copy\\\\@\n", 1) == (None, None)


def test_parse_punch_bad_count_raises():
    with pytest.raises(ValueError, match="Unexpected number"):
        _parse_punch(_punch([0.1, 0.2]), 1)


@requires_cclib
def test_harvest_hessian_prefers_punch_file():
    in_mol = _make_water_mol()
    hess = np.arange(81, dtype=float).reshape(9, 9) * 1.0e-3
    hess = hess + hess.T
    punch = _punch(list(np.zeros(9)) + list(hess[np.tril_indices(9)]))
    with _patch_cclib(_make_ccdata_with_attrs()):
        _, _, out_hess, _, _ = harvest(in_mol, "hf", _g16_log(mol=in_mol, ), punch)
    np.testing.assert_allclose(out_hess, hess)


# ---------------------------------------------------------------------------
# Integration tests (require Gaussian + cclib installed)
# ---------------------------------------------------------------------------


@using("gaussian")
def test_gaussian_energy_water(water):
    """End-to-end HF/STO-3G energy on water via the actual Gaussian executable."""
    import qcengine as qcng

    result = qcng.compute(
        AtomicInput(molecule=water, specification=AtomicSpecification(program="gaussian", driver="energy", model={"method": "HF", "basis": "STO-3G"})),
        "gaussian",
        raise_error=True,
    )
    assert result.success is True
    assert result.return_result < 0.0
    assert "HF TOTAL ENERGY" in result.extras["qcvars"]


@using("gaussian")
def test_gaussian_gradient_water(water):
    """End-to-end HF/STO-3G gradient on water via the actual Gaussian executable."""
    import qcengine as qcng

    result = qcng.compute(
        AtomicInput(molecule=water, specification=AtomicSpecification(program="gaussian", driver="gradient", model={"method": "HF", "basis": "STO-3G"})),
        "gaussian",
        raise_error=True,
    )
    assert result.success is True
    assert np.array(result.return_result).shape == (3, 3)


# rq-17959824
@using("gaussian")
def test_gaussian_b3lyp_d3_water(water):
    """B3LYP-D3/6-31G* energy on water; verify dispersion qcvars are populated and consistent."""
    import qcengine as qcng

    result = qcng.compute(
        AtomicInput(
            molecule=water,
            specification=AtomicSpecification(
                program="gaussian",
                driver="energy",
                model={"method": "b3lyp-d3", "basis": "6-31g*"},
            ),
        ),
        "gaussian",
        raise_error=True,
    )
    assert result.success is True

    qcvars = result.extras["qcvars"]
    # All three Psi4-convention qcvars must be present.
    assert "B3LYP-D3 DISPERSION CORRECTION ENERGY" in qcvars
    assert "B3LYP-D3 TOTAL ENERGY" in qcvars
    assert "B3LYP FUNCTIONAL TOTAL ENERGY" in qcvars
    # Empirical reference values (Gaussian 16). Gaussian's SCF Done energy
    # (-76.4087263383) already includes the dispersion term; plain B3LYP/6-31G*
    # is -76.4087184463. Tolerance absorbs patch-level D3 parameter variation.
    atol = 1.0e-6
    ref_total = -76.4087263383
    ref_disp = -0.0000078919
    ref_func = ref_total - ref_disp
    assert abs(float(qcvars["B3LYP FUNCTIONAL TOTAL ENERGY"]) - ref_func) < atol
    assert abs(float(qcvars["B3LYP-D3 DISPERSION CORRECTION ENERGY"]) - ref_disp) < atol
    assert abs(float(qcvars["B3LYP-D3 TOTAL ENERGY"]) - ref_total) < atol
    # functional + dispersion == total (numerical consistency).
    func = float(qcvars["B3LYP FUNCTIONAL TOTAL ENERGY"])
    disp = float(qcvars["B3LYP-D3 DISPERSION CORRECTION ENERGY"])
    tot = float(qcvars["B3LYP-D3 TOTAL ENERGY"])
    assert abs((func + disp) - tot) < 1.0e-10
    # CURRENT ENERGY and SCF TOTAL ENERGY are the dispersion-corrected total (Psi4 convention).
    assert abs(float(qcvars["CURRENT ENERGY"]) - tot) < 1.0e-10
    assert abs(float(qcvars["SCF TOTAL ENERGY"]) - tot) < 1.0e-10
    assert "HF TOTAL ENERGY" not in qcvars


# ---------------------------------------------------------------------------
# E2E coverage tests (see rqm/gaussian/e2e-coverage.md)
# ---------------------------------------------------------------------------


# rq-54afb7b8
@using("gaussian")
def test_gaussian_hessian_hf_water(water):
    """HF/STO-3G Hessian on water — exercises _parse_force_constants pipeline end-to-end."""
    import qcengine as qcng

    inp = AtomicInput(
        molecule=water,
        specification=AtomicSpecification(
            program="gaussian", driver="hessian",
            model={"method": "HF", "basis": "STO-3G"},
        ),
    )
    result = qcng.compute(inp, "gaussian", raise_error=True)
    assert result.success is True

    hess = np.asarray(result.return_result)
    atol = 1.0e-6
    assert hess.shape == (9, 9)
    assert abs(np.trace(hess) - 3.2315230546) < atol
    assert abs(np.linalg.norm(hess) - 2.2444454404) < atol
    assert abs(hess[0, 0] - (-0.0560540960)) < atol
    assert abs(hess[3, 3] - (-0.0223351423)) < atol
    assert abs(hess[0, 1]) < 1.0e-10
    assert np.allclose(hess, hess.T, atol=1.0e-10)


# rq-70b16c33
@using("gaussian")
def test_gaussian_properties_hf_water(water):
    """HF/STO-3G properties driver: dipole + Mulliken end-to-end."""
    import qcengine as qcng

    inp = AtomicInput(
        molecule=water,
        specification=AtomicSpecification(
            program="gaussian", driver="properties",
            model={"method": "HF", "basis": "STO-3G"},
        ),
    )
    result = qcng.compute(inp, "gaussian", raise_error=True)
    assert result.success is True
    assert isinstance(result.return_result, float)

    qcvars = result.extras["qcvars"]
    atol = 1.0e-6
    assert abs(float(qcvars["HF TOTAL ENERGY"]) - (-74.962963201)) < atol
    # Dipole magnitude (Debye) — looser tol absorbs Gaussian rounding in the log.
    np.testing.assert_allclose(result.properties.scf_dipole_moment, [0.0, 0.0, -0.6787514], atol=1.0e-7)
    mulliken = list(qcvars["MULLIKEN CHARGES"])
    assert len(mulliken) == 3
    assert abs(sum(float(q) for q in mulliken)) < 1.0e-4
    assert abs(float(mulliken[0]) - (-0.366114)) < 1.0e-3


# rq-3727e53b
@using("gaussian")
def test_gaussian_dft_b3lyp_water(water):
    """Plain B3LYP/cc-pVDZ (no dispersion suffix) — confirms the DFT pass-through branch."""
    import qcengine as qcng

    inp = AtomicInput(
        molecule=water,
        specification=AtomicSpecification(
            program="gaussian", driver="energy",
            model={"method": "B3LYP", "basis": "cc-pVDZ"},
        ),
    )
    result = qcng.compute(inp, "gaussian", raise_error=True)
    assert result.success is True

    atol = 1.0e-6
    ref = -76.4203540604
    assert abs(float(result.return_result) - ref) < atol

    qcvars = result.extras["qcvars"]
    # The SCF energy IS the B3LYP energy when no dispersion is requested.
    assert abs(float(qcvars["DFT TOTAL ENERGY"]) - ref) < 1.0e-10
    assert abs(float(qcvars["CURRENT ENERGY"]) - ref) < 1.0e-10
    assert "HF TOTAL ENERGY" not in qcvars
    # No dispersion qcvars when the method has no dispersion suffix.
    assert "DISPERSION CORRECTION ENERGY" not in qcvars
    assert "B3LYP-D3 DISPERSION CORRECTION ENERGY" not in qcvars


# rq-66ca9a20
@using("gaussian")
def test_gaussian_b3lyp_d3_gradient_water(water):
    """B3LYP-D3/6-31G* gradient — confirms the dispersion + gradient combination."""
    import qcengine as qcng

    inp = AtomicInput(
        molecule=water,
        specification=AtomicSpecification(
            program="gaussian", driver="gradient",
            model={"method": "b3lyp-d3", "basis": "6-31g*"},
        ),
    )
    result = qcng.compute(inp, "gaussian", raise_error=True)
    assert result.success is True

    grad = np.asarray(result.return_result)
    assert grad.shape == (3, 3)
    ref_grad = np.array([
        [0.0,  0.0,        -0.01437422],
        [0.0, -0.00800138,  0.00718711],
        [0.0,  0.00800138,  0.00718711],
    ])
    np.testing.assert_allclose(grad, ref_grad, atol=1.0e-6)

    qcvars = result.extras["qcvars"]
    atol = 1.0e-6
    assert abs(float(qcvars["B3LYP-D3 DISPERSION CORRECTION ENERGY"]) - (-0.0000078919)) < atol
    assert abs(float(qcvars["B3LYP FUNCTIONAL TOTAL ENERGY"]) - (-76.4087184464)) < atol
    assert abs(float(qcvars["B3LYP-D3 TOTAL ENERGY"]) - (-76.4087263383)) < atol


# rq-d06d7266
@using("gaussian")
def test_gaussian_uhf_nh2():
    """NH2 radical (charge=0, mult=2) — confirms U-prefix auto-emission for open-shell."""
    import qcengine as qcng

    nh2 = Molecule.from_data(
        """
        0 2
        N  0.000000000  0.000000000  0.140030000
        H  0.000000000  0.802300000 -0.490100000
        H  0.000000000 -0.802300000 -0.490100000
        units angstrom
        """
    )
    inp = AtomicInput(
        molecule=nh2,
        specification=AtomicSpecification(
            program="gaussian", driver="energy",
            model={"method": "HF", "basis": "cc-pVDZ"},
            protocols={"native_files": "all"},
        ),
    )
    result = qcng.compute(inp, "gaussian", raise_error=True)
    assert result.success is True

    atol = 1.0e-6
    assert abs(float(result.return_result) - (-55.5671259096)) < atol

    qcvars = result.extras["qcvars"]
    assert abs(float(qcvars["HF TOTAL ENERGY"]) - (-55.5671259096)) < atol

    com = result.native_files["input"]
    assert "UHF/cc-pVDZ" in com
    # Charge/multiplicity line — look for an exact "0 2" line in the molecule block.
    assert any(ln.strip() == "0 2" for ln in com.splitlines())


# rq-cd65e68a
@using("gaussian")
def test_gaussian_single_atom_he():
    """Single He atom HF/cc-pVDZ — edge case: zero NRE, no bonds."""
    import qcengine as qcng

    he = Molecule.from_data(
        """
        0 1
        He 0.0 0.0 0.0
        units angstrom
        """
    )
    inp = AtomicInput(
        molecule=he,
        specification=AtomicSpecification(
            program="gaussian", driver="energy",
            model={"method": "HF", "basis": "cc-pVDZ"},
        ),
    )
    result = qcng.compute(inp, "gaussian", raise_error=True)
    assert result.success is True
    assert abs(float(result.return_result) - (-2.85516047724)) < 1.0e-6
    assert result.properties.nuclear_repulsion_energy == 0.0


# rq-b8065553
@using("gaussian")
def test_gaussian_b3lyp_d3_hessian_water(water):
    """B3LYP-D3/6-31G* Hessian — confirms DFT-D Hessian succeeds and emits dispersion qcvar."""
    import qcengine as qcng

    inp = AtomicInput(
        molecule=water,
        specification=AtomicSpecification(
            program="gaussian", driver="hessian",
            model={"method": "b3lyp-d3", "basis": "6-31g*"},
        ),
    )
    result = qcng.compute(inp, "gaussian", raise_error=True)
    assert result.success is True

    hess = np.asarray(result.return_result)
    atol = 1.0e-6
    assert hess.shape == (9, 9)
    assert abs(np.trace(hess) - 2.3626048438) < atol
    assert abs(np.linalg.norm(hess) - 1.5979933610) < atol
    assert np.allclose(hess, hess.T, atol=1.0e-10)

    qcvars = result.extras["qcvars"]
    assert abs(float(qcvars["B3LYP-D3 DISPERSION CORRECTION ENERGY"]) - (-0.0000078919)) < atol


# ---------------------------------------------------------------------------
# Basis Set Exchange ("bse:" basis) support
# ---------------------------------------------------------------------------


def _hi_mol(ghost_iodine=False, extra_ghost_hi=False):
    """HI, optionally with the I made a ghost or a ghost HI appended."""
    symbols = ["H", "I"]
    geometry = [0.0, 0.0, 0.0, 0.0, 0.0, 3.04]
    real = [True, not ghost_iodine]
    if extra_ghost_hi:
        symbols += ["H", "I"]
        geometry += [0.0, 0.0, 8.0, 0.0, 0.0, 11.04]
        real += [False, False]
    return Molecule.from_data(
        {
            "symbols": symbols,
            "geometry": geometry,
            "real": real,
            "molecular_charge": 0,
            "molecular_multiplicity": 2 if ghost_iodine else 1,
        }
    )


def _bse_com(harness, mol, basis, keywords=None):
    inp = AtomicInput(molecule=mol, specification=AtomicSpecification(program="gaussian", driver="energy", model={"method": "HF", "basis": basis}, keywords=keywords or {}))
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        return harness.build_input(inp, _default_taskconfig())["infiles"]["gaussian.com"]


@pytest.mark.parametrize(
    "basis, expected",
    [
        ("bse:cc-pvdz", ("cc-pvdz", None)),
        ("BSE:cc-pVDZ:1", ("cc-pVDZ", "1")),
        ("bse:6-31G*:", ("6-31G*", None)),
    ],
)
def test_parse_bse_basis(basis, expected):
    assert parse_bse_basis(basis) == expected


@pytest.mark.parametrize("basis", ["bse:", "bse:cc-pvdz:1:2"])
def test_parse_bse_basis_invalid_raises(basis):
    with pytest.raises(InputError, match="Invalid BSE basis"):
        parse_bse_basis(basis)


def test_bse_cartesian_flags_mixed_raises():
    basis_dict = {
        "elements": {
            "1": {"electron_shells": [{"angular_momentum": [2], "function_type": "gto_spherical"}]},
            "8": {"electron_shells": [{"angular_momentum": [2], "function_type": "gto_cartesian"}]},
        }
    }
    with pytest.raises(InputError, match="mixes Cartesian and spherical d"):
        _cartesian_flags(basis_dict, ["1", "8"])


def test_bse_cartesian_flags_none_without_shells():
    basis_dict = {"elements": {"1": {"electron_shells": [{"angular_momentum": [0], "function_type": "gto"}]}}}
    assert _cartesian_flags(basis_dict, ["1"]) == (None, None)


@using("basis_set_exchange")
def test_bse_build_input_spherical(harness, water):
    com = _bse_com(harness, water, "bse:cc-pvdz")
    lines = com.splitlines()
    assert [ln for ln in lines if ln.startswith("#P")] == ["#P HF/Gen 5D"]
    # Only elements in the molecule are written, after the molecule block.
    assert "O     0" in lines and "H     0" in lines
    assert "He     0" not in lines
    assert "! Basis set: cc-pVDZ  Version: 1" in lines
    assert lines.index("O     0") > max(i for i, ln in enumerate(lines) if ln.lstrip().startswith("H "))
    # The basis section is terminated by exactly one blank line.
    assert com.endswith("****\n\n")


@using("basis_set_exchange")
def test_bse_build_input_cartesian(harness, water):
    com = _bse_com(harness, water, "bse:6-31G*")
    assert [ln for ln in com.splitlines() if ln.startswith("#P")] == ["#P HF/Gen 6D"]


@using("basis_set_exchange")
def test_bse_route_keywords_transition_metal():
    # 6-31G* on Zn: Cartesian d, spherical f
    result = build_bse_basis("bse:6-31G*", ["Zn", "O"], [True, True])
    assert result["route_basis"] == "Gen"
    assert result["route_keywords"] == ["6D", "7F"]


@using("basis_set_exchange")
def test_bse_build_input_ecp(harness):
    com = _bse_com(harness, _hi_mol(), "bse:def2-svp")
    assert [ln for ln in com.splitlines() if ln.startswith("#P")] == ["#P HF/GenECP 5D"]
    assert "I-ECP" in com
    assert com.endswith("\n\n") and not com.endswith("\n\n\n")


@using("basis_set_exchange")
def test_bse_ghost_only_element_has_no_ecp(harness):
    com = _bse_com(harness, _hi_mol(ghost_iodine=True), "bse:def2-svp")
    assert [ln for ln in com.splitlines() if ln.startswith("#P")] == ["#P UHF/Gen 5D"]
    assert "I     0" in com.splitlines()
    assert "I-ECP" not in com


@using("basis_set_exchange")
def test_bse_mixed_real_and_ghost_element_keeps_ecp(harness):
    com = _bse_com(harness, _hi_mol(extra_ghost_hi=True), "bse:def2-svp")
    assert [ln for ln in com.splitlines() if ln.startswith("#P")] == ["#P HF/GenECP 5D"]
    assert "I-ECP" in com


@using("basis_set_exchange")
@pytest.mark.parametrize(
    "basis, match",
    [
        ("bse:not-a-basis", "does not exist"),
        ("bse:cc-pvdz:99", "Version 99 does not exist"),
        ("bse:6-31G(2df,p)", "not found in basis"),
    ],
)
def test_bse_unavailable_basis_raises(harness, basis, match):
    mol = Molecule.from_data("K 0 0 0\nH 0 0 2.2") if "2df" in basis else Molecule.from_data("He 0 0 0")
    with pytest.raises(InputError, match=match):
        _bse_com(harness, mol, basis)


@using("basis_set_exchange")
@pytest.mark.parametrize("keyword", ["6D", "5d", "10F", "7f"])
def test_bse_user_pure_cartesian_keyword_raises(harness, water, keyword):
    with pytest.raises(InputError, match="conflicts with basis"):
        _bse_com(harness, water, "bse:cc-pvdz", keywords={keyword: ""})


def test_bse_missing_library_raises(harness, water):
    with patch.dict("sys.modules", {"basis_set_exchange": None}):
        with pytest.raises(ModuleNotFoundError, match="basis_set_exchange"):
            _bse_com(harness, water, "bse:cc-pvdz")


def test_build_com_file_with_basis_block():
    com = build_com_file(
        link0={"NProcShared": 1, "Mem": "1GB"},
        route_line="#P HF/Gen",
        title="t",
        charge=0,
        multiplicity=1,
        atom_block="He  0.0  0.0  0.0",
        basis_block="He     0\nS    1   1.00\n      1.0   1.0\n****",
    )
    assert com.endswith("He  0.0  0.0  0.0\n\nHe     0\nS    1   1.00\n      1.0   1.0\n****\n\n")


@using("gaussian")
@using("basis_set_exchange")
@pytest.mark.parametrize(
    "bse_basis, builtin_basis, driver",
    [
        ("bse:cc-pvdz", "cc-pVDZ", "gradient"),
        ("bse:6-31G*", "6-31G*", "energy"),
    ],
)
def test_gaussian_bse_matches_builtin_water(water, bse_basis, builtin_basis, driver):
    """A BSE basis written as Gen reproduces Gaussian's built-in definition."""
    import qcengine as qcng

    def run(basis):
        inp = AtomicInput(molecule=water, specification=AtomicSpecification(program="gaussian", driver=driver, model={"method": "HF", "basis": basis}))
        return qcng.compute(inp, "gaussian", raise_error=True)

    bse_res, ref_res = run(bse_basis), run(builtin_basis)
    assert bse_res.properties.calcinfo_nbasis == ref_res.properties.calcinfo_nbasis
    assert abs(bse_res.properties.return_energy - ref_res.properties.return_energy) < 1.0e-9
    assert np.allclose(bse_res.return_result, ref_res.return_result, atol=1.0e-8)


@using("gaussian")
@using("basis_set_exchange")
@pytest.mark.parametrize("extra_ghost_hi", [False, True])
def test_gaussian_bse_ecp_matches_builtin(extra_ghost_hi):
    """def2-SVP on HI (with an ECP on I), with and without a ghost HI."""
    import qcengine as qcng

    mol = _hi_mol(extra_ghost_hi=extra_ghost_hi)

    def run(basis):
        inp = AtomicInput(molecule=mol, specification=AtomicSpecification(program="gaussian", driver="energy", model={"method": "HF", "basis": basis}))
        return qcng.compute(inp, "gaussian", raise_error=True)

    bse_res, ref_res = run("bse:def2-svp"), run("def2SVP")
    assert bse_res.properties.calcinfo_nbasis == ref_res.properties.calcinfo_nbasis
    # BSE and Gaussian's built-in def2-SVP coefficients differ in rounding
    assert abs(bse_res.return_result - ref_res.return_result) < 1.0e-7


@using("gaussian")
@using("psi4")
@using("basis_set_exchange")
def test_gaussian_bse_matches_psi4(water):
    """pcseg-1 is not built into Gaussian; both programs read it from the BSE."""
    import qcengine as qcng

    def run(program, keywords):
        inp = AtomicInput(molecule=water, specification=AtomicSpecification(program=program, driver="energy", model={"method": "HF", "basis": "bse:pcseg-1"}, keywords=keywords))
        return qcng.compute(inp, program, raise_error=True)

    gau = run("gaussian", {})
    psi = run("psi4", {"scf_type": "pk"})
    assert abs(gau.return_result - psi.return_result) < 1.0e-8


# ---------------------------------------------------------------------------
# Route keyword formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "keywords, expected",
    [
        ({"NoSymm": None}, "NoSymm"),
        ({"NoSymm": True}, "NoSymm"),
        ({"SCF": ["Tight", "XQC"]}, "SCF=(Tight,XQC)"),
        ({"Integral": {"Grid": "UltraFine", "Acc2E": 12}}, "Integral=(Grid=UltraFine,Acc2E=12)"),
        ({"Integral": {"NoXCTest": None, "Grid": 99590}}, "Integral=(NoXCTest,Grid=99590)"),
        ({"MaxDisk": 0}, "MaxDisk=0"),
    ],
)
def test_build_route_line_keyword_formats(keywords, expected):
    assert build_route_line("HF", "STO-3G", "", {}, keywords) == f"#P HF/STO-3G {expected}"


def test_build_route_line_false_value_raises():
    with pytest.raises(InputError, match="no off form"):
        build_route_line("HF", "STO-3G", "", {}, {"NoSymm": False})


@pytest.mark.parametrize("keyword", ["Freq", "opt", "Force", "IRC", "Scan"])
def test_build_route_line_job_type_keyword_raises(keyword):
    with pytest.raises(InputError, match="job type"):
        build_route_line("HF", "STO-3G", "Force=NoStep", {}, {keyword: ""})


def test_build_route_line_user_keyword_replaces_harness_extra():
    line = build_route_line("HF", "STO-3G", "", {"Pop": "Full"}, {"pop": "NBO"})
    assert line == "#P HF/STO-3G pop=NBO"


def test_build_input_hessian_requests_punch(harness, water):
    inp = AtomicInput(molecule=water, specification=_spec(driver="hessian"))
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        com = harness.build_input(inp, _default_taskconfig())["infiles"]["gaussian.com"]
    assert [ln for ln in com.splitlines() if ln.startswith("#P")] == ["#P HF/STO-3G Freq Punch=Derivatives"]


# ---------------------------------------------------------------------------
# End-to-end: method coverage, frames, errors
# ---------------------------------------------------------------------------

_PSI4_FC = {"scf_type": "pk", "freeze_core": True, "d_convergence": 10, "e_convergence": 10}


def _cc_water():
    return Molecule.from_data("O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692")


@using("gaussian")
@using("psi4")
@pytest.mark.parametrize(
    "method, psi4_keywords",
    [
        ("mp3", {"mp_type": "conv"}),
        ("mp4", {"qc_module": "fnocc"}),
        ("qcisd(t)", {"qc_module": "fnocc"}),
        ("ccsd(t)", {}),
        ("cisd", {"qc_module": "detci"}),
    ],
)
def test_gaussian_post_hf_matches_psi4(method, psi4_keywords):
    """Gaussian's default frozen core, matched in Psi4."""
    import qcengine as qcng

    def run(program, keywords):
        inp = AtomicInput(molecule=_cc_water(), specification=AtomicSpecification(program=program, driver="energy", model={"method": method, "basis": "cc-pvdz"}, keywords=keywords))
        return qcng.compute(inp, program, raise_error=True)

    assert abs(run("gaussian", {}).return_result - run("psi4", {**_PSI4_FC, **psi4_keywords}).return_result) < 1.0e-7


@using("gaussian")
def test_gaussian_double_hybrid_total_energy():
    """B2PLYP returns the double-hybrid total, not the SCF part."""
    import qcengine as qcng

    inp = AtomicInput(molecule=_cc_water(), specification=AtomicSpecification(program="gaussian", driver="energy", model={"method": "b2plyp", "basis": "sto-3g"}))
    qcvars = qcng.compute(inp, "gaussian", raise_error=True).extras["qcvars"]
    assert abs(float(qcvars["CURRENT ENERGY"]) - (-75.210879736849)) < 1.0e-8
    assert abs(float(qcvars["SCF TOTAL ENERGY"]) - (-75.1988324129)) < 1.0e-8


@using("gaussian")
def test_gaussian_unsupported_method_raises():
    import qcengine as qcng

    inp = AtomicInput(molecule=_cc_water(), specification=AtomicSpecification(program="gaussian", driver="energy", model={"method": "cid", "basis": "sto-3g"}))
    with pytest.raises(InputError, match="RCID"):
        qcng.compute(inp, "gaussian", raise_error=True)


@using("gaussian")
def test_gaussian_numerical_hessian_energy_is_reference():
    """CCSD Freq differentiates numerically; the energy must be the reference-geometry one."""
    import qcengine as qcng

    def run(driver):
        inp = AtomicInput(molecule=_cc_water(), specification=AtomicSpecification(program="gaussian", driver=driver, model={"method": "ccsd", "basis": "sto-3g"}))
        return qcng.compute(inp, "gaussian", raise_error=True)

    assert abs(run("hessian").properties.return_energy - run("energy").properties.return_energy) < 1.0e-8


@using("gaussian")
@using("psi4")
@pytest.mark.parametrize("driver", ["gradient", "hessian"])
def test_gaussian_rotated_molecule_matches_psi4(driver):
    """Off-axis molecule in a fixed frame: derivatives and dipole come back in the input frame."""
    import qcengine as qcng

    mol = Molecule.from_data("O 0.31 -0.42 0.27\nH 1.12 -0.05 0.71\nH -0.29 0.19 0.68\nunits angstrom\nno_com\nno_reorient")

    def run(program, keywords):
        inp = AtomicInput(molecule=mol, specification=AtomicSpecification(program=program, driver=driver, model={"method": "hf", "basis": "6-31g*"}, keywords=keywords))
        return qcng.compute(inp, program, raise_error=True)

    gau, psi = run("gaussian", {}), run("psi4", {"scf_type": "pk", "d_convergence": 10, "e_convergence": 10})
    np.testing.assert_allclose(gau.return_result, psi.return_result, atol=5.0e-8)
    np.testing.assert_allclose(gau.properties.scf_dipole_moment, psi.properties.scf_dipole_moment, atol=1.0e-6)


@using("gaussian")
@pytest.mark.parametrize(
    "molstr, basis, keywords, error",
    [
        ("O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692", "6-31g", {"SCF": {"MaxCycle": 2}}, ConvergenceError),
        ("H 0 0 0\nI 0 0 1.61", "6-31g", {}, InputError),
        ("O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692", "sto-3g", {"NotAKeyword": ""}, InputError),
    ],
)
def test_gaussian_real_errors_classified(molstr, basis, keywords, error):
    import qcengine as qcng

    inp = AtomicInput(molecule=Molecule.from_data(molstr), specification=AtomicSpecification(program="gaussian", driver="energy", model={"method": "hf", "basis": basis}, keywords=keywords))
    with pytest.raises(error):
        qcng.compute(inp, "gaussian", raise_error=True)


# ---------------------------------------------------------------------------
# Harmonic vibrational analysis (VIBRATIONS qcvar)
# ---------------------------------------------------------------------------

# HF/6-31G* stationary point
_WATER_HF_631GS_EQ = """
O  0.0000000000  0.0000000000  0.1089388120
H  0.0000000000  0.7540663912 -0.4644694052
H  0.0000000000 -0.7540663912 -0.4644694052
units angstrom
"""


@requires_cclib
def test_harvest_dipole_derivatives_from_archive():
    in_mol = _make_water_mol()
    values = np.arange(27, dtype=float) * 0.01
    archive = "\\DipoleDeriv=" + ",".join(f"{v:.7f}" for v in values)
    with _patch_cclib(_make_ccdata_with_attrs()):
        *_, dipder = harvest(in_mol, "hf", _g16_log(mol=in_mol, archive=archive))
    # row per Cartesian coordinate, column per dipole component (as Psi4's CURRENT DIPOLE GRADIENT)
    np.testing.assert_allclose(dipder, values.reshape(9, 3))


@requires_cclib
def test_harvest_no_dipole_derivatives_without_freq():
    with _patch_cclib(_make_ccdata_with_attrs()):
        *_, dipder = harvest(_make_water_mol(), "hf", _g16_log())
    assert dipder is None


def test_build_vibrations_layout_and_derived_modes():
    import qcelemental as qcel

    from qcengine.programs.util.vibrations import build_vibrations

    masses = [16.0, 1.0]
    x = np.zeros((6, 2))
    x[2, 0], x[5, 0] = 0.6, -0.8  # stretch
    x[1, 1], x[4, 1] = -0.2, 0.9797958971  # arbitrary second mode
    vib = build_vibrations([-50.0, 3000.0], x, masses=masses, trv=["V", "V"], gamma=[None, "A1"])
    assert list(vib)[:3] == ["omega", "omega_in_wavenumbers", "x"]
    np.testing.assert_allclose(vib["omega"], np.array([-50.0, 3000.0]) / qcel.constants.hartree2wavenumbers)
    assert vib["trv"] == ["v", "v"]
    assert vib["gamma"] == ["-", "A1"]
    # psi4 definitions: q normalized mass-weighted, w = q / sqrt(m), x = w / |w|
    q, w = np.asarray(vib["q"]), np.asarray(vib["w"])
    np.testing.assert_allclose(np.linalg.norm(q, axis=0), 1.0)
    np.testing.assert_allclose(w / np.linalg.norm(w, axis=0), x / np.linalg.norm(x, axis=0), atol=1.0e-12)


def test_build_vibrations_validates_shapes():
    from qcengine.programs.util.vibrations import build_vibrations

    with pytest.raises(ValueError, match="x has shape"):
        build_vibrations([1.0, 2.0], np.zeros((3, 1)))
    with pytest.raises(ValueError, match="values of mu"):
        build_vibrations([1.0], np.ones((3, 1)), mu=[1.0, 2.0])
    with pytest.raises(TypeError, match="Unknown Vibrations fields"):
        build_vibrations([1.0], np.ones((3, 1)), frequencies=[1.0])


@using("gaussian")
@using("psi4")
def test_gaussian_vibrations_match_psi4():
    """Both harnesses run the common analysis on their Hessians; results agree mode by mode."""
    import qcengine as qcng

    mol = Molecule.from_data(_WATER_HF_631GS_EQ)

    def run(program, keywords):
        inp = AtomicInput(molecule=mol, specification=AtomicSpecification(program=program, driver="hessian", model={"method": "hf", "basis": "6-31g*"}, keywords=keywords))
        return qcng.compute(inp, program, raise_error=True).extras["qcvars"]["VIBRATIONS"]

    gau = run("gaussian", {})
    psi = run("psi4", {"scf_type": "pk", "d_convergence": 10})
    assert gau["trv"] == psi["trv"] == ["tr"] * 6 + ["v"] * 3
    vib = slice(6, 9)
    np.testing.assert_allclose(gau["omega_in_wavenumbers"][vib], psi["omega_in_wavenumbers"][vib], atol=1.0e-3)
    for key, atol in (("mu", 1.0e-7), ("k", 1.0e-6), ("infrared_intensity", 1.0e-3)):
        np.testing.assert_allclose(gau[key][vib], psi[key][vib], atol=atol)
    for key in ("x", "q", "w"):
        np.testing.assert_allclose(np.asarray(gau[key])[:, vib], np.asarray(psi[key])[:, vib], atol=1.0e-6)


@using("gaussian")
def test_gaussian_vibrations_frame_invariant():
    """Fixed and free frames give the same frequencies, reduced masses and IR intensities
    (the free frame rotates the Hessian and dipole derivatives into the returned frame)."""
    import qcengine as qcng

    molstr = "O 0.31 -0.42 0.27\nH 1.12 -0.05 0.71\nH -0.29 0.19 0.68\nunits angstrom"

    def run(extra):
        inp = AtomicInput(molecule=Molecule.from_data(molstr + extra), specification=AtomicSpecification(program="gaussian", driver="hessian", model={"method": "hf", "basis": "6-31g*"}))
        return qcng.compute(inp, "gaussian", raise_error=True).extras["qcvars"]["VIBRATIONS"]

    fixed, free = run("\nno_com\nno_reorient"), run("")
    assert fixed["trv"] == free["trv"]
    vib = [i for i, t in enumerate(fixed["trv"]) if t == "v"]
    for key, atol in (("omega_in_wavenumbers", 1.0e-4), ("mu", 1.0e-8), ("infrared_intensity", 1.0e-4)):
        np.testing.assert_allclose(np.asarray(free[key])[vib], np.asarray(fixed[key])[vib], atol=atol)


def test_build_input_states_every_mass(harness):
    """Every atom carries the molecule's mass: Gaussian's defaults differ from QCElemental's for some elements."""
    mol = Molecule.from_data("O 0 0 0.1173\nH@2.014101778 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n@H@2.014101778 0 0 3.0")
    inp = AtomicInput(molecule=mol, specification=_spec())
    with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
        com = harness.build_input(inp, _default_taskconfig())["infiles"]["gaussian.com"]
    atom_lines = [ln.split()[0] for ln in com.splitlines() if ln[:1] in ("O", "H")]
    assert atom_lines == ["O(Iso=15.9949146196)", "H(Iso=2.0141017780)", "H(Iso=1.0078250322)", "H-Bq(Iso=2.0141017780)"]


@using("gaussian")
def test_gaussian_uses_ghost_atom_mass():
    import qcengine as qcng

    mol = Molecule.from_data("O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n@H@2.014101778 0 0 3.0")
    inp = AtomicInput(molecule=mol, specification=AtomicSpecification(program="gaussian", driver="hessian", model={"method": "hf", "basis": "sto-3g"}, protocols={"native_files": "all"}))
    log = qcng.compute(inp, "gaussian", raise_error=True).native_files["gaussian.log"]
    assert re.search(r"AtmWgt=\s+15\.9949146\s+1\.0078250\s+1\.0078250\s+2\.0141018", log)


def test_parse_archive_too_few_fields_raises():
    log = " 1\\1\\GINC-HOST\\SP\\RHF\\\\Version=ES64L-G16RevC.01\\HF=-1.0\\\\@\n"
    with pytest.raises(ValueError, match="Malformed archive"):
        harvest_energies(log)


@using("gaussian")
@using("psi4")
def test_gaussian_isotope_vibrations_match_psi4():
    """Isotopic masses reach Gaussian, and the returned molecule keeps them."""
    import qcengine as qcng

    mol = Molecule.from_data(_WATER_HF_631GS_EQ.replace("H ", "H@2.014101778 "))

    def run(program, keywords):
        inp = AtomicInput(molecule=mol, specification=AtomicSpecification(program=program, driver="hessian", model={"method": "hf", "basis": "6-31g*"}, keywords=keywords))
        return qcng.compute(inp, program, raise_error=True)

    gau = run("gaussian", {})
    psi = run("psi4", {"scf_type": "pk", "d_convergence": 10}).extras["qcvars"]["VIBRATIONS"]
    vib = [i for i, t in enumerate(psi["trv"]) if t == "v"]
    np.testing.assert_allclose(gau.molecule.masses, mol.masses)
    gv = gau.extras["qcvars"]["VIBRATIONS"]
    assert gv["trv"] == psi["trv"]
    np.testing.assert_allclose(np.asarray(gv["omega_in_wavenumbers"])[vib], np.asarray(psi["omega_in_wavenumbers"])[vib], atol=1.0e-3)
    np.testing.assert_allclose(np.asarray(gv["mu"])[vib], np.asarray(psi["mu"])[vib], atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(gv["q"])[:, vib], np.asarray(psi["q"])[:, vib], atol=1.0e-6)



def test_harmonic_analysis_rigid_motions():
    from qcengine.programs.util.vibrations import _rigid_motion_space, harmonic_analysis

    def stiff_hessian(geom, masses):
        """Unit mass-weighted stiffness in every internal direction, zero for rigid motions."""
        rigid = _rigid_motion_space(geom, np.asarray(masses), True, True)
        sqrtm = np.repeat(np.sqrt(masses), 3)
        return np.outer(sqrtm, sqrtm) * (np.identity(3 * len(masses)) - rigid.T @ rigid)

    bent = np.array([[0.0, 0.0, 0.2], [0.0, 1.4, -0.9], [0.0, -1.4, -0.9]])
    linear = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.2], [0.0, 0.0, -2.2]])
    masses = [16.0, 1.0, 1.0]
    assert harmonic_analysis(stiff_hessian(bent, masses), bent, masses)["trv"] == ["tr"] * 6 + ["v"] * 3
    assert harmonic_analysis(stiff_hessian(linear, masses), linear, masses)["trv"] == ["tr"] * 5 + ["v"] * 4
    assert harmonic_analysis(np.zeros((3, 3)), np.zeros((1, 3)), [4.0])["trv"] == ["tr"] * 3
    # without projecting rotations (away from a stationary point they are not zero modes)
    rigid = _rigid_motion_space(bent, np.asarray(masses), True, True)
    trans = _rigid_motion_space(bent, np.asarray(masses), True, False)
    sqrtm = np.repeat(np.sqrt(masses), 3)
    rot_soft = stiff_hessian(bent, masses) + 0.01 * np.outer(sqrtm, sqrtm) * (rigid.T @ rigid - trans.T @ trans)
    assert harmonic_analysis(rot_soft, bent, masses, project_rot=False)["trv"] == ["tr"] * 3 + ["v"] * 6


def test_projection_flags():
    from qcengine.programs.util.vibrations import projection_flags

    assert projection_flags(np.zeros(9)) == {"project_trans": True, "project_rot": True}
    assert projection_flags(np.full(9, 0.05)) == {"project_trans": True, "project_rot": False}
    assert projection_flags(None) == {"project_trans": True, "project_rot": False}
    assert projection_flags(np.zeros(9), external_potential=True) == {"project_trans": False, "project_rot": False}


@using("psi4")
@pytest.mark.parametrize("project_rot", [True, False])
def test_harmonic_analysis_matches_psi4_implementation(project_rot):
    """The common analysis reproduces psi4's qcdb.vib.harmonic_analysis (except irrep labels)."""
    import psi4
    import qcengine as qcng
    from psi4.driver.qcdb.vib import harmonic_analysis as psi4_harmonic_analysis

    from qcengine.programs.util.vibrations import harmonic_analysis

    mol = Molecule.from_data("N 0 0 0.12\nH 0 0.94 -0.27\nH 0.81 -0.47 -0.27\nH -0.81 -0.47 -0.27\nunits angstrom\nno_com\nno_reorient")
    inp = AtomicInput(molecule=mol, specification=AtomicSpecification(program="psi4", driver="hessian", model={"method": "hf", "basis": "sto-3g"}, keywords={"scf_type": "pk"}))
    res = qcng.compute(inp, "psi4", raise_error=True)
    hess, geom, masses = np.asarray(res.return_result), res.molecule.geometry.reshape(-1, 3), res.molecule.masses
    dipder = np.asarray(res.extras["qcvars"]["CURRENT DIPOLE GRADIENT"])

    pmol = psi4.core.Molecule.from_schema({**res.molecule.model_dump(), "fix_com": True, "fix_orientation": True}, nonphysical=True)
    pmol.update_geometry()
    basis = psi4.core.BasisSet.build(pmol, "ORBITAL", "STO-3G", quiet=True)
    ref, _ = psi4_harmonic_analysis(hess, geom, masses, basis, pmol.irrep_labels(), dipder=dipder.T, project_trans=True, project_rot=project_rot)
    ours = harmonic_analysis(hess, geom, masses, dipole_derivatives=dipder, project_trans=True, project_rot=project_rot)

    assert ours["trv"] == [t.lower() for t in ref["TRV"].data]
    vib = [i for i, t in enumerate(ours["trv"]) if t == "v"]
    omega = ref["omega"].data[vib]
    np.testing.assert_allclose(np.asarray(ours["omega_in_wavenumbers"])[vib], np.where(omega.imag > omega.real, -omega.imag, omega.real), atol=1.0e-8)
    for ours_key, ref_key in (("x", "x"), ("q", "q"), ("w", "w")):
        np.testing.assert_allclose(np.asarray(ours[ours_key])[:, vib], np.asarray(ref[ref_key].data)[:, vib], atol=1.0e-10)
    for ours_key, ref_key in (("mu", "mu"), ("k", "k"), ("dq0", "DQ0"), ("qtp0", "Qtp0"), ("xtp0", "Xtp0"), ("theta_vib", "theta_vib"), ("infrared_intensity", "IR_intensity"), ("degeneracy", "degeneracy")):
        np.testing.assert_allclose(np.asarray(ours[ours_key])[vib], np.asarray(ref[ref_key].data)[vib], rtol=1.0e-9, atol=1.0e-10)


# ---------------------------------------------------------------------------
# GaussianDriver: Gaussian's own optimizer
# ---------------------------------------------------------------------------

_DISTORTED_WATER = "O 0.31 -0.42 0.27\nH 1.12 -0.05 0.71\nH -0.29 0.19 0.68\nunits angstrom\nno_com\nno_reorient"


def _opt_input(keywords=None, program="gaussian", method="hf", protocols=None):
    from qcelemental.models.v2 import OptimizationInput

    return OptimizationInput(
        initial_molecule=Molecule.from_data(_DISTORTED_WATER),
        specification={
            "keywords": keywords or {},
            "protocols": protocols or {"trajectory_results": "all"},
            "specification": {"driver": "gradient", "program": program, "model": {"method": method, "basis": "6-31G*"}},
        },
    )


def test_build_input_opt_route(harness, water):
    from qcengine.programs.gaussian.runner import OPT_EXTRAS_KEY

    def route(opt_options):
        spec = _spec(driver="gradient").model_copy(update={"extras": {OPT_EXTRAS_KEY: opt_options}})
        with patch("qcengine.programs.gaussian.runner._find_gaussian", return_value="/g16"):
            com = harness.build_input(AtomicInput(molecule=water, specification=spec), _default_taskconfig())["infiles"]["gaussian.com"]
        return [ln for ln in com.splitlines() if ln.startswith("#P")]

    assert route({}) == ["#P HF/STO-3G Opt NoSymm"]
    assert route({"MaxCycles": 50, "Tight": ""}) == ["#P HF/STO-3G Opt=(MaxCycles=50,Tight) NoSymm"]


def test_gaussiandriver_registered():
    import qcengine as qcng

    assert "gaussiandriver" in qcng.list_all_procedures()


@using("gaussian")
def test_gaussiandriver_rejects_other_programs():
    import qcengine as qcng

    with pytest.raises(InputError, match="only works with Gaussian"):
        qcng.compute(_opt_input(program="psi4"), "gaussiandriver", raise_error=True)


@using("gaussian")
@using("geometric")
def test_gaussiandriver_matches_geometric():
    """Gaussian's optimizer reaches the same minimum as geomeTRIC on Gaussian gradients, in the input frame."""
    import qcengine as qcng

    inp = _opt_input(keywords={"Tight": ""})
    native = qcng.compute(inp, "gaussiandriver", raise_error=True)
    geom = qcng.compute(
        _opt_input(keywords={"coordsys": "tric", "convergence_set": "gau_tight"}), "geometric", raise_error=True
    )
    assert native.success and native.provenance.creator == "GaussianDriver"
    assert abs(native.properties.return_energy - geom.properties.return_energy) < 1.0e-7
    fm, gm = native.final_molecule, geom.final_molecule
    assert abs(fm.measure([0, 1]) - gm.measure([0, 1])) < 2.0e-4
    assert abs(fm.measure([1, 0, 2]) - gm.measure([1, 0, 2])) < 2.0e-2

    traj = native.trajectory_results
    assert native.properties.optimization_iterations == len(traj) > 1
    # input frame: the first step is the input geometry, the molecule keeps its frame flags
    np.testing.assert_allclose(traj[0].molecule.geometry, inp.initial_molecule.geometry, atol=1.0e-5)
    assert fm.fix_com and fm.fix_orientation
    assert traj[-1].properties.return_energy == native.properties.return_energy
    rms = [float(np.sqrt(np.mean(np.square(t.return_result)))) for t in traj]
    assert rms[-1] < 1.0e-4 < rms[0]


@using("gaussian")
def test_gaussiandriver_step_matches_single_point():
    """Energy and gradient of the final step equal a Gaussian gradient calculation at that geometry.

    The single point also uses NoSymm: the DFT grid follows the molecular orientation, so
    Gaussian's standard orientation would change the energy at the 1e-7 Eh level.
    """
    import qcengine as qcng

    native = qcng.compute(_opt_input(method="b3lyp-d3bj"), "gaussiandriver", raise_error=True)
    final = native.trajectory_results[-1]
    sp = qcng.compute(
        AtomicInput(molecule=native.final_molecule, specification=AtomicSpecification(program="gaussian", driver="gradient", model={"method": "b3lyp-d3bj", "basis": "6-31G*"}, keywords={"NoSymm": ""})),
        "gaussian",
        raise_error=True,
    )
    assert abs(final.properties.return_energy - sp.properties.return_energy) < 1.0e-8
    np.testing.assert_allclose(final.return_result, sp.return_result, atol=1.0e-6)
    assert "B3LYP-D3(BJ) DISPERSION CORRECTION ENERGY" in final.extras["qcvars"]


@using("gaussian")
def test_gaussiandriver_max_cycles_raises_convergence_error():
    import qcengine as qcng

    with pytest.raises(ConvergenceError):
        qcng.compute(_opt_input(keywords={"MaxCycles": 2}), "gaussiandriver", raise_error=True)


@using("gaussian")
def test_gaussiandriver_trajectory_protocol():
    import qcengine as qcng

    res = qcng.compute(_opt_input(protocols={"trajectory_results": "final"}), "gaussiandriver", raise_error=True)
    assert len(res.trajectory_results) == 1
    assert res.properties.optimization_iterations > 1



def test_sanitize_for_cclib_rotational_constants():
    from qcengine.programs.gaussian.harvester import _sanitize_for_cclib

    log = " Rotational constants (GHZ):      ************    11.74057    11.74057\n Other ****** line\n"
    assert _sanitize_for_cclib(log) == " Rotational constants (GHZ):      inf    11.74057    11.74057\n Other ****** line\n"


@using("gaussian")
def test_gaussian_nearly_linear_hessian():
    """Overflowed rotational constants (printed as asterisks) must not break parsing."""
    import qcengine as qcng

    mol = Molecule.from_data("C 0.001 0 0\nO 0 0 1.16\nO 0 0 -1.16\nunits angstrom")
    inp = AtomicInput(molecule=mol, specification=AtomicSpecification(program="gaussian", driver="hessian", model={"method": "hf", "basis": "sto-3g"}))
    result = qcng.compute(inp, "gaussian", raise_error=True)
    assert len(result.extras["qcvars"]["VIBRATIONS"]["trv"]) == 9


@using("gaussian")
def test_gaussiandriver_molecule_identity():
    """Optimized geometries keep geometry-independent metadata but not the input's id/hash."""
    import qcengine as qcng
    from qcelemental.models.v2 import OptimizationInput

    initial = Molecule.from_data(_DISTORTED_WATER)
    initial = Molecule(
        **{
            **initial.model_dump(),
            "id": "initial-db-id",
            "identifiers": {"molecule_hash": "initial-hash", "pubchem_conformerid": "initial-conformer", "smiles": "O"},
            "provenance": {"creator": "initial-creator", "version": "0", "routine": "initial-routine"},
        }
    )
    inp = OptimizationInput(
        initial_molecule=initial,
        specification={"protocols": {"trajectory_results": "all"}, "specification": {"driver": "gradient", "program": "gaussian", "model": {"method": "hf", "basis": "6-31G*"}}},
    )
    res = qcng.compute(inp, "gaussiandriver", raise_error=True)
    for mol in [res.final_molecule] + [t.molecule for t in res.trajectory_results]:
        assert mol.id is None
        assert mol.identifiers.molecule_hash != "initial-hash"
        assert mol.identifiers.pubchem_conformerid is None
        assert mol.provenance.creator != "initial-creator"
        assert mol.identifiers.smiles == "O"
    np.testing.assert_allclose(res.properties.return_gradient, res.trajectory_results[-1].return_result)


# Production failures (CCBench): heavy atoms made 6-decimal coordinates miss align()'s NRE
# check, and near-symmetric Li3F3 made align()'s atom-mapping search pick a wrong permutation.
_BBR3_BOHR = [0.0, 0.0, 0.0, 3.5990383, 0.0, 0.0, 5.39855745, -3.1168586, 0.0, 5.39855745, 3.1168586, 0.0]
_LI3F3_BOHR = [0.0, 0.0, 0.0, 2.99617116, 0.0, 0.0, 4.28218146, -2.70691542, 0.0, 2.78445487, -5.30188299, 0.0, -0.20284094, -5.06255615, 0.0, -1.70164442, -2.46821042, 0.0]


@using("gaussian")
@using("psi4")
@pytest.mark.parametrize(
    "symbols, geometry",
    [(["Br", "B", "Br", "Br"], _BBR3_BOHR), (["Li", "F", "Li", "F", "Li", "F"], _LI3F3_BOHR)],
    ids=["BBr3", "Li3F3"],
)
@pytest.mark.parametrize("fixed_frame", [True, False])
def test_gaussian_heavy_and_near_symmetric_molecules(symbols, geometry, fixed_frame):
    import qcengine as qcng

    mol = Molecule(symbols=symbols, geometry=geometry, fix_com=fixed_frame, fix_orientation=fixed_frame)
    fixed = Molecule(symbols=symbols, geometry=geometry, fix_com=True, fix_orientation=True)

    def run(program, molecule, keywords):
        inp = AtomicInput(molecule=molecule, specification=AtomicSpecification(program=program, driver="gradient", model={"method": "hf", "basis": "3-21g"}, keywords=keywords))
        return qcng.compute(inp, program, raise_error=True)

    # Gaussian results are in the input frame either way; Psi4 reference in the fixed input frame
    gau = run("gaussian", mol, {})
    psi = run("psi4", fixed, {"scf_type": "pk", "d_convergence": 10, "puream": False})
    np.testing.assert_array_equal(gau.molecule.geometry, mol.geometry)
    np.testing.assert_allclose(gau.return_result, psi.return_result, atol=1.0e-7)


def test_sanitize_for_cclib_merged_orientation_columns():
    from qcengine.programs.gaussian.harvester import _sanitize_for_cclib

    row = "      1          8           0      500.000000-1000.000000-1500.117300\n"
    assert _sanitize_for_cclib(row) == "      1          8           0      500.000000 -1000.000000 -1500.117300\n"


def test_atom_table_merged_columns():
    from qcengine.programs.gaussian.harvester import _atom_table

    block = "      1          8           0      500.000000-1000.000000 1500.117300"
    np.testing.assert_allclose(_atom_table(block, 1), [[500.0, -1000.0, 1500.1173]])


def test_build_route_line_units_keyword_raises():
    with pytest.raises(InputError, match="Units"):
        build_route_line("HF", "STO-3G", "", {}, {"Units": "Bohr"})


@using("gaussian")
@using("psi4")
def test_gaussian_charged_dipole_origin():
    """Gaussian's dipole origin is the center of nuclear charge; QCSchema's is the coordinate origin."""
    import qcengine as qcng

    mol = Molecule.from_data("1 1\nHe 0.43 -0.31 0.22\nH 0.43 -0.31 1.0\nunits angstrom\nno_com\nno_reorient")

    def run(program, keywords):
        inp = AtomicInput(molecule=mol, specification=AtomicSpecification(program=program, driver="properties", model={"method": "hf", "basis": "sto-3g"}, keywords=keywords))
        return qcng.compute(inp, program, raise_error=True)

    gau, psi = run("gaussian", {}), run("psi4", {"scf_type": "pk", "d_convergence": 10})
    np.testing.assert_allclose(gau.properties.scf_dipole_moment, psi.properties.scf_dipole_moment, atol=1.0e-6)


@using("gaussian")
def test_gaussian_symmetrized_geometry_raises():
    """Symmetry=Loose makes Gaussian compute at a symmetrized geometry; that must not be reported for the input."""
    import qcengine as qcng

    mol = Molecule.from_data("O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.46925\nunits angstrom\nno_com\nno_reorient")
    inp = AtomicInput(molecule=mol, specification=AtomicSpecification(program="gaussian", driver="gradient", model={"method": "hf", "basis": "sto-3g"}, keywords={"Symmetry": "Loose"}))
    with pytest.raises(InputError, match="symmetrized"):
        qcng.compute(inp, "gaussian", raise_error=True)


@using("gaussian")
def test_gaussian_far_translated_molecule():
    """Fixed-width coordinate columns run together for large coordinates."""
    import qcengine as qcng

    mol = Molecule.from_data("O 500 -1000 1500.1173\nH 500 -999.2428 1499.5308\nH 500 -1000.7572 1499.5308\nunits angstrom\nno_com\nno_reorient")
    ref = Molecule.from_data("O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\nunits angstrom\nno_com\nno_reorient")

    def run(molecule):
        inp = AtomicInput(molecule=molecule, specification=AtomicSpecification(program="gaussian", driver="gradient", model={"method": "hf", "basis": "sto-3g"}))
        return qcng.compute(inp, "gaussian", raise_error=True)

    np.testing.assert_allclose(run(mol).return_result, run(ref).return_result, atol=1.0e-6)
