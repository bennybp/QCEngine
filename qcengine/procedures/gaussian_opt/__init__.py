"""Geometry optimization with Gaussian's own optimizer (Opt)."""

from decimal import Decimal
from typing import Any, ClassVar, Dict, Union

from qcelemental.models.v2 import AtomicInput, AtomicResult, OptimizationInput, OptimizationResult, Provenance

from qcengine.config import TaskConfig
from qcengine.exceptions import ConvergenceError, InputError
from qcengine.procedures.gaussian_opt.harvester import harvest_optimization
from qcengine.procedures.model import ProcedureHarness
from qcengine.programs.gaussian.harvester import is_normal_termination
from qcengine.programs.gaussian.runner import OPT_EXTRAS_KEY, GaussianHarness, _classify_error
from qcengine.programs.qcvar_identities_resources import build_atomicproperties, build_out
from qcengine.programs.util import error_stamp


class GaussianDriverProcedure(ProcedureHarness):
    """Structural relaxation using Gaussian's optimizer.

    The optimization keywords are passed through verbatim as Gaussian ``Opt``
    options, e.g. ``{"MaxCycles": 50, "Tight": ""}`` -> ``Opt=(MaxCycles=50,Tight)``.
    The keywords of the inner specification are route keywords as for the
    Gaussian program harness. Like geomeTRIC and optking, the optimization
    runs in the frame of the input molecule: the route gets ``NoSymm``, so
    Gaussian neither reorients the molecule nor imposes its point group.
    """

    _defaults: ClassVar[Dict[str, Any]] = {"name": "GaussianDriver", "procedure": "optimization"}

    def found(self, raise_error: bool = False) -> bool:
        return GaussianHarness().found(raise_error)

    def get_version(self) -> str:
        return GaussianHarness().get_version()

    def build_input_model(
        self, data: Union[Dict[str, Any], "OptimizationInput"], *, return_input_schema_version: bool = False
    ) -> "OptimizationInput":
        return self._build_model(data, "OptimizationInput", return_input_schema_version=return_input_schema_version)

    def compute(self, input_model: OptimizationInput, config: TaskConfig) -> OptimizationResult:
        self.found(raise_error=True)
        harness = GaussianHarness()

        spec = input_model.specification.specification
        if spec.program and spec.program.lower() != "gaussian":
            raise InputError(f"GaussianDriver procedure only works with Gaussian, not {spec.program}")
        if spec.driver.value != "gradient":
            raise InputError(f"GaussianDriver needs a gradient specification, not {spec.driver.value}")

        opt_options = dict(input_model.specification.keywords)
        atomic_input = AtomicInput(
            molecule=input_model.initial_molecule,
            specification=spec.model_copy(update={"extras": {**spec.extras, OPT_EXTRAS_KEY: opt_options}}),
        )
        job_inputs = harness.build_input(atomic_input, config)
        _, dexe = harness.execute(job_inputs)

        log_text = dexe["outfiles"].get("gaussian.log") or ""
        stdin = job_inputs["infiles"]["gaussian.com"]
        stderr = dexe.get("stderr", "") or ""
        if not is_normal_termination(log_text):
            if "Number of steps exceeded" in log_text:
                raise ConvergenceError(error_stamp(stdin, log_text, stderr))
            raise _classify_error(log_text)(error_stamp(stdin, log_text, stderr))

        try:
            steps = harvest_optimization(input_model.initial_molecule, spec.model.method, log_text)
        except Exception as e:
            raise _classify_error(log_text)(error_stamp(stdin, log_text, stderr)) from e

        version = self.get_version()
        method_upper = spec.model.method.upper()
        step_spec = spec.model_dump(exclude={"schema_name", "schema_version"})
        trajectory = []
        for step in steps:
            qcvars = step.qcvars
            qcvars[f"{method_upper} TOTAL GRADIENT"] = step.gradient
            qcvars["CURRENT GRADIENT"] = step.gradient
            build_out(qcvars)
            trajectory.append(
                AtomicResult(
                    input_data={"molecule": step.molecule, "specification": step_spec},
                    molecule=step.molecule,
                    properties=build_atomicproperties(qcvars),
                    return_result=step.gradient,
                    extras={"qcvars": {k: str(v) if isinstance(v, Decimal) else v for k, v in qcvars.items()}},
                    provenance=Provenance(creator="Gaussian", version=version, routine="gaussian_opt"),
                    success=True,
                )
            )

        final = trajectory[-1]
        return OptimizationResult(
            input_data=input_model,
            final_molecule=final.molecule,
            trajectory_results=trajectory,
            trajectory_properties=[
                {k: getattr(r.properties, k) for k in ("calcinfo_natom", "return_energy", "return_gradient")}
                for r in trajectory
            ],
            properties={
                "return_energy": final.properties.return_energy,
                "return_gradient": final.return_result,
                "nuclear_repulsion_energy": final.molecule.nuclear_repulsion_energy(),
                "optimization_iterations": len(trajectory),
            },
            native_files={"input": stdin, "gaussian.log": log_text},
            stdout=log_text,
            stderr=stderr,
            success=True,
            provenance=Provenance(creator="GaussianDriver", version=version, routine="gaussian_opt"),
        )
