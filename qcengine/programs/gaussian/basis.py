"""Build Gaussian general-basis (Gen/GenECP) input from the Basis Set Exchange.

A basis named ``bse:<name>`` or ``bse:<name>:<version>`` is fetched from the
``basis_set_exchange`` library, using the same naming as Psi4's ``bse:`` prefix,
and written inline after the molecule block.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import qcelemental as qcel
from qcelemental.util import which_import

from ...exceptions import InputError

_BSE_PREFIX = "bse:"


def is_bse_basis(basis: str) -> bool:
    """Return True if ``basis`` requests a basis set from the Basis Set Exchange."""
    return basis.lower().startswith(_BSE_PREFIX)


def parse_bse_basis(basis: str) -> Tuple[str, Optional[str]]:
    """Split ``bse:<name>[:<version>]`` into ``(name, version)``.

    ``version`` is None when not given, which selects the latest version.
    """
    parts = basis.split(":")
    if len(parts) not in (2, 3) or not parts[1]:
        raise InputError(f"Invalid BSE basis '{basis}'. Expected 'bse:<name>' or 'bse:<name>:<version>'.")
    name = parts[1]
    version = parts[2] if len(parts) == 3 and parts[2] else None
    return name, version


def _cartesian_flags(basis_dict: Dict[str, Any], zs: Iterable[str]) -> Tuple[Optional[bool], Optional[bool]]:
    """Return whether the d shells and the f-and-higher shells are Cartesian.

    Gaussian has a single global switch for d (5D/6D) and one for f and higher
    (7F/10F), so all elements in the molecule must agree. None means the
    molecule has no shells of that kind.
    """
    d_types = set()
    f_types = set()
    for z in zs:
        for shell in basis_dict["elements"][z].get("electron_shells", []):
            for am in shell["angular_momentum"]:
                if am == 2:
                    d_types.add(shell["function_type"])
                elif am >= 3:
                    f_types.add(shell["function_type"])

    flags = []
    for label, types in (("d", d_types), ("f and higher", f_types)):
        if len(types) > 1:
            raise InputError(
                f"Basis mixes Cartesian and spherical {label} functions across the elements in this "
                "molecule, which Gaussian cannot represent."
            )
        flags.append(None if not types else types.pop() == "gto_cartesian")
    return flags[0], flags[1]


def build_bse_basis(basis: str, symbols: Iterable[str], real: Iterable[bool]) -> Dict[str, Any]:
    """Fetch a BSE basis for the elements in a molecule and format it for Gaussian.

    Parameters
    ----------
    basis:
        Basis string with the ``bse:`` prefix.
    symbols:
        Element symbol of each atom, including ghost atoms.
    real:
        Whether each atom is real (False for ghost atoms).

    Returns
    -------
    dict
        ``"route_basis"``: ``"Gen"`` or ``"GenECP"``, used in place of the basis name.
        ``"route_keywords"``: bare route keywords (``5D``/``6D``, ``7F``/``10F``).
        ``"basis_block"``: general-basis (and ECP) section text, without trailing blank lines.
    """
    which_import(
        "basis_set_exchange",
        raise_error=True,
        raise_msg="For 'bse:' basis sets, please install basis_set_exchange: "
        "`conda install -c conda-forge basis_set_exchange` or `pip install basis_set_exchange`.",
    )
    import basis_set_exchange as bse
    from basis_set_exchange import writers

    name, version = parse_bse_basis(basis)

    symbols = list(symbols)
    real = list(real)
    all_zs = sorted({str(qcel.periodictable.to_Z(s)) for s in symbols}, key=int)
    real_zs = {str(qcel.periodictable.to_Z(s)) for s, r in zip(symbols, real) if r}

    try:
        basis_dict = bse.get_basis(name, elements=all_zs, version=version)
    except KeyError as e:
        raise InputError(f"Basis Set Exchange could not provide basis '{basis}': {e.args[0]}") from e

    # Gaussian never places an ECP on a ghost (-Bq) atom, and an ECP section
    # entry for an element with no real atoms is an input error.
    for z in all_zs:
        if z not in real_zs:
            basis_dict["elements"][z].pop("ecp_potentials", None)
            basis_dict["elements"][z].pop("ecp_electrons", None)
    has_ecp = any("ecp_potentials" in basis_dict["elements"][z] for z in all_zs)

    # Gen defaults to 5D 7F, so always state the convention the basis was defined in.
    cart_d, cart_f = _cartesian_flags(basis_dict, all_zs)
    route_keywords: List[str] = []
    if cart_d is not None:
        route_keywords.append("6D" if cart_d else "5D")
    if cart_f is not None:
        route_keywords.append("10F" if cart_f else "7F")

    # Comment lines only: a blank line would end the basis section.
    header = [
        f"! Basis Set Exchange {bse.version()}",
        f"! Basis set: {basis_dict['name']}  Version: {basis_dict['version']}",
    ]
    body = writers.write_formatted_basis_str(basis_dict, "gaussian94").strip("\n")

    return {
        "route_basis": "GenECP" if has_ecp else "Gen",
        "route_keywords": route_keywords,
        "basis_block": "\n".join(header) + "\n" + body,
    }
