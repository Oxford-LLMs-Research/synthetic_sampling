"""Leakage rules for profile / target construction.

Target variables (and ESS country-sibling concept vars) must not appear as
profile features. The generator enforces this via exclusion sets; this module
documents and centralises the rule surface used by DatasetBuilder.
"""

from __future__ import annotations

from typing import Iterable, Optional, Set

from .country_specific import CountrySpecificHandler


def target_exclusions(
    target_codes: Iterable[str],
    cs_handler: Optional[CountrySpecificHandler] = None,
    concept_ids: Optional[Iterable[str]] = None,
) -> Set[str]:
    """Variables that must be excluded from profile features for these targets."""
    excl: Set[str] = set(target_codes)
    if cs_handler is not None and concept_ids:
        for concept_id in concept_ids:
            excl.update(cs_handler.get_all_vars_for_concept(concept_id))
    return excl


def pool_exclusions(
    cs_handler: Optional[CountrySpecificHandler],
) -> Set[str]:
    """Country-specific vars replaced by concept representatives in the target pool."""
    if cs_handler is None:
        return set()
    return set(cs_handler.get_variables_to_exclude_from_pool())
