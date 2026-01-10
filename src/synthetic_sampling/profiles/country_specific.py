"""
Country-Specific Variable Handler.

Handles surveys like ESS where certain questions (education, religion, party vote) 
have different variable versions per country.

Key insight: Country-specific variables represent CONCEPTS that can be:
- Sampled as targets (resolve to country's variable for answer)
- Sampled as features (resolve to country's variable for profile info)
- Properly excluded when selected as targets (same logic as regular variables)

Detection Strategy:
- Country codes appear in variable names but positions vary
- Most reliable: last 2 characters before any numeric suffix
- Examples: prtvtdat (AT), prtvgde1 (DE), edlvebe (BE)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Tuple
from collections import defaultdict
import re


@dataclass
class ConceptGroup:
    """
    A group of country-specific variables for a single concept.
    
    Example: Party vote has prtvtdat (Austria), prtvtebe (Belgium), etc.
    All represent the same underlying concept: "which party did you vote for"
    """
    concept_id: str  # e.g., "party_voted" - used as identifier
    concept_name: str  # e.g., "Party voted for" - human readable
    prefixes: List[str]  # Variable prefixes (may be multiple, e.g., ['prtvt', 'prtvg', 'prtvc'])
    country_variables: Dict[str, List[str]] = field(default_factory=dict)
    
    # Representative info for sampling (from any country's version)
    representative_question: Optional[str] = None
    representative_description: Optional[str] = None
    section: Optional[str] = None
    
    @property
    def countries_covered(self) -> List[str]:
        return list(self.country_variables.keys())
    
    @property
    def all_variables(self) -> Set[str]:
        """All country-specific variables in this group."""
        result = set()
        for vars_list in self.country_variables.values():
            result.update(vars_list)
        return result
    
    def get_variables_for_country(self, country_code: str) -> List[str]:
        """Get all variables for a specific country (may be multiple)."""
        return self.country_variables.get(country_code.upper(), [])
    
    def get_primary_variable_for_country(self, country_code: str) -> Optional[str]:
        """Get the primary (first) variable for a country."""
        vars_list = self.get_variables_for_country(country_code)
        return vars_list[0] if vars_list else None


class CountrySpecificHandler:
    """
    Detects and manages country-specific variables in survey metadata.
    
    Uses robust country code detection that handles:
    - Standard suffix: prtvtdat (prtv + t + d + AT)
    - Suffix with number: prtvgde1 (prtv + g + DE + 1)
    - Multiple vars per country: Germany has prtvgde1, prtvgde2
    
    Parameters
    ----------
    metadata : dict
        Survey metadata (section -> var_code -> info)
    country_codes : list[str]
        Valid country codes in this survey
    concept_configs : list[dict]
        List of concept configurations, each with:
        - 'concept_id': Unique identifier
        - 'concept_name': Human readable name  
        - 'prefixes': List of variable prefixes to match
    min_countries : int
        Minimum countries for a concept to be detected
    """
    
    CONCEPT_MARKER = "__concept__"
    
    def __init__(
        self,
        metadata: dict,
        country_codes: List[str],
        concept_configs: List[Dict],
        min_countries: int = 3
    ):
        self.metadata = metadata
        self.country_codes = set(code.upper() for code in country_codes)
        self.concept_configs = concept_configs
        self.min_countries = min_countries
        
        # Build indices
        self._all_vars = self._collect_all_vars()
        self._var_to_section = self._build_var_to_section()
        
        # Detect concept groups
        self._concept_groups: Dict[str, ConceptGroup] = {}
        self._var_to_concept: Dict[str, str] = {}
        self._detect_concept_groups()
    
    def _collect_all_vars(self) -> List[str]:
        """Collect all variable codes from metadata."""
        all_vars = []
        for section, variables in self.metadata.items():
            if isinstance(variables, dict):
                all_vars.extend(variables.keys())
        return all_vars
    
    def _build_var_to_section(self) -> Dict[str, str]:
        """Map variable codes to their sections."""
        var_to_section = {}
        for section, variables in self.metadata.items():
            if isinstance(variables, dict):
                for var_code in variables:
                    var_to_section[var_code] = section
        return var_to_section
    
    def _extract_country_code(self, var_code: str, prefix: str) -> Optional[str]:
        """
        Extract country code from variable name.
        
        Strategy: Look for 2-letter country code, handling various patterns:
        - edlveat -> AT (last 2 chars)
        - prtvgde1 -> DE (last 2 chars before digit)
        - prtvclt3 -> LT (last 2 chars before digit)
        """
        if not var_code.startswith(prefix):
            return None
        
        remainder = var_code[len(prefix):]
        if len(remainder) < 2:
            return None
        
        # Strip trailing digits
        remainder_no_digits = remainder.rstrip('0123456789')
        
        if len(remainder_no_digits) < 2:
            return None
        
        # Take last 2 characters
        potential_cc = remainder_no_digits[-2:].upper()
        
        if potential_cc in self.country_codes:
            return potential_cc
        
        return None
    
    def _get_var_info(self, var_code: str) -> Optional[dict]:
        """Get metadata info for a variable."""
        section = self._var_to_section.get(var_code)
        if section and section in self.metadata:
            return self.metadata[section].get(var_code)
        return None
    
    def _detect_concept_groups(self):
        """Detect country-specific variable groups based on concept configs."""
        for config in self.concept_configs:
            concept_id = config['concept_id']
            concept_name = config['concept_name']
            prefixes = config['prefixes']
            
            country_variables = defaultdict(list)
            
            # Find all matching variables
            for prefix in prefixes:
                matching_vars = [v for v in self._all_vars if v.startswith(prefix)]
                
                for var in matching_vars:
                    country = self._extract_country_code(var, prefix)
                    if country:
                        country_variables[country].append(var)
                        self._var_to_concept[var] = concept_id
            
            if len(country_variables) >= self.min_countries:
                # Get representative info from first available variable
                rep_var = None
                for vars_list in country_variables.values():
                    if vars_list:
                        rep_var = vars_list[0]
                        break
                
                rep_question = None
                rep_description = None
                section = None
                
                if rep_var:
                    var_info = self._get_var_info(rep_var)
                    if var_info:
                        rep_question = var_info.get('question')
                        rep_description = var_info.get('description')
                        section = self._var_to_section.get(rep_var)
                
                group = ConceptGroup(
                    concept_id=concept_id,
                    concept_name=concept_name,
                    prefixes=prefixes,
                    country_variables=dict(country_variables),
                    representative_question=rep_question,
                    representative_description=rep_description,
                    section=section
                )
                self._concept_groups[concept_id] = group
    
    # -------------------------------------------------------------------------
    # Core Access Methods
    # -------------------------------------------------------------------------
    
    def get_all_country_specific_vars(self) -> Set[str]:
        """Get all individual country-specific variables."""
        all_vars = set()
        for group in self._concept_groups.values():
            all_vars.update(group.all_variables)
        return all_vars
    
    def get_concept_groups(self) -> Dict[str, ConceptGroup]:
        """Get all detected concept groups."""
        return self._concept_groups.copy()
    
    def is_country_specific_var(self, var_code: str) -> bool:
        """Check if a variable is a country-specific variant."""
        return var_code in self._var_to_concept
    
    def is_concept_code(self, code: str) -> bool:
        """Check if a code is a concept marker."""
        return code.startswith(self.CONCEPT_MARKER)
    
    def get_concept_id_from_code(self, code: str) -> Optional[str]:
        """Extract concept_id from a concept code."""
        if self.is_concept_code(code):
            return code[len(self.CONCEPT_MARKER):]
        return None
    
    def make_concept_code(self, concept_id: str) -> str:
        """Create a concept code from concept_id."""
        return f"{self.CONCEPT_MARKER}{concept_id}"
    
    # -------------------------------------------------------------------------
    # Pool Construction
    # -------------------------------------------------------------------------
    
    def get_variables_to_exclude_from_pool(self) -> Set[str]:
        """Get individual country-specific variables to exclude (replaced by concepts)."""
        return self.get_all_country_specific_vars()
    
    def get_concept_representatives_for_pool(self) -> List[Dict]:
        """Get concept representatives to add to the sampling pool."""
        representatives = []
        for concept_id, group in self._concept_groups.items():
            representatives.append({
                'var_code': self.make_concept_code(concept_id),
                'section': group.section or 'unknown',
                'question': group.representative_question or group.concept_name,
                'description': group.representative_description or group.concept_name,
                'concept_id': concept_id,
                'is_concept': True,
                'countries_covered': group.countries_covered,
            })
        return representatives
    
    def get_all_vars_for_concept(self, concept_id: str) -> Set[str]:
        """Get all country-specific variables for a concept (for exclusion)."""
        group = self._concept_groups.get(concept_id)
        if group:
            return group.all_variables
        return set()
    
    # -------------------------------------------------------------------------
    # Per-Respondent Resolution
    # -------------------------------------------------------------------------
    
    def resolve_to_variable(
        self,
        concept_id: str,
        country_code: str
    ) -> Optional[str]:
        """Resolve a concept to the primary country-specific variable."""
        group = self._concept_groups.get(concept_id)
        if group is None:
            return None
        return group.get_primary_variable_for_country(country_code)
    
    def resolve_to_all_variables(
        self,
        concept_id: str,
        country_code: str
    ) -> List[str]:
        """Resolve a concept to ALL country-specific variables (for multi-vote countries)."""
        group = self._concept_groups.get(concept_id)
        if group is None:
            return []
        return group.get_variables_for_country(country_code)
    
    def get_variable_metadata(
        self,
        concept_id: str,
        country_code: str
    ) -> Optional[dict]:
        """Get metadata for the primary country-specific variable."""
        var_code = self.resolve_to_variable(concept_id, country_code)
        if var_code is None:
            return None
        return self._get_var_info(var_code)
    
    def get_value_label(
        self,
        concept_id: str,
        country_code: str,
        raw_value
    ) -> Optional[str]:
        """Get the label for a raw value in a country-specific variable."""
        var_meta = self.get_variable_metadata(concept_id, country_code)
        if var_meta is None:
            return None
        
        values_map = var_meta.get('values', {})
        
        # Try various key formats
        for key in [raw_value, str(raw_value)]:
            if key in values_map:
                return values_map[key]
        
        # Try int conversion for float values
        try:
            int_key = str(int(float(raw_value)))
            if int_key in values_map:
                return values_map[int_key]
        except (ValueError, TypeError):
            pass
        
        return str(raw_value)
    
    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------
    
    def get_summary(self) -> str:
        """Get a summary of detected country-specific variables."""
        lines = [f"Country-specific variable groups ({len(self._concept_groups)} concepts):"]
        for concept_id, group in sorted(self._concept_groups.items()):
            # Count countries with multiple vars
            multi_var_countries = [c for c, v in group.country_variables.items() if len(v) > 1]
            multi_note = f" ({len(multi_var_countries)} with multiple vars)" if multi_var_countries else ""
            
            lines.append(
                f"  {concept_id} ({group.concept_name}): "
                f"{len(group.all_variables)} vars across {len(group.countries_covered)} countries{multi_note}"
            )
        
        total = len(self.get_all_country_specific_vars())
        lines.append(f"  Total country-specific variables: {total}")
        lines.append(f"  These are replaced by {len(self._concept_groups)} concept representatives in pool")
        return '\n'.join(lines)
    
    def get_detailed_coverage(self, concept_id: str) -> str:
        """Get detailed country coverage for a concept."""
        group = self._concept_groups.get(concept_id)
        if not group:
            return f"Concept '{concept_id}' not found"
        
        lines = [f"{concept_id} ({group.concept_name}):"]
        for country in sorted(group.country_variables.keys()):
            vars = group.country_variables[country]
            lines.append(f"  {country}: {vars}")
        
        return '\n'.join(lines)


# =============================================================================
# ESS Configuration
# =============================================================================

ESS_CONCEPT_CONFIGS = [
    {
        'concept_id': 'education_level',
        'concept_name': 'Highest level of education',
        'prefixes': ['edlv'],
    },
    {
        'concept_id': 'religion_denomination', 
        'concept_name': 'Religious denomination',
        'prefixes': ['rlgdn'],
    },
    {
        'concept_id': 'religion_raised',
        'concept_name': 'Religion raised in',
        'prefixes': ['rlgde'],
    },
    {
        'concept_id': 'party_voted',
        'concept_name': 'Party voted for in last election',
        'prefixes': ['prtvt', 'prtvg', 'prtvc'],  # Multiple prefixes for different patterns
    },
    {
        'concept_id': 'party_close',
        'concept_name': 'Party feels closest to',
        'prefixes': ['prtcl'],
    },
]


# =============================================================================
# Factory Function
# =============================================================================

def create_handler_for_survey(
    metadata: dict,
    survey_config  # SurveyConfig with country_specific field
) -> Optional[CountrySpecificHandler]:
    """
    Create a CountrySpecificHandler from survey config.
    
    Returns None if the survey doesn't have country-specific handling enabled.
    """
    if not survey_config.has_country_specific_vars():
        return None
    
    cs_config = survey_config.country_specific
    
    # Extract country codes from metadata
    country_codes = _extract_country_codes(metadata, cs_config.country_var)
    if not country_codes:
        return None
    
    # Convert old format (prefix -> name) to new format (concept configs)
    if hasattr(cs_config, 'get_prefixes_dict'):
        # Old format - convert
        prefix_dict = cs_config.get_prefixes_dict()
        concept_configs = []
        for prefix, name in prefix_dict.items():
            # For party_voted, use the extended prefixes
            if prefix == 'prtvt':
                concept_configs.append({
                    'concept_id': name,
                    'concept_name': name.replace('_', ' ').title(),
                    'prefixes': ['prtvt', 'prtvg', 'prtvc'],
                })
            else:
                concept_configs.append({
                    'concept_id': name,
                    'concept_name': name.replace('_', ' ').title(),
                    'prefixes': [prefix],
                })
    else:
        # Assume new format or use ESS defaults
        concept_configs = ESS_CONCEPT_CONFIGS
    
    return CountrySpecificHandler(
        metadata=metadata,
        country_codes=country_codes,
        concept_configs=concept_configs,
        min_countries=cs_config.min_countries
    )


def _extract_country_codes(metadata: dict, country_var: str) -> List[str]:
    """Extract country codes from the country variable in metadata."""
    for section, variables in metadata.items():
        if not isinstance(variables, dict):
            continue
        if country_var in variables:
            values = variables[country_var].get('values', {})
            if values:
                return list(values.keys())
    return []