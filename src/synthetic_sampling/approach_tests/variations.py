"""
Answer Variation Rules and Generator for Surface Form Sensitivity Testing.

This module provides:
1. Comprehensive variation rules (synonyms, reordering, pronouns)
2. Special values and ineligibility patterns for filtering
3. AnswerVariationGenerator class for generating variations

Used in Test 1 (Surface Form Sensitivity) of the perplexity validation tests.

Coverage Statistics (from analysis of 6 surveys, ~12,000 unique options):
- Synonym coverage: ~25% of instances
- Reorder coverage: ~12%
- Pronoun coverage: ~6%
"""

import re
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Set


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AnswerVariation:
    """Represents a single variation of an answer option."""
    original: str
    variation: str
    variation_type: str  # "pronoun", "synonym", "reorder"
    rule_applied: str    # Human-readable description of the rule
    
    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# 1. SPECIAL VALUES (to filter before variation)
# =============================================================================
# These indicate missing data/refusal - NOT substantive answers.
# They should be filtered out before applying variations.

SPECIAL_VALUES: Set[str] = {
    # Don't Know variants
    "don't know",
    "dont know",
    "do not know",
    "do not know / no answer",
    "don't know/haven't heard",
    "don't know/haven't heard enough",
    "don't know him",
    "hard to say",
    "difficult to say",
    "can't say",
    "can't choose",
    "uncertain",
    "not sure",
    "haven't thought about it",
    "haven't thought much about it",
    "do not understand",
    
    # Refused variants
    "refused",
    "refused to answer",
    "refuse to answer",
    "decline to answer",
    "prefer not to say",
    "prefer not to answer",
    "refusal",
    
    # Missing/NA variants
    "missing",
    "not applicable",
    "not asked",
    "not asked in survey",
    "not asked in this country",
    "not available",
    "no data",
    "no answer",
    "no response",
    "na",
    "n/a",
    "inapplicable",
    "does not apply",
    "does not apply at all",
    "skipped",
    "not stated",
    
    # Other/Open-ended
    "other",
    "others",
    "other (specify)",
    "other (please specify)",
    "other - specify",
    "other- specify",
    "other (please name)",
    "others (please specify)",
    
    # Administrative
    "extra-regio nuts 1",
    "extra-regio nuts 2",
    "extra-regio nuts 3",
}


# =============================================================================
# 2. INELIGIBILITY PATTERNS (for question filtering)
# =============================================================================
# Questions with options matching these patterns should be EXCLUDED.
# These are specific categories (parties, ethnicities, religions),
# not attitudes where surface form matters.

INELIGIBILITY_PATTERNS: Dict[str, List[str]] = {
    # Political Parties
    "party_country_prefix": [
        r"^[A-Z]{2,3}:\s+",  # "ALB: ", "ARM: " country prefixes
    ],
    "party_names": [
        r"\bparty\s*\(",  # "Party (acronym)"
        r"\([A-Z]{2,6}\)$",  # Ends with party acronym "(ANC)", "(PDP)"
        r"\bparty\s+of\s+\w+",
        r"\bdemocratic\s+party\b",
        r"\brepublican\s+party\b",
        r"\blabour\s+party\b",
        r"\bliberal\s+party\b",
        r"\bconservative\s+party\b",
        r"\bsocialist\s+party\b",
        r"\bcommunist\s+party\b",
        r"\bgreen\s+party\b",
        r"\bpeople'?s\s+party\b",
        r"\bnational\s+front\b",
        r"\bmovement\s+for\b",
        r"\balliance\s+for\b",
        r"\bcoalition\s+of\b",
        r"\bcongress\s+party\b",
    ],
    
    # Ethnic Groups / Tribes
    "ethnicity_general": [
        r"\btribe\b",
        r"\btribal\b",
        r"\bethnic\s+group\b",
        r"igna$",  # Ethiopian: Oromigna, Tigrigna
        r"\bcluster\b",  # Regional cluster
    ],
    
    # Religious Denominations
    "religion_denominations": [
        r"\bcatholic\b",
        r"\bprotestant\b",
        r"\borthodox\b",
        r"\bsunni\b",
        r"\bshia\b",
        r"\bhindu\b",
        r"\bbuddhist\b",
        r"\bmuslim\b",
        r"\bchristian\b",
        r"\bpentecostal\b",
        r"\bevangelical\b",
        r"\bdenomination\b",
    ],
    
    # Languages (as standalone options)
    "languages": [
        r"^arabic$",
        r"^english$",
        r"^french$",
        r"^spanish$",
        r"^portuguese$",
        r"^swahili$",
        r"^mandarin$",
        r"^cantonese$",
    ],
}

# Question-level patterns (check question TEXT, not just options)
INELIGIBLE_QUESTION_PATTERNS: List[str] = [
    r"which\s+party",
    r"vote\s+for\s+which",
    r"party\s+did\s+you",
    r"ethnic\s+group",
    r"ethnicity",
    r"your\s+tribe",
    r"religious\s+denomination",
    r"what\s+religion",
    r"language.*speak",
    r"mother\s+tongue",
    r"native\s+language",
]


# =============================================================================
# 3. SYNONYM WORD PAIRS (intensity modifiers)
# =============================================================================
# Word-level synonyms applied with word boundary matching.
# These are intensity modifiers that can be substituted.

SYNONYM_WORDS: List[Tuple[str, str]] = [
    # === High intensity ===
    ("Strongly", "Completely"),
    ("Strongly", "Totally"),
    ("Completely", "Totally"),
    ("Completely", "Absolutely"),
    ("Extremely", "Very"),
    ("Extremely", "Absolutely"),
    ("Absolutely", "Completely"),
    
    # === Medium-high intensity ===
    ("Very", "Quite"),
    ("Very", "Highly"),
    ("Quite", "Fairly"),
    ("Fairly", "Reasonably"),
    ("Rather", "Fairly"),
    ("Rather", "Quite"),
    
    # === Medium intensity ===
    ("Somewhat", "Fairly"),
    ("Somewhat", "Moderately"),
    ("Somewhat", "Partially"),
    ("Somewhat", "Rather"),
    ("Mostly", "Generally"),
    ("Mostly", "Largely"),
    ("Mostly", "Mainly"),
    ("Moderately", "Fairly"),
    ("Moderately", "Reasonably"),
    
    # === Low intensity ===
    ("Slightly", "A little"),
    ("Slightly", "Marginally"),
    ("Slightly", "Somewhat"),
    ("A little", "Marginally"),
    ("A little", "Slightly"),
    
    # === Frequency ===
    ("Always", "Every time"),
    ("Always", "Invariably"),
    ("Often", "Frequently"),
    ("Often", "Regularly"),
    ("Sometimes", "Occasionally"),
    ("Sometimes", "At times"),
    ("Rarely", "Seldom"),
    ("Rarely", "Hardly ever"),
    ("Rarely", "Infrequently"),
    ("Never", "Not at all"),
    ("Never", "Not ever"),
    
    # === Amount ===
    ("Much", "A lot"),
    ("Much", "Considerably"),
    ("Little", "Not much"),
    ("Little", "Slightly"),
    
    # === Evaluation ===
    ("Good", "Positive"),
    ("Bad", "Negative"),
    ("Excellent", "Outstanding"),
    ("Excellent", "Very good"),
    ("Poor", "Bad"),
    ("Poor", "Inadequate"),
    ("Fair", "Adequate"),
    ("Fair", "Average"),
    ("Fair", "Reasonable"),
    ("Terrible", "Awful"),
    ("Terrible", "Very bad"),
    
    # === Trust/Confidence ===
    ("Complete", "Total"),
    ("Complete", "Full"),
    ("Total", "Full"),
    ("Great", "High"),
    ("Great", "Considerable"),
    
    # === Likelihood ===
    ("Likely", "Probable"),
    ("Unlikely", "Improbable"),
    
    # === Support/Oppose ===
    ("Support", "Favor"),
    ("Oppose", "Against"),
    
    # === Difficulty ===
    ("Difficult", "Hard"),
    ("Easy", "Simple"),
    
    # === Safety ===
    ("Safe", "Secure"),
    ("Unsafe", "Insecure"),
    ("Dangerous", "Unsafe"),
]


# =============================================================================
# 4. SYNONYM PHRASES (full phrase replacements)
# =============================================================================
# Complete phrase synonyms, applied only on exact match.

SYNONYM_PHRASES: List[Tuple[str, str]] = [
    # === Agreement neutrals ===
    ("Neither agree nor disagree", "Neutral"),
    ("Neither agree nor disagree", "No opinion"),
    ("Neither agree or disagree", "Neutral"),
    ("Agree with neither", "Neutral"),
    
    # === Satisfaction neutrals ===
    ("Neither satisfied nor dissatisfied", "Neutral"),
    
    # === Good/bad neutrals ===
    ("Neither good nor bad", "Neutral"),
    ("Neither good, nor bad", "Neutral"),
    ("Neither bad nor good", "Neutral"),
    
    # === Other neutrals ===
    ("Neither important nor unimportant", "Neutral"),
    ("Neither likely nor unlikely", "Neutral"),
    ("Neither positive nor negative", "Neutral"),
    ("Neither trust nor distrust", "Neutral"),
    ("Neither approve nor disapprove", "Neutral"),
    ("Neither happy nor unhappy", "Neutral"),
    ("Neither worse nor better", "No change"),
    ("Neither better nor worse", "No change"),
    
    # === Importance scale ===
    ("Not at all important", "Not important"),
    ("Not at all important", "Unimportant"),
    ("Very important", "Extremely important"),
    ("Somewhat important", "Fairly important"),
    ("Not very important", "Not so important"),
    ("Absolutely essential", "Essential"),
    
    # === Trust scale ===
    ("Trust completely", "Complete trust"),
    ("Trust completely", "Full trust"),
    ("Trust somewhat", "Some trust"),
    ("No trust at all", "Complete distrust"),
    ("Do not trust at all", "No trust"),
    ("Do not trust very much", "Little trust"),
    ("A great deal of trust", "Complete trust"),
    ("Quite a lot of trust", "High trust"),
    ("Not a lot of trust", "Low trust"),
    ("Not very much trust", "Little trust"),
    
    # === Satisfaction scale ===
    ("Not at all satisfied", "Completely dissatisfied"),
    ("Not very satisfied", "Somewhat dissatisfied"),
    ("Completely satisfied", "Fully satisfied"),
    ("Fairly satisfied", "Quite satisfied"),
    
    # === Frequency scale ===
    ("Very often", "Frequently"),
    ("Fairly often", "Often"),
    ("Not very often", "Rarely"),
    ("Not at all often", "Never"),
    ("Hardly ever", "Rarely"),
    ("Almost never", "Rarely"),
    
    # === Likelihood scale ===
    ("Very likely", "Extremely likely"),
    ("Not very likely", "Somewhat unlikely"),
    ("Not at all likely", "Very unlikely"),
    ("Not likely at all", "Very unlikely"),
    
    # === Evaluation scale ===
    ("Very good", "Excellent"),
    ("Very bad", "Terrible"),
    ("Fairly good", "Good"),
    ("Fairly bad", "Bad"),
    
    # === Agreement intensity ===
    ("Strongly agree", "Completely agree"),
    ("Strongly disagree", "Completely disagree"),
    ("Agree strongly", "Completely agree"),
    ("Disagree strongly", "Completely disagree"),
    
    # === Extent scale ===
    ("To a great extent", "Very much"),
    ("To some extent", "Somewhat"),
    ("To a small extent", "A little"),
    ("Not at all", "Not in any way"),
    ("To a medium extent", "To a moderate extent"),
    ("To a medium extent", "Moderately"),
    ("To a limited extent", "To a small extent"),
    ("To a large extent", "To a great extent"),
    
    # === Amount scale ===
    ("A great deal", "A lot"),
    ("A great deal", "Very much"),
    ("A fair amount", "Quite a bit"),
    ("Not very much", "Little"),
    ("None at all", "Nothing"),
    
    # === Quantity ===
    ("Some", "A few"),
    ("Some", "Several"),
    ("All of them", "Every one of them"),
    ("Most of them", "The majority"),
    ("Some of them", "A few of them"),
    ("Few of them", "Not many of them"),
    
    # === Like me scale (Schwartz values) ===
    ("Very much like me", "Exactly like me"),
    ("Like me", "Similar to me"),
    ("Somewhat like me", "A bit like me"),
    ("A little like me", "Slightly like me"),
    ("Not like me", "Unlike me"),
    ("Not like me at all", "Completely unlike me"),
    
    # === Interest scale ===
    ("Very interested", "Extremely interested"),
    ("Fairly interested", "Quite interested"),
    ("Not very interested", "Somewhat uninterested"),
    ("Not at all interested", "Completely uninterested"),
    
    # === Worry/Concern scale ===
    ("Very worried", "Extremely worried"),
    ("Fairly worried", "Quite worried"),
    ("Not very worried", "Somewhat unworried"),
    ("Not at all worried", "Not worried"),
    ("Very concerned", "Extremely concerned"),
    ("Not concerned at all", "Not worried at all"),
    
    # === Democracy scale ===
    ("Very democratic", "Fully democratic"),
    ("Fairly democratic", "Quite democratic"),
    ("Not very democratic", "Somewhat undemocratic"),
    ("Not at all democratic", "Completely undemocratic"),
    ("No democracy at all", "Not democratic at all"),
    ("Slight democracy", "Slightly democratic"),
    ("Some democracy", "Somewhat democratic"),
    ("A lot of democracy", "Very democratic"),
    ("Democratic to the greatest extent", "Fully democratic"),
    ("Much less democratic", "Far less democratic"),
    ("Much more democratic", "Far more democratic"),
    
    # === Better/Worse ===
    ("Much better", "A lot better"),
    ("A little better", "Slightly better"),
    ("About the same", "No change"),
    ("A little worse", "Slightly worse"),
    ("Much worse", "A lot worse"),
    ("Become stronger", "Strengthen"),
    ("Become weaker", "Weaken"),
    ("Become stronger", "Get stronger"),
    ("Become weaker", "Get weaker"),
    ("Remain the same", "Stay the same"),
    ("Become stronger than in previous years", "Strengthen compared to before"),
    ("Become weaker than in previous years", "Weaken compared to before"),
    ("Remain the same as in previous years", "Stay the same as before"),
    ("Stayed the same", "Remained the same"),
    
    # === Influence ===
    ("Some influence", "Moderate influence"),
    ("No influence", "No impact"),
    
    # === Frequency with intervals ===
    ("Several times a day", "Multiple times daily"),
    ("Several times a week", "Multiple times weekly"),
    ("Several times a month", "Multiple times monthly"),
    ("Once a day", "Daily"),
    ("Once a week", "Weekly"),
    ("Once a month", "Monthly"),
    ("Every day", "Daily"),
    ("Every week", "Weekly"),
    ("Every month", "Monthly"),
    ("At least once a week", "Weekly or more"),
    ("At least once a month", "Monthly or more"),
    ("More than once a week", "Multiple times weekly"),
    ("A few times a week", "Several times weekly"),
    ("A few times a month", "Several times monthly"),
    ("A few times a year", "Several times yearly"),
    ("Less than once a month", "Less than monthly"),
    ("A few times", "Several times"),
    ("Many times", "Numerous times"),
    ("Once or twice", "A couple of times"),
    ("Just once or twice", "Only once or twice"),
    ("Only once", "Just once"),
    ("Most of the time", "Usually"),
    ("Most of the time", "Mostly"),
    
    # === Guarantee scale ===
    ("Fully guaranteed", "Completely guaranteed"),
    ("Somewhat guaranteed", "Partially guaranteed"),
    ("Not guaranteed", "Unguaranteed"),
    ("Not at all guaranteed", "Completely unguaranteed"),
    ("Guaranteed to a medium extent", "Moderately guaranteed"),
    ("Guaranteed to a limited extent", "Somewhat guaranteed"),
    ("Not guaranteed at all", "Completely unguaranteed"),
    ("Guaranteed to a great extent", "Highly guaranteed"),
    
    # === Justifiable scale ===
    ("Never justifiable", "Never justified"),
    ("Rarely justifiable", "Rarely justified"),
    ("Sometimes justifiable", "Sometimes justified"),
    ("Often justifiable", "Often justified"),
    ("Always justifiable", "Always justified"),
    ("Never justifiable", "Not justifiable"),
    ("Always justifiable", "Completely justifiable"),
    
    # === Essential/Critical scale ===
    ("Not essential at all", "Completely unessential"),
    ("Not very essential", "Not particularly essential"),
    ("Somewhat essential", "Fairly essential"),
    ("Very essential", "Extremely essential"),
    ("Moderately essential", "Fairly essential"),
    ("Critical", "Crucial"),
    ("Important but not critical", "Important but not crucial"),
    
    # === Membership/Belonging ===
    ("Active member", "Actively participating"),
    ("Inactive member", "Non-active member"),
    ("Don't belong", "Not a member"),
    
    # === Neither patterns (additional) ===
    ("Neither applies nor not", "Neutral"),
    ("Neither acceptable nor unacceptable", "Neutral"),
    ("Neither undermined nor enriched", "No effect"),
    ("Neither bad nor good for the economy", "Neutral economic effect"),
    ("Neither gone too far nor should go further", "About right"),
    
    # === Statement agreement ===
    ("Agree with Statement 1", "Support Statement 1"),
    ("Agree with Statement 2", "Support Statement 2"),
    
    # === Contact/Interaction ===
    ("No contact", "No interaction"),
    ("Some contact", "Some interaction"),
    
    # === Balanced ===
    ("Balanced", "Even"),
    
    # === Behavioral ===
    ("Have done", "Did this"),
    ("Might do", "Would consider doing"),
    
    # === Close scale ===
    ("Not close at all", "Not at all close"),
    ("Not very close", "Not particularly close"),
    ("Quite close", "Fairly close"),
    
    # === Trust patterns ===
    ("Most people can be trusted", "People can generally be trusted"),
    ("You can't be too careful", "You need to be careful"),
    
    # === Care scale ===
    ("Would not care", "Would be indifferent"),
    ("Would care a lot", "Would be very concerned"),
    
    # === None patterns ===
    ("None of them", "None"),
    ("None of these", "None of the above"),
    ("None of these are challenges", "None of the above"),
    
    # === Gender comparison ===
    ("Men more than women", "More men than women"),
    ("Women more than men", "More women than men"),
    
    # === Effective scale ===
    ("Not effective at all", "Completely ineffective"),
    ("Not very effective", "Somewhat ineffective"),
    ("Quite effective", "Fairly effective"),
    
    # === Left-Right political ===
    ("Far left", "Extreme left"),
    ("Far right", "Extreme right"),
]


# =============================================================================
# 5. REORDER PATTERNS
# =============================================================================
# Compound phrases where word order can be swapped.
# Format: (original, reordered)

REORDER_PATTERNS: List[Tuple[str, str]] = [
    # === Agreement ===
    ("Strongly agree", "Agree strongly"),
    ("Strongly disagree", "Disagree strongly"),
    ("Somewhat agree", "Agree somewhat"),
    ("Somewhat disagree", "Disagree somewhat"),
    ("Completely agree", "Agree completely"),
    ("Completely disagree", "Disagree completely"),
    ("Totally agree", "Agree totally"),
    ("Totally disagree", "Disagree totally"),
    
    # === Satisfaction ===
    ("Very satisfied", "Satisfied, very"),
    ("Very dissatisfied", "Dissatisfied, very"),
    ("Completely satisfied", "Satisfied completely"),
    ("Completely dissatisfied", "Dissatisfied completely"),
    ("Extremely satisfied", "Satisfied, extremely"),
    ("Extremely dissatisfied", "Dissatisfied, extremely"),
    ("Somewhat satisfied", "Satisfied, somewhat"),
    ("Somewhat dissatisfied", "Dissatisfied, somewhat"),
    
    # === Importance ===
    ("Very important", "Important, very"),
    ("Extremely important", "Important, extremely"),
    ("Not at all important", "Not important at all"),
    ("Somewhat important", "Important, somewhat"),
    
    # === Support/Oppose ===
    ("Strongly support", "Support strongly"),
    ("Strongly oppose", "Oppose strongly"),
    ("Somewhat support", "Support somewhat"),
    ("Somewhat oppose", "Oppose somewhat"),
    ("Strongly favor", "Favor strongly"),
    
    # === Approve/Disapprove ===
    ("Strongly approve", "Approve strongly"),
    ("Strongly disapprove", "Disapprove strongly"),
    
    # === Trust ===
    ("Highly trust", "Trust highly"),
    ("Completely trust", "Trust completely"),
    ("Somewhat trust", "Trust somewhat"),
    ("Slightly trust", "Trust slightly"),
    ("Mostly trust", "Trust mostly"),
    
    # === Likelihood ===
    ("Very likely", "Likely, very"),
    ("Very unlikely", "Unlikely, very"),
    ("Extremely likely", "Likely, extremely"),
    ("Somewhat likely", "Likely, somewhat"),
    
    # === Frequency ===
    ("Very often", "Often, very"),
    ("Quite often", "Often, quite"),
    ("Fairly often", "Often, fairly"),
    ("Very frequently", "Frequently, very"),
    ("Quite frequently", "Frequently, quite"),
    
    # === Good/Bad ===
    ("Very good", "Good, very"),
    ("Very bad", "Bad, very"),
    ("Extremely good", "Good, extremely"),
    ("Extremely bad", "Bad, extremely"),
    ("Fairly good", "Good, fairly"),
    ("Fairly bad", "Bad, fairly"),
    
    # === Interest ===
    ("Very interested", "Interested, very"),
    ("Somewhat interested", "Interested, somewhat"),
    
    # === Happy ===
    ("Very happy", "Happy, very"),
    ("Quite happy", "Happy, quite"),
    ("Not very happy", "Not happy, very"),
]


# =============================================================================
# 6. PRONOUN EXACT MATCHES
# =============================================================================
# Standalone options where pronouns can be added or removed.
# Only applied to EXACT matches to avoid substring issues.

PRONOUN_EXACT: List[Tuple[str, str]] = [
    # Add "I" to bare forms
    ("Agree", "I agree"),
    ("Disagree", "I disagree"),
    ("Satisfied", "I am satisfied"),
    ("Dissatisfied", "I am dissatisfied"),
    ("Approve", "I approve"),
    ("Disapprove", "I disapprove"),
    ("Support", "I support"),
    ("Oppose", "I oppose"),
    
    # Yes/No with pronouns
    ("Yes", "Yes, I do"),
    ("No", "No, I don't"),
    ("Yes", "Yes, I have"),
    ("No", "No, I haven't"),
    
    # Trust forms
    ("Trust", "I trust"),
    ("Trust completely", "I trust completely"),
    ("Trust somewhat", "I trust somewhat"),
    
    # Interest
    ("Interested", "I am interested"),
    ("Not interested", "I am not interested"),
]


# =============================================================================
# 7. ANSWER VARIATION GENERATOR CLASS
# =============================================================================

class AnswerVariationGenerator:
    """
    Generates variations of answer options for surface form sensitivity testing.
    
    Supports three types of variations:
    - Synonym: Word-level and phrase-level synonym substitutions
    - Reorder: Word order permutations ("Strongly agree" <-> "Agree strongly")
    - Pronoun: Adding/removing first-person pronouns ("Agree" <-> "I agree")
    
    Example usage:
        generator = AnswerVariationGenerator()
        options = ["Strongly agree", "Agree", "Disagree", "Strongly disagree"]
        variations = generator.generate_all_variations(options)
        # Returns: {"synonym": [...], "reorder": [...], "pronoun": [...]}
    """
    
    def __init__(self):
        """Initialize generator with compiled patterns for efficiency."""
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        """Build lookup tables for fast variation matching."""
        # Phrase synonyms (case-insensitive exact match)
        self._phrase_synonym_map: Dict[str, List[Tuple[str, str]]] = {}
        for p1, p2 in SYNONYM_PHRASES:
            key1 = p1.lower()
            key2 = p2.lower()
            if key1 not in self._phrase_synonym_map:
                self._phrase_synonym_map[key1] = []
            if key2 not in self._phrase_synonym_map:
                self._phrase_synonym_map[key2] = []
            self._phrase_synonym_map[key1].append((p1, p2))
            self._phrase_synonym_map[key2].append((p2, p1))
        
        # Reorder patterns (bidirectional)
        self._reorder_map: Dict[str, Tuple[str, str]] = {}
        for orig, reord in REORDER_PATTERNS:
            self._reorder_map[orig.lower()] = (orig, reord)
            self._reorder_map[reord.lower()] = (reord, orig)
        
        # Pronoun patterns (bidirectional)
        self._pronoun_add_map: Dict[str, Tuple[str, str]] = {}
        self._pronoun_remove_map: Dict[str, Tuple[str, str]] = {}
        for without, with_pronoun in PRONOUN_EXACT:
            self._pronoun_add_map[without.lower()] = (without, with_pronoun)
            self._pronoun_remove_map[with_pronoun.lower()] = (with_pronoun, without)
    
    def is_special_value(self, option: str) -> bool:
        """Check if option is a special value (DK/Refused/NA)."""
        return option.lower().strip() in SPECIAL_VALUES
    
    def is_option_ineligible(self, option: str) -> bool:
        """Check if option matches ineligibility patterns (parties, ethnicities, etc.)."""
        option_lower = option.lower()
        for category, patterns in INELIGIBILITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, option_lower):
                    return True
        return False
    
    def is_question_ineligible(self, question_text: str) -> bool:
        """Check if question text indicates ineligibility."""
        question_lower = question_text.lower()
        for pattern in INELIGIBLE_QUESTION_PATTERNS:
            if re.search(pattern, question_lower):
                return True
        return False
    
    def generate_synonym_variations(self, option: str) -> List[AnswerVariation]:
        """
        Generate synonym variations for an answer option.
        
        Checks phrase-level matches first, then word-level substitutions.
        """
        if self.is_special_value(option) or self.is_option_ineligible(option):
            return []
        
        variations = []
        option_lower = option.lower()
        
        # 1. Check phrase-level synonyms (exact match)
        if option_lower in self._phrase_synonym_map:
            for orig, synonym in self._phrase_synonym_map[option_lower]:
                # Preserve original case pattern if possible
                if option_lower == orig.lower():
                    variation_text = synonym
                else:
                    variation_text = orig
                
                variations.append(AnswerVariation(
                    original=option,
                    variation=variation_text,
                    variation_type="synonym",
                    rule_applied=f"'{orig}' <-> '{synonym}'"
                ))
        
        # 2. Check word-level synonyms (boundary matching)
        for word1, word2 in SYNONYM_WORDS:
            pattern1 = rf'\b{re.escape(word1)}\b'
            pattern2 = rf'\b{re.escape(word2)}\b'
            
            # Check if word1 is in option -> replace with word2
            if re.search(pattern1, option, re.IGNORECASE):
                variation_text = re.sub(pattern1, word2, option, flags=re.IGNORECASE)
                if variation_text != option:  # Only add if actually changed
                    variations.append(AnswerVariation(
                        original=option,
                        variation=variation_text,
                        variation_type="synonym",
                        rule_applied=f"'{word1}' -> '{word2}'"
                    ))
            
            # Check if word2 is in option -> replace with word1
            elif re.search(pattern2, option, re.IGNORECASE):
                variation_text = re.sub(pattern2, word1, option, flags=re.IGNORECASE)
                if variation_text != option:
                    variations.append(AnswerVariation(
                        original=option,
                        variation=variation_text,
                        variation_type="synonym",
                        rule_applied=f"'{word2}' -> '{word1}'"
                    ))
        
        return variations
    
    def generate_reorder_variations(self, option: str) -> List[AnswerVariation]:
        """Generate word-order variations for an answer option."""
        if self.is_special_value(option) or self.is_option_ineligible(option):
            return []
        
        variations = []
        option_lower = option.lower()
        
        if option_lower in self._reorder_map:
            orig, reordered = self._reorder_map[option_lower]
            # Determine which direction the variation goes
            if option_lower == orig.lower():
                variation_text = reordered
            else:
                variation_text = orig
            
            variations.append(AnswerVariation(
                original=option,
                variation=variation_text,
                variation_type="reorder",
                rule_applied=f"'{orig}' <-> '{reordered}'"
            ))
        
        return variations
    
    def generate_pronoun_variations(self, option: str) -> List[AnswerVariation]:
        """Generate pronoun addition/removal variations for an answer option."""
        if self.is_special_value(option) or self.is_option_ineligible(option):
            return []
        
        variations = []
        option_lower = option.lower()
        
        # Check for pronoun addition (bare form -> with pronoun)
        if option_lower in self._pronoun_add_map:
            without, with_pronoun = self._pronoun_add_map[option_lower]
            variations.append(AnswerVariation(
                original=option,
                variation=with_pronoun,
                variation_type="pronoun",
                rule_applied=f"'{without}' -> '{with_pronoun}'"
            ))
        
        # Check for pronoun removal (with pronoun -> bare form)
        if option_lower in self._pronoun_remove_map:
            with_pronoun, without = self._pronoun_remove_map[option_lower]
            variations.append(AnswerVariation(
                original=option,
                variation=without,
                variation_type="pronoun",
                rule_applied=f"'{with_pronoun}' -> '{without}'"
            ))
        
        return variations
    
    def generate_all_variations(
        self, 
        options: List[str]
    ) -> Dict[str, List[AnswerVariation]]:
        """Generate all variations, deduplicated by (original, variation) pair."""
        all_variations = {
            "synonym": [],
            "reorder": [],
            "pronoun": [],
        }
        
        for option in options:
            # Track seen (original, variation) pairs within each type
            seen_synonym = set()
            for v in self.generate_synonym_variations(option):
                key = (v.original, v.variation)
                if key not in seen_synonym:
                    seen_synonym.add(key)
                    all_variations["synonym"].append(v)
            
            # Reorder and pronoun are already 1:1, but for consistency:
            seen_reorder = set()
            for v in self.generate_reorder_variations(option):
                key = (v.original, v.variation)
                if key not in seen_reorder:
                    seen_reorder.add(key)
                    all_variations["reorder"].append(v)
            
            seen_pronoun = set()
            for v in self.generate_pronoun_variations(option):
                key = (v.original, v.variation)
                if key not in seen_pronoun:
                    seen_pronoun.add(key)
                    all_variations["pronoun"].append(v)
        
        return all_variations
    
    def get_coverage_stats(self, options: List[str]) -> Dict[str, int]:
        """
        Get coverage statistics for a list of options.
        
        Returns count of options that have at least one variation of each type.
        """
        stats = {"synonym": 0, "reorder": 0, "pronoun": 0, "any": 0}
        
        for option in options:
            has_any = False
            if self.generate_synonym_variations(option):
                stats["synonym"] += 1
                has_any = True
            if self.generate_reorder_variations(option):
                stats["reorder"] += 1
                has_any = True
            if self.generate_pronoun_variations(option):
                stats["pronoun"] += 1
                has_any = True
            if has_any:
                stats["any"] += 1
        
        return stats
    
    def filter_substantive_options(self, options: List[str]) -> List[str]:
        """Filter out special values from options list."""
        return [opt for opt in options if not self.is_special_value(opt)]


# =============================================================================
# 8. UTILITY FUNCTIONS
# =============================================================================

def get_rule_statistics() -> Dict[str, int]:
    """Get counts of all rule types."""
    return {
        "special_values": len(SPECIAL_VALUES),
        "ineligibility_pattern_groups": len(INELIGIBILITY_PATTERNS),
        "ineligible_question_patterns": len(INELIGIBLE_QUESTION_PATTERNS),
        "synonym_word_pairs": len(SYNONYM_WORDS),
        "synonym_phrase_pairs": len(SYNONYM_PHRASES),
        "reorder_patterns": len(REORDER_PATTERNS),
        "pronoun_patterns": len(PRONOUN_EXACT),
    }


def print_rule_statistics():
    """Print statistics about the variation rules."""
    stats = get_rule_statistics()
    print("=" * 60)
    print("VARIATION RULES STATISTICS")
    print("=" * 60)
    print(f"\nSpecial values to filter:     {stats['special_values']:4d}")
    print(f"Ineligibility pattern groups: {stats['ineligibility_pattern_groups']:4d}")
    print(f"Question-level patterns:      {stats['ineligible_question_patterns']:4d}")
    print(f"\nSynonym word pairs:           {stats['synonym_word_pairs']:4d}")
    print(f"Synonym phrase pairs:         {stats['synonym_phrase_pairs']:4d}")
    print(f"Reorder patterns:             {stats['reorder_patterns']:4d}")
    print(f"Pronoun exact matches:        {stats['pronoun_patterns']:4d}")
    print("=" * 60)


if __name__ == "__main__":
    print_rule_statistics()
    
    # Demo usage
    print("\n" + "=" * 60)
    print("DEMO: Generating variations")
    print("=" * 60)
    
    generator = AnswerVariationGenerator()
    demo_options = [
        "Strongly agree",
        "Agree", 
        "Neither agree nor disagree",
        "Disagree",
        "Strongly disagree",
        "Don't know",  # Should be filtered
    ]
    
    print(f"\nInput options: {demo_options}")
    
    variations = generator.generate_all_variations(demo_options)
    
    for var_type, var_list in variations.items():
        print(f"\n{var_type.upper()} variations ({len(var_list)}):")
        for v in var_list:
            print(f"  '{v.original}' -> '{v.variation}' [{v.rule_applied}]")
    
    print(f"\nCoverage stats: {generator.get_coverage_stats(demo_options)}")