# Missing Target Questions: Analysis and Explanation

## Summary

The dataset description reports **261 unique target questions**, while model results contain **249 unique target questions**. This discrepancy of 12 questions is actually the result of:

- **25 target questions** that are in the dataset but missing from model results
- **13 "extra" target codes** that appear in results but are actually **parsing artifacts** (fragments of incorrectly parsed target codes)

**Net difference: 25 missing - 13 artifacts = 12 actual missing targets**

## The 25 Missing Target Questions

These targets are present in the dataset description but absent from model results:

### Contemporary Issues (3 targets)
1. `Q534_2` - Media information security (2,499 instances)
2. `Q547_4` - Climate environment policy (2,508 instances)
3. `Q553B_6` - Climate environment, Western corporations (1,299 instances)

### Institutional Trust (5 targets)
4. `Q201B_13` - Trust in religious leaders (2,235 instances)
5. `Q204A_2` - Healthcare system satisfaction (2,505 instances)
6. `Q204_2` - Government performance evaluation (2,502 instances)
7. `Q82_CUSMA` - Canada-United States-Mexico Agreement confidence (147 instances)
8. `Q82_NAFTA` - NAFTA confidence (87 instances)

### Political Attitudes (10 targets)
9. `Q2061A_KUW` - Kuwait political priorities (108 instances)
10. `Q550A_2` - Corruption as pillar of human dignity (1,209 instances)
11. `Q550A_6` - Civil rights as pillar of human dignity (1,209 instances)
12. `Q725_4` - Chinese President Xi Jinping foreign policies (2,490 instances)
13. `Q725_5` - Iranian Supreme Leader foreign policies (2,487 instances)
14. `Q725_6` - Saudi Crown Prince foreign policies (2,256 instances)
15. `Q728_5B` - Iran's nuclear program threat (1,419 instances)
16. `Q729B_6` - China's foreign aid (1,050 instances)
17. `Q731_5` - Chinese policy on Israeli-Palestinian conflict (2,502 instances)
18. `QMOR8_2` - Earthquake-resistant building law (447 instances)

### Political Participation (1 target)
19. `Q303B_1` - Votes counted fairly in elections (2,022 instances)

### Social Attitudes (3 targets)
20. `Q601_21B` - Gender attitudes, minimum wage statement (1,290 instances)
21. `Q628_4A` - Gender attitudes, marriage choice freedom (1,212 instances)
22. `Q916_1` - Migration attitudes, foreign domestic workers law (2,223 instances)

### Values/Identity (1 target)
23. `Q610_6B` - Religious values, Bible reading (129 instances)

### Wellbeing (2 targets)
24. `Q129A_1` - Household food security (2,271 instances)
25. `Q277_2` - Neighborhood violence frequency (2,508 instances)

## The 13 "Extra" Target Codes (Parsing Artifacts)

These are **not real target questions** but rather fragments created by incorrect parsing of target codes that contain underscores. The parsing bug splits target codes on underscores, extracting only the suffix portion.

### Direct Fragment Matches

| Extra Fragment | Source Target Code | Instances Match |
|----------------|-------------------|-----------------|
| `21B` | `Q601_21B` | 1,290 ✓ |
| `4A` | `Q628_4A` | 1,212 ✓ |
| `5B` | `Q728_5B` | 1,419 ✓ |
| `6B` | `Q610_6B` | 129 ✓ |
| `CUSMA` | `Q82_CUSMA` | 147 ✓ |
| `KUW` | `Q2061A_KUW` | 108 ✓ |
| `NAFTA` | `Q82_NAFTA` | 87 ✓ |

### Partial Fragment Matches

The following fragments are suffixes from multiple missing targets:

| Fragment | Appears in Missing Targets | Total Instances |
|----------|----------------------------|-----------------|
| `1` | `Q129A_1`, `Q201B_13`, `Q2061A_KUW`, `Q303B_1`, `Q731_5`, `Q916_1` | 6,516 |
| `2` | `Q129A_1`, `Q201B_13`, `Q204A_2`, `Q204_2`, `Q2061A_KUW`, `Q277_2`, `Q534_2`, `Q550A_2`, `Q601_21B`, `Q628_4A`, `Q725_4`, `Q725_5`, `Q725_6`, `Q728_5B`, `Q729B_6`, `Q82_CUSMA`, `Q82_NAFTA`, `QMOR8_2` | 11,670 |
| `4` | `Q204A_2`, `Q204_2`, `Q534_2`, `Q547_4`, `Q628_4A`, `Q725_4` | 4,998 |
| `5` | `Q534_2`, `Q547_4`, `Q550A_2`, `Q550A_6`, `Q553B_6`, `Q725_4`, `Q725_5`, `Q725_6`, `Q728_5B`, `Q731_5` | 4,989 |
| `6` | `Q2061A_KUW`, `Q550A_6`, `Q553B_6`, `Q601_21B`, `Q610_6B`, `Q628_4A`, `Q725_6`, `Q729B_6`, `Q916_1` | 5,814 |
| `13` | `Q201B_13` | 2,235 |

## Root Cause: Parsing Bug

The issue originates in the `parse_example_id()` function in `src/synthetic_sampling/evaluation/evaluation.py` (line 181). 

**Problem:** The function uses `rsplit('_', 1)` to split the example_id, which incorrectly handles target codes that contain underscores themselves.

**Example:**
- Correct target code: `Q725_4`
- Example ID format: `arabbarometer_700512_Q725_4_s3m2`
- Incorrect parsing: Splits on last underscore → extracts `4` instead of `Q725_4`

**Affected target codes:** All targets with underscores in their codes (e.g., `Q725_4`, `Q601_21B`, `Q82_CUSMA`) were incorrectly parsed, creating fragment artifacts.

## Where the Targets Were Lost

The 25 missing targets likely failed during one of these stages:

1. **Instance Generation**: Some targets may have failed validation or had insufficient data
2. **Evaluation Filtering**: Targets with no valid predictions may have been filtered out
3. **Country-Specific Filtering**: Some targets are country-specific:
   - `Q2061A_KUW` - Kuwait only
   - `Q82_CUSMA`, `Q82_NAFTA` - North America only
   - `Q610_6B` - Likely country-specific (Bible reading)

## Impact on Analysis

- **Total instances affected**: ~35,000+ instances across the 25 missing targets
- **Survey distribution**: Missing targets are primarily from:
  - Arab Barometer (international relations questions)
  - ESS Wave 11 (wellbeing questions)
  - WVS (country-specific questions)

## Recommendations for Documentation

1. **Acknowledge the parsing bug**: The 13 "extra" targets are artifacts, not real questions
2. **Report the 25 missing targets**: These represent actual data loss
3. **Note the net difference**: 12 targets (25 - 13) represents the effective reduction
4. **Document country-specific exclusions**: Some targets are intentionally country-specific

## Files Generated

- `missing_targets.json` - Complete analysis with all missing and extra targets
- This document - Explanation for paper documentation
