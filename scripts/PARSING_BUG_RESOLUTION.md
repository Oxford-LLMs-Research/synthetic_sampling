# Parsing Bug Resolution: Complete Analysis

## Summary

**The discrepancy was entirely due to a parsing bug, not missing data.**

After fixing the `parse_example_id()` function in `src/synthetic_sampling/evaluation/evaluation.py`:

- **Dataset description**: 261 unique targets
- **Model results (after fix)**: 261 unique targets  
- **Missing targets**: 0
- **Extra targets**: 0

**All 261 target questions are present in the model results.**

## What Was Happening

### Before the Fix

The `parse_example_id()` function used `rsplit('_', 1)` which incorrectly split target codes containing underscores:

- Example: `arabbarometer_700512_Q725_4_s3m2`
- Buggy parsing: Extracted `4` instead of `Q725_4`
- Result: 25 targets with underscores were split into 13 fragments

### The Numbers

- **Before fix**: Found 249 target codes (236 real + 13 fragments)
- **After fix**: Found 261 target codes (all real, no fragments)
- **Net effect**: The 12 "missing" targets (261 - 249) were actually 25 targets incorrectly parsed as 13 fragments

## The 25 Previously "Missing" Targets

These targets were always in the results but were incorrectly parsed:

1. `Q129A_1` - Wellbeing (2,271 instances)
2. `Q201B_13` - Institutional trust (2,235 instances)
3. `Q204A_2` - Institutional trust (2,505 instances)
4. `Q204_2` - Institutional trust (2,502 instances)
5. `Q2061A_KUW` - Political attitudes (108 instances)
6. `Q277_2` - Wellbeing (2,508 instances)
7. `Q303B_1` - Political participation (2,022 instances)
8. `Q534_2` - Contemporary issues (2,499 instances)
9. `Q547_4` - Contemporary issues (2,508 instances)
10. `Q550A_2` - Political attitudes (1,209 instances)
11. `Q550A_6` - Political attitudes (1,209 instances)
12. `Q553B_6` - Contemporary issues (1,299 instances)
13. `Q601_21B` - Social attitudes (1,290 instances)
14. `Q610_6B` - Values/identity (129 instances)
15. `Q628_4A` - Social attitudes (1,212 instances)
16. `Q725_4` - Political attitudes (2,490 instances)
17. `Q725_5` - Political attitudes (2,487 instances)
18. `Q725_6` - Political attitudes (2,256 instances)
19. `Q728_5B` - Political attitudes (1,419 instances)
20. `Q729B_6` - Political attitudes (1,050 instances)
21. `Q731_5` - Political attitudes (2,502 instances)
22. `Q82_CUSMA` - Institutional trust (147 instances)
23. `Q82_NAFTA` - Institutional trust (87 instances)
24. `Q916_1` - Social attitudes (2,223 instances)
25. `QMOR8_2` - Political attitudes (447 instances)

## The 13 Parsing Artifacts (Now Resolved)

These were fragments created by the buggy parsing:

- `1`, `2`, `4`, `5`, `6`, `13` - Numeric suffixes from multiple targets
- `21B`, `4A`, `5B`, `6B` - Partial target codes
- `CUSMA`, `KUW`, `NAFTA` - Suffixes from country-specific targets

**After the fix, these artifacts no longer appear** - they're correctly parsed as part of their full target codes.

## Impact on Previous Calculations

### Before Fix (Incorrect)

- **Target-specific metrics**: Artifacts created fake target groups
- **Baselines**: Calculated per artifact instead of real target
- **Variance ratios**: Split incorrectly across fragments
- **JS divergence**: Distribution comparisons on fragments, not real targets
- **Any analysis grouping by target**: Affected by incorrect grouping

### After Fix (Correct)

- All calculations now group by correct target codes
- All 261 targets properly identified
- No parsing artifacts
- All metrics calculated correctly

## Files Fixed

1. **`src/synthetic_sampling/evaluation/evaluation.py`**
   - Fixed `parse_example_id()` function (line 147)
   - Now correctly handles target codes with underscores

2. **`scripts/generate_main_results_figure.py`**
   - Fixed `extract_question_id()` function (line 145)
   - Now correctly extracts question IDs with underscores

## Verification

After the fix:
- ✅ All 261 targets found in results
- ✅ No missing targets
- ✅ No parsing artifacts
- ✅ All target codes with underscores correctly parsed

## Conclusion

**No target questions were actually lost.** The entire discrepancy was due to the parsing bug. All 261 target questions are present in the model results and are now correctly identified.
