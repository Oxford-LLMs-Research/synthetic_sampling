# Identifying Missing Target Questions

## Summary

Your dataset description reports **261 unique target questions**, but your model results only show **249 target questions**. This means **12 target questions are missing** from the model results.

## How to Find the Missing Targets

### Option 1: Using the find_missing_targets.py script

If you have model results files (JSONL format), run:

```bash
python scripts/find_missing_targets.py outputs/dataset_description.md <path_to_results> --output missing_targets.json
```

Where `<path_to_results>` can be:
- A directory containing `*_results.jsonl` files
- One or more specific JSONL result files
- A glob pattern like `results/*/*.jsonl`

### Option 2: Manual comparison

1. Extract all 261 targets from the dataset description:
   ```bash
   python scripts/extract_targets_from_description.py > all_261_targets.txt
   ```

2. Extract targets from your model results (if you have the ResultsAnalyzer):
   ```python
   from synthetic_sampling.evaluation import ResultsAnalyzer
   analyzer = ResultsAnalyzer.from_jsonl("your_results.jsonl")
   result_targets = set(analyzer.targets)
   print(f"Found {len(result_targets)} targets in results")
   ```

3. Compare the two sets to find missing ones.

## Possible Reasons for Missing Targets

1. **Filtering during generation**: Some targets might have been excluded if they had insufficient data or failed validation
2. **Failed generation**: Some target questions might have failed to generate instances
3. **Evaluation filtering**: Some targets might have been filtered out during evaluation (e.g., if they had no valid predictions)
4. **Country-specific targets**: Some targets might be country-specific and not present in all surveys

## Next Steps

1. Run the comparison script to identify the exact 12 missing targets
2. Check the generation logs to see if any targets were skipped
3. Verify if the missing targets are country-specific or survey-specific
4. Check if the missing targets have very few instances (might have been filtered out)
