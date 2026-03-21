# Analysis Optimization Guide

## Problem

Running multiple analysis scripts (main figure, disaggregated analysis, profile richness) involves:
1. **Redundant data loading**: Each script loads the same JSONL files
2. **Redundant enrichment**: Each script enriches instances with metadata (slow!)
3. **Redundant metric computation**: Some metrics are computed multiple times

## Solution: Shared Cache System

We've created a **shared caching system** (`shared_data_cache.py`) that:
- Caches enriched instances after first load
- Shares cache across all analysis scripts
- Automatically invalidates cache when input files change
- Dramatically speeds up subsequent analyses

## How It Works

### Cache Location
- **Default: `analysis/.cache/enriched/`** (shared across all scripts)
- Cache files: `{model_name}_{cache_key}.json`
- Cache key includes: model name, filters, input file paths + modification times
- **All analysis scripts now use the same shared cache by default** for maximum reuse

### Cache Invalidation
Cache is automatically invalidated when:
- Input files are modified (detected via file modification time)
- Different filters are used (profile, survey)
- Different model is analyzed

## Usage

### 1. Profile Richness Analysis (with cache)

```bash
# First run: Computes and caches enriched data
python scripts/analyze_profile_richness.py \
    --results-dir "C:\Users\murrn\cursor\synthetic_sampling\results" \
    --inputs "outputs/main_data" \
    --output analysis/profile_richness

# Subsequent runs: Uses cache (much faster!)
python scripts/analyze_profile_richness.py \
    --results-dir "C:\Users\murrn\cursor\synthetic_sampling\results" \
    --inputs "outputs/main_data" \
    --output analysis/profile_richness
```

### 2. Disaggregated Analysis (uses same cache)

```bash
# Uses cache from previous analysis if available
python scripts/analyze_disaggregated.py \
    --results-dir results \
    --inputs outputs/main_data \
    --output analysis/disaggregated
```

### 3. Profile Richness by Topic (new, uses cache)

```bash
# Tests if harder topics benefit more from profile richness
python scripts/analyze_profile_richness_by_topic.py \
    --results-dir "C:\Users\murrn\cursor\synthetic_sampling\results" \
    --inputs "outputs/main_data" \
    --output analysis/profile_richness_by_topic
```

## Performance Benefits

### Without Cache
- **First analysis**: ~5-10 minutes (load + enrich + compute)
- **Second analysis**: ~5-10 minutes (reload + re-enrich + recompute)
- **Total for 3 analyses**: ~15-30 minutes

### With Cache
- **First analysis**: ~5-10 minutes (load + enrich + compute + cache)
- **Second analysis**: ~1-2 minutes (load from cache + compute)
- **Third analysis**: ~1-2 minutes (load from cache + compute)
- **Total for 3 analyses**: ~7-14 minutes (**50% faster!**)

## Cache Management

### Clear Cache
```bash
# Delete cache directory
rm -rf analysis/.cache/enriched
```

### Force Reload
```bash
# Use --no-cache or --force-reload flag
python scripts/analyze_profile_richness.py \
    --results-dir results \
    --inputs outputs/main_data \
    --no-cache
```

### Check Cache Status
```bash
# List cache files
ls analysis/.cache/enriched/
```

## Best Practices

1. **Always provide `--inputs`** when you have main_data available
   - Enables caching
   - Provides metadata for disaggregated analyses
   
2. **Use consistent cache directory** across scripts
   - Default: `{output_dir}/.cache/enriched/`
   - Or specify: `--cache-dir analysis/.cache/enriched`

3. **Run analyses in sequence** to maximize cache reuse
   - First: Main figure or profile richness (creates cache)
   - Then: Disaggregated or by-topic analyses (reuses cache)

4. **Share cache across team members**
   - Cache files are JSON (portable)
   - Can be committed to git (if small) or shared via cloud storage

## New Analysis: Profile Richness by Topic

The new `analyze_profile_richness_by_topic.py` script:
- Tests if harder topics benefit more/less from profile richness
- Computes metrics for each topic × profile combination
- Analyzes benefit by topic difficulty
- Uses shared cache (fast!)

### Example Output
```
PROFILE RICHNESS BENEFIT BY TOPIC DIFFICULTY
================================================================================

Topic difficulty threshold: 45.2% accuracy

Easy topics (≥45.2% accuracy):
  Average improvement: +1.2%
  N topics: 15

Hard topics (<45.2% accuracy):
  Average improvement: +2.8%
  N topics: 12

→ Harder topics benefit MORE from profile richness (+1.6% difference)
```

## Migration Guide

### Existing Scripts
All existing scripts continue to work without `--inputs`:
- They fall back to direct loading (no enrichment)
- No breaking changes

### New Scripts
New scripts should use the shared cache:
```python
from shared_data_cache import get_all_models_enriched

instances_by_model = get_all_models_enriched(
    results_dir=results_dir,
    input_paths=[Path("outputs/main_data")],
    cache_dir=cache_dir,
    profile_filter=None,
    model_whitelist=None,
    force_reload=False,
    verbose=True,
)
```

## Troubleshooting

### Cache not being used?
- Check that `--inputs` is provided
- Check that cache directory exists and is writable
- Check that input file paths are correct

### Stale cache?
- Use `--no-cache` or `--force-reload` to refresh
- Or delete cache directory manually

### Cache too large?
- Cache files are per model (can be large)
- Consider using `--models` to limit which models are cached
- Cache can be deleted and regenerated anytime
