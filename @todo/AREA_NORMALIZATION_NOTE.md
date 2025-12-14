# NOTE: Area Normalization Clarification

## Current Implementation (COMPLETED)

The hardness analysis uses **constant TOTAL area** (100 ha across all tests):
- 3 farms = 100 ha total (33.3 ha/farm)
- 10 farms = 100 ha total (10.0 ha/farm)
- 50 farms = 100 ha total (2.0 ha/farm)

This was done to isolate the effect of farm count on hardness.

## Requested Implementation (FOR FUTURE)

User prefers **constant area PER FARM** (e.g., 1 ha/farm):
- 10 farms = 10 ha total (1.0 ha/farm)
- 20 farms = 20 ha total (1.0 ha/farm)  
- 50 farms = 50 ha total (1.0 ha/farm)

This would better represent real-world scenarios where farms don't shrink as you add more.

## To Re-run with Constant Area Per Farm

Modify `hardness_comprehensive_analysis.py`:

```python
# Change this:
TARGET_TOTAL_AREA = 100.0  # hectares (constant TOTAL)

# To this:
TARGET_AREA_PER_FARM = 1.0  # hectares (constant PER FARM)

# Then in sample_farms_constant_area():
target_total_area = n_farms * TARGET_AREA_PER_FARM
```

This will make:
- 3 farms → 3 ha total
- 25 farms → 25 ha total
- 100 farms → 100 ha total

The solve times will likely increase more dramatically with this approach since both:
1. Number of farms increases
2. Total area (and thus problem scale) increases

## Impact on Results

With constant area per farm:
- Quadratic terms will grow even faster
- Hardness curves will be steeper
- May hit practical limits sooner (memory, time)
- More realistic for actual deployment scenarios

Current results are still valid for understanding scaling behavior, just with a different normalization.
