# Testing Guide - SimulatedAnnealing Fallback

## Overview

Both alternative implementations now support **automatic fallback to SimulatedAnnealingSampler** when no D-Wave token is provided. This enables extensive testing without QPU access!

## How It Works

### Conditional Logic
```python
# Both solvers detect missing token
use_simulated_annealing = (token is None or token == 'YOUR_DWAVE_TOKEN_HERE')

if use_simulated_annealing:
    sampler = neal.SimulatedAnnealingSampler()  # Classical simulation
else:
    sampler = DWaveSampler(token=token)  # Real QPU
```

### What Changes
- **With Token**: Uses actual D-Wave QPU hardware
- **Without Token**: Uses `neal.SimulatedAnnealingSampler` (classical simulation)
- **Automatic**: No code changes needed, just omit the token

## Testing Steps

### 1. Install Neal (if not already installed)
```powershell
conda activate oqi
pip install dwave-neal
```

### 2. Test Alternative 1: Custom Hybrid Workflow

#### Run Unit Tests
```powershell
cd @todo
python test_custom_hybrid.py
```

Expected output:
```
[TEST 3: Hybrid Framework Availability]
  ✓ dwave-hybrid imported successfully
  ✓ HYBRID_AVAILABLE flag is True
```

#### Run Benchmark (SimulatedAnnealing Mode)
```powershell
python comprehensive_benchmark_CUSTOM_HYBRID.py --config 10
```

Expected behavior:
- No D-Wave token required
- Uses `neal.SimulatedAnnealingSampler` instead of QPU
- Message: "Using SimulatedAnnealingSampler (neal) for testing"
- Both farm and patch scenarios complete successfully

### 3. Test Alternative 2: Decomposed QPU

#### Run Unit Tests
```powershell
python test_decomposed.py
```

Expected output:
```
[TEST 4: Low-Level QPU Sampler Availability]
  ✓ DWaveSampler imported successfully
  ✓ LOWLEVEL_QPU_AVAILABLE flag is True

[TEST 5: Decomposed Solver Function]
  ✓ solve_with_decomposed_qpu function imported
```

#### Run Benchmark (Hybrid: Gurobi + SimulatedAnnealing)
```powershell
python comprehensive_benchmark_DECOMPOSED.py --config 10
```

Expected behavior:
- **Farm scenario**: Uses Gurobi (classical MINLP) ✓
- **Patch scenario**: Uses `neal.SimulatedAnnealingSampler` instead of QPU
- Message: "SOLVING WITH SIMULATED ANNEALING (Testing Mode - No QPU)"
- Results saved to `Benchmarks/DECOMPOSED/`

### 4. Compare Results

Both benchmarks create JSON files:
```
Benchmarks/
├── CUSTOM_HYBRID/
│   └── results_config_10_TIMESTAMP.json
└── DECOMPOSED/
    └── results_config_10_TIMESTAMP.json
```

Check solver performance:
```powershell
# View results
Get-Content "Benchmarks\CUSTOM_HYBRID\results_config_10_*.json" | ConvertFrom-Json | Format-List
Get-Content "Benchmarks\DECOMPOSED\results_config_10_*.json" | ConvertFrom-Json | Format-List
```

## Expected Performance

### Alternative 1: Custom Hybrid (SimulatedAnnealing Mode)
```
Farm Scenario:
  - gurobi: ~0.1-1s (classical MINLP)
  - custom_hybrid: ~1-5s (racing: Tabu + SA, no QPU)

Patch Scenario:
  - gurobi: ~0.1-1s (classical BIP)
  - custom_hybrid: ~1-5s (racing: Tabu + SA, no QPU)
```

### Alternative 2: Decomposed (Hybrid Mode)
```
Farm Scenario:
  - gurobi: ~0.1-1s (classical MINLP)

Patch Scenario:
  - simulated_annealing: ~1-5s (neal, no QPU)
```

## Output Indicators

### SimulatedAnnealing Mode Active
Look for these messages:

**Alternative 1:**
```
SOLVING WITH CUSTOM HYBRID WORKFLOW (Simulated Annealing - No QPU)
Note: Using neal.SimulatedAnnealingSampler for testing without D-Wave token
    ✅ Workflow structure (Testing Mode):
       Racing Branches:
         - Tabu Search (interruptible)
         - Simulated Annealing (neal)
```

**Alternative 2:**
```
SOLVING WITH SIMULATED ANNEALING (Testing Mode - No QPU)
Note: Using neal.SimulatedAnnealingSampler for testing without D-Wave token
  ✓ SimulatedAnnealingSampler ready (neal)
  ✓ Testing mode: No QPU required
```

### JSON Output Differences

**With SimulatedAnnealing:**
```json
{
  "solver_name": "simulated_annealing",
  "sampler_type": "simulated_annealing",
  "qpu_access_time": 0.0,
  "sampler_config": {
    "num_reads": 1000,
    "sampler": "neal.SimulatedAnnealingSampler"
  }
}
```

**With Real QPU:**
```json
{
  "solver_name": "dwave_decomposed_qpu",
  "sampler_type": "qpu",
  "qpu_access_time": 0.0234,
  "qpu_config": {
    "chip_id": "Advantage_system6.3",
    "topology": "pegasus",
    "num_reads": 1000
  }
}
```

## Testing Different Configurations

### Small Scale (Fast Testing)
```powershell
python comprehensive_benchmark_CUSTOM_HYBRID.py --config 5
python comprehensive_benchmark_DECOMPOSED.py --config 5
```

### Medium Scale
```powershell
python comprehensive_benchmark_CUSTOM_HYBRID.py --config 10
python comprehensive_benchmark_DECOMPOSED.py --config 10
```

### Larger Scale (Stress Test)
```powershell
python comprehensive_benchmark_CUSTOM_HYBRID.py --config 25
python comprehensive_benchmark_DECOMPOSED.py --config 25
```

## Optional: Test with Real D-Wave QPU

If you have a D-Wave token:

```powershell
# Set token
$env:DWAVE_API_TOKEN = "YOUR_REAL_TOKEN"

# Run with QPU
python comprehensive_benchmark_CUSTOM_HYBRID.py --config 10
python comprehensive_benchmark_DECOMPOSED.py --config 10

# Or pass directly
python comprehensive_benchmark_CUSTOM_HYBRID.py --config 10 --token YOUR_TOKEN
python comprehensive_benchmark_DECOMPOSED.py --config 10 --token YOUR_TOKEN
```

## Troubleshooting

### "neal not available"
```powershell
pip install dwave-neal
```

### "dwave-hybrid not available"
```powershell
pip install dwave-hybrid
```

### ImportError in tests
```powershell
# Ensure oqi environment is active
conda activate oqi

# Verify packages
conda list | Select-String "dwave|neal"
```

## Advantages of SimulatedAnnealing Testing

✅ **No QPU Required**: Test complete workflows without quantum hardware  
✅ **Unlimited Testing**: No QPU time limits or costs  
✅ **Reproducible**: Deterministic results for debugging  
✅ **Fast Iteration**: Quick testing during development  
✅ **Validates Logic**: Ensures code structure is correct before QPU deployment  

## Next Steps After Testing

1. ✅ Verify both alternatives work with SimulatedAnnealing
2. ✅ Compare solution quality: Gurobi vs SimulatedAnnealing
3. ✅ Analyze timing characteristics
4. ✅ Test with larger configurations (n=25, 50)
5. (Optional) Test with real D-Wave QPU if token available
6. Document findings and performance comparisons

---

**Status**: Both alternatives ready for extensive testing without QPU! 🎉
