import ast, sys
for fp in ['Benchmarks/decomposition_scaling/benchmark_solvers_comparison.py', 'Benchmarks/decomposition_scaling/generate_plots.py']:
    with open(fp) as f: src = f.read()
    try:
        ast.parse(src); print('OK:', fp)
    except SyntaxError as e:
        print('ERROR:', fp, e); sys.exit(1)
