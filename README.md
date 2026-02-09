
# Project Work — Weighted Traveling Collector Problem (WTCP)

## Structure

```
project-work/
├── Problem.py                          # Problem class (provided)
├── s322796.py                          # Main solution entry point
├── src/
│   └── solvers.py                      # Full implementation (EA, mutations, crossover, representation)
├── mysolution.ipynb                    # Experiments and results for all configurations
├── Sina_Behnam_Sharbafan.pdf           # Report
├── base_requirements.txt               # Required libraries
└── README.md
```

## Solution

The solution uses an **Evolutionary Algorithm** with β-conditional mutation operators to solve the WTCP. The implementation is in `src/solvers.py` and follows the required interface through `s322796.py`.

Entry point: `s322796.py` → `solution()` returns the path as `[(c1, g1), (c2, g2), ..., (0, 0)]`.

## How to Run

```bash
python s322796.py --num_cities 100 --alpha 1 --beta 2 --density 1.0 --num_generations 50
```

## Requirements

```bash
pip install -r base_requirements.txt
```
