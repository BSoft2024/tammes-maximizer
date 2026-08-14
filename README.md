# Independent Computational Experiments on the Tammes Problem

This repository contains a simple, fully reproducible multi-start maximizer for the classical **Tammes problem** (packing equal circles on a sphere / maximizing the minimum distance among \(n\) points on the unit sphere \(S^2\)).

The code was developed as part of an independent exploration of open cases of the Tammes problem. It recovers (to the published precision) several of the current best-known numerical records for small open \(n\) and produces contact graphs and visualisations. It also runs comfortably into the uncertain range \(n \approx 24\)–\(35\).

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

```bash
pip install numpy scipy matplotlib
```

## Quick start

```bash
python tammes_maximizer.py --n 24 25 27 28 30 --starts 0 --no-polish --outdir .
```

- `--starts 0` (default) automatically chooses the number of random starts according to \(n\):
  - 50 starts for \(n \le 25\)
  - 120 starts for \(n \le 30\)
  - 200 starts for \(n > 30\)
- `--no-polish` skips the expensive SLSQP polishing stage (recommended for \(n \gtrsim 30\)).
- The script also attempts a simple constructive warm-start: when processing consecutive values of \(n\) it can seed from the best configuration found for \(n-1\).

Typical outputs for each \(n\):

- `coords_nXX.txt` – unit-vector coordinates
- `plot_nXX.png` – 3-D visualisation with contact edges
- console summary of the achieved angle, contact-graph degree sequence, and comparison with the reference value (when available)

## Method (short description)

1. **Local search – progressive potential**  
   Minimise a high-power repulsive potential \(\sum_{i<j}\|x_i-x_j\|^{-p}\) with increasing exponents \(p\) (L-BFGS-B).

2. **Optional polishing – auxiliary-variable SLSQP**  
   Maximise an auxiliary variable \(t\) subject to \(\|x_i-x_j\|\ge t\) and unit-norm constraints. Disabled by default for larger \(n\).

3. **Global exploration**  
   Many independent random starts + repeated light perturbations of the best candidate. Optional warm-start from a good \((n-1)\)-point configuration.

## Reference values used for comparison

The code carries a small table of published / best-known angles together with short provenance notes:

| \(n\) | Angle (°)   | Status |
|-------|-------------|--------|
| 24    | 43.6907671  | snub cube; proven optimal (Robinson) |
| 25    | 41.6344612  | Sloane table; unproven |
| 27    | 39.6824560  | Sloane table; unproven |
| 28    | 38.6770790  | Sloane table; unproven |
| 30    | 37.3773682  | Sloane table; unproven |

(Values for some larger \(n\) are deliberately omitted when sources disagree.)

## Citation / licence

This is an independent computational experiment. The code is released for reproducibility and further exploration. You are free to use, modify and redistribute it. If you find the package useful, a brief acknowledgement is appreciated.

## Further work

The same pipeline can be applied to larger \(n\), to variants (e.g. totally separable packings), or can be strengthened with more sophisticated global-search strategies. The modular structure of `tammes_maximizer.py` makes such extensions straightforward.
