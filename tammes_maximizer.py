#!/usr/bin/env python3
"""
Simple multi-start maximizer for the Tammes problem
(maximize the minimum angular/Euclidean distance of n points on the unit sphere).

Supports both small open cases and the range n≈24–35 where optima are
still uncertain or only numerically known. Uses only NumPy, SciPy and Matplotlib.

Author: independent computational experiment (2026)
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import time
import os


# ---------------------------------------------------------------------------
# Core geometry utilities
# ---------------------------------------------------------------------------

def random_unit_points(n, seed=None):
    """Generate n random points uniformly on the unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def angular_min_deg(pts):
    """Return the minimum angular separation in degrees."""
    dots = pts @ pts.T
    np.fill_diagonal(dots, -np.inf)
    max_cos = np.max(dots)
    return float(np.degrees(np.arccos(np.clip(max_cos, -1.0, 1.0))))


def min_euclidean(pts):
    """Return the minimum Euclidean distance between distinct points."""
    dots = pts @ pts.T
    np.fill_diagonal(dots, -1.0)
    return float(np.sqrt(np.min(2.0 - 2.0 * dots)))


def warm_start(n, base_config=None, seed=None):
    """
    Constructive warm-start helper.
    If base_config has fewer than n points, add the missing k = n - base.shape[0]
    random points (reproducibly when a seed is supplied) and return an
    n-point configuration. Otherwise return None.
    """
    if base_config is not None and base_config.shape[0] < n:
        k = n - base_config.shape[0]
        extra = random_unit_points(k, seed=seed)
        return np.vstack([base_config, extra])
    return None


# ---------------------------------------------------------------------------
# Local optimizers
# ---------------------------------------------------------------------------

def pot_obj(x, n, p):
    """High-power repulsive potential (to be minimized)."""
    pts = x.reshape(n, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    dots = pts @ pts.T
    iu = np.triu_indices(n, 1)
    d2 = np.maximum(2.0 - 2.0 * dots[iu], 1e-16)
    return np.sum(d2 ** (-p / 2.0))


def progressive_local(pts0, p_seq=(5.0, 12.0, 25.0, 50.0), maxiter=45):
    """
    Progressive high-power potential minimization (L-BFGS-B).
    Starts with moderate p and increases it to sharpen the minimum-distance objective.
    """
    n = pts0.shape[0]
    x = pts0.ravel().copy()
    for p in p_seq:
        res = minimize(
            lambda xx: pot_obj(xx, n, p),
            x,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-12},
        )
        x = res.x
    pts = x.reshape(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def polish_aux(pts0, maxiter=100):
    """
    Polish a configuration by maximizing an auxiliary variable t
    subject to all pairwise Euclidean distances >= t and unit-norm constraints (SLSQP).
    Can be slow for n ≳ 30; use --no-polish for larger instances.
    """
    n = pts0.shape[0]
    dots = pts0 @ pts0.T
    np.fill_diagonal(dots, 1.0)
    t0 = float(np.sqrt(np.min(2.0 - 2.0 * dots)))
    x0 = np.concatenate([pts0.ravel(), [max(t0 * 0.97, 0.25)]])

    def objective(x):
        return -x[-1]

    cons = []
    for i in range(n):
        def unit(x, i=i):
            return np.dot(x[3 * i : 3 * i + 3], x[3 * i : 3 * i + 3]) - 1.0
        cons.append({"type": "eq", "fun": unit})

    for i in range(n):
        for j in range(i + 1, n):
            def dge(x, i=i, j=j):
                return np.linalg.norm(x[3 * i : 3 * i + 3] - x[3 * j : 3 * j + 3]) - x[-1]
            cons.append({"type": "ineq", "fun": dge})

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        constraints=cons,
        options={"maxiter": maxiter, "ftol": 1e-11, "disp": False},
    )
    pts = res.x[:-1].reshape(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts, bool(res.success)


# ---------------------------------------------------------------------------
# Multi-start + perturbation campaign
# ---------------------------------------------------------------------------

def perturb(pts, scale=0.05, seed=None):
    """Add Gaussian noise and re-project to the sphere."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=scale, size=pts.shape)
    pts2 = pts + noise
    return pts2 / np.linalg.norm(pts2, axis=1, keepdims=True)


def run_campaign(n, n_random=40, n_pert=6, seed_base=42, do_polish=True,
                 base_config=None, verbose=True):
    """
    Full multi-start + perturbation campaign for a given n.
    Optionally accepts a base_config of (n-1) points for a constructive warm-start.
    Returns (best_points, best_angular_degrees).
    """
    if verbose:
        polish_str = "with polish" if do_polish else "NO polish"
        print(f"Campaign n={n}  ({n_random} random + {n_pert} perts, {polish_str})")
    t0 = time.time()
    best_ang = -1.0
    best_pts = None

    # Optional constructive warm-start (handles any gap size, reproducible seed)
    if base_config is not None:
        ws = warm_start(n, base_config, seed=seed_base + n * 9999)
        if ws is not None:
            pts = progressive_local(ws)
            if do_polish:
                pts, _ = polish_aux(pts, maxiter=70)
            ang = angular_min_deg(pts)
            best_ang = ang
            best_pts = pts.copy()
            if verbose:
                print(f"  warm-start: {ang:.7f}°")

    # Phase 1: random starts
    for s in range(n_random):
        pts0 = random_unit_points(n, seed=seed_base + n * 100 + s)
        pts = progressive_local(pts0)
        if do_polish:
            pts, _ = polish_aux(pts, maxiter=70)
        ang = angular_min_deg(pts)
        if ang > best_ang + 1e-8:
            best_ang = ang
            best_pts = pts.copy()
            if verbose and (s < 3 or ang > best_ang - 0.01):
                print(f"  new best (random {s}): {ang:.7f}°")

    # Phase 2: perturbations of the current best
    for r in range(n_pert):
        pts0 = perturb(best_pts, scale=0.035 + 0.015 * (r % 4), seed=seed_base + 2000 + r)
        pts = progressive_local(pts0)
        if do_polish:
            pts, _ = polish_aux(pts, maxiter=80)
        ang = angular_min_deg(pts)
        if ang > best_ang + 1e-8:
            best_ang = ang
            best_pts = pts.copy()
            if verbose:
                print(f"  new best (pert {r}): {ang:.7f}°")

    # Optional final polish
    if do_polish and best_pts is not None:
        pts_final, _ = polish_aux(best_pts, maxiter=120)
        ang_final = angular_min_deg(pts_final)
        if ang_final > best_ang:
            best_ang = ang_final
            best_pts = pts_final

    if verbose:
        print(f"  → {best_ang:.7f}°   ({time.time() - t0:.1f}s)")
    return best_pts, best_ang


# ---------------------------------------------------------------------------
# Contact graph
# ---------------------------------------------------------------------------

def contact_graph(pts, tol=1e-4):
    """Return edges achieving nearly the minimal angular separation + degrees."""
    n = pts.shape[0]
    dots = pts @ pts.T
    np.fill_diagonal(dots, -np.inf)
    max_cos = np.max(dots)
    edges = []
    degrees = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(dots[i, j] - max_cos) < tol:
                edges.append((i, j))
                degrees[i] += 1
                degrees[j] += 1
    return edges, degrees


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_configuration(pts, edges, n, ang, filename):
    """Save a 3-D plot of the points and their contact edges."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="cyan", alpha=0.08, linewidth=0)

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="crimson", s=50, depthshade=True)

    for i, j in edges:
        ax.plot(
            [pts[i, 0], pts[j, 0]],
            [pts[i, 1], pts[j, 1]],
            [pts[i, 2], pts[j, 2]],
            "k-", lw=1.2, alpha=0.7,
        )

    ax.set_xlim([-1.15, 1.15])
    ax.set_ylim([-1.15, 1.15])
    ax.set_zlim([-1.15, 1.15])
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(f"Tammes n = {n}\nmin angle ≈ {ang:.5f}°   |   contacts = {len(edges)}")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {filename}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tammes multi-start maximizer (supports n up to ~35)"
    )
    parser.add_argument(
        "--n", type=int, nargs="+",
        default=[24, 25, 27, 28, 30, 32],
        help="Values of n to optimise (default: 24 25 27 28 30 32)",
    )
    parser.add_argument(
        "--starts", type=int, default=0,
        help="Random starts (0 = auto: 50 for n≤25, 120 for n≤30, 200 for n>30)",
    )
    parser.add_argument("--perts", type=int, default=6,
                        help="Number of perturbation rounds")
    parser.add_argument("--no-polish", action="store_true",
                        help="Skip the expensive SLSQP polish (recommended for n≥30)")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Published / best-known reference values with provenance notes
    # Format: n -> (angle_degrees, status_string)
    published = {
        15: (53.6578501, "Sloane table; unproven"),
        16: (52.2443957, "Sloane table; unproven"),
        17: (51.0903285, "Sloane table; unproven"),
        18: (49.5566548, "Sloane table; unproven"),
        20: (47.4310362, "Sloane table; unproven"),
        21: (45.6132231, "Sloane table; unproven"),
        24: (43.6907671, "snub cube; proven optimal (Robinson)"),
        25: (41.6344612, "Sloane table; unproven"),
        27: (39.6824560, "Sloane table; unproven"),
        28: (38.6770790, "Sloane table; unproven"),
        30: (37.3773682, "Sloane table; unproven"),
        # n=32 omitted: best-known value uncertain across sources
    }

    print("=" * 64)
    print("Tammes multi-start maximizer  (extended range)")
    print("=" * 64)
    print(f"Polish: {'DISABLED' if args.no_polish else 'enabled'}")
    print()

    prev_best = None  # for optional sequential warm-start

    for n in args.n:
        # Auto-scale number of starts
        if args.starts == 0:
            n_random = 50 if n <= 25 else (120 if n <= 30 else 200)
        else:
            n_random = args.starts

        # Automatically disable polish for larger n unless user forces it
        do_polish = (not args.no_polish) and (n <= 28)
        if n > 28 and not args.no_polish:
            print(f"(n={n} > 28 → polish auto-disabled for speed)")
            do_polish = False

        print(f"Starts for n={n}: {n_random}")

        pts, ang = run_campaign(
            n,
            n_random=n_random,
            n_pert=args.perts,
            seed_base=args.seed,
            do_polish=do_polish,
            base_config=prev_best,  # try warm-start from previous n if available
            verbose=True,
        )
        edges, degrees = contact_graph(pts)

        # save coordinates
        coord_file = os.path.join(args.outdir, f"coords_n{n}.txt")
        np.savetxt(coord_file, pts, fmt="%.12f")
        print(f"  Coordinates → {coord_file}")

        # contact-graph summary
        print(f"  Contacts: {len(edges)} edges")
        print(f"  Degrees:  {sorted(degrees, reverse=True)}")

        if n in published:
            ref_ang, status = published[n]
            delta = ang - ref_ang
            print(f"  Reference: {ref_ang:.7f}°  ({status})")
            print(f"  Delta:     {delta:+.7f}°")

        # plot
        plot_file = os.path.join(args.outdir, f"plot_n{n}.png")
        plot_configuration(pts, edges, n, ang, plot_file)
        print()

        # keep for possible warm-start of n+1
        prev_best = pts

    print("Done.")


if __name__ == "__main__":
    main()
