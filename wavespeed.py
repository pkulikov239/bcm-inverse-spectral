"""wavespeed.py — the last step of the reconstruction, V ↦ c.

The chain is  V ↦ V̇ ↦ χ ↦ τ ↦ c.  Only the first arrow is delicate:
differentiating V inverts the Volterra operator Y, so it goes through a
regularisation, one of three.  CPU only; the total variation problem is handed
to cvxpy, which states it in the notation of the paper and solves it with the
interior-point solver CLARABEL.

    V(r), V̇(r)             the volume ∫_{Ω(r)} c⁻² dx and its derivative
    r, Δt                  the travel-time grid
    τ(x) = ∫_0^x dy/c(y)   travel time;  χ = τ⁻¹,  χ(r) = ∫_0^r ds/V̇(s)
    c(χ(r)) = 1/V̇(r)       the identity the reconstruction rests on
    ĉ                      the recovered wave speed

    Y, (Yg)(r) = ∫_0^r g   the Volterra operator, discretised by trapezoids
    D_t                    the forward difference,  (D_t g)_i = g_{i+1} − g_i
    α_TV                   min_g ‖Yg − V‖² + α_TV‖D_t g‖₁
    α_H1                   min_g ‖Yg − V‖² + α_H1‖D_t g‖²

Y and D_t are named after the symbols of the paper.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import CubicSpline, UnivariateSpline

__all__ = ["travel_time", "Y", "D_t", "solve_tv", "α_TV_lcurve", "print_errors",
           "differentiate_V", "solve_tikhonov_h1", "reconstruct_c"]


def travel_time(c):
    """τ(1) = ∫_0^1 dx/c(x) of the TRUE wave speed, rounded up to 1e-5.

    For scoring only: τ(1) is not data of the inverse problem.  The
    reconstruction reads it from the spectrum, `bcm.InvProb.τ_1`."""
    return float(np.ceil(quad(lambda x: 1.0/c(np.array([x]))[0], 0, 1)[0]*1e5) / 1e5)


def Y(N, Δt):
    """The Volterra operator (Yg)(r) = ∫_0^r g(t) dt, trapezoidal on N points:
    (Yg)_0 = 0,  (Yg)_i = Δt(½g_0 + g_1 + … + g_{i−1} + ½g_i),  i ≥ 1."""
    out = Δt * (np.tril(np.ones((N, N))) - 0.5*np.eye(N))
    out[:, 0] = 0.5*Δt
    out[0, 0] = 0.0
    return out


def D_t(N):
    """The forward difference of shape (N−1, N):  (D_t g)_i = g_{i+1} − g_i."""
    return np.diff(np.eye(N), axis=0)


# =============================================================================
# Regularised differentiation
# =============================================================================

def solve_tv(Y_h, V, α_TV):
    """min_g ‖Yg − V‖² + α_TV‖D_t g‖₁, through cvxpy.

    `cp.tv` is the discrete total variation ‖D_t g‖₁ itself, so the code states
    the problem of the paper and nothing else.  CLARABEL is an interior-point
    solver: on the profiles of the paper it reaches a lower objective than the
    first-order splittings and does it in a fraction of the time.  cvxpy is
    imported here rather than at module level, so the two smooth options work
    without it."""
    import cvxpy as cp

    g = cp.Variable(Y_h.shape[1])
    cp.Problem(cp.Minimize(cp.sum_squares(Y_h @ g - V) + α_TV*cp.tv(g))
               ).solve(solver=cp.CLARABEL)
    return g.value


def α_TV_lcurve(Y_h, V, n_α=100, α_max=2.0, solver=solve_tv):
    """α_TV by the L-curve: the point of α_TV ↦ (‖Yg − V‖², ‖D_t g‖₁) farthest
    from the chord joining the endpoints of that curve."""
    α = np.linspace(0, α_max, n_α)
    curve = np.array([[np.linalg.norm(Y_h @ g - V)**2, np.abs(np.diff(g)).sum()]
                      for g in (solver(Y_h, V, a) for a in α)])

    p1, p2 = curve[0], curve[-1]
    d = p1 - curve
    dist = np.abs((p2[0]-p1[0])*d[:, 1] - (p2[1]-p1[1])*d[:, 0])
    return α[np.argmax(dist / np.linalg.norm(p1 - p2))]


def solve_tikhonov_h1(Y_h, V, α_H1):
    """(YᵀY + α_H1 D_tᵀD_t) V̇ = YᵀV."""
    D = D_t(Y_h.shape[1])
    return np.linalg.solve(Y_h.T @ Y_h + α_H1 * D.T @ D, Y_h.T @ V)


def differentiate_V(V, r, smoothing=None):
    """V̇, by fitting a smoothing spline to V and differentiating it."""
    s = smoothing if smoothing is not None else len(r)*np.var(V)*1e-6
    return UnivariateSpline(r, V, s=s, k=4).derivative()(r)


# =============================================================================
# The reconstruction
# =============================================================================

def print_errors(ĉ, c, x):
    """The TV, L^∞, L² and L¹ errors of ĉ against c, absolute and relative."""
    Δx = x[1] - x[0]
    norms = {"TV":  lambda v: np.abs(np.diff(v)).sum(),
             "Loo": lambda v: np.abs(v).max(),
             "L2":  lambda v: np.sqrt((v**2).sum() * Δx),
             "L1":  lambda v: np.abs(v).sum() * Δx}

    print("*-----------------")
    print("Errors:")
    for name, norm in norms.items():
        err, scale = norm(ĉ - c(x)), norm(c(x))
        rel = "n/a" if scale < 1e-14 else f"{100*err/scale:.3f}%"
        print(f"{name} = {err:.5f}   ({rel})")
    print("*-----------------")


def reconstruct_c(V, r, ip, smoothing=None, α_reg=1e-6, method="spline"):
    """ĉ from the volume profile, through  χ(r) = ∫_0^r ds/V̇(s),  τ = χ⁻¹,
    c(x) = 1/V̇(τ(x)).

    (V, r) is the output of `bcm.V`: r runs over t_1, …, stops at τ̂(1), so
    len(V) may be smaller than N; everything below uses len(V).

    `method` picks the regularisation for V̇: "spline" fits a smoothing spline
    and differentiates it, "h1" is Tikhonov-H¹, "tv" is total variation.  The
    paper uses the first two for smooth profiles and TV for piecewise constant
    ones, where V̇ itself jumps.  `α_reg` scales the spline smoothing, or is
    α_H1, or is α_TV.  Returns (ĉ, x) on N = ip.discr.N points of Ω̄."""
    N, n = ip.discr.N, len(V)

    if method == "spline":
        s = smoothing if smoothing is not None else (n - 1)*α_reg
        inner = differentiate_V(V[1:], r[1:], smoothing=s)
        V̇ = np.concatenate([[inner[0]], inner])
    elif method in ("h1", "tv"):
        # row i = 0 is dropped: Y's first row is identically 0
        Y_h = Y(n, r[1] - r[0])[1:, :]
        solve = solve_tikhonov_h1 if method == "h1" else solve_tv
        V̇ = solve(Y_h, V[1:], α_reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    V̇ = np.clip(V̇, 1e-6, None)
    χ = cumulative_trapezoid(1.0/V̇, r, initial=0.0)

    # χ must be strictly increasing to be inverted; keep the part that is
    rising = np.diff(χ) > 0
    stop = len(χ) if rising.all() else np.argmin(rising) + 1

    τ = CubicSpline(χ[:stop], r[:stop], extrapolate=False)      # τ = χ⁻¹
    V̇_spline = CubicSpline(r[:stop], V̇[:stop], extrapolate=False)

    x = np.linspace(0, 1, N)
    inside = (x >= χ[0]) & (x <= χ[stop - 1])
    ĉ = np.full(N, np.nan)
    ĉ[inside] = 1.0 / V̇_spline(τ(x[inside]))

    # continue ĉ constantly outside the interval the waves reached
    ok = np.flatnonzero(~np.isnan(ĉ))
    if len(ok):
        ĉ[:ok[0]], ĉ[ok[-1]:] = ĉ[ok[0]], ĉ[ok[-1]]
    return ĉ, x
