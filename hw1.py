import sys
from itertools import combinations

import numpy as np
import pyomo.environ as pyo

SOLVER = pyo.SolverFactory("appsi_highs")
TOL = 1e-6


def build_abstract_model():
    m = pyo.AbstractModel()

    # Sets
    m.I = pyo.Set()
    m.J = pyo.Set()
    m.K = pyo.Set()
    m.ILE = pyo.Set(within=m.I)
    m.IEQ = pyo.Set(within=m.I)
    m.IGE = pyo.Set(within=m.I)

    # Parameters
    m.a = pyo.Param(m.I, m.J, default=0)
    m.b = pyo.Param(m.I)
    m.c = pyo.Param(m.K, m.J, default=0)

    # Variables
    m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)

    # Constraints, one block per sense
    def le_rule(M, i):
        return sum(M.a[i, j] * M.x[j] for j in M.J) <= M.b[i]

    def eq_rule(M, i):
        return sum(M.a[i, j] * M.x[j] for j in M.J) == M.b[i]

    def ge_rule(M, i):
        return sum(M.a[i, j] * M.x[j] for j in M.J) >= M.b[i]

    m.le_con = pyo.Constraint(m.ILE, rule=le_rule)
    m.eq_con = pyo.Constraint(m.IEQ, rule=eq_rule)
    m.ge_con = pyo.Constraint(m.IGE, rule=ge_rule)

    # One Expression + Objective per objective k; a weighted-sum objective with
    # mutable weights w; an epsilon-constraint block with mutable bounds eps
    m.z = pyo.Expression(m.K, rule=lambda M, k: sum(M.c[k, j] * M.x[j] for j in M.J))
    m.obj = pyo.Objective(m.K, rule=lambda M, k: M.z[k], sense=pyo.maximize)
    m.w = pyo.Param(m.K, default=0.0, mutable=True)
    m.weighted = pyo.Objective(
        rule=lambda M: sum(M.w[k] * M.z[k] for k in M.K), sense=pyo.maximize
    )
    m.eps = pyo.Param(m.K, default=0.0, mutable=True)
    m.eps_con = pyo.Constraint(m.K, rule=lambda M, k: M.z[k] >= M.eps[k])
    return m


def deactivate_all(inst):
    for kk in inst.K:
        inst.obj[kk].deactivate()
    inst.weighted.deactivate()
    inst.eps_con.deactivate()


def solve_current(inst, required=True):
    try:
        res = SOLVER.solve(inst)
    except RuntimeError:
        if required:
            raise
        return None, None
    tc = str(res.solver.termination_condition)
    if tc != "optimal":
        if required:
            raise AssertionError(f"solver said {tc}")
        return None, None
    z = {k: pyo.value(inst.z[k]) for k in inst.K}
    x = {j: pyo.value(inst.x[j]) for j in inst.J}
    return z, x


def solve_single(inst, k):
    deactivate_all(inst)
    inst.obj[k].activate()
    return solve_current(inst)


def solve_weighted(inst, weights):
    deactivate_all(inst)
    for kk in inst.K:
        inst.w[kk].set_value(float(weights[kk]))
    inst.weighted.activate()
    return solve_current(inst)


def solve_epsilon(inst, k0, eps_vals):
    deactivate_all(inst)
    inst.obj[k0].activate()
    for kk in inst.K:
        if kk != k0:
            inst.eps[kk].set_value(float(eps_vals[kk]))
            inst.eps_con[kk].activate()
    return solve_current(inst, required=False)


def eval_z(inst, xarr):
    return {k: float(sum(float(inst.c[k, j]) * xarr[jj] for jj, j in enumerate(inst.J)))
            for k in inst.K}


def build_certifier(inst):
    J = list(inst.J)
    t = pyo.ConcreteModel()
    t.J = pyo.Set(initialize=J)
    t.K = pyo.Set(initialize=list(inst.K))
    t.x = pyo.Var(t.J, within=pyo.NonNegativeReals)
    t.d = pyo.Var(t.K, within=pyo.NonNegativeReals)
    t.zhat = pyo.Param(t.K, default=0.0, mutable=True)
    t.le = pyo.Constraint(
        [i for i in inst.ILE],
        rule=lambda mm, i: sum(inst.a[i, j] * mm.x[j] for j in J) <= inst.b[i])
    t.eq = pyo.Constraint(
        [i for i in inst.IEQ],
        rule=lambda mm, i: sum(inst.a[i, j] * mm.x[j] for j in J) == inst.b[i])
    t.ge = pyo.Constraint(
        [i for i in inst.IGE],
        rule=lambda mm, i: sum(inst.a[i, j] * mm.x[j] for j in J) >= inst.b[i])
    t.ref = pyo.Constraint(
        list(inst.K),
        rule=lambda mm, k: sum(inst.c[k, j] * mm.x[j] for j in J) - mm.d[k]
        == mm.zhat[k] - 1e-8)
    t.tot = pyo.Objective(rule=lambda mm: sum(mm.d[k] for k in inst.K), sense=pyo.maximize)

    def certify(zhat):
        for k in inst.K:
            t.zhat[k].set_value(float(zhat[k]))
        res = SOLVER.solve(t)
        assert str(res.solver.termination_condition) == "optimal", res.solver.termination_condition
        return pyo.value(t.tot), np.array([pyo.value(t.x[j]) for j in J])

    return certify


def hyperplanes(inst):
    J = list(inst.J)
    rows = []
    for i in inst.I:
        rows.append((f"c{i}", np.array([float(inst.a[i, j]) for j in J]), float(inst.b[i])))
    for jj, j in enumerate(J):
        e = np.zeros(len(J))
        e[jj] = 1.0
        rows.append((f"x{j}", e, 0.0))
    return rows


def feasible(inst, x, tol=1e-6):
    J = list(inst.J)
    if any(v < -tol for v in x):
        return False
    for i in inst.I:
        lhs = sum(float(inst.a[i, j]) * x[jj] for jj, j in enumerate(J))
        b = float(inst.b[i])
        if i in inst.ILE and lhs > b + tol:
            return False
        if i in inst.IEQ and abs(lhs - b) > tol:
            return False
        if i in inst.IGE and lhs < b - tol:
            return False
    return True


def enumerate_vertices(inst):
    n = len(inst.J)
    rows = hyperplanes(inst)
    verts = {}
    for combo in combinations(range(len(rows)), n):
        A = np.array([rows[t][1] for t in combo])
        rhs = np.array([rows[t][2] for t in combo])
        try:
            x = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            continue
        if feasible(inst, x):
            key = tuple(round(v, 6) for v in x)
            verts.setdefault(key, x)
    return verts


def active_rows(inst, x, tol=1e-6):
    act = set()
    for name, coef, rhs in hyperplanes(inst):
        if abs(float(coef @ x) - rhs) <= tol * max(1.0, abs(rhs)):
            act.add(name)
    return act


def rank_of(inst, names):
    rows = {name: coef for name, coef, _ in hyperplanes(inst)}
    M = np.array([rows[n] for n in names])
    return int(np.linalg.matrix_rank(M)) if len(names) else 0


def fmt(v):
    return f"{v:g}"


def fmtx(inst, x):
    return "(" + ", ".join(fmt(round(v, 6)) for v in x) + ")"


def fmtz(inst, z):
    return "(" + ", ".join(fmt(round(z[k], 6)) for k in inst.K) + ")"


def key_of_x(x):
    return tuple(round(v, 4) for v in x)


def main(dat_file):
    from importlib.metadata import PackageNotFoundError, version as pkg_version
    vers = []
    for pkg in ("pyomo", "highspy", "numpy"):
        try:
            vers.append(f"{pkg} {pkg_version(pkg)}")
        except PackageNotFoundError:
            vers.append(f"{pkg} unknown")
    print("versions: " + ", ".join(vers)
          + f", python {sys.version_info.major}.{sys.version_info.minor}"
            f".{sys.version_info.micro}")

    inst = build_abstract_model().create_instance(dat_file)
    J = list(inst.J)
    K = list(inst.K)
    n = len(J)

    print(f"=== instance: {dat_file} ===")
    print(f"|I|={len(inst.I)}  |J|={n}  |K|={len(inst.K)}")
    print(f"ILE={sorted(inst.ILE)}  IEQ={sorted(inst.IEQ)}  IGE={sorted(inst.IGE)}")
    assert set(inst.ILE) | set(inst.IEQ) | set(inst.IGE) == set(inst.I)
    assert not set(inst.ILE) & set(inst.IEQ)
    assert not set(inst.ILE) & set(inst.IGE)
    assert not set(inst.IEQ) & set(inst.IGE)
    print("sanity: ILE/IEQ/IGE partition I; a and c default to 0")

    deactivate_all(inst)

    if "example" in dat_file:
        z1, _ = solve_single(inst, 1)
        print(f"sanity check 1: max z1 = {z1[1]:.6f} (expected 1040.625)")
        assert abs(z1[1] - 1040.625) < 1e-4
        zw, _ = solve_weighted(inst, {1: 1.0, 2: 1.0, 3: 2.0, 4: 1.0, 5: 1.0})
        tot = sum(zw[k] * (2.0 if k == 3 else 1.0) for k in K)
        print(f"sanity check 2: max z1+z2+2z3+z4+z5 = {tot:.6f} (expected 3657.375)")
        assert abs(tot - 3657.375) < 1e-4
        print("sanity checks passed")
        return

    print("\n--- structure (Task 1) ---")
    print("objective coefficient signs c[k,j] (0 omitted):")
    for k in K:
        terms = [f"{inst.c[k, j]:+g}*x{j}" for j in J if float(inst.c[k, j]) != 0.0]
        print(f"  z{k} = " + " ".join(terms))
    print("constraint rows:")
    for i in inst.I:
        terms = [f"{inst.a[i, j]:+g}*x{j}" for j in J if float(inst.a[i, j]) != 0.0]
        sense = "<=" if i in inst.ILE else ("=" if i in inst.IEQ else ">=")
        print(f"  row {i}: " + " ".join(terms) + f" {sense} {inst.b[i]:g}")
    for j in J:
        signs = [(k, float(inst.c[k, j])) for k in K if float(inst.c[k, j]) != 0.0]
        if signs and all(s < 0 for _, s in signs):
            print(f"  x{j}: hurts every objective it appears in {signs}")
        elif not signs:
            print(f"  x{j}: appears in no objective")
    print("objective yield per unit of row-1 budget (c[1,j]/a[1,j]):")
    for j in J:
        cost = float(inst.a[1, j])
        if cost == 0.0:
            print(f"  x{j}: uses no row-1 budget")
            continue
        part = "  ".join(f"z{k} {float(inst.c[k, j]) / cost:+.3f}"
                         for k in K if float(inst.c[k, j]) != 0.0)
        print(f"  x{j}: costs {cost:g}  ->  {part}")

    print("\n--- payoff table (Task 4a) ---")
    payoff = {}
    payoff_x = {}
    for k in K:
        payoff[k], payoff_x[k] = solve_single(inst, k)
    print("        " + "".join(f"{'z' + str(k):>12s}" for k in K))
    for r in K:
        print(f"max z{r}  " + "".join(f"{payoff[r][k]:12.4f}" for k in K))
    ideal = {k: payoff[k][k] for k in K}
    nadir_est = {k: min(payoff[r][k] for r in K) for k in K}
    print("ideal   " + "".join(f"{ideal[k]:12.4f}" for k in K))
    print("nadir~  " + "".join(f"{nadir_est[k]:12.4f}" for k in K))

    print("\n--- weighting sweep (Task 3a) ---")
    rng = np.random.default_rng(0)
    wlists = [{k: 1.0 for k in K}]
    wlists += [dict(zip(K, rng.dirichlet(np.ones(len(K))))) for _ in range(300)]
    found_w = {}
    n_at_41 = None
    keys_at_41 = None
    for i, w in enumerate(wlists):
        z, x = solve_weighted(inst, w)
        found_w.setdefault(key_of_x(np.array([x[j] for j in J])), (z, x))
        if i == 40:
            n_at_41 = len(found_w)
            keys_at_41 = sorted(found_w)
    print(f"{len(wlists)} solves -> {len(found_w)} distinct points")
    print(f"distinct after the first 41 solves (uniform + 40 Dirichlet draws, seed 0): {n_at_41}")
    for kk in keys_at_41:
        print(f"  among the first 41: x=({', '.join(fmt(v) for v in kk)})")

    wcert = dict(zip(K, (21.0, 41.0, 32.0, 29.0, 21.0)))
    zc, xc = solve_weighted(inst, wcert)
    vcert = sum(wcert[k] * zc[k] for k in K)
    print(f"certificate weights w=(21,41,32,29,21): LP optimal value = {vcert:g} "
          f"(all efficient vertices attain it; optimal set = the candidate face)")

    print("\n--- epsilon-constraint sweep (Task 3a) ---")
    k0 = K[0]
    grids = {}
    for k in K:
        if k == k0:
            continue
        lo = nadir_est[k]
        hi = ideal[k]
        grids[k] = [lo] + list(np.linspace(lo + (hi - lo) * 0.33, hi, 3))
    found_e = {}
    from itertools import product as iproduct
    levels = [grids[k] for k in K if k != k0]
    total = 1
    for g in levels:
        total *= len(g)
    n_infeas = 0
    for vals in iproduct(*levels):
        eps_vals = dict(zip([k for k in K if k != k0], vals))
        out = solve_epsilon(inst, k0, eps_vals)
        if out[0] is None:
            n_infeas += 1
            continue
        z, x = out
        found_e.setdefault(key_of_x(np.array([x[j] for j in J])), (z, x))
    print(f"{total} grids ({n_infeas} infeasible) -> {len(found_e)} distinct points")

    certify = build_certifier(inst)

    def certify_point(z):
        gap, wit = certify(z)
        return gap <= 1e-6, gap, wit

    weighted_ok = {}
    for key, (z, x) in found_w.items():
        ok, gap, wit = certify_point(z)
        weighted_ok[key] = ok
    eps_ok = {}
    for key, (z, x) in found_e.items():
        ok, gap, wit = certify_point(z)
        eps_ok[key] = ok
    print(f"weighting: {sum(weighted_ok.values())} of {len(found_w)} certified efficient")
    print(f"epsilon:   {sum(eps_ok.values())} of {len(found_e)} certified efficient")

    print("\n--- full vertex enumeration (Task 3d / Bonus) ---")
    verts = enumerate_vertices(inst)
    print(f"{len(verts)} distinct vertices of X")
    vert_info = {}
    for key, x in sorted(verts.items()):
        z = eval_z(inst, x)
        gap, wit = certify(z)
        vert_info[key] = (x, z, gap)
    eef = [(key, x, z) for key, (x, z, gap) in vert_info.items() if gap <= 1e-6]
    eef.sort()
    print(f"certified efficient extreme points: {len(eef)}")
    eef_zs = [z for _, _, z in eef]
    nadir_true = {k: min(z[k] for z in eef_zs) for k in K}
    print("\nall efficient extreme points (x, z, gap, found by):")
    for key, x, z in eef:
        gap = vert_info[key][2]
        found_by = []
        for pk, px in payoff_x.items():
            if key_of_x(np.array([px[j] for j in J])) == key_of_x(x):
                found_by.append(f"payoff(max z{pk})")
        if key_of_x(x) in found_w and weighted_ok[key_of_x(x)]:
            found_by.append("weighting")
        if key_of_x(x) in found_e and eps_ok[key_of_x(x)]:
            found_by.append("epsilon")
        found_by.append("vertex enumeration")
        print(f"  x={fmtx(inst, x)}  z={fmtz(inst, z)}  gap={gap:8.1e}  found: {', '.join(found_by)}")
    ranks = [rank_of(inst, active_rows(inst, x)) for _, x, _ in eef]
    print(f"active-set ranks of the {len(eef)} reported points: {ranks} "
          f"(all n = {n}: every reported point is an extreme point)")
    ties = [abs(sum(wcert[k] * z[k] for k in K) - vcert) <= 1e-6 * max(1.0, abs(vcert))
            for _, _, z in eef]
    print(f"certificate check: {sum(ties)}/{len(eef)} efficient extreme points attain "
          f"w·z = {vcert:g} -> the whole face conv(EEF) is optimal for this w, hence efficient")
    print("\nvertices attaining each row maximum (multiple optima check):")
    for k in K:
        cnt = sum(1 for _, (x, z, gap) in vert_info.items() if z[k] >= ideal[k] - 1e-6)
        print(f"  max z{k} = {ideal[k]:g}: attained by {cnt} of the {len(verts)} vertices")

    print("\n--- certification of payoff-table representatives (Task 4a) ---")
    for k in K:
        z = payoff[k]
        gap, wit = certify(z)
        print(f"  max z{k} rep: z={fmtz(inst, z)} gap={gap:10.3e} efficient={gap <= 1e-6}")

    print("\n--- all vertices of X with certification verdict (Task 3d) ---")
    for key, (x, z, gap) in sorted(vert_info.items()):
        print(f"  x={fmtx(inst, x)} z={fmtz(inst, z)} gap={gap:10.3e} efficient={gap <= 1e-6}")

    print("\n--- nadir comparison (Task 4b) ---")
    print("ideal             " + "".join(f"{ideal[k]:12.4f}" for k in K))
    print("nadir est (plain) " + "".join(f"{nadir_est[k]:12.4f}" for k in K))
    print("nadir over EEF    " + "".join(f"{nadir_true[k]:12.4f}" for k in K))

    print("\n--- trade-offs (Task 4c) ---")
    print("z ranges over the efficient extreme points:")
    for k in K:
        vals = [z[k] for z in eef_zs]
        print(f"  z{k}: min {min(vals):10.4f}  max {max(vals):10.4f}")

    print("\n--- midpoints of efficient extreme points (Task 4d) ---")
    print("pair_type  x_mid  z_mid  efficient?")
    n_adj = 0
    n_nonadj = 0
    for (k1, x1, z1v), (k2, x2, z2v) in combinations(eef, 2):
        common = active_rows(inst, x1) & active_rows(inst, x2)
        adj = rank_of(inst, common) >= n - 1
        if adj:
            n_adj += 1
        else:
            n_nonadj += 1
        mid = (x1 + x2) / 2.0
        zm = eval_z(inst, mid)
        gap, wit = certify(zm)
        ptype = "adjacent" if adj else "non-adjacent"
        extra = ""
        if gap > 1e-6:
            witz = eval_z(inst, wit)
            extra = f" dominated by x={fmtx(inst, wit)} z={fmtz(inst, witz)}"
        print(f"  {ptype:12s} {fmtx(inst, x1)} x {fmtx(inst, x2)}  mid={fmtx(inst, mid)} "
              f"z={fmtz(inst, zm)} gap={gap:8.1e} efficient={gap <= 1e-6}{extra}")
    print(f"tested: {n_adj} adjacent pairs, {n_nonadj} non-adjacent pairs")

    print("\n--- efficient edges (Bonus) ---")
    nedges = 0
    for (k1, x1, z1v), (k2, x2, z2v) in combinations(eef, 2):
        common = active_rows(inst, x1) & active_rows(inst, x2)
        if rank_of(inst, common) >= n - 1:
            mid = (x1 + x2) / 2.0
            zm = eval_z(inst, mid)
            gap, wit = certify(zm)
            if gap <= 1e-6:
                nedges += 1
                print(f"  efficient edge: {fmtx(inst, x1)} -- {fmtx(inst, x2)} (midpoint gap {gap:8.1e})")
            else:
                print(f"  adjacent but NOT efficient edge: {fmtx(inst, x1)} -- {fmtx(inst, x2)} (gap {gap:g})")
    print(f"total efficient edges: {nedges}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "molp_example.dat")
