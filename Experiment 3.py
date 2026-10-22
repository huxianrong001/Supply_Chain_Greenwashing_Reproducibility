"""Experiment 3 (3A boundary extremes; 3B interval widths), JIMO round-2.
Run: python experiment3_second_round_rewritten.py --steps 1000 --repeats 20 --networks 20
Quick: python experiment3_second_round_rewritten.py --steps 30 --repeats 3 --networks 2
Only triangular sampling is probabilistic; fuzzy memberships themselves are NOT probability distributions.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.stats import friedmanchisquare, wilcoxon, t as student_t

N, M, HUB_COUNT = 100, 3, 5
ETA, UPDATE_FRACTION = 1.2, 0.2
RHO_MIN, RHO_MAX = 0.3, 0.7
PERIPHERAL_B_RATE, WINDOW, SD_TOL = .30, 20, .005
S = (-.04, -.02, .00);
P = (-.06, -.03, .00)
SCENARIOS = {
    '3A': {
        'A': dict(R=(.08, .12, .16), T=(.14, .20, .26), S=S, P=P),
        'B': dict(R=(.08, .12, .16), T=(.14, .20, .45), S=S, P=P),
        'C': dict(R=(.01, .12, .16), T=(.14, .20, .26), S=S, P=P)},
    '3B': {
        'L': dict(R=(.10, .12, .14), T=(.18, .20, .22), S=S, P=P),
        'M': dict(R=(.08, .12, .16), T=(.14, .20, .26), S=S, P=P),
        'H': dict(R=(.04, .12, .20), T=(.10, .20, .30), S=S, P=P)}
}
PAYOFF_KEYS = ('R', 'S', 'T', 'P')
SNAP_TIMES = (0, 100, 500, 1000)


def seed_for(base, net_idx, rep, stream, condition=0):
    return int(np.random.SeedSequence([base, net_idx, rep, stream, condition]).generate_state(1, dtype=np.uint64)[0])


def weighted(tri): return (tri[0] + 2 * tri[1] + tri[2]) / 4


def mean_field(fuzz):
    r, s, tempt, p = [weighted(fuzz[k]) for k in PAYOFF_KEYS]
    denom = tempt - r + s - p
    return None if abs(denom) < 1e-12 else float((tempt - r) / denom)


def initialize(hubs, seed):
    rng = np.random.default_rng(seed)
    state = np.zeros(N, dtype=np.int8)
    peripheral = np.setdiff1d(np.arange(N), hubs)
    picks = rng.choice(peripheral, size=round(PERIPHERAL_B_RATE * len(peripheral)), replace=False)
    state[picks] = 1
    state[hubs[:2]] = 1  # matches original Experiment 3: first two hubs initially B
    return state


def simulate(adj, inertia, hubs, initial, fuzzy, seed, steps, save_snapshots=False):
    rng = np.random.default_rng(seed)
    state = initial.copy()
    density = np.empty(steps + 1);
    density[0] = state.mean()
    snap = {0: state.copy()} if save_snapshots else {}
    absorb = 0 if state.min() == state.max() else None
    quasi = None

    def payoff(node):
        labels = 2 * int(state[node]) + state[adj[node]].astype(int)
        return sum(float(rng.triangular(*fuzzy[PAYOFF_KEYS[int(k)]])) for k in labels)

    updates = max(1, round(UPDATE_FRACTION * N))
    for step in range(1, steps + 1):
        for node in rng.choice(N, size=updates, replace=False):
            nbrs = adj[int(node)]
            if len(nbrs) == 0: continue
            j = int(rng.choice(nbrs));
            i = int(node)
            if state[i] == state[j]: continue
            delta = payoff(j) - payoff(i)
            p = (1 - inertia[i]) / (1 + np.exp(-np.clip(ETA * delta, -700, 700)))
            if rng.random() < p: state[i] = state[j]
        density[step] = state.mean()
        if save_snapshots and step in SNAP_TIMES: snap[step] = state.copy()
        if absorb is None and state.min() == state.max(): absorb = step
        if quasi is None and step >= WINDOW and density[step - WINDOW + 1:step + 1].std() < SD_TOL: quasi = step
    klass = 'all_green' if state.max() == 0 else ('all_B' if state.min() == 1 else 'not_absorbed')
    return dict(density=density, snapshots=snap, terminal=float(state.mean()),
                absorption_step=absorb, quasi_step=quasi, absorb_class=klass,
                final_B_hubs=int(state[hubs].sum()), tail100_mean=float(density[-min(100, steps):].mean()))


def ci(values):
    a = np.asarray(values, dtype=float)
    if len(a) < 2: return [None, None]
    half = student_t.ppf(.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return [float(max(0, a.mean() - half)), float(min(1, a.mean() + half))]


def summary(rows, conditions, networks):
    out = {}
    for cond in conditions:
        sub = [r for r in rows if r['scenario'] == cond]
        vals = [r['terminal_B_ratio'] for r in sub]
        nm = [np.mean([r['terminal_B_ratio'] for r in sub if r['network_index'] == k]) for k in range(networks)]
        absorbed = [r['absorption_step'] for r in sub if r['absorption_step'] is not None]
        q = [r['quasi_step'] for r in sub if r['quasi_step'] is not None]
        out[cond] = dict(n_simulations=len(sub), n_networks=networks, terminal_mean=float(np.mean(vals)),
                         ci95_network_level_t=ci(nm) if networks > 1 else None,
                         ci95_conditional_t=ci(vals) if networks == 1 else None,
                         absorption_rate=len(absorbed) / len(sub), quasi_rate=len(q) / len(sub),
                         absorption_mean_among_absorbed=float(np.mean(absorbed)) if absorbed else None,
                         quasi_mean_among_observed=float(np.mean(q)) if q else None,
                         n_all_green=sum(r['absorb_class'] == 'all_green' for r in sub),
                         n_all_B=sum(r['absorb_class'] == 'all_B' for r in sub),
                         n_not_absorbed=sum(r['absorb_class'] == 'not_absorbed' for r in sub),
                         final_B_hubs_mean=float(np.mean([r['final_B_hubs'] for r in sub])))
    return out


def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]);
    ans = [None] * len(ps);
    current = 0.
    for rank, i in enumerate(idx):
        current = max(current, min(1., ps[i] * (len(ps) - rank)));
        ans[i] = float(current)
    return ans


def stats(rows, conditions, networks, repeats):
    lookup = {(r['network_index'], r['replicate'], r['scenario']): r['terminal_B_ratio'] for r in rows}
    if networks > 1:
        data = np.array([[np.mean([lookup[(k, r, c)] for r in range(1, repeats + 1)]) for c in conditions] for k in
                         range(networks)])
    else:
        data = np.array([[lookup[(0, r, c)] for c in conditions] for r in range(1, repeats + 1)])
    out = dict(unit='network_mean' if networks > 1 else 'replicate_conditional_on_fixed_network', n_units=len(data),
               conditions=conditions)
    if len(conditions) >= 3 and len(data) >= 3:
        st, p = friedmanchisquare(*data.T);
        out['friedman'] = {'statistic': float(st), 'p': float(p)}
    comparisons = []
    for j, a in enumerate(conditions):
        for l in range(j + 1, len(conditions)):
            b = conditions[l];
            dif = data[:, j] - data[:, l]
            if np.all(dif == 0):
                W, p = 0., 1.
            else:
                W, p = wilcoxon(dif, zero_method='wilcox', alternative='two-sided');
                W, p = float(W), float(p)
            dif_ci = ci_unbounded(dif)
            comparisons.append(dict(comparison=f'{a}_vs_{b}', difference_mean=float(dif.mean()),
                                    difference_ci95_t=dif_ci, W=W, p_raw=p))
    adj = holm([x['p_raw'] for x in comparisons])
    for x, p in zip(comparisons, adj): x['p_holm'] = p
    out['paired_wilcoxon_holm'] = comparisons
    return out


def ci_unbounded(a):
    a = np.asarray(a, float)
    if len(a) < 2: return [None, None]
    half = student_t.ppf(.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return [float(a.mean() - half), float(a.mean() + half)]


def save_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));
        w.writeheader();
        w.writerows(rows)


def make_plots(out, mode, curves, snap, graph, hubs, repeats, steps, networks):
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 20, 'axes.linewidth': 1.2, 'mathtext.fontset': 'stix',
    })
    conds = list(SCENARIOS[mode]);
    x = np.arange(steps + 1)

    if mode == '3A':
        sc_labels = {'A': 'a) Baseline', 'B': 'b) High $T^o$', 'C': 'c) Low $R^p$'}
    else:
        sc_labels = {'L': 'a) Low width', 'M': 'b) Medium width', 'H': 'c) High width'}

    # ==========================================
    # 1. Evolution curves (演化曲线图)
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), dpi=300, sharey=True)
    for ax, c in zip(axes, conds):
        arr = np.array(curves[c]);
        mu = arr.mean(axis=0)
        unit = arr.reshape(networks, repeats, steps + 1).mean(axis=1) if networks > 1 else arr
        sem = unit.std(axis=0, ddof=1) / np.sqrt(len(unit)) if len(unit) > 1 else np.zeros(steps + 1)
        critical = student_t.ppf(.975, len(unit) - 1) if len(unit) > 1 else 0

        ax.plot(x, mu, color='black', lw=1.8, label='Mean')
        ax.fill_between(x, np.clip(mu - critical * sem, 0, 1), np.clip(mu + critical * sem, 0, 1), color='#8daac0',
                        alpha=0.35, label='95% t CI')
        ax.plot(x, arr[0], color='#ce8041', alpha=0.5, lw=0.8, label='Representative run')

        for t in (100, 500, 1000):
            if t <= steps:
                ax.axvline(x=t, color='#d62728', linestyle='--', linewidth=1.0, alpha=0.5)

        ax.set_xlabel(r'Time step $t$')
        ax.set_xlim(0, steps)
        ax.set_ylim(0, 1.05)

        # abc 标签放在子图正下方
        ax.text(0.5, -0.20, sc_labels[c], ha='center', va='top',
                transform=ax.transAxes, fontsize=20, color='#333333')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out', length=4, width=1.2, labelsize=20)

    axes[0].set_ylabel(r'Greenwashing density $f(t)$')
    handles, labels = axes[0].get_legend_handles_labels()
    # 图例放在最底部居中
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=18)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.30, wspace=0.08)

    # 只保存 PNG
    fig.savefig(out / f'Experiment_{mode}_evolution.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ==========================================
    # 2. Network snapshots (网络快照图)
    # ==========================================
    times = [t for t in SNAP_TIMES if t <= steps]
    pos = nx.spring_layout(graph, seed=42, k=0.55 / np.sqrt(N), iterations=200)

    NODE_MIN, NODE_MAX = 55.0, 420.0
    degree = np.array([graph.degree(i) for i in range(N)])
    d_min, d_max = float(degree.min()), float(degree.max())
    sizes = NODE_MIN + (NODE_MAX - NODE_MIN) * (degree - d_min) / max(1.0, d_max - d_min)

    cmap = plt.get_cmap('RdYlGn_r')
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(len(conds), len(times), figsize=(4.6 * len(times), 3.9 * len(conds)), dpi=300,
                             squeeze=False)
    non_hubs = [i for i in range(N) if i not in set(hubs.tolist())]

    for ri, c in enumerate(conds):
        for j, t in enumerate(times):
            ax = axes[ri, j];
            freq = snap[c][t] / repeats
            node_colors = [cmap(norm(freq[i])) for i in range(N)]

            nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='#c8ccd1', alpha=0.28, width=0.45)

            nx.draw_networkx_nodes(graph, pos, nodelist=non_hubs, ax=ax,
                                   node_color=[node_colors[i] for i in non_hubs],
                                   node_size=[sizes[i] for i in non_hubs],
                                   edgecolors='#5a5a5a', linewidths=0.35)
            nx.draw_networkx_nodes(graph, pos, nodelist=hubs.tolist(), ax=ax,
                                   node_color=[node_colors[i] for i in hubs],
                                   node_size=[sizes[i] for i in hubs],
                                   edgecolors='black', linewidths=1.8)

            ax.axis('off')
            ax.margins(0.06)

            if ri == len(conds) - 1:
                ax.text(0.5, -0.08, f't = {t}', transform=ax.transAxes,
                        ha='center', va='top', fontsize=20, color='#333333')

            if j == 0:
                ax.text(0.02, 0.98, sc_labels[c], transform=ax.transAxes,
                        ha='left', va='top', fontsize=20, color='#333333',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none'))

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.035, 0.50, 0.018])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Average greenwashing frequency (0 = all green, 1 = all B)', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Hub node',
                              markerfacecolor='#bbbbbb', markersize=16,
                              markeredgewidth=2.0, markeredgecolor='black')]
    fig.legend(handles=legend_elements, loc='lower right',
               bbox_to_anchor=(0.98, 0.055), frameon=False, fontsize=16)

    fig.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.16,
                        wspace=0.03, hspace=0.14)

    fig.savefig(out / f'Experiment_{mode}_network_frequencies.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main(args):
    base = Path(args.output);
    base.mkdir(parents=True, exist_ok=True)
    modes = ['3A', '3B'] if args.mode == 'both' else [args.mode]
    net_seeds = [42 + 100 * k for k in range(args.networks)]
    for mode in modes:
        out = base / mode;
        out.mkdir(parents=True, exist_ok=True)
        conds = list(SCENARIOS[mode]);
        rows = [];
        curves = {c: [] for c in conds}
        times = [t for t in SNAP_TIMES if t <= args.steps]
        snaps = {c: {t: np.zeros(N) for t in times} for c in conds}
        rep_graph = rep_hubs = None
        for ni, ns in enumerate(net_seeds):
            graph = nx.barabasi_albert_graph(N, M, seed=ns)
            adj = [np.fromiter(graph.neighbors(i), dtype=int) for i in range(N)]
            degrees = np.array([len(a) for a in adj]);
            hubs = np.argsort(-degrees, kind='stable')[:HUB_COUNT]
            inertia = RHO_MIN + (RHO_MAX - RHO_MIN) * degrees / degrees.max()
            if ni == 0: rep_graph, rep_hubs = graph, hubs
            for rep in range(1, args.repeats + 1):
                # Identical initialization for all scenarios AND across 3A/3B.
                init_seed = seed_for(args.seed, ni, rep, 10)
                initial = initialize(hubs, init_seed)
                signature = ','.join(map(str, np.flatnonzero(initial)))
                for ci, c in enumerate(conds):
                    sim_seed = seed_for(args.seed, ni, rep, 20 if mode == '3A' else 30, ci)
                    result = simulate(adj, inertia, hubs, initial, SCENARIOS[mode][c], sim_seed, args.steps,
                                      ni == 0 and args.plot)
                    rows.append(dict(mode=mode, network_index=ni, network_seed=ns, replicate=rep, scenario=c,
                                     initialization_seed=init_seed, initial_B_signature=signature,
                                     simulation_seed=sim_seed,
                                     initial_B_count=int(initial.sum()), terminal_B_ratio=result['terminal'],
                                     tail100_mean=result['tail100_mean'], quasi_step=result['quasi_step'],
                                     absorption_step=result['absorption_step'], absorb_class=result['absorb_class'],
                                     final_B_hubs=result['final_B_hubs']))
                    curves[c].append(result['density'])
                    if ni == 0 and args.plot:
                        for t, v in result['snapshots'].items(): snaps[c][t] += v
            print(f'Experiment {mode}: completed network {ni + 1}/{args.networks} (seed={ns})', flush=True)
        summ = summary(rows, conds, args.networks);
        infer = stats(rows, conds, args.networks, args.repeats)
        save_csv(out / 'replicate_results.csv', rows)
        (out / 'summary.json').write_text(json.dumps(summ, indent=2), encoding='utf-8')
        (out / 'paired_statistics.json').write_text(json.dumps(infer, indent=2), encoding='utf-8')
        meta = dict(N=N, m=M, hub_count=HUB_COUNT, eta=ETA, update_fraction=UPDATE_FRACTION,
                    rho_bounds=[RHO_MIN, RHO_MAX], network_seeds=net_seeds, base_seed=args.seed,
                    steps=args.steps, repeats_per_network=args.repeats,
                    initial_B_hubs=2, initial_peripheral_B_rate=PERIPHERAL_B_RATE,
                    scenario_payoffs=SCENARIOS[mode],
                    weighted_defuzzified_payoffs={c: {k: weighted(SCENARIOS[mode][c][k]) for k in PAYOFF_KEYS} for c in
                                                  conds},
                    homogeneous_reference_f_star={c: mean_field(SCENARIOS[mode][c]) for c in conds},
                    inference_unit='independent BA network means' if args.networks > 1 else 'fixed-network matched replicates',
                    quasi_convergence_definition=f'first {WINDOW}-step window with density sd < {SD_TOL}',
                    note='Triangular probability sampling is an additional assumption, not the fuzzy membership itself; non-absorbing runs are right-censored.')
        (out / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        if args.plot: make_plots(out, mode, curves, snaps, rep_graph, rep_hubs, args.repeats, args.steps, args.networks)
        print(mode, {k: round(v['terminal_mean'], 4) for k, v in summ.items()})
    print('Results:', base.resolve())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['3A', '3B', 'both'], default='both')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--networks', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='experiment3_second_round_results')
    p.add_argument('--no-plot', dest='plot', action='store_false')
    args = p.parse_args()
    if min(args.steps, args.repeats, args.networks) < 1: p.error('steps, repeats, networks must be positive')
    main(args)