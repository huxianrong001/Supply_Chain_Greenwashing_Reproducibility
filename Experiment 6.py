"""Experiment 6: matched initial greenwashing ratio robustness.

Optimized for publication-quality plots:
- Only saves PNG.
- Uses LaTeX formatting ($ $) for mathematical symbols.
- No titles for single plots, tight layout.
- Clear, uniform labels for multi-panel snapshots.
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from scipy.stats import t as student_t, friedmanchisquare, wilcoxon

# --- 全局字体与绘图设置 ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'pdf.fonttype': 42,
})

PAY = {'R': (.08, .12, .16), 'S': (-.04, -.02, 0.), 'T': (.14, .20, .26), 'P': (-.06, -.03, 0.)}
KEY = ('R', 'S', 'T', 'P')
RATIOS = (.10, .30, .50, .70)


def derive_seed(root, *parts):
    return int(np.random.SeedSequence([root, *map(int, parts)]).generate_state(1, dtype=np.uint32)[0])


def network(n, m, s):
    g = nx.barabasi_albert_graph(n, m, seed=s)
    adj = [np.fromiter(g.neighbors(i), dtype=np.int32) for i in range(n)]
    degree = np.array([len(v) for v in adj], dtype=float)
    hubs = np.argsort(-degree, kind='stable')[:5]
    return g, adj, degree, hubs


def nested_states(n, hubs, init_seed):
    """Two initial greenwashing hubs; nested peripheral B sets at all four ratios."""
    gen = np.random.default_rng(init_seed)
    periphery = np.setdiff1d(np.arange(n), hubs)
    permutation = gen.permutation(periphery)
    initial = {}
    for ratio in RATIOS:
        state = np.zeros(n, dtype=np.int8)
        state[hubs[:2]] = 1
        state[permutation[:int(round(ratio * len(periphery)))]] = 1
        initial[ratio] = state
    return initial


def evolve(adj, degree, x0, rng_seed, steps, eta, update_fraction, rho_bounds, snapshot_steps):
    gen = np.random.default_rng(rng_seed)
    n = len(degree)
    state = x0.copy()
    rho = rho_bounds[0] + (rho_bounds[1] - rho_bounds[0]) * degree / degree.max()
    path = np.empty(steps + 1, dtype=float)
    path[0] = state.mean()
    snap = {0: state.copy()} if 0 in snapshot_steps else {}
    absorb = 0 if state.min() == state.max() else None
    quasi = None
    win = 20
    recent = np.empty(win, dtype=float)
    for t in range(1, steps + 1):
        for i_ in gen.choice(n, size=max(1, int(round(n * update_fraction))), replace=False):
            i = int(i_)
            j = int(gen.choice(adj[i]))
            if state[i] == state[j]:
                continue

            def payoff(u):
                ids = 2 * int(state[u]) + state[adj[u]].astype(int)
                return sum(float(gen.triangular(*PAY[KEY[int(z)]])) for z in ids)

            delta = payoff(j) - payoff(i)
            probability = (1 - rho[i]) / (1 + np.exp(-np.clip(eta * delta, -700, 700)))
            if gen.random() < probability:
                state[i] = state[j]
        path[t] = state.mean()
        recent[(t - 1) % win] = path[t]
        if quasi is None and t >= win and recent.std() < .005:
            quasi = t
        if absorb is None and state.min() == state.max():
            absorb = t
        if t in snapshot_steps:
            snap[t] = state.copy()
    outcome = 'all_B' if state.all() else 'all_green' if not state.any() else 'not_absorbed'
    return dict(terminal=float(path[-1]), absorption_step=absorb, quasi_step=quasi, outcome=outcome), path, snap


def tci(vals):
    arr = np.asarray(vals, dtype=float)
    if len(arr) < 2:
        return (None, None)
    half = float(student_t.ppf(.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr)))
    return float(arr.mean() - half), float(arr.mean() + half)


def holm(pvalues):
    order = np.argsort(pvalues)
    ans = np.empty(len(pvalues))
    running = 0
    for k, idx in enumerate(order):
        running = max(running, min(1., pvalues[idx] * (len(pvalues) - k)))
        ans[idx] = running
    return ans.tolist()


# ==========================================
# 绘图模块 (只保存 PNG, 无标题, 紧凑)
# ==========================================
def plot_evolution(folder, curve_means):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    colors = ['#1f77b4', '#e67e22', '#2ca02c', '#c43c39']
    for ratio, color in zip(RATIOS, colors):
        x = np.asarray(curve_means[ratio])
        mu = x.mean(axis=0)
        if len(x) > 1:
            half = student_t.ppf(.975, len(x) - 1) * x.std(axis=0, ddof=1) / np.sqrt(len(x))
            ax.fill_between(np.arange(len(mu)), np.clip(mu - half, 0, 1), np.clip(mu + half, 0, 1), color=color,
                            alpha=0.12)
        ax.plot(np.arange(len(mu)), mu, label=f'{int(ratio * 100)}%', color=color, lw=2)

    ax.set_xlabel(r'Time step $t$')
    ax.set_ylabel(r'Greenwashing density $f(t)$')
    ax.set_ylim(0, 1.03)
    ax.set_xlim(0, len(mu) - 1)
    ax.legend(frameon=False, ncol=2, loc='lower right')
    fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.15)
    fig.savefig(folder / 'evolution_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_terminal(folder, summary):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    xx = np.arange(4)
    m = np.array([summary[str(int(r * 100))]['mean_f1000'] for r in RATIOS])
    ci = np.array([summary[str(int(r * 100))]['ci95_network_t'] for r in RATIOS])
    err = np.vstack([np.clip(m - ci[:, 0], 0, None), np.clip(ci[:, 1] - m, 0, None)])

    ax.errorbar(xx, m, yerr=err, fmt='o', lw=1.6, capsize=5, markersize=8, color='#284e68')
    ax.set_xticks(xx, [f'{int(r * 100)}%' for r in RATIOS])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Initial peripheral greenwashing ratio')
    ax.set_ylabel(r'Terminal greenwashing density $f(T)$')
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    fig.savefig(folder / 'terminal_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_snapshots(folder, g, hubs, per_ratio_snap, steps):
    """4x4 网格快照：行表示比例，列表示时间，紧凑排版。"""
    ts = sorted(per_ratio_snap[RATIOS[0]])
    coords = nx.spring_layout(g, seed=42)
    fig, axes = plt.subplots(4, len(ts), figsize=(4.2 * len(ts), 11.5), dpi=300)
    cmap = plt.get_cmap('RdYlGn_r')

    for row, ratio in enumerate(RATIOS):
        for col, t in enumerate(ts):
            ax = axes[row, col]
            values = np.asarray(per_ratio_snap[ratio][t]).mean(axis=0)
            sizes = np.full(len(values), 35.)
            sizes[hubs] = 115
            edgecolors = ['black' if i in set(hubs) else '#999999' for i in range(len(values))]

            nx.draw_networkx_edges(g, pos=coords, ax=ax, alpha=0.1, width=0.5)
            nx.draw_networkx_nodes(g, pos=coords, ax=ax, node_size=sizes, node_color=values,
                                   cmap=cmap, vmin=0, vmax=1, edgecolors=edgecolors, linewidths=0.8)
            ax.axis('off')

            # 时间标签放在最上面一行的顶部
            if row == 0:
                ax.text(0.5, 1.05, rf'$t={t}$', transform=ax.transAxes, ha='center', va='bottom', fontsize=16)
            # 比例标签放在最左侧一行的左边
            if col == 0:
                ax.text(-0.15, 0.5, f'{int(ratio * 100)}%', rotation=90, transform=ax.transAxes, va='center',
                        ha='right', fontsize=16)

    # 手动留出底部空间放 colorbar
    fig.subplots_adjust(bottom=0.08, wspace=0.02, hspace=0.05, left=0.1, right=0.95, top=0.95)

    cax = fig.add_axes([0.35, 0.03, 0.3, 0.015])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=mc.Normalize(0, 1), cmap=cmap), cax=cax, orientation='horizontal')
    cb.set_label('Average greenwashing frequency', fontsize=14)
    cb.ax.tick_params(labelsize=12)

    fig.savefig(folder / 'network_snapshots.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main(o):
    folder = Path(o.output)
    folder.mkdir(parents=True, exist_ok=True)

    tsteps = sorted(set([0, min(100, o.steps), min(500, o.steps), o.steps]))
    rows = []
    curves = {r: [] for r in RATIOS}
    snapshots = {r: {t: [] for t in tsteps} for r in RATIOS}

    first_g = None
    first_hubs = None

    for ni in range(o.networks):
        netseed = derive_seed(o.seed, 10, ni)
        g, adj, deg, hubs = network(o.n, o.m, netseed)
        if ni == 0:
            first_g, first_hubs = g, hubs

        network_paths = {r: [] for r in RATIOS}

        for rep in range(o.repeats):
            initseed = derive_seed(o.seed, 20, ni, rep)
            states = nested_states(o.n, hubs, initseed)
            dynseed = derive_seed(o.seed, 30, ni, rep)

            for ratio in RATIOS:
                result, path, snaps = evolve(adj, deg, states[ratio], dynseed, o.steps, o.eta, o.update_fraction,
                                             (o.rho_min, o.rho_max), tsteps if ni == 0 else ())
                network_paths[ratio].append(path)
                if ni == 0:
                    for t in tsteps:
                        snapshots[ratio][t].append(snaps[t])
                rows.append(dict(network_index=ni, network_seed=netseed, replicate=rep + 1,
                                 initialization_seed=initseed, simulation_seed=dynseed, ratio=ratio,
                                 initial_B_count=int(states[ratio].sum()), **result))

        for ratio in RATIOS:
            curves[ratio].append(np.mean(network_paths[ratio], axis=0))
        print(f'Completed network {ni + 1} of {o.networks}', flush=True)

    # --- 数据导出 ---
    with (folder / 'replicate_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    matrix = np.empty((o.networks, len(RATIOS)))
    for k, ratio in enumerate(RATIOS):
        rr = [r for r in rows if r['ratio'] == ratio]
        network_means = [np.mean([r['terminal'] for r in rr if r['network_index'] == ni]) for ni in range(o.networks)]
        matrix[:, k] = network_means
        summary[str(int(ratio * 100))] = {
            'mean_f1000': float(np.mean(network_means)),
            'ci95_network_t': tci(network_means),
            'n_networks': o.networks, 'n_runs': len(rr),
            'all_green': sum(r['outcome'] == 'all_green' for r in rr),
            'all_B': sum(r['outcome'] == 'all_B' for r in rr),
            'not_absorbed': sum(r['outcome'] == 'not_absorbed' for r in rr),
            'absorption_rate': float(np.mean([r['absorption_step'] is not None for r in rr])),
            'quasi_rate': float(np.mean([r['quasi_step'] is not None for r in rr]))
        }
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    # --- 统计检验 ---
    stats = {'n_units': o.networks, 'inference_unit': 'independent_network_mean',
             'conditions': [f'{int(x * 100)}%' for x in RATIOS]}
    if o.networks >= 3:
        a, p = friedmanchisquare(*matrix.T)
        stats['friedman'] = {'statistic': float(a), 'p': float(p)}

    tests = []
    for i in range(4):
        for j in range(i + 1, 4):
            diff = matrix[:, i] - matrix[:, j]
            if np.allclose(diff, 0):
                w, p = 0., 1.
            else:
                w, p = wilcoxon(diff)
            tests.append({'comparison': f'{int(RATIOS[i] * 100)}% vs {int(RATIOS[j] * 100)}%',
                          'difference_mean': float(diff.mean()), 'ci95_difference_t': tci(diff),
                          'W': float(w), 'p_raw': float(p)})
    for d, p in zip(tests, holm([d['p_raw'] for d in tests])):
        d['p_holm'] = p
    stats['paired_wilcoxon_holm'] = tests
    (folder / 'paired_statistics.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')

    with (folder / 'evolution_network_mean_ci.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['ratio', 'step', 'mean', 'ci_low', 'ci_high'])
        for ratio in RATIOS:
            vals = np.asarray(curves[ratio])
            mu = vals.mean(axis=0)
            if len(vals) > 1:
                half = student_t.ppf(.975, len(vals) - 1) * vals.std(axis=0, ddof=1) / np.sqrt(len(vals))
            else:
                half = np.zeros(len(mu))
            for t in range(len(mu)):
                writer.writerow([ratio, t, mu[t], max(0, mu[t] - half[t]), min(1, mu[t] + half[t])])

    meta = vars(o).copy()
    meta.update({'network_seeds': [derive_seed(o.seed, 10, i) for i in range(o.networks)],
                 'payoff_triangles': PAY,
                 'initialization': 'two greenwashing hubs, three green hubs; nested peripheral sets',
                 'seed_policy': 'single root seed; all streams deterministically derived',
                 'inference_unit': 'independent BA network means', 'right_censor': 'no absorption by T'})
    (folder / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    # --- 画图 ---
    plot_evolution(folder, curves)
    plot_terminal(folder, summary)
    if not o.no_snapshots:
        plot_snapshots(folder, first_g, first_hubs, snapshots, tsteps)

    print('Saved:', folder.resolve())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42, help='Change only this root seed.')
    p.add_argument('--networks', type=int, default=20)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--m', type=int, default=3)
    p.add_argument('--eta', type=float, default=1.2)
    p.add_argument('--rho-min', type=float, default=.3)
    p.add_argument('--rho-max', type=float, default=.7)
    p.add_argument('--update-fraction', type=float, default=.2)
    p.add_argument('--output', default='experiment6_results')
    p.add_argument('--no-snapshots', action='store_true')
    p.add_argument('--smoke', action='store_true')
    o = p.parse_args()

    if o.smoke:
        o.networks = 2
        o.repeats = 2
        o.steps = 30
        if o.output == 'experiment6_results':
            o.output = 'experiment6_smoke'

    main(o)