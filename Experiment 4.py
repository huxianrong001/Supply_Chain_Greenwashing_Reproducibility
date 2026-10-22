"""Experiment 4: learning intensity x strategic inertia (round-2 JIMO).

Standard: 20 BA networks x 20 replicates x 6 conditions x 1000 steps.
Run: python experiment4_second_round_rewritten.py --networks 20 --repeats 20 --steps 1000
Test: python experiment4_second_round_rewritten.py --networks 2 --repeats 3 --steps 30
Interpret interaction ONLY from difference-of-differences tests, not main effects.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon, t as student_t

N, M, K, UPDATE_FRAC = 100, 3, 5, .20
ETA_LEVELS = (.5, 1.2, 2.5)
INERTIA = {'low': (.2, .4), 'high': (.6, .8)}
FUZZY = {'R': (.08, .12, .16), 'S': (-.04, -.02, .00),
         'T': (.14, .20, .26), 'P': (-.06, -.03, .00)}
PAYOFF_KEYS = ('R', 'S', 'T', 'P')
CONDITIONS = [(eta, level) for eta in ETA_LEVELS for level in ('low', 'high')]
WINDOW, TOL, PERIPHERAL_B_RATE = 20, .005, .3
SNAPS = (0, 100, 500, 1000)


def seed_for(base, ni, rep, stream, condition=0):
    return int(np.random.SeedSequence([base, ni, rep, stream, condition]).generate_state(1, dtype=np.uint64)[0])


def condition_name(eta, level):
    return f'eta_{eta:g}_{level}_inertia'


def init_strategy(hubs, seed):
    rng = np.random.default_rng(seed)
    strategy = np.zeros(N, dtype=np.int8)
    peripheral = np.setdiff1d(np.arange(N), hubs)
    strategy[rng.choice(peripheral, size=round(len(peripheral) * PERIPHERAL_B_RATE), replace=False)] = 1
    strategy[hubs[:2]] = 1
    return strategy


def simulate(adj, inertia, initial, eta, seed, steps, save_snap=False):
    rng = np.random.default_rng(seed)
    state = initial.copy()
    path = np.empty(steps + 1, dtype=float);
    path[0] = state.mean()
    snapshots = {0: state.copy()} if save_snap else {}
    absorb = 0 if state.min() == state.max() else None
    quasi = None
    switches = np.zeros(steps, dtype=int)

    def payoff(i):
        labels = 2 * int(state[i]) + state[adj[i]].astype(int)
        return sum(float(rng.triangular(*FUZZY[PAYOFF_KEYS[int(x)]])) for x in labels)

    for t in range(1, steps + 1):
        changed = 0
        for item in rng.choice(N, size=round(UPDATE_FRAC * N), replace=False):
            i = int(item)
            j = int(rng.choice(adj[i]))
            if state[i] == state[j]: continue
            gain = payoff(j) - payoff(i)
            imitation = (1 - inertia[i]) / (1 + np.exp(-np.clip(eta * gain, -700, 700)))
            if rng.random() < imitation:
                state[i] = state[j];
                changed += 1
        path[t] = state.mean();
        switches[t - 1] = changed
        if save_snap and t in SNAPS: snapshots[t] = state.copy()
        if absorb is None and state.min() == state.max(): absorb = t
        if quasi is None and t >= WINDOW and path[t - WINDOW + 1:t + 1].std() < TOL: quasi = t
    category = 'all_green' if state.max() == 0 else ('all_B' if state.min() == 1 else 'not_absorbed')
    return dict(path=path, snaps=snapshots, terminal=float(state.mean()),
                absorb_step=absorb, quasi_step=quasi, absorb_class=category,
                final_switches_tail100=float(switches[-min(100, steps):].mean()))


def t_interval(x, clip=False):
    x = np.asarray(x, dtype=float)
    if len(x) < 2: return [None, None]
    half = student_t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    lo, hi = float(x.mean() - half), float(x.mean() + half)
    return [max(0, lo), min(1, hi)] if clip else [lo, hi]


def adjust_holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i]);
    adjusted = [1.] * len(pvals);
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., float(pvals[i]) * (len(pvals) - rank)))
        adjusted[i] = running
    return adjusted


def paired_test(matrix, labels):
    """Rows are INDEPENDENT NETWORKS, columns are matched conditions."""
    a = np.asarray(matrix, dtype=float)
    result = {'n_units': len(a), 'unit': 'independent_BA_network_mean', 'conditions': list(labels)}
    if a.shape[1] >= 3 and len(a) >= 3:
        test, p = friedmanchisquare(*a.T)
        result['friedman'] = {'statistic': float(test), 'p': float(p)}
    pairs = []
    for i in range(a.shape[1]):
        for j in range(i + 1, a.shape[1]):
            delta = a[:, i] - a[:, j]
            if np.allclose(delta, 0, rtol=0, atol=1e-15):
                stat, p = 0., 1.
            else:
                stat, p = wilcoxon(delta, alternative='two-sided', zero_method='wilcox')
            pairs.append({'comparison': f'{labels[i]}_vs_{labels[j]}',
                          'difference_mean': float(delta.mean()),
                          'difference_ci95_t': t_interval(delta),
                          'W': float(stat), 'p_raw': float(p)})
    for row, p in zip(pairs, adjust_holm([x['p_raw'] for x in pairs])): row['p_holm'] = p
    result['paired_wilcoxon_holm'] = pairs
    return result


def analyze(rows, networks, repeats):
    means = {}
    summaries = {}
    for eta, level in CONDITIONS:
        cond = condition_name(eta, level)
        part = [r for r in rows if r['condition'] == cond]
        per_net = np.array(
            [np.mean([r['terminal_B_ratio'] for r in part if r['network_index'] == ni]) for ni in range(networks)])
        means[cond] = per_net
        absorbed = [r['absorption_step'] for r in part if r['absorption_step'] is not None]
        quasi = [r['quasi_step'] for r in part if r['quasi_step'] is not None]
        summaries[cond] = {'eta': eta, 'inertia_level': level, 'inertia_bounds': INERTIA[level],
                           'n_networks': networks, 'repeats_per_network': repeats, 'n_simulations': len(part),
                           'terminal_mean': float(per_net.mean()),
                           'ci95_network_level_t': t_interval(per_net, clip=True) if networks > 1 else None,
                           'ci95_single_network_replicates_t': t_interval([r['terminal_B_ratio'] for r in part],
                                                                          clip=True) if networks == 1 else None,
                           'absorption_rate': len(absorbed) / len(part),
                           'absorption_time_mean_among_absorbed': float(np.mean(absorbed)) if absorbed else None,
                           'quasi_rate': len(quasi) / len(part),
                           'quasi_time_mean_among_observed': float(np.mean(quasi)) if quasi else None,
                           'n_all_green': sum(r['absorb_class'] == 'all_green' for r in part),
                           'n_all_B': sum(r['absorb_class'] == 'all_B' for r in part),
                           'n_not_absorbed': sum(r['absorb_class'] == 'not_absorbed' for r in part)}
    # Explicit interaction contrasts: inertia effect at each eta, then test differences in that effect across eta.
    inertia_effect = np.column_stack(
        [means[condition_name(eta, 'high')] - means[condition_name(eta, 'low')] for eta in ETA_LEVELS])
    contrasts = {f'eta_{eta:g}': {'mean_high_minus_low': float(inertia_effect[:, i].mean()),
                                  'ci95_t': t_interval(inertia_effect[:, i])} for i, eta in enumerate(ETA_LEVELS)}
    interaction = paired_test(inertia_effect, [f'eta_{eta:g}' for eta in ETA_LEVELS]) if networks > 1 else {
        'warning': 'Only one network: network-level interaction inference is not available.'}
    # Main effects, descriptive marginal averages; paired tests at the network level.
    eta_matrix = np.column_stack(
        [(means[condition_name(eta, 'low')] + means[condition_name(eta, 'high')]) / 2 for eta in ETA_LEVELS])
    inertia_matrix = np.column_stack([
        np.mean([means[condition_name(e, 'low')] for e in ETA_LEVELS], axis=0),
        np.mean([means[condition_name(e, 'high')] for e in ETA_LEVELS], axis=0)])
    main_effects = {'eta': paired_test(eta_matrix, [f'eta_{x:g}' for x in ETA_LEVELS]) if networks > 1 else None,
                    'inertia': paired_test(inertia_matrix, ['low', 'high']) if networks > 1 else None}
    return summaries, {'unit': 'independent_BA_network_means' if networks > 1 else 'fixed_network_descriptive',
                       'inertia_effect_by_eta': contrasts, 'interaction_test': interaction,
                       'main_effect_tests': main_effects,
                       'note': 'Interaction is assessed by differences between high-minus-low inertia effects across eta; no interaction claim follows from visual differences alone.'}


def export_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));
        writer.writeheader();
        writer.writerows(rows)


def plots(output, curve_dict, steps, networks, repeats, summary):
    # ==========================================
    # 1. Evolution curves (1x3 子图)
    # ==========================================
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 20, 'axes.linewidth': 1.2, 'mathtext.fontset': 'stix',
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), dpi=300, sharey=True)
    t = np.arange(steps + 1)

    palette = {'low': '#1f77b4', 'high': '#d62728'}
    linestyle = {'low': '-', 'high': '--'}

    for ax, eta in zip(axes, ETA_LEVELS):
        for level in ('low', 'high'):
            name = condition_name(eta, level)
            arr = np.asarray(curve_dict[name])
            mu = arr.mean(axis=0)
            net = arr.reshape(networks, repeats, steps + 1).mean(axis=1) if networks > 1 else arr

            if len(net) > 1:
                critical = student_t.ppf(.975, len(net) - 1)
                half = critical * net.std(axis=0, ddof=1) / np.sqrt(len(net))
            else:
                half = np.zeros_like(mu)

            ax.plot(t, mu, color=palette[level], linestyle=linestyle[level], lw=1.8,
                    label=f'{level.capitalize()} inertia')
            ax.fill_between(t, np.clip(mu - half, 0, 1), np.clip(mu + half, 0, 1),
                            color=palette[level], alpha=0.18)

        ax.set_xlabel(r'Time step $t$')  # 修正为 $t$
        ax.set_xlim(0, steps)
        ax.set_ylim(0, 1.05)
        for t_val in (100, 500, 1000):
            if t_val <= steps:
                ax.axvline(x=t_val, color='#d62728', linestyle='--', linewidth=1.0, alpha=0.5)

        # abc 标签放在子图正下方
        ax.text(0.5, -0.20, f'{chr(97 + ETA_LEVELS.index(eta))}) $\\eta={eta:g}$', ha='center', va='top',
                transform=ax.transAxes, fontsize=20, color='#333333')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out', length=4, width=1.2, labelsize=20)

    axes[0].set_ylabel(r'Greenwashing density $f(t)$')  # 修正为 $f(t)$
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=18)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.30, wspace=0.08)

    fig.savefig(output / 'Experiment4_evolution_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ==========================================
    # 2. Terminal comparison bar chart (柱状图)
    # ==========================================
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    xs = np.arange(3)
    w = 0.36

    for level, offset, color in [('low', -w / 2, '#1f77b4'), ('high', w / 2, '#d62728')]:
        names = [condition_name(e, level) for e in ETA_LEVELS]
        vals = [summary[c]['terminal_mean'] for c in names]

        err = np.array([
            [vals[i] - summary[c]['ci95_network_level_t'][0] if summary[c]['ci95_network_level_t'] else 0 for i, c in
             enumerate(names)],
            [summary[c]['ci95_network_level_t'][1] - vals[i] if summary[c]['ci95_network_level_t'] else 0 for i, c in
             enumerate(names)]
        ])

        ax.bar(xs + offset, vals, width=w, label=f'{level.capitalize()} inertia', color=color, alpha=0.85)
        ax.errorbar(xs + offset, vals, yerr=err, fmt='none', ecolor='black', capsize=6)

    ax.set_xticks(xs, [f'$\\eta={e:g}$' for e in ETA_LEVELS])
    ax.set_ylabel(r'Terminal greenwashing density $f(T)$')  # 修正为 $f(T)$
    ax.set_ylim(0, 1.05)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out', length=4, width=1.2)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.18)
    fig.savefig(output / 'Experiment4_terminal_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main(args):
    output = Path(args.output);
    output.mkdir(parents=True, exist_ok=True)
    network_seeds = [42 + 100 * ni for ni in range(args.networks)]
    rows = [];
    curves = {condition_name(*c): [] for c in CONDITIONS}
    for ni, network_seed in enumerate(network_seeds):
        graph = nx.barabasi_albert_graph(N, M, seed=network_seed)
        adj = [np.fromiter(graph.neighbors(i), dtype=int) for i in range(N)]
        degrees = np.array([len(a) for a in adj]);
        hubs = np.argsort(-degrees, kind='stable')[:K]
        for rep in range(1, args.repeats + 1):
            init_seed = seed_for(args.seed, ni, rep, 10)
            initial = init_strategy(hubs, init_seed)
            signature = ','.join(map(str, np.flatnonzero(initial)))
            for index, (eta, level) in enumerate(CONDITIONS):
                lo, hi = INERTIA[level]
                inertia = lo + (hi - lo) * degrees / degrees.max()
                dynamic_seed = seed_for(args.seed, ni, rep, 20, index)
                res = simulate(adj, inertia, initial, eta, dynamic_seed, args.steps)
                cond = condition_name(eta, level)
                rows.append(dict(network_index=ni, network_seed=network_seed, replicate=rep,
                                 condition=cond, eta=eta, inertia_level=level, rho_min=lo, rho_max=hi,
                                 initialization_seed=init_seed, initial_B_signature=signature,
                                 simulation_seed=dynamic_seed, initial_B_count=int(initial.sum()),
                                 initial_B_hubs=int(initial[hubs].sum()), terminal_B_ratio=res['terminal'],
                                 quasi_step=res['quasi_step'], absorption_step=res['absorb_step'],
                                 absorb_class=res['absorb_class'], tail100_switches=res['final_switches_tail100']))
                if args.plot: curves[cond].append(res['path'])
        print(f'Experiment 4: completed network {ni + 1}/{args.networks} (seed={network_seed})', flush=True)
    summaries, tests = analyze(rows, args.networks, args.repeats)
    export_csv(output / 'replicate_results.csv', rows)
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding='utf-8')
    (output / 'interaction_statistics.json').write_text(json.dumps(tests, indent=2, ensure_ascii=False),
                                                        encoding='utf-8')
    metadata = {'N': N, 'm': M, 'hub_count': K, 'network_seeds': network_seeds, 'base_seed': args.seed,
                'networks': args.networks, 'repeats_per_network': args.repeats, 'steps': args.steps,
                'eta_levels': ETA_LEVELS, 'inertia_levels': INERTIA, 'payoff_triangles': FUZZY,
                'update_fraction': UPDATE_FRAC, 'initial_peripheral_B_rate': PERIPHERAL_B_RATE,
                'initial_B_hubs': 2, 'shared_initial_states_across_6_conditions': True,
                'inference_unit': 'independent network mean' if args.networks > 1 else 'fixed network matched replicates',
                'payoff_sampling': 'new triangular probability sample at each local payoff evaluation; probability sampling is an additional assumption, not fuzzy membership',
                'quasi_definition': f'first {WINDOW} density observations with SD < {TOL}',
                'absorption_censoring': f'runs without absorption at T={args.steps} are right-censored'}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    if args.plot: plots(output, curves, args.steps, args.networks, args.repeats, summaries)
    print('Terminal means:', {k: round(v['terminal_mean'], 4) for k, v in summaries.items()})
    print('Results:', output.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--networks', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='experiment4_second_round_results')
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    args = parser.parse_args()
    if min(args.steps, args.repeats, args.networks) < 1: parser.error('steps, repeats and networks must be >= 1')
    main(args)