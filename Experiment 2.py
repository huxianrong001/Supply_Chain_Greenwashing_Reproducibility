"""Experiment 2 — second-round revision.

Main comparison: low/medium/high triangular payoff uncertainty and a
weighted-defuzzification deterministic benchmark.
Additional reviewer checks: stochastic sampling timescales (comparison,
edge-fixed, run-fixed), asymmetric weighted vs centroid defuzzification.

Run (baseline + sensitivity with 20 networks):
 python experiment2_second_round_rewritten.py --steps 1000 --repeats 20 --networks 20
Quick smoke test:
 python experiment2_second_round_rewritten.py --steps 30 --repeats 3 --networks 1
"""
from __future__ import annotations
import argparse
import csv
import json
import os
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
PERIPHERAL_B_RATE = 0.30
OBS_WINDOW, OBS_SD = 20, 0.005
BASE_KEYS = ('R', 'S', 'T', 'P')
NETWORK_BASE_SEED = 42
FUZZ_LEVELS = {
    'low': {'R': (.11, .12, .13), 'S': (-.025, -.02, -.015),
            'T': (.18, .20, .22), 'P': (-.035, -.03, -.025)},
    'medium': {'R': (.08, .12, .16), 'S': (-.04, -.02, .00),
               'T': (.14, .20, .26), 'P': (-.06, -.03, .00)},
    'high': {'R': (.02, .12, .22), 'S': (-.08, -.02, .04),
             'T': (.06, .20, .34), 'P': (-.12, -.03, .06)},
}
ASYMMETRIC = {'R': (.07, .12, .19), 'S': (-.06, -.02, .01),
              'T': (.11, .20, .32), 'P': (-.09, -.03, .02)}
MAIN_CONDITIONS = ['low', 'medium', 'high', 'benchmark']
SAMPLING_CONDITIONS = ['comparison', 'edge_fixed', 'run_fixed']
OPERATOR_CONDITIONS = ['weighted', 'centroid']


def stable_seed(base, network_index, replicate, stream, condition=0):
    """Deterministic, non-overlapping independent streams."""
    ss = np.random.SeedSequence([base, network_index, replicate, stream, condition])
    return int(ss.generate_state(1, dtype=np.uint64)[0])


def payoff_array(fuzzy, operator='weighted'):
    data = np.array([fuzzy[k] for k in BASE_KEYS], dtype=float)
    if operator == 'weighted':
        return (data[:, 0] + 2*data[:, 1] + data[:, 2]) / 4
    if operator == 'centroid':
        return data.mean(axis=1)
    raise ValueError(operator)


def reference_equilibrium(arr):
    r, s, temptation, p = arr
    denominator = (temptation - r) + (s - p)
    return float((temptation-r)/denominator) if abs(denominator)>1e-12 else None


def network_setup(network_seed):
    g = nx.barabasi_albert_graph(N, M, seed=network_seed)
    adj = [np.fromiter(g.neighbors(i), dtype=int) for i in range(N)]
    deg = np.array([len(x) for x in adj])
    hubs = np.argsort(-deg, kind='stable')[:HUB_COUNT]
    inertia = RHO_MIN + (RHO_MAX-RHO_MIN)*deg/deg.max()
    return g, adj, hubs, inertia, deg


def shared_initial(hubs, rng):
    initial = np.zeros(N, dtype=np.int8)
    peripheral = np.array([i for i in range(N) if i not in set(hubs.tolist())])
    n_b = int(round(PERIPHERAL_B_RATE * len(peripheral)))
    initial[rng.choice(peripheral, size=n_b, replace=False)] = 1
    initial[hubs[3:]] = 1  # first 3 hub green; other 2 greenwashing
    return initial


def simulate(adj, inertia, initial, fuzz, mode, seed, steps, operator='weighted',
             save_snapshots=False):
    """mode: comparison / edge_fixed / run_fixed / deterministic."""
    rng = np.random.default_rng(seed)
    state = initial.copy()
    density = np.empty(steps+1, dtype=float)
    density[0] = state.mean()
    snapshots = {0: state.copy()} if save_snapshots else {}
    switches = np.zeros(steps, dtype=int)
    absorb_at = 0 if state.min()==state.max() else None
    quasi_at = None
    fixed_payoff = payoff_array(fuzz, operator) if mode=='deterministic' else None
    if mode == 'run_fixed':
        fixed_payoff = np.array([rng.triangular(*fuzz[k]) for k in BASE_KEYS])
    edge_effects = None
    if mode == 'edge_fixed':
        edge_effects = {(i,int(j)): np.array([rng.triangular(*fuzz[k])
                          for k in BASE_KEYS]) for i, ns in enumerate(adj) for j in ns}
    elif mode not in ('comparison','run_fixed','deterministic'):
        raise ValueError(mode)

    def payoff(i):
        ids = 2*int(state[i]) + state[adj[i]].astype(int)
        if mode == 'edge_fixed':
            return float(sum(edge_effects[i,int(j)][int(code)]
                             for j, code in zip(adj[i], ids)))
        if fixed_payoff is not None:
            return float(fixed_payoff[ids].sum())
        return float(sum(rng.triangular(*fuzz[BASE_KEYS[int(k)]]) for k in ids))

    updates = max(1, round(UPDATE_FRACTION * N))
    for step in range(1, steps+1):
        changed = 0
        # Note on Common Random Numbers (CRN): The node selection and imitation RNG
        # are coupled to the passed 'seed'. Because different conditions receive
        # different stable_seeds, CRN is NOT strictly enforced across conditions.
        # However, they share the exact same initial state per network-replicate.
        for i in rng.choice(N, size=updates, replace=False):
            if not len(adj[i]):
                continue
            j = int(rng.choice(adj[i]))
            if state[i] == state[j]:
                continue
            d = payoff(j) - payoff(i)
            probability = (1-inertia[i])/(1+np.exp(-np.clip(ETA*d, -700,700)))
            if rng.random() < probability:
                state[i] = state[j]
                changed += 1
        switches[step-1] = changed
        density[step] = state.mean()
        if save_snapshots and step in (100,500,1000):
            snapshots[step] = state.copy()
        if absorb_at is None and state.min()==state.max():
            absorb_at = step
        if quasi_at is None and step>=OBS_WINDOW:
            recent = density[step-OBS_WINDOW+1:step+1]
            if recent.std() < OBS_SD:
                quasi_at = step
    category = 'all_green' if state.max()==0 else ('all_B' if state.min()==1 else 'mixed')
    tail = min(100,steps)
    return {'density': density, 'snapshots': snapshots,
            'initial_density': float(initial.mean()),
            'terminal_density': float(state.mean()),
            'tail_mean': float(density[-tail:].mean()),
            'tail_sd': float(density[-tail:].std()),
            'tail_switches': float(switches[-tail:].mean()),
            'quasi_step': quasi_at, 'absorption_step': absorb_at,
            'absorbed_state': category}


def mean_ci(a):
    arr = np.asarray(a,dtype=float)
    if len(arr)<2:
        return [None,None]
    # 使用 t 临界值而非 1.96
    h = student_t.ppf(.975,len(arr)-1)*arr.std(ddof=1)/np.sqrt(len(arr))
    return [float(max(0,arr.mean()-h)),float(min(1,arr.mean()+h))]


def summary_by_condition(rows, networks):
    out={}
    for cond in sorted(set(row['condition'] for row in rows)):
        subset=[r for r in rows if r['condition']==cond]
        vals=np.array([r['terminal_B_ratio'] for r in subset])
        network_means=np.array([np.mean([r['terminal_B_ratio'] for r in subset
                            if r['network_index']==i]) for i in range(networks)])
        abs_rows=[r for r in subset if r.get('absorption_step') is not None]
        quasi_rows=[r for r in subset if r.get('quasi_step') is not None]
        out[cond]={'n_simulations':len(subset),'n_networks':networks,
                   'terminal_mean':float(vals.mean()),
                   'ci95_run_conditional_t':mean_ci(vals) if networks==1 else None,
                   'ci95_network_level_t':mean_ci(network_means) if networks>1 else None,
                   'absorption_rate':len(abs_rows)/len(subset),
                   'absorption_mean_among_absorbed':float(np.mean([r.get('absorption_step') for r in abs_rows])) if abs_rows else None,
                   'quasi_rate':len(quasi_rows)/len(subset),
                   'quasi_mean_among_observed':float(np.mean([r.get('quasi_step') for r in quasi_rows])) if quasi_rows else None,
                   'all_green_n':sum(r['absorbed_state']=='all_green' for r in subset),
                   'all_B_n':sum(r['absorbed_state']=='all_B' for r in subset),
                   'mixed_at_T_n':sum(r['absorbed_state']=='mixed' for r in subset)}
    return out


def holm(pvalues):
    indexed=sorted(enumerate(pvalues),key=lambda z:z[1])
    res=[None]*len(pvalues)
    running=0
    for rank,(i,p) in enumerate(indexed):
        running=max(running,min(1.,p*(len(pvalues)-rank)))
        res[i]=running
    return res


def matched_stats(rows, conditions, networks, repeats):
    by={(r['network_index'],r['replicate'],r['condition']):r['terminal_B_ratio'] for r in rows}
    units=[]
    for i in range(networks):
        vals=[]
        for cond in conditions:
            observations=[by[(i,r,cond)] for r in range(1,repeats+1)]
            vals.append(float(np.mean(observations)) if networks>1 else observations)
        if networks>1:
            units.append(vals)
        else:
            units=list(map(list,zip(*vals)))
    matrix=np.array(units)
    result={'unit':'network_mean' if networks>1 else 'replicate_on_fixed_network',
            'n_units':int(len(matrix)),'conditions':conditions,
            'note':'Fixed-network inference is conditional on graph seed.' if networks==1 else 'Network realization is the inference unit.'}

    # 修正：Friedman 检验只有在超过 2 组条件（且大于等于 3 个样本）时才能使用
    if len(conditions) >= 3 and len(matrix) >= 3:
        try:
            st,p=friedmanchisquare(*matrix.T)
            result['friedman']={'statistic':float(st),'p':float(p)}
        except ValueError as exc:
            result['friedman']={'error':str(exc)}

    pair=[]
    for ix,a in enumerate(conditions):
        for iy in range(ix+1,len(conditions)):
            b=conditions[iy]
            d=matrix[:,ix]-matrix[:,iy]
            if np.all(d == 0):
                p=1.; W=0.
            else:
                try:
                    W,p=wilcoxon(d,zero_method='wilcox',alternative='two-sided')
                    p=float(p); W=float(W)
                    if not np.isfinite(p): p=1.
                except ValueError:
                    p=1.; W=0.
            pair.append({'comparison':f'{a}_vs_{b}', 'difference_mean':float(d.mean()),
                         'W':W,'p_raw':p})
    adjusted=holm([p['p_raw'] for p in pair])
    for row,value in zip(pair,adjusted):
        row['p_holm']=value
    result['paired_wilcoxon_holm']=pair
    return result


def write_csv(path,rows):
    if not rows: return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def make_figures(out,curves,snaps,graph,deg,hubs,summary,steps,networks,repeats):
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 20, 'axes.linewidth': 1.2, 'mathtext.fontset': 'stix',
    })

    # 1. Evolution curves
    sc_labels = {
        'low': 'a) Low uncertainty',
        'medium': 'b) Medium uncertainty',
        'high': 'c) High uncertainty'
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), dpi=300, sharey=True)
    x = np.arange(steps + 1)

    for ax, cond in zip(axes, ['low', 'medium', 'high']):
        vals = np.array(curves[cond])
        mu = vals.mean(axis=0)
        if networks > 1:
            bynet = vals.reshape(networks, repeats, steps + 1).mean(axis=1)
            sem = bynet.std(axis=0, ddof=1) / np.sqrt(networks)
            # 使用 t 临界值
            t_crit = student_t.ppf(0.975, networks - 1)
            name = '95% network-level t CI'
        else:
            sem = vals.std(axis=0, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.zeros_like(mu)
            t_crit = student_t.ppf(0.975, len(vals) - 1) if len(vals) > 1 else 0
            name = '95% conditional t CI'

        ax.plot(x, mu, color='black', lw=1.8, label='Mean')
        ax.fill_between(x, np.clip(mu - t_crit * sem, 0, 1), np.clip(mu + t_crit * sem, 0, 1),
                        color='#8daac0', alpha=0.35, label=name)
        ax.plot(x, vals[0], color='#ce8041', alpha=0.5, lw=0.8, label='Representative run')

        # 统一 LaTeX 符号为 $t$
        ax.set_xlabel(r'Time step $t$')
        ax.set_xlim(0, steps)
        ax.set_ylim(0, 1.05)
        for t in (100, 500, 1000):
            if t <= steps:
                ax.axvline(x=t, color='#d62728', linestyle='--', linewidth=1.0, alpha=0.5)

        ax.text(0.5, -0.20, sc_labels[cond], ha='center', va='top',
                transform=ax.transAxes, fontsize=20, color='#333333')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out', length=4, width=1.2, labelsize=20)

    # 统一 LaTeX 符号为 $f(t)$
    axes[0].set_ylabel(r'Greenwashing density $f(t)$')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=18)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.30, wspace=0.08)
    fig.savefig(out / 'Fig5-2_mean_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2. Terminal comparison bar chart
    labels = MAIN_CONDITIONS
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    means = [summary[k]['terminal_mean'] for k in labels]
    cis = [summary[k]['ci95_network_level_t'] if networks > 1 else summary[k]['ci95_run_conditional_t'] for k in labels]
    yerr = np.array([[means[i]-cis[i][0] if cis[i][0] is not None else 0 for i in range(4)],
                     [cis[i][1]-means[i] if cis[i][1] is not None else 0 for i in range(4)]])

    ax.bar(labels, means, color=['#43658b', '#b67846', '#9a4545', '#666666'])
    ax.errorbar(labels, means, yerr=yerr, fmt='none', ecolor='black', capsize=6)
    # 统一 LaTeX 符号
    ax.set_ylabel(r'Terminal greenwashing density $f(1000)$')
    ax.set_ylim(0, 1.05)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out', length=4, width=1.2)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.18)
    fig.savefig(out / 'Fig5-4_terminal_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3. Network snapshots
    pos = nx.spring_layout(graph, seed=42, k=0.55 / np.sqrt(N), iterations=200)
    times = [t for t in (0, 100, 500, 1000) if t <= steps]
    cond_list = ['low', 'medium', 'high']

    NODE_MIN, NODE_MAX = 55.0, 420.0
    d_min, d_max = float(deg.min()), float(deg.max())
    sizes = NODE_MIN + (NODE_MAX - NODE_MIN) * (deg - d_min) / max(1.0, d_max - d_min)

    cmap = plt.get_cmap('RdYlGn_r')
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(len(cond_list), len(times),
                             figsize=(4.6 * len(times), 3.9 * len(cond_list)),
                             dpi=300, squeeze=False)

    for row, cond in enumerate(cond_list):
        for col, t in enumerate(times):
            ax = axes[row, col]
            avg = snaps[cond][t] / repeats
            node_colors = [cmap(norm(avg[i])) for i in range(N)]

            nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='#c8ccd1', alpha=0.28, width=0.45)

            non_hubs = [i for i in range(N) if i not in set(hubs.tolist())]
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

            if row == len(cond_list) - 1:
                ax.text(0.5, -0.08, f't = {t}', transform=ax.transAxes,
                        ha='center', va='top', fontsize=20, color='#333333')

            if col == 0:
                ax.text(0.02, 0.98, sc_labels[cond], transform=ax.transAxes,
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

    fig.savefig(out / 'Fig5-3_network_frequencies.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main(opt):
    out=Path(opt.output);out.mkdir(parents=True,exist_ok=True)

    # 当跳过敏感性分析时，清理旧的结果文件，避免混淆
    if opt.skip_sensitivity:
        for f in ['sampling_mechanism_replicates.csv', 'asymmetric_operator_comparison.csv', 'sensitivity_statistics.json']:
            p = out / f
            if p.exists():
                p.unlink()
                print(f'Removed stale file: {f}')

    net_seeds=[NETWORK_BASE_SEED+100*i for i in range(opt.networks)]
    all_main=[]; all_sampling=[]; all_operators=[]
    curves={c:[] for c in MAIN_CONDITIONS}
    snaps={c:{t:np.zeros(N) for t in (0,100,500,1000) if t<=opt.steps} for c in ('low','medium','high')}
    rep_graph=None;rep_deg=None;rep_hubs=None

    for net_i,net_seed in enumerate(net_seeds):
        graph,adj,hubs,inertia,deg=network_setup(net_seed)
        if rep_graph is None: rep_graph,rep_deg,rep_hubs=graph,deg,hubs

        for rep in range(1,opt.repeats+1):
            init_seed=stable_seed(opt.seed,net_i,rep,10)
            initial=shared_initial(hubs,np.random.default_rng(init_seed))

            for ci,cond in enumerate(MAIN_CONDITIONS):
                fuzzy=FUZZ_LEVELS['medium'] if cond=='benchmark' else FUZZ_LEVELS[cond]
                mode='deterministic' if cond=='benchmark' else 'comparison'
                run_seed=stable_seed(opt.seed,net_i,rep,20,ci)
                res=simulate(adj,inertia,initial,fuzzy,mode,run_seed,opt.steps,
                             save_snapshots=(net_i==0))
                all_main.append({'network_index':net_i,'network_seed':net_seed,
                                 'replicate':rep,'condition':cond,
                                 'initialization_seed':init_seed,'simulation_seed':run_seed,
                                 'initial_B_count':int(initial.sum()),
                                 'initial_peripheral_B_count':int(initial.sum()-2),
                                 'terminal_B_ratio':res['terminal_density'],
                                 'tail100_mean':res['tail_mean'],
                                 'tail100_sd':res['tail_sd'],
                                 'tail100_switches_mean':res['tail_switches'],
                                 'quasi_step':res['quasi_step'],
                                 'absorption_step':res['absorption_step'],
                                 'absorbed_state':res['absorbed_state']})
                curves[cond].append(res['density'])
                if net_i==0 and cond in snaps:
                    for t,arr in res['snapshots'].items():
                        snaps[cond][t]+=arr

            if not opt.skip_sensitivity:
                for ci,mode in enumerate(SAMPLING_CONDITIONS):
                    run_seed=stable_seed(opt.seed,net_i,rep,30,ci)
                    res=simulate(adj,inertia,initial,FUZZ_LEVELS['medium'],mode,
                                 run_seed,opt.steps)
                    all_sampling.append({'network_index':net_i,'network_seed':net_seed,
                        'replicate':rep,'condition':mode,'initialization_seed':init_seed,
                        'simulation_seed':run_seed,'initial_B_count':int(initial.sum()),
                        'terminal_B_ratio':res['terminal_density'],
                        'quasi_step':res['quasi_step'],'absorption_step':res['absorption_step'],
                        'absorbed_state':res['absorbed_state']})

                common_seed=stable_seed(opt.seed,net_i,rep,40)
                for operator in OPERATOR_CONDITIONS:
                    res=simulate(adj,inertia,initial,ASYMMETRIC,'deterministic',
                                 common_seed,opt.steps,operator=operator)
                    all_operators.append({'network_index':net_i,'network_seed':net_seed,
                        'replicate':rep,'condition':operator,'initialization_seed':init_seed,
                        'simulation_seed':common_seed,
                        'reference_f_star':reference_equilibrium(payoff_array(ASYMMETRIC,operator)),
                        'terminal_B_ratio':res['terminal_density'],
                        'absorbed_state':res['absorbed_state'],
                        'quasi_step':res['quasi_step'],'absorption_step':res['absorption_step']})

        print(f'Completed network {net_i+1}/{opt.networks} (seed={net_seed})',flush=True)

    write_csv(out/'replicate_results.csv',all_main)
    summary=summary_by_condition(all_main,opt.networks)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    stats=matched_stats(all_main,MAIN_CONDITIONS,opt.networks,opt.repeats)
    (out/'paired_statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')

    if not opt.skip_sensitivity:
        write_csv(out/'sampling_mechanism_replicates.csv',all_sampling)
        write_csv(out/'asymmetric_operator_comparison.csv',all_operators)

        sample_summary=summary_by_condition(all_sampling,opt.networks)
        op_summary=summary_by_condition(all_operators,opt.networks)

        sensitivity={'sampling_summary':sample_summary,
                     'sampling_matched_tests':matched_stats(all_sampling,SAMPLING_CONDITIONS,opt.networks,opt.repeats),
                     'operator_summary':op_summary,
                     'operator_matched_tests':matched_stats(all_operators,OPERATOR_CONDITIONS,opt.networks,opt.repeats),
                     'asymmetric_parameters':{k:list(v) for k,v in ASYMMETRIC.items()},
                     'operator_reference_f_star':{op:reference_equilibrium(payoff_array(ASYMMETRIC,op)) for op in OPERATOR_CONDITIONS}}
        (out/'sensitivity_statistics.json').write_text(json.dumps(sensitivity,indent=2),encoding='utf-8')

    metadata={'N':N,'m':M,'hub_count':HUB_COUNT,'network_seeds':net_seeds,
              'repeats_per_network':opt.repeats,'steps':opt.steps,'base_seed':opt.seed,
              'eta':ETA,'update_fraction':UPDATE_FRACTION,'rho_bounds':[RHO_MIN,RHO_MAX],
              'peripheral_B_ratio':PERIPHERAL_B_RATE,
              'fuzzy_levels':FUZZ_LEVELS,'asymmetric_parameters':ASYMMETRIC,
              'simulation_modes':{'comparison':'new triangular samples for each local payoff evaluation',
                  'edge_fixed':'directed edge-by-strategy-pair fixed triangular draws per run',
                  'run_fixed':'one triangular draw per R/S/T/P each run',
                  'deterministic':'weighted / centroid point values'},
              'inference_unit':'network means' if opt.networks>1 else 'replicates conditional on fixed network',
              'fixed_initialization_within_network_replicate':True,
              'sampling_sensitivity_enabled':not opt.skip_sensitivity,
              'note':'Fuzzy membership functions are NOT probability distributions; triangular sampling is an explicit additional stochastic modelling assumption. Non-absorption times are right-censored at T.'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')

    if opt.plot:
        make_figures(out,curves,snaps,rep_graph,rep_deg,rep_hubs,summary,opt.steps,opt.networks,opt.repeats)

    print('Main terminal means:',{k:round(v['terminal_mean'],4) for k,v in summary.items()})
    print('Output:',out.resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--networks',type=int,default=20, help='Number of independent BA network realizations (default 20)')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',default='experiment2_second_round_results')
    parser.add_argument('--skip-sensitivity',action='store_true',help='Only run original four conditions')
    parser.add_argument('--no-plot',dest='plot',action='store_false')
    opt=parser.parse_args()
    if min(opt.steps,opt.repeats,opt.networks)<1:
        parser.error('steps, repeats and networks must be >=1')
    main(opt)