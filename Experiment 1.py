"""Experiment 1 — Hub initial-strategy configurations (second-round revision).

A: five green hubs; B: five greenwashing hubs; C: three green + two B hubs.
Within a (network, replicate) block the peripheral initial state is identical.
Only the initial hub strategies change. Independent random evolution follows.

Install: pip install numpy scipy networkx matplotlib
Run: python experiment1_second_round_rewritten.py --steps 1000 --repeats 20
Optional network robustness: --networks 20  (new sample, not old Table 3)
"""
from __future__ import annotations
import argparse
import csv
import json
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
WINDOW, DELTA = 20, 0.005
FUZZY = {'R': (.08,.12,.16), 'S': (-.04,-.02,0.),
         'T': (.14,.20,.26), 'P': (-.06,-.03,0.)}
CASES = ('A', 'B', 'C')
CASE_NAMES = {'A':'Hub all green','B':'Hub all greenwashing','C':'Hub 3G 2B'}
SNAPSHOTS = (0,100,500,1000)


def hub_configured(base_state, hubs, case):
    state = base_state.copy()
    if case == 'A':
        state[hubs] = 0
    elif case == 'B':
        state[hubs] = 1
    elif case == 'C':
        state[hubs[:3]] = 0
        state[hubs[3:]] = 1
    else:
        raise ValueError(case)
    return state


def payoff(node, state, adj, rng):
    keys = ('R','S','T','P')
    indices = 2 * int(state[node]) + state[adj[node]]
    return sum(float(rng.triangular(*FUZZY[keys[int(x)]])) for x in indices)


def simulate(graph, initial, sim_seed, steps, snapshots):
    rng = np.random.default_rng(sim_seed)
    adj = [np.fromiter(graph.neighbors(i), dtype=int) for i in range(N)]
    degree = np.array([len(a) for a in adj], dtype=float)
    inertia = RHO_MIN + (RHO_MAX-RHO_MIN)*degree/degree.max()
    state = initial.copy()
    density = np.empty(steps+1)
    density[0] = state.mean()
    shots = {0: state.copy()} if 0 in snapshots else {}
    conv_step = None
    absorb_step = 0 if state.min() == state.max() else None
    for t in range(1, steps+1):
        for i in rng.choice(N, size=max(1,round(UPDATE_FRACTION*N)), replace=False):
            neighbors = adj[i]
            if not len(neighbors):
                continue
            j = int(rng.choice(neighbors))
            if state[i] == state[j]:
                continue
            diff = payoff(j, state, adj, rng) - payoff(i, state, adj, rng)
            z = np.clip(ETA*diff, -700.,700.)
            switch_prob = (1.-inertia[i]) / (1.+np.exp(-z))
            if rng.random() < switch_prob:
                state[i] = state[j]
        density[t] = state.mean()
        if t in snapshots:
            shots[t] = state.copy()
        if absorb_step is None and state.min() == state.max():
            absorb_step = t
        if conv_step is None and t >= WINDOW:
            if density[t-WINDOW+1:t+1].std() < DELTA:
                conv_step = t
    cls = 'all_green' if state.max()==0 else ('all_B' if state.min()==1 else 'not_absorbed')
    return dict(density=density, shots=shots, terminal=float(state.mean()),
                quasi_step=conv_step, absorb_step=absorb_step, absorb_class=cls,
                tail_mean=float(density[-min(100,steps):].mean()),
                tail_sd=float(density[-min(100,steps):].std()))


def interval(values):
    a = np.asarray(values, dtype=float)
    mu = a.mean()
    if len(a)<2:
        return [float(mu),None,None]
    h = float(student_t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return [float(mu),float(max(0,mu-h)),float(min(1,mu+h))]


def holm(pvalues):
    result = [None]*len(pvalues)
    running=0.
    for rank,idx in enumerate(np.argsort(pvalues)):
        running=max(running,min(1.,(len(pvalues)-rank)*pvalues[idx]))
        result[idx]=float(running)
    return result


def make_plots(curves, shots, graph, hubs, steps, out, block_count):
    # ==========================================
    # 1. Evolution curves (演化曲线图)
    # ==========================================
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 20, 'axes.linewidth': 1.2, 'mathtext.fontset': 'stix',
    })
    sc_labels = {'A': 'a) Hub all green', 'B': 'b) Hub all greenwashing', 'C': 'c) Hub 3G 2B'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), dpi=300, sharey=True)
    tt = np.arange(steps + 1)

    for ax, case in zip(axes, CASES):
        a = np.asarray(curves[case])
        mu = a.mean(axis=0)
        if len(a) > 1:
            # 正态近似 95% CI (mean ± 1.96 * SEM)，回应审稿人 Comment 8
            sem = a.std(axis=0, ddof=1) / np.sqrt(len(a))
            lo, hi = mu - 1.96 * sem, mu + 1.96 * sem
        else:
            lo = hi = mu

        ax.plot(tt, mu, color='black', lw=1.8, label='Mean')
        ax.fill_between(tt, np.maximum(0, lo), np.minimum(1, hi),
                        color='#8daac0', alpha=0.35, label='95% CI')
        ax.plot(tt, a[0], color='#ce8041', lw=0.8, alpha=0.5, label='Representative run')

        ax.set_xlabel(r'Time step $(t)$')
        ax.set_xlim(0, steps)
        ax.set_ylim(0, 1.05)
        for t in (100, 500, 1000):
            if t <= steps:
                ax.axvline(x=t, color='#d62728', linestyle='--', linewidth=1.0, alpha=0.5)

        # abc 标签放在子图正下方
        ax.text(0.5, -0.20, sc_labels[case], ha='center', va='top',
                transform=ax.transAxes, fontsize=20, color='#333333')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out', length=4, width=1.2, labelsize=20)

    axes[0].set_ylabel(r'Greenwashing density $(f(t)$)')
    handles, labels = axes[0].get_legend_handles_labels()
    # 图例放在最底部居中
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=18)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.30, wspace=0.08)

    # 只保存 PNG
    fig.savefig(out / 'evolution_three_scenarios.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ==========================================
    # 2. Network snapshots (网络快照图)
    # ==========================================
    times = [t for t in SNAPSHOTS if t <= steps]
    if not times: return

    # 更好的弹簧布局，避免节点过度拥挤
    pos = nx.spring_layout(graph, seed=42, k=0.55 / np.sqrt(N), iterations=200)
    degree = np.array([graph.degree(i) for i in range(N)])
    NODE_MIN, NODE_MAX = 55.0, 420.0
    d_min, d_max = float(degree.min()), float(degree.max())
    node_sizes = NODE_MIN + (NODE_MAX - NODE_MIN) * (degree - d_min) / max(1.0, d_max - d_min)

    hub_list = hubs.tolist()
    non_hub = [i for i in range(N) if i not in set(hub_list)]
    cmap = plt.get_cmap('RdYlGn_r')
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(len(CASES), len(times),
                             figsize=(4.6 * len(times), 3.9 * len(CASES)),
                             dpi=300, squeeze=False)
    for row, case in enumerate(CASES):
        for col, t in enumerate(times):
            ax = axes[row, col]
            freq = shots[case][t] / block_count
            node_colors = [cmap(norm(freq[i])) for i in range(N)]

            nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='#c8ccd1', alpha=0.28, width=0.45)

            nx.draw_networkx_nodes(
                graph, pos, nodelist=non_hub, ax=ax,
                node_color=[node_colors[i] for i in non_hub],
                node_size=[node_sizes[i] for i in non_hub],
                edgecolors='#5a5a5a', linewidths=0.35)

            nx.draw_networkx_nodes(
                graph, pos, nodelist=hub_list, ax=ax,
                node_color=[node_colors[i] for i in hub_list],
                node_size=[node_sizes[i] for i in hub_list],
                edgecolors='black', linewidths=1.8)

            ax.axis('off')
            ax.margins(0.06)

            # 时间标签只在最下面一行显示
            if row == len(CASES) - 1:
                ax.text(0.5, -0.08, f't = {t}', transform=ax.transAxes,
                        ha='center', va='top', fontsize=20, color='#333333')

            # abc 标签在每个子图的左上角
            if col == 0:
                ax.text(0.02, 0.98, sc_labels[case], transform=ax.transAxes,
                        ha='left', va='top', fontsize=20, color='#333333',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none'))

    # 渐变色图例（colorbar）放在整张图的最下面居中
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.035, 0.50, 0.018])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Average greenwashing frequency (0 = all green, 1 = all B)', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    # Hub 节点图例放在右下角
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Hub node',
                              markerfacecolor='#bbbbbb', markersize=16,
                              markeredgewidth=2.0, markeredgecolor='black')]
    fig.legend(handles=legend_elements, loc='lower right',
               bbox_to_anchor=(0.98, 0.055), frameon=False, fontsize=16)

    fig.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.16,
                        wspace=0.03, hspace=0.14)

    # 只保存 PNG
    fig.savefig(out / 'snapshot_three_scenarios.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[];curves={c:[] for c in CASES}
    snapshot_times=[t for t in SNAPSHOTS if t<=args.steps]
    first_shots={c:{t:np.zeros(N) for t in snapshot_times} for c in CASES}
    first_graph=None;first_hubs=None
    seeds=[args.seed+100*i for i in range(args.networks)]
    for net_id,net_seed in enumerate(seeds):
        graph=nx.barabasi_albert_graph(N,M,seed=net_seed)
        degree=np.array([graph.degree(i) for i in range(N)])
        hubs=np.argsort(-degree,kind='stable')[:HUB_COUNT]
        if net_id==0:first_graph,first_hubs=graph,hubs.copy()
        peripheral=np.setdiff1d(np.arange(N),hubs)
        n_b=round(.30*len(peripheral))
        for rep in range(args.repeats):
            init_seed=args.seed+10_000_000+net_id*100_000+rep
            init_rng=np.random.default_rng(init_seed)
            b_nodes=init_rng.choice(peripheral,size=n_b,replace=False)
            base=np.zeros(N,dtype=np.int8);base[b_nodes]=1
            # All cases in this block have IDENTICAL peripheral B nodes.
            for case_id,case in enumerate(CASES):
                initial=hub_configured(base,hubs,case)
                sim_seed=args.seed+20_000_000+net_id*100_000+rep*10+case_id
                result=simulate(graph,initial,sim_seed,args.steps,snapshot_times)
                curves[case].append(result['density'])
                if net_id==0:
                    for t,state in result['shots'].items():first_shots[case][t]+=state
                rows.append(dict(network_id=net_id,network_seed=net_seed,
                     replicate=rep+1,case=case,initialization_seed=init_seed,simulation_seed=sim_seed,
                     initial_peripheral_B_count=int(base.sum()),initial_hub_B_count=int(initial[hubs].sum()),
                     initial_B_count=int(initial.sum()),initial_peripheral_signature=','.join(map(str,sorted(b_nodes))),
                     terminal_B_ratio=result['terminal'],last100_mean=result['tail_mean'],
                     last100_sd=result['tail_sd'],quasi_convergence_step=result['quasi_step'],
                     first_absorption_step=result['absorb_step'],absorption_class=result['absorb_class']))
    with (out/'replicate_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    summary={}
    for c in CASES:
        subset=[r for r in rows if r['case']==c]
        ci=interval([r['terminal_B_ratio'] for r in subset])
        summary[c]=dict(n=len(subset),terminal_mean=ci[0],terminal_ci95_pooled_t=ci[1:],
            # With multiple networks, use network-level means for conservative uncertainty.
            terminal_ci95_network_t=interval([np.mean([r['terminal_B_ratio'] for r in subset
                            if r['network_id']==k]) for k in range(args.networks)])[1:] if args.networks>1 else None,
            quasi_convergence_rate=np.mean([r['quasi_convergence_step'] is not None for r in subset]),
            quasi_step_mean_observed=np.mean([r['quasi_convergence_step'] for r in subset if r['quasi_convergence_step'] is not None])
                            if any(r['quasi_convergence_step'] is not None for r in subset) else None,
            absorption_rate=np.mean([r['first_absorption_step'] is not None for r in subset]),
            absorption_time_mean_among_absorbed=np.mean([r['first_absorption_step'] for r in subset
                            if r['first_absorption_step'] is not None])
                            if any(r['first_absorption_step'] is not None for r in subset) else None)
        print(f'{c}: mean={ci[0]:.4f}; CI (conditional/pooled) = [{ci[1]:.4f},{ci[2]:.4f}]')

    # Fixed-network comparison: pairs by repeat; multi-network: pairs by network means.
    if args.networks==1:
        block_keys=[(0,r) for r in range(1,args.repeats+1)]
        unit='within-fixed-network matched replicate'
    else:
        block_keys=[(net,None) for net in range(args.networks)]
        unit='network realization (means of matched replicates)'
    data={}
    for c in CASES:
        data[c]=np.array([np.mean([row['terminal_B_ratio'] for row in rows
                                  if row['case']==c and row['network_id']==net
                                  and (rep is None or row['replicate']==rep)])
                          for net,rep in block_keys])
    statistics={'inference_unit':unit,'n_units':len(block_keys),
                'warning':'Single-network p-values are conditional on that graph, not topology-generalizable.'
                          if args.networks==1 else 'Network-level paired analysis across BA realizations.'}
    if len(block_keys)>=3:
        try:
            stat,p=friedmanchisquare(*(data[c] for c in CASES))
            statistics['friedman']={'statistic':float(stat),'p':float(p)}
        except ValueError as e:statistics['friedman_error']=str(e)
        tests=[]
        for a,b in [('A','B'),('A','C'),('B','C')]:
            dif=data[a]-data[b]
            if np.allclose(dif,0):p=1.;stat=0.
            else:
                stat,p=wilcoxon(dif,zero_method='wilcox',alternative='two-sided')
            tests.append(dict(comparison=f'{a}-{b}',W=float(stat),p_raw=float(p),
                              difference_mean=float(dif.mean())))
        adjusted=holm([x['p_raw'] for x in tests])
        for x,p in zip(tests,adjusted):x['p_holm']=p
        statistics['paired_wilcoxon']=tests
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=float),encoding='utf-8')
    (out/'paired_statistics.json').write_text(json.dumps(statistics,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'metadata.json').write_text(json.dumps(dict(N=N,m=M,hub_count=HUB_COUNT,eta=ETA,
         update_fraction=UPDATE_FRACTION,rho_bounds=[RHO_MIN,RHO_MAX],fuzzy=FUZZY,
         network_seeds=seeds,repeats=args.repeats,steps=args.steps,base_seed=args.seed,
         initial_peripheral_greenwashing_fraction=.30,
         shared_peripheral_initialization=True,comparison_design=unit),indent=2),encoding='utf-8')

    make_plots(curves,first_shots,first_graph,first_hubs,args.steps,out,args.repeats)
    print('Results saved to:',out.resolve())

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--networks',type=int,default=1)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',default='experiment1_second_round_results')
    args=parser.parse_args()
    if min(args.steps,args.repeats,args.networks)<1:parser.error('steps, repeats and networks must be >= 1')
    main(args)