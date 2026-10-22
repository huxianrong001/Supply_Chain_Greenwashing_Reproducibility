"""Experiment 5: network topology, hub placement, payoff normalization and reviewer robustness.

Main sections:
  5A: BA attachment m=2,3,5 (K=5); accumulation baseline
  5B: fixed 30 initial B on high/random/low degree positions (N=100,m=3)
  5C: m=2,3,5 accumulated vs degree-normalized payoffs; the accumulated arm
      is REUSED BIT-FOR-BIT from 5A (no duplicate baseline simulation).
  5D: topology K=3,5,10; N=100,200,300; inertia degree/constant/reverse/random
  5E: hub mechanism ablation, positions x accumulated/normalized x inertia on/off

Example:
 python experiment5_second_round_rewritten.py --sections 5A,5B,5C,5D,5E --networks 20 --repeats 20 --steps 1000
 Quick check:
 python experiment5_second_round_rewritten.py --networks 2 --repeats 2 --steps 30

All means/p-values for multi-network comparisons use network as inference unit.
B denotes greenwashing; A denotes green strategy. One asynchronous update per
selected node per round, without replacement. Payoff draws use the triangular
probability sampler as an EXTRA assumption, not a fuzzy membership PDF.
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t, wilcoxon, friedmanchisquare

PAY={'R':(.08,.12,.16),'S':(-.04,-.02,0.),'T':(.14,.20,.26),'P':(-.06,-.03,0.)}
KEY=('R','S','T','P')
ETA=1.2
RHO=(.3,.7)
UPDATE=.2
WINDOW=20
TOL=.005

def seed(base,*parts):
    return int(np.random.SeedSequence([base,*[int(v) for v in parts]]).generate_state(1,dtype=np.uint64)[0])

def create_graph(n,m,netseed):
    g=nx.barabasi_albert_graph(n,m,seed=netseed)
    adj=[np.fromiter(g.neighbors(i),dtype=np.int32) for i in range(n)]
    deg=np.array([len(v) for v in adj],dtype=float)
    order=np.argsort(-deg,kind='stable')
    return g,adj,deg,order

def initial_baseline(n,order,k,seed_):
    """Two initially B hub nodes (when available), 30% B of non-hub nodes."""
    rng=np.random.default_rng(seed_)
    hubs=order[:k]; other=np.setdiff1d(np.arange(n),hubs)
    state=np.zeros(n,dtype=np.int8)
    state[rng.choice(other,size=round(.30*n)-min(2,k),replace=False)]=1
    state[hubs[:min(2,k)]]=1
    return state

def initial_position(n,order,count,placement,seed_):
    """Exactly count B nodes; placement changes degree exposure, not B count."""
    state=np.zeros(n,dtype=np.int8)
    if placement=='high': chosen=order[:count]
    elif placement=='low': chosen=order[-count:]
    elif placement=='random': chosen=np.random.default_rng(seed_).choice(n,count,replace=False)
    else: raise ValueError(placement)
    state[chosen]=1
    return state

def inertia_vector(deg,kind,seed_):
    lo,hi=RHO
    ratio=deg/deg.max()
    if kind=='degree': return lo+(hi-lo)*ratio
    if kind=='constant': return np.full(len(deg), float(np.mean(lo+(hi-lo)*ratio)))
    if kind=='reverse':
        # Reassign the SAME inertia values to nodes in reverse degree order,
        # preserving the exact inertia distribution while reversing correlation.
        base=lo+(hi-lo)*ratio
        out=np.empty(len(deg),dtype=float)
        out[np.argsort(deg,kind='stable')]=np.sort(base)[::-1]
        return out
    if kind=='random':
        # Permutation retains EXACTLY the empirical inertia distribution.
        return np.random.default_rng(seed_).permutation(lo+(hi-lo)*ratio)
    if kind=='none':return np.zeros(len(deg),dtype=float)
    raise ValueError(kind)

def simulate(adj,deg,initial,rho,mode,dynseed,steps,save_path=False):
    rng=np.random.default_rng(dynseed)
    state=initial.copy()
    n=len(state)
    path=np.empty(steps+1,dtype=float) if save_path else None
    if save_path:path[0]=state.mean()
    absorption=0 if state.min()==state.max() else None
    quasi=None
    recent=np.empty(WINDOW,dtype=float)
    recent[0]=state.mean()
    updates=max(1,round(UPDATE*n))
    def payoff(i):
        ids=2*int(state[i])+state[adj[i]].astype(int)
        val=sum(float(rng.triangular(*PAY[KEY[int(z)]])) for z in ids)
        return val/len(adj[i]) if mode=='normalized' else val
    for t in range(1,steps+1):
        for i_ in rng.choice(n,updates,replace=False):
            i=int(i_); j=int(rng.choice(adj[i]))
            if state[i]==state[j]:continue
            d=payoff(j)-payoff(i)
            p=(1-rho[i])/(1+np.exp(-np.clip(ETA*d,-700,700)))
            if rng.random()<p:state[i]=state[j]
        f=float(state.mean())
        recent[(t-1)%WINDOW]=f
        if save_path:path[t]=f
        if absorption is None and state.min()==state.max():absorption=t
        if quasi is None and t>=WINDOW and recent.std()<TOL:quasi=t
    label='all_green' if not state.any() else 'all_B' if state.all() else 'not_absorbed'
    return {'terminal_B_ratio':float(state.mean()),'absorption_step':absorption,
            'quasi_step':quasi,'absorbed_state':label,
            'final_B_hubs':int(state[np.argsort(-deg,kind='stable')[:5]].sum())},path

def experiment_row(section,condition,network_index,network_seed,rep,n,m,k,initseed,dynseed,initial,result,deg):
    selected=np.flatnonzero(initial)
    return dict(section=section,condition=condition,network_index=network_index,
        network_seed=network_seed,replicate=rep,N=n,m=m,K=k,
        initialization_seed=initseed,simulation_seed=dynseed,
        initial_B_count=int(initial.sum()),initial_B_degree_sum=int(deg[selected].sum()),
        initial_B_neighbor_exposure=float(deg[selected].sum()/deg.sum()),
        **result)

def write_rows(p,rows):
    if not rows:return
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def clean_num(x):return None if not np.isfinite(x) else float(x)

def ci(vals):
    z=np.asarray(vals,dtype=float)
    if len(z)<2:return [None,None]
    half=student_t.ppf(.975,len(z)-1)*z.std(ddof=1)/np.sqrt(len(z))
    return [clean_num(z.mean()-half),clean_num(z.mean()+half)]

def cond_summary(rows,n_networks):
    out={}
    for cond in sorted({r['condition'] for r in rows}):
        rr=[r for r in rows if r['condition']==cond]
        nm=[np.mean([r['terminal_B_ratio'] for r in rr if r['network_index']==ni]) for ni in range(n_networks)]
        absorbed=[r['absorption_step'] for r in rr if r['absorption_step'] is not None]
        quasi=[r['quasi_step'] for r in rr if r['quasi_step'] is not None]
        out[cond]=dict(n_runs=len(rr),n_networks=n_networks,
            terminal_mean=float(np.mean(nm)),ci95_network_t=ci(nm) if n_networks>1 else None,
            ci95_fixed_network_t=ci([r['terminal_B_ratio'] for r in rr]) if n_networks==1 else None,
            absorption_rate=len(absorbed)/len(rr),quasi_rate=len(quasi)/len(rr),
            absorption_mean_observed=float(np.mean(absorbed)) if absorbed else None,
            quasi_mean_observed=float(np.mean(quasi)) if quasi else None,
            all_B_count=sum(r['absorbed_state']=='all_B' for r in rr),
            all_green_count=sum(r['absorbed_state']=='all_green' for r in rr),
            censored_count=sum(r['absorbed_state']=='not_absorbed' for r in rr))
    return out

def holm(pvalues):
    ranked=sorted(enumerate(pvalues),key=lambda x:x[1]); adjusted=[1.]*len(pvalues); prev=0
    for rank,(idx,p) in enumerate(ranked):
        prev=max(prev,min(1.,p*(len(pvalues)-rank)));adjusted[idx]=prev
    return adjusted

def group_stats(rows,conditions,n_networks,repeats):
    lookup={(r['network_index'],r['replicate'],r['condition']):r['terminal_B_ratio'] for r in rows}
    matrix=[]
    if n_networks>1:
        for ni in range(n_networks):
            matrix.append([np.mean([lookup[(ni,rep,c)] for rep in range(1,repeats+1)]) for c in conditions])
    else:
        for rep in range(1,repeats+1):matrix.append([lookup[(0,rep,c)] for c in conditions])
    mat=np.array(matrix)
    out={'inference_unit':'independent_network_mean' if n_networks>1 else 'within_fixed_network_replicate',
         'n_units':len(mat),'conditions':conditions}
    if len(conditions)>2 and len(mat)>=3:
        try:
            stat,p=friedmanchisquare(*mat.T)
            out['friedman']={'statistic':clean_num(stat),'p':clean_num(p)}
        except ValueError as e:out['friedman']={'error':str(e)}
    tests=[]
    for a in range(len(conditions)):
        for b in range(a+1,len(conditions)):
            dif=mat[:,a]-mat[:,b]
            if np.allclose(dif,0):W,p=0.,1.
            else:
                W,p=wilcoxon(dif)
                if not np.isfinite(p):p=1.
            tests.append(dict(comparison=f'{conditions[a]}_vs_{conditions[b]}',
                difference_mean=float(dif.mean()),ci95_difference_t=ci(dif),W=float(W),p_raw=float(p)))
    for item,adj in zip(tests,holm([x['p_raw'] for x in tests])):item['p_holm']=adj
    out['paired_wilcoxon_holm']=tests
    return out

def set_figure_style():
    plt.rcParams.update({
        'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix','axes.labelsize':13,'xtick.labelsize':10,
        'ytick.labelsize':10,'legend.fontsize':10,'axes.linewidth':1.1,
        'axes.spines.top':False,'axes.spines.right':False,
        'savefig.dpi':400,
    })

def condition_name(c):
    # All labels horizontal; underscore is not a slash or math markup.
    labels={'high':'High degree','random':'Random','low':'Low degree',
            'degree':'Degree-dependent','constant':'Constant',
            'reverse':'Reverse-degree','random_inertia':'Random'}
    if c.startswith('m_'):
        parts=c.split('_')
        if len(parts)==2:return rf'$m={parts[1]}$'
        return rf'$m={parts[1]}$'+'\n'+ ('Accumulated' if parts[2]=='accumulated' else 'Normalized')
    if c.startswith('K_'):return rf'$K={c.split("_")[1]}$'
    if c.startswith('N_'):return rf'$N={c.split("_")[1]}$'
    if c in labels:return labels[c]
    if c in ('high','low','random'):return c.capitalize()
    if c.count('_')>=2:
        pieces=c.split('_')
        position=pieces[0].capitalize()
        mode='Normalized' if 'normalized' in pieces else 'Accumulated'
        inertia='No inertia' if 'none' in pieces else 'Degree inertia'
        return position+'\n'+mode+'\n'+inertia
    return c.replace('_',' ')

def panel_subtitle(section):
    return None  # each exported figure is a standalone plot; no a/b/c labels

def save_figure(fig, path):
    fig.savefig(path,dpi=400,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)

def plot_comparison(path,rows,conditions,label):
    set_figure_style()
    means=[]; lower=[]; upper=[]; nnet=len({r['network_index'] for r in rows})
    for c in conditions:
        vals=np.array([np.mean([r['terminal_B_ratio'] for r in rows if r['network_index']==ni and r['condition']==c]) for ni in range(nnet)])
        mu=float(vals.mean()); lo,hi=ci(vals)
        means.append(mu);lower.append(mu-lo if lo is not None else 0.)
        upper.append(hi-mu if hi is not None else 0.)
    fig,ax=plt.subplots(figsize=(max(7,1.15*len(conditions)),4.8))
    xx=np.arange(len(conditions))
    ax.errorbar(xx,means,yerr=[lower,upper],fmt='o',capsize=4,color='#284e68',lw=1.7,markersize=6)
    ax.set_xticks(xx,[condition_name(c) for c in conditions])
    ax.tick_params(axis='x',labelrotation=0,labelsize=8 if len(conditions)>6 else 10)
    ax.set_ylabel(r'Terminal greenwashing density $f(T)$')
    ax.set_ylim(0,1.05)
    subtitle=panel_subtitle(label)
    fig.subplots_adjust(bottom=.37 if len(conditions)>6 else .26)
    if subtitle:ax.text(.5,-.30,subtitle,transform=ax.transAxes,ha='center',va='top',fontsize=12)
    save_figure(fig,path)

def record_curve(curve_sums,section,condition,ni,path):
    key=(section,condition,ni)
    if key not in curve_sums:curve_sums[key]=[np.zeros_like(path),0]
    curve_sums[key][0]+=path
    curve_sums[key][1]+=1

def plot_evolution(path,section,conditions,curve_sums,nnet,steps):
    set_figure_style()
    fig,ax=plt.subplots(figsize=(8.5,5.0))
    colors=plt.get_cmap('tab10').colors
    x=np.arange(steps+1)
    export=[]
    for j,c in enumerate(conditions):
        mat=np.stack([curve_sums[(section,c,ni)][0]/curve_sums[(section,c,ni)][1] for ni in range(nnet)])
        mean=mat.mean(axis=0)
        if nnet>1:
            half=student_t.ppf(.975,nnet-1)*mat.std(axis=0,ddof=1)/np.sqrt(nnet)
            lo=np.maximum(0,mean-half);hi=np.minimum(1,mean+half)
        else:lo=mean.copy();hi=mean.copy()
        col=colors[j%len(colors)]
        ax.plot(x,mean,lw=2,color=col,label=condition_name(c).replace('\n','  '))
        ax.fill_between(x,lo,hi,color=col,alpha=.14,linewidth=0)
        export.extend([{'condition':c,'t':int(t),'mean':float(mean[t]),'ci_low':float(lo[t]),'ci_high':float(hi[t])} for t in x])
    ax.set(xlim=(0,steps),ylim=(0,1.05),xlabel=r'Time step $t$',ylabel=r'Greenwashing density $f(t)$')
    ax.legend(frameon=False,loc='best',ncol=2 if len(conditions)>5 else 1)
    subtitle=panel_subtitle(section)
    if subtitle:
        fig.subplots_adjust(bottom=.22)
        ax.text(.5,-.22,subtitle,transform=ax.transAxes,ha='center',va='top',fontsize=12)
    else:fig.subplots_adjust(bottom=.16)
    save_figure(fig,path)
    write_rows(path.with_name('evolution_network_mean_ci.csv'),export)

def plot_5e_panels(folder, rows, curve_sums, nnet, steps):
    """Three actual subplots; position captions only here, below each panel."""
    set_figure_style()
    kinds=[('accumulated','degree'),('normalized','degree'),('accumulated','none'),('normalized','none')]
    desc=['Accumulated, inertia','Normalized, inertia','Accumulated, no inertia','Normalized, no inertia']
    colors=plt.get_cmap('tab10').colors
    export=[]
    for target in ('evolution','terminal'):
        fig,axes=plt.subplots(1,3,figsize=(16.2,4.65),sharey=True)
        for panel,(ax,pos) in enumerate(zip(axes,['high','random','low'])):
            conds=[f'{pos}_{mode}_{iner}' for mode,iner in kinds]
            if target=='evolution':
                x=np.arange(steps+1)
                for j,cond in enumerate(conds):
                    mat=np.stack([curve_sums[('5E',cond,ni)][0]/curve_sums[('5E',cond,ni)][1] for ni in range(nnet)])
                    mu=mat.mean(axis=0)
                    half=(student_t.ppf(.975,nnet-1)*mat.std(axis=0,ddof=1)/np.sqrt(nnet)) if nnet>1 else np.zeros_like(mu)
                    ax.plot(x,mu,color=colors[j],lw=1.8,label=desc[j]);ax.fill_between(x,np.clip(mu-half,0,1),np.clip(mu+half,0,1),color=colors[j],alpha=.13)
                    export.extend([{'condition':cond,'t':int(tt),'mean':float(mu[tt]),'ci_low':float(np.clip(mu[tt]-half[tt],0,1)),'ci_high':float(np.clip(mu[tt]+half[tt],0,1))} for tt in x])
                ax.set(xlim=(0,steps),xlabel=r'Time step $t$')
            else:
                av=[];down=[];up=[]
                for cond in conds:
                    z=np.array([np.mean([r['terminal_B_ratio'] for r in rows if r['condition']==cond and r['network_index']==ni]) for ni in range(nnet)])
                    mu=z.mean();lo,hi=ci(z);av.append(mu);down.append(mu-lo if lo is not None else 0);up.append(hi-mu if hi is not None else 0)
                ax.bar(range(4),av,color=colors[:4],edgecolor='white',linewidth=.5)
                ax.errorbar(range(4),av,yerr=[down,up],fmt='none',ecolor='black',capsize=3)
                ax.set_xticks(range(4),['Accum.','Norm.','No inertia','Both'])
                ax.tick_params(axis='x',labelrotation=0,labelsize=8)
            ax.set_ylim(0,1.05)
            ax.text(.5,-.24,f'({chr(97+panel)}) {pos.capitalize()}-degree placement',transform=ax.transAxes,ha='center',va='top',fontsize=11)
        axes[0].set_ylabel(r'Greenwashing density $f(t)$' if target=='evolution' else r'Terminal greenwashing density $f(T)$')
        if target=='evolution':fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.02))
        fig.subplots_adjust(bottom=.32,wspace=.12)
        save_figure(fig,folder/('evolution_curves.png' if target=='evolution' else 'terminal_comparison.png'))
    write_rows(folder/'evolution_network_mean_ci.csv',export)

def main(opt):
    sections={x.strip().upper() for x in opt.sections.split(',')}
    valid={'5A','5B','5C','5D','5E'}
    if not sections<=valid:raise ValueError(f'Unknown sections: {sections-valid}')
    # 5C requires 5A baseline. The user can request only 5C; we auto-run 5A as a dependency.
    if '5C' in sections:sections.add('5A')
    out=Path(opt.output);out.mkdir(parents=True,exist_ok=True)
    results=defaultdict(list); curve_sums={}
    meta={'N_main':100,'m_main':[2,3,5],'K_main':5,'eta':ETA,'inertia_range':RHO,
          'payoffs':PAY,'networks':opt.networks,'repeats_per_network':opt.repeats,
          'steps':opt.steps,'network_seeds':[opt.seed+100*i for i in range(opt.networks)],
          'seed_base':opt.seed,'sections':sorted(sections),
          'cluster_unit':'independent BA network; repeated runs nested within graph',
          '5C_accumulated':'EXACT reuse of 5A baseline records (no rerun)',
          'censoring':'nonabsorbed at steps are right-censored',
          'payoff_probability_note':'triangular RNG is an added probability model, not fuzzy membership distribution'}
    for ni in range(opt.networks):
        netseed=opt.seed+100*ni
        graphs={m:create_graph(100,m,netseed) for m in (2,3,5)} if (sections&{'5A','5B','5C','5D','5E'}) else {}
        for rep in range(1,opt.repeats+1):
            if '5A' in sections:
                for m in (2,3,5):
                    graph,adj,deg,order=graphs[m]
                    inseed=seed(opt.seed,ni,rep,101,m)
                    initial=initial_baseline(100,order,5,inseed)
                    rho=inertia_vector(deg,'degree',seed(opt.seed,ni,rep,199,m))
                    dyn=seed(opt.seed,ni,rep,201,m)
                    res,path=simulate(adj,deg,initial,rho,'accumulated',dyn,opt.steps,save_path=(not opt.no_plot))
                    row=experiment_row('5A',f'm_{m}',ni,netseed,rep,100,m,5,inseed,dyn,initial,res,deg)
                    results['5A'].append(row)
                    if path is not None:record_curve(curve_sums,'5A',f'm_{m}',ni,path)
                    if '5C' in sections:
                        # This is the EXACT baseline value used in 5A.
                        results['5C'].append(dict(row,section='5C',condition=f'm_{m}_accumulated'))
                        if path is not None:record_curve(curve_sums,'5C',f'm_{m}_accumulated',ni,path)
                        dyn_norm=seed(opt.seed,ni,rep,202,m)
                        res_norm,path_norm=simulate(adj,deg,initial,rho,'normalized',dyn_norm,opt.steps,save_path=(not opt.no_plot))
                        if path_norm is not None:record_curve(curve_sums,'5C',f'm_{m}_normalized',ni,path_norm)
                        results['5C'].append(experiment_row('5C',f'm_{m}_normalized',ni,netseed,rep,100,m,5,inseed,dyn_norm,initial,res_norm,deg))
            if '5B' in sections or '5E' in sections:
                graph,adj,deg,order=graphs[3]
                initseed=seed(opt.seed,ni,rep,301)
                initstates={p:initial_position(100,order,30,p,initseed) for p in ('high','random','low')}
                degree_rho=inertia_vector(deg,'degree',0)
                b_reference={}
                if '5B' in sections:
                    for ix,p in enumerate(('high','random','low')):
                        dyn=seed(opt.seed,ni,rep,311,ix)
                        rs,path_b=simulate(adj,deg,initstates[p],degree_rho,'accumulated',dyn,opt.steps,save_path=(not opt.no_plot))
                        if path_b is not None:record_curve(curve_sums,'5B',p,ni,path_b)
                        base_row=experiment_row('5B',p,ni,netseed,rep,100,3,5,initseed,dyn,initstates[p],rs,deg)
                        results['5B'].append(base_row)
                        b_reference[p]=(base_row,path_b)
                if '5E' in sections:
                    for ix,p in enumerate(('high','random','low')):
                        for jx,(paymode,rhokind) in enumerate((('accumulated','degree'),('normalized','degree'),('accumulated','none'),('normalized','none'))):
                            cond=f'{p}_{paymode}_{rhokind}'
                            if jx==0 and p in b_reference:
                                results['5E'].append(dict(b_reference[p][0],section='5E',condition=cond))
                                if b_reference[p][1] is not None:record_curve(curve_sums,'5E',cond,ni,b_reference[p][1])
                            else:
                                dyn=seed(opt.seed,ni,rep,401,ix,jx)
                                rho=inertia_vector(deg,rhokind,0)
                                rs,path_e=simulate(adj,deg,initstates[p],rho,paymode,dyn,opt.steps,save_path=(not opt.no_plot))
                                if path_e is not None:record_curve(curve_sums,'5E',cond,ni,path_e)
                                results['5E'].append(experiment_row('5E',cond,ni,netseed,rep,100,3,5,initseed,dyn,initstates[p],rs,deg))
            if '5D' in sections:
                # K changes REAL initial strategies, rather than top-K descriptive metrics.
                g,adj,deg,order=graphs[3]
                for k in (3,5,10):
                    inseed=seed(opt.seed,ni,rep,501)
                    init=initial_baseline(100,order,k,inseed)
                    dyn=seed(opt.seed,ni,rep,510,k)
                    rs,path_k=simulate(adj,deg,init,inertia_vector(deg,'degree',0),'accumulated',dyn,opt.steps,save_path=(not opt.no_plot))
                    if path_k is not None:record_curve(curve_sums,'5D_K',f'K_{k}',ni,path_k)
                    results['5D_K'].append(experiment_row('5D_K',f'K_{k}',ni,netseed,rep,100,3,k,inseed,dyn,init,rs,deg))
                for n in (100,200,300):
                    if n==100:gg,aa,dd,oo=graphs[3]
                    else:gg,aa,dd,oo=create_graph(n,3,netseed)
                    inseed=seed(opt.seed,ni,rep,601,n)
                    init=initial_baseline(n,oo,5,inseed)
                    dyn=seed(opt.seed,ni,rep,610,n)
                    rs,path_n=simulate(aa,dd,init,inertia_vector(dd,'degree',0),'accumulated',dyn,opt.steps,save_path=(not opt.no_plot))
                    if path_n is not None:record_curve(curve_sums,'5D_N',f'N_{n}',ni,path_n)
                    results['5D_N'].append(experiment_row('5D_N',f'N_{n}',ni,netseed,rep,n,3,5,inseed,dyn,init,rs,dd))
                initseed=seed(opt.seed,ni,rep,701)
                init=initial_baseline(100,order,5,initseed)
                for jx,rhokind in enumerate(('degree','constant','reverse','random')):
                    dyn=seed(opt.seed,ni,rep,710,jx)
                    rho=inertia_vector(deg,rhokind,seed(opt.seed,ni,rep,702))
                    rs,path_i=simulate(adj,deg,init,rho,'accumulated',dyn,opt.steps,save_path=(not opt.no_plot))
                    if path_i is not None:record_curve(curve_sums,'5D_inertia',rhokind,ni,path_i)
                    results['5D_inertia'].append(experiment_row('5D_inertia',rhokind,ni,netseed,rep,100,3,5,initseed,dyn,init,rs,deg))
        print(f'Completed network {ni+1}/{opt.networks} (seed={netseed})',flush=True)
    for section,rows in results.items():
        folder=out/section;folder.mkdir(parents=True,exist_ok=True)
        conditions=sorted({r['condition'] for r in rows})
        write_rows(folder/'replicate_results.csv',rows)
        (folder/'summary.json').write_text(json.dumps(cond_summary(rows,opt.networks),indent=2),encoding='utf-8')
        stats=group_stats(rows,conditions,opt.networks,opt.repeats)
        (folder/'paired_statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
        if not opt.no_plot:
            if section=='5E':
                plot_5e_panels(folder,rows,curve_sums,opt.networks,opt.steps)
            else:
                plot_comparison(folder/'terminal_comparison.png',rows,conditions,section)
                plot_evolution(folder/'evolution_curves.png',section,conditions,curve_sums,opt.networks,opt.steps)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    if '5A' in results and '5C' in results:
        baseline={(r['network_index'],r['replicate'],r['m']):r['terminal_B_ratio'] for r in results['5A']}
        actual={(r['network_index'],r['replicate'],r['m']):r['terminal_B_ratio'] for r in results['5C'] if r['condition'].endswith('_accumulated')}
        assert baseline==actual,'5A versus 5C accumulated baseline mismatch!'
        (out/'baseline_consistency_check.json').write_text(json.dumps({'equal':True,'records':len(baseline),'comment':'Table15 and Table17 baseline values agree exactly by construction'},indent=2),encoding='utf-8')
    print('Output directory:',out.resolve())
    for sec,rows in results.items():print(sec, 'runs:',len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--sections',default='5A,5B,5C,5D,5E')
    p.add_argument('--networks',type=int,default=20)
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',default='experiment5_second_round_results')
    p.add_argument('--no-plot',action='store_true')
    o=p.parse_args()
    if min(o.networks,o.repeats,o.steps)<1:p.error('All run counts must be >=1')
    main(o)
