#!/usr/bin/env python3
"""Reproduce the complete-flight MPPI experiment and committed teaching figures.

No optimization runs in the ordinary book build. Use --quick for development;
the default uses three independent gust realizations with matched solver seeds.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import re
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from aircraft_mppi import (Aircraft, FlightConfig, MeanWind, audit, make_actual_gust,
    simulate, optimize, conditional_gusts, evaluate_tapes)

ART = ROOT/'artifacts/aircraft_mppi'
STATIC = ROOT/'_static/aircraft_mppi'
COLORS = {'full_trip':'#555555','frozen':'#D55E00','mean':'#0072B2','stochastic':'#009E73'}
LABELS = {'full_trip':'Mean-wind plan','frozen':'Frozen plan + gusts','mean':'Mean-wind replanning','stochastic':'Stochastic replanning'}
STYLES = {'full_trip':':','frozen':'--','mean':'-','stochastic':'-.'}


def serial(run, aircraft):
    states = run['states']
    return {'mode':run['mode'],'states':states.tolist(),'controls':run['controls'].tolist(),
            'gusts':run['gusts'].tolist(),'dts':run['dts'].tolist(),
            'lonlat':aircraft.lonlat(states[:,:2]).tolist(),'replans':run['replans'],
            'failure':run.get('failure')}


def convergence(aircraft, cfg):
    """Vary candidate count and scenario count separately at the first decision.

    Score the selected plan on 128 fresh common scenarios, unseen by every
    optimizer. Multiple planning seeds expose sampling variability.
    """
    test_wind = conditional_gusts(np.zeros(2), 70, 128, cfg, np.random.default_rng(70191))
    settings = [(k, cfg.scenarios) for k in (16,32,64,128)]
    settings += [(cfg.samples,l) for l in (1,3,12,24)]
    records=[]
    for k,l in settings:
        for seed in range(4):
            setting = replace(cfg,samples=k,scenarios=l)
            tape,dt,_,diag=optimize(aircraft,aircraft.initial,np.zeros(2),setting,0,cfg.duration,True,np.random.default_rng(seed+471))
            costs,checks=evaluate_tapes(aircraft,aircraft.initial,tape[None],dt[None],test_wind,cfg)
            records.append({'candidates':k,'scenarios':l,'planning_seed':seed+471,
                'held_out_expected_cost':float(costs.mean()),'held_out_cost_se':float(costs.std(ddof=1)/np.sqrt(128)),
                'held_out_mean_arrival_error_m':float(checks['horizontal'].mean()),
                'ess':diag['ess'],'runtime_s':diag['runtime_s'],'accepted':diag['accepted']})
    return {'evaluation_scenarios':128,'evaluation_seed':70191,'records':records}


def save_figure(fig,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(STATIC/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def figures(runs, metrics, aircraft):
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,
        'axes.titlesize':10,'legend.fontsize':8,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.6})
    fig,axs=plt.subplots(2,2,figsize=(8,5.8),constrained_layout=True)
    for mode,run in runs.items():
        x=run['states']; tm=x[:,4]/60
        relative=x[:,:2]-aircraft.start_xy
        along=relative@aircraft.direction/1000
        cross=relative@np.array([-aircraft.direction[1],aircraft.direction[0]])/1000
        kw=dict(color=COLORS[mode],ls=STYLES[mode],label=LABELS[mode])
        axs[0,0].plot(along,x[:,2]/1000,**kw)
        axs[0,1].plot(along,cross,**kw)
        axs[1,0].plot(tm,aircraft.mass0-x[:,3],**kw)
        error=np.linalg.norm(x[:,:2]-aircraft.end_xy,axis=1)/1000
        axs[1,1].plot(tm,error,**kw)
    axs[0,0].set(xlabel='Along-route distance (km)',ylabel='Altitude (km)',title='Climb, cruise, and descent')
    axs[0,1].set(xlabel='Along-route distance (km)',ylabel='Cross-track displacement (km)',title='Gusts displace the frozen plan')
    axs[1,0].set(xlabel='Elapsed time (min)',ylabel='Fuel burned (kg)',title='Fuel and execution time')
    axs[1,1].set(xlabel='Elapsed time (min)',ylabel='Distance to destination (km)',title='Final approach',xlim=(50,62),ylim=(0,60))
    axs[1,1].axhline(1,color='.4',lw=.8,ls=':')
    axs[0,0].legend(loc='lower center',ncol=1)
    save_figure(fig,'flight_comparison')

    fig,axs=plt.subplots(1,3,figsize=(8,2.9),constrained_layout=True)
    modes=['frozen','mean','stochastic']
    labels=['Frozen','Mean\nreplanning','Stochastic\nreplanning']
    for ax,key,title,ylabel in zip(axs,['fuel_kg','horizontal_error_m','max_normalized_violation'],['Fuel consumption','Arrival accuracy','Constraint audit'],['Fuel (kg)','Horizontal error (m)','Maximum normalized violation']):
        for i,mode in enumerate(modes):
            vals=np.array([m[key] for m in metrics if m['mode']==mode])
            ax.errorbar(i,vals.mean(),yerr=vals.std(ddof=1) if len(vals)>1 else 0,color=COLORS[mode],fmt='o',capsize=4)
            ax.scatter(i+np.linspace(-.07,.07,len(vals)),vals,color=COLORS[mode],s=14,alpha=.45)
        ax.set(xticks=range(3),xticklabels=labels,ylabel=ylabel,title=title)
        ax.tick_params(axis='x',labelsize=8)
    axs[1].axhline(1000,color='.5',ls=':',lw=1)
    axs[1].text(.02,.95,'Arrival tolerance: 1000 m',transform=axs[1].transAxes,va='top',fontsize=8)
    axs[2].set_ylim(-.01,max(.05,axs[2].get_ylim()[1]))
    save_figure(fig,'metrics')

    stochastic=runs['stochastic']; idx=min(25,len(stochastic['replans'])-1)
    record=stochastic['replans'][idx]
    endpoints=np.array(record['predicted_terminal_xy'])-aircraft.end_xy
    fig,axs=plt.subplots(1,2,figsize=(8,2.9),constrained_layout=True)
    for mode in ['mean','stochastic']:
        records=runs[mode]['replans']
        t=np.array([r['time'] for r in records])/60
        ranges=np.array([r['predicted_horizontal_p10_p90'] for r in records])
        axs[0].plot(t,ranges.mean(axis=1)/1000,color=COLORS[mode],ls=STYLES[mode],label=LABELS[mode])
        axs[0].fill_between(t,ranges[:,0]/1000,ranges[:,1]/1000,color=COLORS[mode],alpha=.18)
    axs[0].set(xlabel='Replanning time (min)',ylabel='Predicted terminal error (km)',title='Conditional wind rollouts')
    axs[0].legend()
    axs[1].scatter(endpoints[:,0]/1000,endpoints[:,1]/1000,color=COLORS['stochastic'],s=25,label='Scenario endpoint')
    axs[1].scatter([0],[0],color='black',marker='+',s=70,label='Destination')
    circle=plt.Circle((0,0),1,fill=False,color='.5',ls=':');axs[1].add_patch(circle)
    axs[1].set(aspect='equal',xlabel='East error (km)',ylabel='North error (km)',title=f'Forecast at {record["time"]/60:.0f} min')
    axs[1].legend(fontsize=7)
    save_figure(fig,'wind_uncertainty')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    cfg=FlightConfig(samples=12,scenarios=3,iterations=1) if args.quick else FlightConfig()
    seeds=[cfg.seed+1] if args.quick else [cfg.seed+1,cfg.seed+2,cfg.seed+3]
    ART.mkdir(exist_ok=True,parents=True);STATIC.mkdir(exist_ok=True,parents=True)
    aircraft=Aircraft(MeanWind()); metrics=[];representative={};start=time.perf_counter()
    for seed in seeds:
        actual=make_actual_gust(cfg,seed)
        for mode in ('full_trip','frozen','mean','stochastic'):
            if mode=='full_trip' and seed!=seeds[0]:continue
            run=simulate(aircraft,cfg,mode,actual)
            report=audit(aircraft,run,cfg);report['gust_seed']=seed;metrics.append(report)
            print(json.dumps(report),flush=True)
            if seed==seeds[0]:representative[mode]=run
    convergence_results = None if args.quick else convergence(aircraft,cfg)
    document={'config':asdict(cfg),'gust_seeds':seeds,'metrics':metrics,
        'convergence':convergence_results,
        'runtime_s':time.perf_counter()-start,
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['code/aircraft_mppi.py','code/mppi_control.py','data/aircraft/era5_wind.npz']},
        'wind':'ERA5 2023-06-01 12:00 UTC; pressure/lat/lon interpolation, held at 200/925 hPa boundaries',
        'uncertainty':'OU gusts: tau=300 s, stationary standard deviations east=5 m/s, north=4 m/s; independent actual-future and optimizer RNGs',
        'aggregation':'scenario costs averaged before weighting; reference/proposal Gaussian correction applied once per candidate'}
    (ART/'metrics.json').write_text(json.dumps(document,indent=2)+'\n')
    for mode,run in representative.items():
        (ART/f'{mode}.json').write_text(json.dumps(serial(run,aircraft),separators=(',',':'))+'\n')
    replay={'origin':aircraft.origin.tolist(),'destination':aircraft.destination_lonlat.tolist(),
        'labels':LABELS,'runs':{k:serial(v,aircraft) for k,v in representative.items()},'metrics':metrics[:4]}
    (ROOT/'interactive/aircraft-mppi-data.json').write_text(json.dumps(replay,separators=(',',':'))+'\n')
    replay_file=ROOT/'interactive/aircraft-mppi.html'
    if replay_file.exists():
        html=replay_file.read_text()
        embedded=json.dumps(replay,separators=(',',':'))
        html=re.sub(r'(<script id="data" type="application/json">).*?(</script>)',
                    lambda m:m.group(1)+embedded+m.group(2),html,flags=re.DOTALL)
        replay_file.write_text(html)
    figures(representative,metrics,aircraft)
    lines=['| Controller | Fuel (kg) | Arrival error (m) | Feasible arrivals | Largest constraint violation |','|---|---:|---:|---:|---:|']
    for mode in ('full_trip','frozen','mean','stochastic'):
        rows=[m for m in metrics if m['mode']==mode]
        fuel=np.array([m['fuel_kg'] for m in rows]);arrival=np.array([m['horizontal_error_m'] for m in rows])
        lines.append(f'| {LABELS[mode]} | {fuel.mean():.1f} | {arrival.mean():.1f} | {sum(m["arrival_pass"] and m["max_normalized_violation"]<1e-7 for m in rows)}/{len(rows)} | {max(m["max_normalized_violation"] for m in rows):.3g} |')
    lines += ['',f'The repeated runs use {len(seeds)} independent gust seeds. The figures show mean and sample standard deviation across those seeds, with individual runs also plotted. The mean-wind plan is one deterministic reference run. Prediction uses {cfg.samples} candidates, {cfg.scenarios} shared wind scenarios, and {cfg.iterations} updates per decision. Arrival tolerances are 1000 m horizontally and 30 m vertically. Constraint residuals and replay differences are available in the downloadable metrics.', '']
    (ART/'results.md').write_text('\n'.join(lines))
    print('artifacts complete',flush=True)

if __name__=='__main__':main()
