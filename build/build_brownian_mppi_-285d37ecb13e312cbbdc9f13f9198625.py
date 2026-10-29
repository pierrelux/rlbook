#!/usr/bin/env python3
"""Regenerate the Brownian MPPI experiments locally; ordinary builds load outputs.

Run: MPLCONFIGDIR=/private/tmp/mppi-mpl .venv/bin/python scripts/build_brownian_mppi_artifacts.py
Only NumPy, SciPy and Matplotlib are needed. No downloads or credentials.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import re
from pathlib import Path
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from brownian_mppi import (PassageProblem,scalar_benchmark,solve_reference,
    paired_trials,sample_path_integral,analytic_slit,narrow_slit_control,sample_slit_control)

BLUE,ORANGE,GREEN="#0072B2","#D55E00","#009E73"
LABELS={"path_integral":"Sampled PI","mean_dynamics":"Mean dynamics","reference":"Numerical reference"}
COLORS={"path_integral":BLUE,"mean_dynamics":ORANGE,"reference":GREEN}


def mean_ci(values):
    values=np.asarray(values,dtype=float)
    return {"mean":float(values.mean()),"ci95_halfwidth":float(1.96*values.std(ddof=1)/np.sqrt(values.size))}


def wilson(values):
    values=np.asarray(values,dtype=float)
    n=values.size; p=values.mean();z=1.96;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {"mean":float(p),"ci95":[float(center-half),float(center+half)]}


def configure_style():
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],
      "font.size":9,"axes.labelsize":9,"axes.titlesize":9,"xtick.labelsize":8,"ytick.labelsize":8,
      "legend.fontsize":8,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7,
      "lines.linewidth":1.3,"figure.dpi":150,"savefig.dpi":150,"svg.fonttype":"none",
      "savefig.facecolor":"white","axes.prop_cycle":matplotlib.cycler(color=[BLUE,ORANGE,GREEN]),
      "pdf.fonttype":42})


def save(fig,name):
    directory=ROOT/"_static"/"brownian_mppi"
    directory.mkdir(parents=True,exist_ok=True)
    for suffix in ("svg","pdf","png"):
        fig.savefig(directory/f"{name}.{suffix}",bbox_inches="tight",pad_inches=.04)
    plt.close(fig)


def geometry(ax,p,limits=(-2.1,.8)):
    ax.axvspan(p.enter,p.leave,color=".9",zorder=-5)
    for lo,hi in p.intervals:
        ax.fill_between([p.enter,p.leave],lo,hi,color="white",zorder=-4)
        ax.plot([p.enter,p.leave],[lo,lo],color=".5",lw=.6)
        ax.plot([p.enter,p.leave],[hi,hi],color=".5",lw=.6)
    ax.set(xlim=(0,p.final_time),ylim=limits,xlabel="Time",ylabel="Position $x$")


def figures(cases,summary,convergence,snapshot,slit_validation):
    configure_style()
    fig,axs=plt.subplots(1,2,figsize=(6.3,2.7),sharex=True,sharey=True,layout="constrained")
    for ax,(key,data) in zip(axs,cases.items()):
        p=data["problem"];geometry(ax,p)
        tt=np.arange(p.steps+1)*p.dt
        for j in range(8):
            ax.plot(tt,data["trials"]["path_integral"]["states"][j],color=BLUE,alpha=.7,lw=.8)
        for j in range(3):
            ax.plot(tt,data["trials"]["mean_dynamics"]["states"][j],color=ORANGE,ls="--",lw=.9,alpha=.85)
        ax.set_title(rf"$\nu={p.nu:g},\ \lambda={p.temperature:g}$")
    axs[1].set_ylabel("")
    axs[0].plot([],[],color=BLUE,label="Sampled PI (8 paths)")
    axs[0].plot([],[],color=ORANGE,ls="--",label="Mean dynamics (3 paired paths)")
    axs[0].legend(loc="lower left",fontsize=7)
    save(fig,"brownian-passages")

    fig,axs=plt.subplots(1,2,figsize=(6.3,2.6),layout="constrained")
    xx=np.linspace(-.7,.7,301)
    for T,color,style in [(2,BLUE,"-"),(1,GREEN,"--"),(.5,ORANGE,"-.")]:
        axs[0].plot(xx,narrow_slit_control(xx,T,1),color=color,ls=style,label=f"Time to slit = {T:g}")
    axs[0].axhline(0,color=".65",lw=.6)
    axs[0].axvline(0,color=".65",lw=.6)
    axs[0].set(xlabel="Position $x$",ylabel="Control $u$",title="Commitment depends on time remaining")
    axs[0].legend(loc="upper left",fontsize=7)
    points=slit_validation["points"]
    ts=np.array([d["remaining"] for d in points]); exact=np.array([d["exact_control"] for d in points])
    estimates=np.array([d["estimate"]["mean"] for d in points]);errors=np.array([d["estimate"]["ci95_halfwidth"] for d in points])
    axs[1].plot(ts,exact,color=".2",label="Exact finite-width slit")
    axs[1].errorbar(ts,estimates,yerr=errors,color=BLUE,fmt="o",ms=3,capsize=2,label="Sampled PI, 95% CI")
    axs[1].axhline(0,color=".65",lw=.6)
    axs[1].set(xlabel="Time remaining to slit",ylabel="First control at $x=0.2$",title="Forward sampling checks the formula")
    axs[1].legend(fontsize=7)
    save(fig,"brownian-feedback")

    fig,axs=plt.subplots(1,2,figsize=(6.3,2.7),layout="constrained")
    keys=list(cases)
    for m,method in enumerate(LABELS):
        vals=[summary[key][method]["cost"]["mean"] for key in keys]
        cis=[summary[key][method]["cost"]["ci95_halfwidth"] for key in keys]
        xpos=np.arange(len(keys))+(m-1)*.13
        axs[0].errorbar(xpos,vals,yerr=cis,fmt=["o","s","^"][m],color=COLORS[method],capsize=2,ms=4,label=LABELS[method])
    axs[0].set(xticks=np.arange(len(keys)),xticklabels=[str(cases[k]["problem"].nu) for k in keys],xlabel="Physical noise variance $\nu$",ylabel="Expected total cost",ylim=(0,None))
    axs[0].legend(fontsize=7,loc="upper left")
    high=summary[keys[-1]]
    for m,method in enumerate(LABELS):
        for j,metric in enumerate(("failure","wide")):
            stat=high[method][metric];p=stat["mean"];low,hi=stat["ci95"]
            axs[1].errorbar(j+(m-1)*.13,p,yerr=[[p-low],[hi-p]],fmt=["o","s","^"][m],color=COLORS[method],capsize=2,ms=4)
    axs[1].set(xticks=[0,1],xticklabels=["Any outside visit","In wide passage"],ylabel="Fraction of trials",ylim=(-.05,1.05),title=r"High noise ($\nu=0.3$)")
    save(fig,"brownian-outcomes")

    fig,axs=plt.subplots(1,2,figsize=(6.3,2.6),layout="constrained")
    p=cases["high_noise"]["problem"];geometry(axs[0],p,(-2.5,1.2))
    tt=np.arange(snapshot["paths"].shape[1])*p.dt
    order=np.argsort(snapshot["weights"])
    chosen=np.r_[order[-14:],order[np.linspace(0,len(order)-15,14,dtype=int)]]
    for index in chosen:
        strong=index in order[-14:]
        axs[0].plot(tt,snapshot["paths"][index],color=BLUE if strong else ".7",lw=.9 if strong else .5,alpha=.85 if strong else .4)
    axs[0].set_title("Future paths used for one update")
    axs[0].set_xlim(0,p.leave)
    ranked=snapshot["weights"][order[::-1]]
    axs[1].plot(np.arange(1,len(ranked)+1),np.cumsum(ranked),color=BLUE)
    axs[1].set(xlabel="Candidates, in decreasing weight",ylabel="Cumulative normalized weight",xscale="log",ylim=(0,1.03),title=f"ESS = {snapshot['ess']:.0f} of {len(ranked):,}")
    axs[1].axhline(.9,color=".65",ls="--",lw=.7)
    save(fig,"brownian-samples")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials",type=int,default=128)
    parser.add_argument("--samples",type=int,default=4096)
    args=parser.parse_args()
    if args.trials<2 or args.samples<2:parser.error("at least two trials and samples required")
    start=time.perf_counter();cases={};summary={};npz={};replay={}
    for name,nu in (("low_noise",.015),("high_noise",.3)):
        p=PassageProblem(nu=nu)
        result=paired_trials(p,trials=args.trials,samples=args.samples,seed=8102)
        cases[name]={"problem":p,"trials":result}
        summary[name]={}
        replay[name]={"problem":asdict(p),"methods":{}}
        for method,data in result.items():
            summary[name][method]={**{k:mean_ci(data[k]) for k in ("cost","effort","penalty")},
                **{k:wilson(data[k]) for k in ("failure","wide")},"update_failures":data["update_failures"]}
            if method=="path_integral":
                summary[name][method]["median_ess"]=float(np.median(data["ess"][:,:round(p.leave/p.dt)]))
            for key in ("states","controls","cost","effort","penalty","failure","wide","ess"):
                npz[f"{name}_{method}_{key}"]=data[key]
            replay[name]["methods"][method]={key:np.round(data[key][:8],6).tolist() for key in ("states","controls")}
        summary[name]["paired_cost_gap_pi_minus_mean"]=mean_ci(result["path_integral"]["cost"]-result["mean_dynamics"]["cost"])
        print(name,json.dumps(summary[name]),flush=True)

    convergence={"sampling":[],"time_step":[],"space_and_domain":[]}
    p=PassageProblem(nu=.3);ref=solve_reference(p)
    for K in (256,1024,4096,16384):
        estimates=[sample_path_integral(p.x0,0,p,np.random.default_rng(seed),samples=K)["action"] for seed in range(32)]
        convergence["sampling"].append({"samples":K,"seeds":list(range(32)),"reference_control":float(ref.action(p.x0,0)),
            "estimate":mean_ci(estimates),"rmse":float(np.sqrt(np.mean((np.array(estimates)-ref.action(p.x0,0))**2)))})
    for nu in (.015,.15,.3):
        for h in (.05,.025,.0125,.00625):
            pp=PassageProblem(nu=nu,dt=h);r=solve_reference(pp,dx=.0025)
            convergence["time_step"].append({"nu":nu,"dt":h,"dx":.0025,"control":float(r.action(pp.x0,0)),"value":float(np.interp(pp.x0,r.grid,r.value[0]))})
    for dx,extent in ((.01,6),(.005,6),(.0025,6),(.005,8)):
        r=solve_reference(p,dx=dx,extent=extent)
        convergence["space_and_domain"].append({"dx":dx,"extent":extent,"control":float(r.action(p.x0,0))})
    slit={"samples":4096,"replicates":32,"nu":1,"x":.2,"penalty":20,"points":[]}
    for remaining in (.25,.5,.75,1,1.5,2):
        value,action=analytic_slit(.2,2-remaining,penalty=20)
        draws=[sample_slit_control(.2,2-remaining,np.random.default_rng(seed),samples=4096) for seed in range(32)]
        slit["points"].append({"remaining":remaining,"exact_control":float(action),"exact_value":float(value),
             "estimate":mean_ci([d["action"] for d in draws]),"value_estimate":mean_ci([d["value"] for d in draws])})
    snapshot=sample_path_integral(p.x0,0,p,np.random.default_rng(2026),samples=args.samples,keep_paths=True)
    figures(cases,summary,convergence,snapshot,slit)
    output={"schema_version":1,"source":"Kappen (2005), sections 6.1 and 6.2; finite-penalty teaching adaptation",
        "source_url":"https://arxiv.org/abs/physics/0505066","plant_seed":8102,"planner_seed":1008102,
        "trials":args.trials,"samples_per_update":args.samples,"settings":{k:asdict(v["problem"]) for k,v in cases.items()},
        "metrics":summary,"scalar":scalar_benchmark(),"convergence":convergence,"slit_validation":slit,
        "runtime_seconds":round(time.perf_counter()-start,3),
        "notes":["Intervals include endpoints; failure means any sampled outside visit for enter<t<=leave.",
            "Wide-passage frequency is occupancy at t=1.5, not conditioning on successful survival.",
            "Cost uncertainty is normal-approximation 95% CI over independent trials; proportions use Wilson intervals.",
            "Each trial shares plant increments across controllers. Planner RNG is independent and never sees future plant increments.",
            "The sampled first-increment estimator and convolution reference use the same time discretization.",
            "Between-step crossings are not counted; the time-step sweep documents discretization sensitivity.",
            "Guided trajectories determine only proposal densities; executed controls are weighted first-increment means."]}
    out=ROOT/"artifacts"/"brownian_mppi";out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(output,indent=2)+"\n")
    np.savez_compressed(out/"trials.npz",**npz)
    replay["summary"]=output
    (out/"replay.json").write_text(json.dumps(replay,separators=(",",":"))+"\n")
    html=ROOT/"interactive"/"brownian-mppi.html"
    template=html.read_text()
    embedded=json.dumps(replay,separators=(",",":"))
    template=re.sub(r'(<script id="data" type="application/json">).*?(</script>)',
                    lambda m:m.group(1)+embedded+m.group(2),template,flags=re.DOTALL)
    html.write_text(template)
    (out/"README.md").write_text("# Brownian MPPI artifacts\n\nRegenerate with `MPLCONFIGDIR=/private/tmp/mppi-mpl .venv/bin/python scripts/build_brownian_mppi_artifacts.py`.\n\n`results.json` records all settings, seeds, definitions, numerical checks, confidence intervals, and runtime. `trials.npz` preserves every trajectory and trial metric. `replay.json` contains eight paths per method for the offline replay. Static SVG, PDF, and PNG figures are in `_static/brownian_mppi/`. The experiment needs NumPy, SciPy, and Matplotlib and makes no network requests.\n\nThese are finite-penalty, finite-width teaching adaptations of Kappen's slit and finite-thickness passage constructions, not a reproduction of his figure parameters. Samples represent physical Brownian uncertainty; the proposal correction accounts for guided sampling. The closed-loop experiment uses actual forward samples at every nonlinear control update and an analytically integrated terminal tail. The convolution solution is an independent numerical reference.\n")
    print(f"Finished in {output['runtime_seconds']:.1f} seconds",flush=True)


if __name__=="__main__":main()
