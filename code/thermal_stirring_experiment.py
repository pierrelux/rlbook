"""Explicit offline experiments and replay artifacts for controlled stirring."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import itertools
import json
import time
import jax
import jax.numpy as jnp
import numpy as np

from thermal_stirring import (CONFIG, PLAN, StirringController, stirring_step, flow_basis,
    face_velocity, warm_state, mixing_components, mixing_error, physical_checks,
    recorded_rollout, minmod)


def binary(path, values):
    a=np.asarray(values,dtype='<f4')
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite movie')
    a.tofile(path)
    return {'file':path.name,'shape':list(a.shape),'dtype':'float32-le',
            'channels':['temperature_K','unreacted_fraction'],
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def json_save(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def source_signature():
    here=Path(__file__).parent
    return hashlib.sha256(b''.join((here/p).read_bytes() for p in
        ('thermal_stirring.py','thermal_stirring_metal.py','thermal_stirring_kernel.py','mppi_control.py','thermal_reactor.py'))).hexdigest()


def metrics(fields, tape, config=CONFIG, planner=PLAN):
    ok,peak,conversion=map(np.asarray,physical_checks(jnp.asarray(fields),config))
    et,ec=map(np.asarray,mixing_components(jnp.asarray(fields),config))
    weights=flow_basis(config)[0][0,:,-1].astype(float);weights/=weights.sum()
    outlet=fields[:,:,:,-1]
    mean=np.sum(outlet*weights,axis=-1)
    variance=np.sum((outlet-mean[...,None])**2*weights,axis=-1)
    duration=(len(fields)-1)*planner.frame_dt
    integrated=float(np.trapezoid(et+ec,dx=planner.frame_dt))
    effort=np.sum(tape.astype(float)**2,axis=-1)
    slew=np.sum(np.diff(np.concatenate([np.zeros((1,3)),tape]),axis=0)**2,axis=-1)
    return {'peak_temperature_K':peak.tolist(),'outlet_conversion':conversion.tolist(),
            'temperature_mixing_error':et.tolist(),'conversion_mixing_error':ec.tolist(),
            'outlet_temperature_std_K':np.sqrt(variance[:,0]).tolist(),
            'outlet_conversion_std':np.sqrt(variance[:,1]).tolist(),
            'summary':{'duration_s':duration,'integrated_mixing_error_s':integrated,
                'mean_mixing_error':integrated/duration,
                'running_objective':integrated/duration+planner.effort_weight*float(effort.mean())
                                   +planner.slew_weight*float(slew.mean()),
                'terminal_mixing_error':float(et[-1]+ec[-1]),
                'downstream_temperature_rms_K':float(np.sqrt(np.trapezoid(et,dx=planner.frame_dt)/duration)*20),
                'downstream_conversion_rms':float(np.sqrt(np.trapezoid(ec,dx=planner.frame_dt)/duration)*.1),
                'mean_outlet_temperature_std_K':float(np.trapezoid(np.sqrt(variance[:,0]),dx=planner.frame_dt)/duration),
                'effort_s':float(effort.sum()*planner.control_dt),'slew_sum':float(slew.sum()),
                'maximum_temperature_K':float(peak.max()),'minimum_outlet_conversion':float(conversion.min()),
                'temperature_violation_s':float(np.sum(peak[:-1]>config.temperature_limit)*planner.frame_dt),
                'conversion_violation_s':float(np.sum(conversion[:-1]<config.conversion_target)*planner.frame_dt)}}


def constant_search(initial, config=CONFIG, planner=PLAN):
    vectors=np.array([a for a in itertools.product(np.linspace(-1,1,9),repeat=3) if np.dot(a,a)<=1+1e-9],np.float32)
    count=int(round(planner.duration/planner.control_dt))
    tapes=np.broadcast_to(vectors[:,None,:],(len(vectors),count,3)).copy()
    ctl=StirringController(config,planner)
    started=time.perf_counter();masked,raw,ok=ctl.score(initial,tapes,0.)
    rows=[{'control':a.tolist(),'objective':float(c),'coarse_feasible':bool(k)} for a,c,k in zip(vectors,raw,ok)]
    chosen=None
    for i in np.argsort(masked):
        if not np.isfinite(masked[i]):break
        valid,check=ctl.validate(initial,tapes[i],0.)
        rows[i]['fine_validation']=check
        if valid:
            chosen=int(i);break
    if chosen is None:raise RuntimeError('No feasible constant reference on the declared grid')
    return tapes[chosen],{'grid_spacing':.25,'candidate_count':len(vectors),'selected_index':chosen,
                         'controls':vectors[chosen].tolist(),'runtime_s':time.perf_counter()-started,'rows':rows,
                         'selection_note':'Lowest full-record mean stage cost plus terminal mixing error among finely validated grid vectors; not a continuous global optimum.'}


def controlled_run(initial, cache, config=CONFIG, planner=PLAN, capture=False):
    signature=source_signature()+json.dumps(asdict(planner),sort_keys=True)
    path=cache/f'seed-{planner.seed}.npz'
    ctl=StirringController(config,planner)
    frames=[initial];actions=[];decisions=[]
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            if str(saved['signature'])==signature:
                frames=list(saved['fields']);actions=list(saved['controls'])
                decisions=json.loads(str(saved['decisions']))
                ctl.incumbent=saved['incumbent'];ctl.previous=saved['previous']
                ctl.rng.bit_generator.state=json.loads(str(saved['rng']))
                print(f'Resumed seed {planner.seed} at {len(actions)*planner.control_dt}s',flush=True)
    steps=int(round(planner.duration/planner.control_dt))
    for k in range(len(actions),steps):
        t=k*planner.control_dt
        action,record,snapshot=ctl.plan(frames[-1],t,capture and t in (16.,40.,72.))
        decisions.append(record)
        if action is None:
            json_save(cache/f'failure-{planner.seed}.json',record)
            raise RuntimeError(f'No feasible continuation at {t}s, seed{planner.seed}')
        if snapshot:
            encoded={key:np.asarray(value) for key,value in snapshot.items() if key!='iteration'}
            encoded['iteration']=json.dumps(snapshot['iteration'])
            np.savez_compressed(cache/f'snapshot-{int(t)}.npz',**encoded)
        future=np.asarray(recorded_rollout(jnp.asarray(frames[-1]),jnp.asarray(action[None]),t,config,planner))
        frames.extend(future[1:]);actions.append(action)
        temporary=path.with_suffix('.pending.npz')
        np.savez_compressed(temporary,signature=signature,fields=np.asarray(frames),controls=np.asarray(actions),
            decisions=json.dumps(decisions),incumbent=ctl.incumbent,previous=ctl.previous,
            rng=json.dumps(ctl.rng.bit_generator.state))
        temporary.replace(path)
        print(json.dumps({'seed':planner.seed,'time_s':t,'accepted':record['accepted'],
            'objective':record['final_objective'],'ess':[round(x['ess'],1) for x in record['iterations']],
            'action':action.tolist(),'runtime_s':round(record['runtime_s'],2)}),flush=True)
    return np.asarray(frames),np.asarray(actions),decisions


def advection_benchmark():
    """One diagonal circuit of a smooth periodic scalar, with diffusion zero."""
    results=[]
    for n in (32,64,128):
        x=(jnp.arange(n)+.5)/n
        xx,yy=jnp.meshgrid(x,x)
        initial=.5+.2*jnp.sin(2*jnp.pi*xx)*jnp.cos(2*jnp.pi*yy)
        for order in (1,2):
            @jax.jit
            def evolve(f):
                dt=1/(8*n)
                def rhs(a):
                    flux=[]
                    for axis in (-1,-2):
                        d=a-jnp.roll(a,1,axis)
                        s=minmod(d,jnp.roll(a,-1,axis)-a) if order==2 else jnp.zeros_like(a)
                        q=a+.5*s
                        flux.append(-(q-jnp.roll(q,1,axis))*n)
                    return flux[0]+flux[1]
                def advance(a,_):
                    first=a+dt*rhs(a)
                    second=.5*a+.5*(first+dt*rhs(first))
                    return second,None
                return jax.lax.scan(advance,f,None,length=8*n)[0]
            final=np.asarray(evolve(initial));original=np.asarray(initial)
            results.append({'mesh':n,'order':order,'rmse':float(np.sqrt(np.mean((final-original)**2))),
                'mass_error':float(abs(final.mean()-original.mean())),
                'variance_retained':float(final.var()/original.var()),
                'minimum':float(final.min()),'maximum':float(final.max())})
    return {'problem':'Periodic unit square, velocity(1,1), duration1, zero diffusion; exact field returns to initial state.',
            'results':results,'note':'Variance lost in this benchmark is numerical smoothing, not physical mixing.'}


def refinement(initial, values, tape, config=CONFIG, planner=PLAN):
    fine_t=np.asarray(recorded_rollout(jnp.asarray(initial),jnp.asarray(tape),0.,replace(config,max_dt=config.max_dt/2),planner))
    finer=replace(config,nx=config.nx*2,ny=config.ny*2,max_dt=config.max_dt/4)
    finer_initial=np.repeat(np.repeat(initial,2,-1),2,-2)
    fine_x=np.asarray(recorded_rollout(jnp.asarray(finer_initial),jnp.asarray(tape),0.,finer,planner))
    restricted=fine_x.reshape(len(fine_x),2,config.ny,2,config.nx,2).mean(axis=(3,5))
    original=metrics(values,tape,config,planner)['summary']
    def compare(mapped,full,cfg):
        result=metrics(full,tape,cfg,planner)['summary']
        return {'temperature_rmse_K':float(np.sqrt(np.mean((mapped[:,0]-values[:,0])**2))),
            'mean_mixing_error':result['mean_mixing_error'],
            'mixing_error_relative_change':result['mean_mixing_error']/max(original['mean_mixing_error'],1e-12)-1,
            'maximum_temperature_K':result['maximum_temperature_K'],
            'minimum_outlet_conversion':result['minimum_outlet_conversion'],
            'temperature_violation_s':result['temperature_violation_s'],'conversion_violation_s':result['conversion_violation_s']}
    return {'time_refinement':compare(fine_t,fine_t,replace(config,max_dt=config.max_dt/2)),
            'space_refinement':compare(restricted,fine_x,finer),'fine_mesh':[finer.ny,finer.nx],
            'initial_transfer':'Conservative piecewise-constant prolongation; controls and forcing unchanged.'},fine_x


def generate_stirring(root,backend='jax'):
    root=Path(root);assets=root/'interactive/thermal-stirring-data';reports=root/'artifacts/thermal_stirring'
    cache=root/'outputs/thermal_stirring';assets.mkdir(parents=True,exist_ok=True);reports.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    cfg,p=CONFIG,replace(PLAN,backend=backend);initial=warm_state(cfg)
    stationary=np.asarray(stirring_step(initial,jnp.zeros(3),0.,2.,cfg,False))
    if np.max(np.abs(stationary[0]-initial[0]))>.01:raise RuntimeError('Unsettled warm state')
    print('Warm jacketed state ready; searching constant controls',flush=True)
    signature=source_signature()+json.dumps(asdict(p),sort_keys=True)
    refpath=cache/'constant.npz'
    constant=None
    if refpath.exists():
        with np.load(refpath,allow_pickle=False) as saved:
            if str(saved['signature'])==signature:
                constant=saved['tape'];search=json.loads(str(saved['search']))
    if constant is None:
        constant,search=constant_search(initial,cfg,p)
        np.savez_compressed(refpath,signature=signature,tape=constant,search=json.dumps(search))
    json_save(reports/'constant-search.json',search)
    print('Selected constant '+str(search['controls']),flush=True)
    steps=int(p.duration/p.control_dt)
    tapes={'unstirred':np.zeros((steps,3),np.float32),'constant':constant}
    runs={name:np.asarray(recorded_rollout(jnp.asarray(initial),jnp.asarray(tape),0.,cfg,p)) for name,tape in tapes.items()}
    records={name:[] for name in runs}
    for seed in (19744,19745,19746):
        name=f'mppi-{seed}'
        runs[name],tapes[name],records[name]=controlled_run(initial,cache,cfg,replace(p,seed=seed),seed==19744)
    import matplotlib
    manifest={'version':1,'scenario':'jacketed-controlled-stirring','config':cfg.to_dict(),'planner':asdict(p),
        'start_s':0,'end_s':p.duration,'mode_centers_m':[2,4,6],'mode_amplitude_m2_s':.15,
        'runs':[],'snapshots':[],'constant_search':{k:v for k,v in search.items() if k!='rows'},
        'limits':{'temperature_K':[800,1070],'conversion':[0,1],'temperature_error_K':[-60,60],'conversion_error':[-.2,.2]},
        'colormaps':{name:(matplotlib.colormaps[name](np.linspace(0,1,256))[:,:3]*255).astype(int).tolist() for name in ('magma','viridis','RdBu_r')},
        'method_note':'Direct numerical PDE planning, with prescribed incompressible velocity modes. No momentum solver, surrogate, or random inlet. Costs correct for the Gaussian proposal in independent latent coordinates.'}
    # Small cell-centered bases support exactly synchronized flow arrows.
    ub,vb=flow_basis(cfg)
    arrows=np.stack([.5*(ub[...,1:]+ub[...,:-1]),.5*(vb[...,1:,:]+vb[...,:-1,:])],axis=1)
    manifest['arrow_basis']=arrows[:,:,::3,::4].tolist()
    manifest['arrow_x_m']=((np.arange(cfg.nx)[::4]+.5)*cfg.length/cfg.nx).tolist()
    manifest['arrow_y_m']=((np.arange(cfg.ny)[::3]+.5)*cfg.width/cfg.ny).tolist()
    for name,values in runs.items():
        manifest['runs'].append({'id':name,'fields':binary(assets/f'{name}.bin',values),
            'controls':tapes[name].tolist(),'times_s':(np.arange(len(values))*p.frame_dt).tolist(),
            'completed':len(tapes[name])==steps,'decisions':records[name],**metrics(values,tapes[name],cfg,p)})
    for t in (16,40,72):
        with np.load(cache/f'snapshot-{t}.npz',allow_pickle=False) as saved:
            s={key:saved[key] for key in saved.files}
        descriptor={'time_s':t,'times_ahead_s':(np.arange(161)*p.frame_dt).tolist(),
            'iteration':json.loads(str(s['iteration'])),'weighted_proposal_exists':bool(s['weighted_proposal_exists']),
            'reference':s['reference'].tolist(),'candidates':[]}
        init=s['initial'];ct=s['candidate_controls']
        cf=np.asarray(recorded_rollout(jnp.broadcast_to(init,(6,)+init.shape),jnp.asarray(ct.swapaxes(0,1)),float(t),cfg,p))
        for j in range(6):
            descriptor['candidates'].append({'index':int(s['candidate_indices'][j]),'objective':float(s['candidate_costs'][j]),
                'weight':float(s['candidate_weights'][j]),'coarse_feasible':bool(s['candidate_feasible'][j]),
                'controls':ct[j].tolist(),'fields':binary(assets/f'decision-{t}-candidate-{j}.bin',cf[:,j])})
        for key in ('weighted','used'):
            tape=s[key+'_controls'];movie=np.asarray(recorded_rollout(jnp.asarray(init),jnp.asarray(tape),float(t),cfg,p))
            descriptor[key]={'controls':tape.tolist(),'fields':binary(assets/f'decision-{t}-{key}.bin',movie)}
        manifest['snapshots'].append(descriptor)
    print('Recordings complete; sampling sensitivity and transport benchmarks',flush=True)
    sensitivity=[]
    for count in (128,256,512):
        for seed in (19744,19745,19746):
            ctl=StirringController(cfg,replace(p,candidates=count,seed=seed))
            _,record,_=ctl.plan(initial,0.)
            sensitivity.append({'candidates':count,'seed':seed,**record})
            print('Sensitivity '+str(count)+'/'+str(seed),flush=True)
    manifest['sensitivity']=sensitivity
    manifest['advection_benchmark']=advection_benchmark()
    print('Checking every recorded control sequence on a doubled mesh',flush=True)
    numerical={};fine_runs={}
    for name in runs:
        numerical[name],fine_runs[name]=refinement(initial,runs[name],tapes[name],cfg,p)
    manifest['numerical_validation']=numerical
    errors=np.array([metrics(runs[f'mppi-{seed}'],tapes[f'mppi-{seed}'],cfg,p)['summary']['mean_mixing_error'] for seed in (19744,19745,19746)])
    manifest['optimizer_variability']={'seeds':[19744,19745,19746],'mean_mixing_error_mean':float(errors.mean()),
        'mean_mixing_error_sample_sd':float(errors.std(ddof=1)),
        'note':'Optimizer randomness on one deterministic feed history; not physical-disturbance uncertainty.'}
    json_save(assets/'manifest.json',manifest)
    json_save(reports/'summary.json',{r['id']:r['summary'] for r in manifest['runs']})
    json_save(reports/'sensitivity.json',sensitivity);json_save(reports/'numerical-validation.json',numerical)
    json_save(reports/'advection-benchmark.json',manifest['advection_benchmark'])
    from thermal_stirring_replay import validate_assets, make_figures
    json_save(reports/'validation.json',validate_assets(assets))
    make_figures(root,manifest,runs)
    print('Stirring artifacts complete',flush=True)
    return manifest
