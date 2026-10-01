"""Artifact checks and static figures for the stirring-control experiment."""
from pathlib import Path
import hashlib
import json
import numpy as np
from thermal_stirring import CONFIG, PLAN, face_velocity, mixing_error
from thermal_stirring_experiment import metrics
from thermal_reactor import valid_fields


def validate_assets(assets):
    assets=Path(assets);m=json.loads((assets/'manifest.json').read_text());arrays={};count=0
    def read(desc):
        nonlocal count
        raw=(assets/desc['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==desc['sha256']
        a=np.frombuffer(raw,dtype='<f4').reshape(desc['shape']);count+=1
        assert valid_fields(a)
        return a
    for r in m['runs']:
        a=read(r['fields']);arrays[r['id']]=a
        assert r['completed'] and len(a)==641 and len(r['controls'])==80
        np.testing.assert_allclose(np.diff(r['times_s']),.25)
        assert np.max(np.linalg.norm(r['controls'],axis=-1))<=1+1e-6
        expected=metrics(a,np.asarray(r['controls']))
        for key in ('peak_temperature_K','outlet_conversion','temperature_mixing_error',
                    'conversion_mixing_error','outlet_temperature_std_K','outlet_conversion_std'):
            np.testing.assert_allclose(expected[key],r[key],atol=2e-6)
        for decision in r['decisions']:
            assert not decision['planning_failure']
            for it in decision['iterations']:
                assert it['incumbent_after_objective']<=it['incumbent_before_objective']+1e-9
    for a in arrays.values():np.testing.assert_array_equal(a[0],arrays['unstirred'][0])
    if np.count_nonzero(m['constant_search']['controls']) == 0:
        np.testing.assert_array_equal(arrays['constant'], arrays['unstirred'])
    for s in m['snapshots']:
        initial=arrays['mppi-19744'][int(s['time_s']/.25)]
        for c in s['candidates']+[s['weighted'],s['used']]:
            a=read(c['fields']);assert len(a)==len(s['times_ahead_s'])==161
            np.testing.assert_array_equal(a[0],initial)
            assert np.max(np.linalg.norm(c['controls'],axis=-1))<=1+1e-6
        assert sum(c['weight'] for c in s['candidates'])<=1+1e-6
        assert all(c['coarse_feasible'] or c['weight']==0 for c in s['candidates'])
    root=assets.parent.parent
    sources=['code/thermal_stirring.py','code/thermal_stirring_metal.py',
             'code/thermal_stirring_kernel.py','code/thermal_stirring_experiment.py',
             'code/thermal_stirring_replay.py','code/thermal_reactor.py','code/mppi_control.py',
             'scripts/build_thermal_reactor.py']
    return {'completed':True,'arrays_checked':count,'run_count':len(arrays),'snapshot_count':len(m['snapshots']),
            'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
            'field_hashes_timestamps_controls_and_diagnostics_verified':True,
            'numerical_validation':m['numerical_validation'],'optimizer_variability':m['optimizer_variability']}


def make_figures(root,manifest,runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,
                         'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    dest=Path(root)/'_static/thermal_reactor';dest.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        for ext in ('png','pdf','svg'):fig.savefig(dest/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    zero_constant=np.count_nonzero(manifest['constant_search']['controls']) == 0
    comparison=('unstirred','mppi-19744') if zero_constant else ('unstirred','constant','mppi-19744')
    fig,axes=plt.subplots(len(comparison),4,figsize=(12,3.3 if zero_constant else 5.0),layout='compressed')
    if zero_constant:fig.suptitle('No stirring is also the best grid constant',fontsize=9)
    for col,t in enumerate((24,40,48,56)):
        k=int(t/.25)
        for row,name in enumerate(comparison):
            f=runs[name][k,0];f=f-f.mean(axis=-2,keepdims=True)
            ax=axes[row,col]
            im=ax.imshow(f,origin='lower',extent=(0,8,0,2),cmap='RdBu_r',vmin=-60,vmax=60)
            ax.set(xticks=[0,2,4,6,8],yticks=[0,1,2],xlabel='Downstream position (m)')
            if row==0:ax.set_title(f'{t} s')
            if col==0:ax.set_ylabel({'unstirred':'No stirring',
                                    'constant':'Best grid constant','mppi-19744':'MPPI'}[name]+'\nWidth (m)')
            ax.axvline(6,color='.5',ls=':',lw=.7)
    fig.colorbar(im,ax=axes,fraction=.018,pad=.02,label='Temperature departure (K)')
    save(fig,'stirring-storyboard')
    fig,axes=plt.subplots(2,2,figsize=(10,5.6),layout='constrained')
    lookup={r['id']:r for r in manifest['runs']};ts=np.array(lookup['unstirred']['times_s'])
    for name,color,style,label in [('unstirred','#6b7280','--','No stirring'),('constant','#D55E00','-.','Best grid constant')]:
        if zero_constant and name=='constant':continue
        if zero_constant:label='No stirring = best grid constant'
        r=lookup[name];e=np.array(r['temperature_mixing_error'])+r['conversion_mixing_error']
        axes[0,0].plot(ts,e,color=color,ls=style,label=label)
        axes[0,1].plot(ts,r['outlet_temperature_std_K'],color=color,ls=style)
    seeds=[lookup[f'mppi-{s}'] for s in (19744,19745,19746)]
    for ax,key in ((axes[0,0],'mixing'),(axes[0,1],'outlet_temperature_std_K')):
        y=np.array([np.array(r['temperature_mixing_error'])+r['conversion_mixing_error'] if key=='mixing' else r[key] for r in seeds])
        mean,sd=y.mean(0),y.std(0,ddof=1)
        ax.plot(ts,mean,color='#0072B2',label='MPPI mean ± 1 s.d. (3 seeds)')
        ax.fill_between(ts,mean-sd,mean+sd,color='#0072B2',alpha=.18)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=2,fontsize=8)
    selected=lookup['mppi-19744'];tape=np.array(selected['controls']);t=np.arange(81)*2
    for j,(color,style) in enumerate(zip(('#0072B2','#D55E00','#009E73'),('-','--','-.'))):
        axes[1,0].step(t,np.r_[tape[:,j],tape[-1,j]],where='post',color=color,ls=style,label=f'S{j+1}: x={2+2*j} m')
    axes[1,0].set_ylim(-1,1)
    axes[1,0].legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,fontsize=8)
    for name,color,style,label in [('unstirred','#6b7280','--','No stirring'),('constant','#D55E00','-.','Constant')]:
        if zero_constant and name=='constant':continue
        axes[1,1].plot(ts,lookup[name]['outlet_conversion'],color=color,ls=style,label=label)
    quality=np.array([r['outlet_conversion'] for r in seeds]);mean=quality.mean(0);sd=quality.std(0,ddof=1)
    axes[1,1].plot(ts,mean,color='#0072B2')
    axes[1,1].fill_between(ts,mean-sd,mean+sd,color='#0072B2',alpha=.18)
    axes[1,1].axhline(.95,color='black',ls=':',lw=.8)
    for ax,label in zip(axes.flat,('Normalized downstream variation','Outlet temperature std. (K)',
                                 'Stirring control (seed 19744)','Flow-weighted outlet conversion')):
        ax.set(xlabel='Time (s)',ylabel=label)
    save(fig,'stirring-control')
