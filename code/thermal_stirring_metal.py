"""Optional MLX execution of the same minmod/SSPRK2 candidate PDE.

No learned dynamics: JAX still executes and validates selected controls.
The fixed step is proved below to respect the whole admissible control ball.
"""
import itertools
import numpy as np
import mlx.core as mx
from thermal_stirring import flow_basis
from thermal_stirring_kernel import build_substep


class MetalStirringScorer:
    def __init__(self, cfg, planner):
        self.cfg,self.planner=cfg,planner
        ub,vb=flow_basis(cfg)
        dx,dy=cfg.length/cfg.nx,cfg.width/cfg.ny
        terms=np.stack([ub[...,1:]/dx,-ub[...,:-1]/dx,vb[...,1:,:]/dy,-vb[...,:-1,:]/dy])
        maximum=0.
        for selector in itertools.product((0,1),repeat=4):
            summed=np.einsum('i,ijyx->jyx',selector,terms)
            maximum=max(maximum,float(np.max(summed[0]+np.linalg.norm(summed[1:],axis=0))))
        bound=.9/(2*maximum+2*cfg.diffusivity*(dx**-2+dy**-2)+cfg.loss+.1*np.exp(12000*(1/950-1/cfg.wall_temperature)))
        if cfg.max_dt>bound:
            raise ValueError('GPU fixed timestep exceeds the admissible-flow stability bound')
        self.stability_bound=bound
        self.substeps=int(round(planner.control_dt/cfg.max_dt))
        if abs(self.substeps*cfg.max_dt-planner.control_dt)>1e-6:
            raise ValueError('Control interval must divide into fixed GPU steps')
        ubase,vbase=mx.array(ub),mx.array(vb)
        @mx.compile
        def velocity(a):
            return ((ubase[0]+mx.einsum('bj,jyx->byx',a,ubase[1:]))[:,None],
                    (vbase[0]+mx.einsum('bj,jyx->byx',a,vbase[1:]))[:,None])

        def mixing(x):
            part=x[:,:,:,int(round(6/cfg.length*cfg.nx)):]
            var=mx.mean(mx.var(part,axis=-2),axis=-1)
            return var[:,0]/400+var[:,1]/.01
        weights=mx.array(ub[0,:,-1]);weights=weights/mx.sum(weights)

        @mx.compile
        def accumulate(x,u,old,total,valid):
            conv=1-mx.sum(x[:,1,:,-1]*weights,-1)
            ok=(mx.all(mx.isfinite(x),axis=(1,2,3)) & (mx.max(x[:,0],axis=(1,2))<=cfg.temperature_limit)
                & (mx.min(x[:,0],axis=(1,2))>0) & (mx.min(x[:,1],axis=(1,2))>=-2e-5)
                & (mx.max(x[:,1],axis=(1,2))<=1+2e-5) & (conv>=cfg.conversion_target)
                & (mx.sum(u*u,-1)<=1+1e-6))
            cost=mixing(x)+planner.effort_weight*mx.sum(u*u,-1)+planner.slew_weight*mx.sum((u-old)**2,-1)
            return total+cost,valid & ok
        self.velocity,self.substep,self.accumulate,self.mixing=velocity,build_substep(cfg),accumulate,mixing

    def run(self,initial,tapes,start,previous,return_fields=False):
        count,horizon,_=tapes.shape
        x=mx.broadcast_to(mx.array(np.asarray(initial,np.float32)),(count,)+initial.shape)
        actions=mx.array(np.asarray(tapes,np.float32))
        old=mx.broadcast_to(mx.array(np.asarray(previous,np.float32)),(count,3))
        total=mx.zeros(count);valid=mx.ones(count,dtype=mx.bool_)
        for k in range(horizon):
            a=actions[:,k];u,v=self.velocity(a)
            for sub in range(self.substeps):
                t=mx.array(start+k*self.planner.control_dt+sub*self.cfg.max_dt,mx.float32)
                x=self.substep(x,t,u,v)
                # Bound graph size while keeping the GPU asynchronously occupied.
                if sub%8==7:mx.async_eval(x)
            total,valid=self.accumulate(x,a,old,total,valid);old=a
            mx.eval(x,total,valid)
        raw=total/horizon+self.planner.terminal_weight*self.mixing(x)
        valid=valid & mx.isfinite(raw)
        mx.eval(raw,valid)
        out=(np.where(np.asarray(valid),np.asarray(raw),np.inf),np.asarray(raw),np.asarray(valid))
        return (*out,np.asarray(x)) if return_fields else out

    def score(self,initial,tapes,start,previous):
        return self.run(initial,tapes,start,previous)
