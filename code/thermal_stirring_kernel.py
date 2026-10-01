"""Fused Metal stencil for the same conservative minmod SSPRK2 stages."""
import mlx.core as mx


def build_substep(cfg):
    header = f'''
constant int NX = {cfg.nx};
constant int NY = {cfg.ny};
constant int CELLS = NX*NY;
constant float DX = {float(cfg.length/cfg.nx)}f;
constant float DY = {float(cfg.width/cfg.ny)}f;
constant float DIFF = {float(cfg.diffusivity)}f;
constant float LOSS = {float(cfg.loss)}f;
constant float WALL = {float(cfg.wall_temperature)}f;
constant float COOL = {float(cfg.reaction_cooling)}f;
constant float DT = {float(cfg.max_dt)}f;
inline float2 field(device const float *state, int batch, int x, int y) {{
    int p=batch*2*CELLS+clamp(y,0,NY-1)*NX+clamp(x,0,NX-1);
    return float2(state[p],state[p+CELLS]);
}}
inline float2 mm(float2 a,float2 b) {{
    return select(float2(0.0f),sign(a)*min(abs(a),abs(b)),a*b>0.0f);
}}
inline float2 sx(device const float *s,int b,int x,int y) {{
    if(x<=0 || x>=NX-1) return float2(0.0f);
    float2 c=field(s,b,x,y);
    return mm(c-field(s,b,x-1,y),field(s,b,x+1,y)-c);
}}
inline float2 sy(device const float *s,int b,int x,int y) {{
    if(y<=0 || y>=NY-1) return float2(0.0f);
    float2 c=field(s,b,x,y);
    return mm(c-field(s,b,x,y-1),field(s,b,x,y+1)-c);
}}
inline float2 fx(device const float *s,device const float *u,int b,int i,int y,float t) {{
    float velocity=u[b*NY*(NX+1)+y*(NX+1)+i];
    float2 left,right,gradient=float2(0.0f);
    if(i==0) {{
        float tin={float(cfg.inlet_temperature)}f+({float(cfg.transverse_amplitude)}f+60.0f*sin(2.0f*M_PI_F*t/32.0f))*cos(M_PI_F*(float(y)+0.5f)/float(NY));
        left=float2(tin,1.0f);right=field(s,b,0,y);
    }} else if(i==NX) {{left=right=field(s,b,NX-1,y);}}
    else {{
        float2 a=field(s,b,i-1,y),c=field(s,b,i,y);
        gradient=(c-a)/DX;left=a+0.5f*sx(s,b,i-1,y);right=c-0.5f*sx(s,b,i,y);
    }}
    return velocity*(velocity>=0.0f?left:right)-DIFF*gradient;
}}
inline float2 fy(device const float *s,device const float *v,int b,int x,int j) {{
    if(j==0 || j==NY) return float2(0.0f);
    float velocity=v[b*(NY+1)*NX+j*NX+x];
    float2 a=field(s,b,x,j-1),c=field(s,b,x,j);
    float2 below=a+0.5f*sy(s,b,x,j-1),above=c-0.5f*sy(s,b,x,j);
    return velocity*(velocity>=0.0f?below:above)-DIFF*(c-a)/DY;
}}
'''
    source='''
    uint idx=thread_position_in_grid.x;
    int b=idx/CELLS,p=idx%CELLS,x=p%NX,y=p/NX;
    float2 z=field(state,b,x,y);
    float t=clock+(SECOND?DT:0.0f);
    float2 derivative=-(fx(state,uface,b,x+1,y,t)-fx(state,uface,b,x,y,t))/DX
                      -(fy(state,vface,b,x,y+1)-fy(state,vface,b,x,y))/DY;
    float rate=0.1f*exp(clamp(12000.0f*(1.0f/950.0f-1.0f/z.x),-60.0f,30.0f))*z.y;
    derivative+=float2(LOSS*(WALL-z.x)-COOL*rate,-rate);
    float2 value=z+DT*derivative;
    if(SECOND) value=0.5f*field(original,b,x,y)+0.5f*value;
    out[b*2*CELLS+p]=value.x;out[b*2*CELLS+CELLS+p]=value.y;
    '''
    kernel=mx.fast.metal_kernel(name='stirring_ssprk2',input_names=['state','original','uface','vface','clock'],
        output_names=['out'],source=source,header=header)
    def substep(x,t,u,v):
        args={'grid':(x.shape[0]*cfg.nx*cfg.ny,1,1),'threadgroup':(256,1,1),
              'output_shapes':[x.shape],'output_dtypes':[mx.float32]}
        first=kernel(inputs=[x,x,u,v,t],template=[('SECOND',False)],**args)[0]
        return kernel(inputs=[first,x,u,v,t],template=[('SECOND',True)],**args)[0]
    return substep
