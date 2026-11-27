"""NextDraw pad-friction rig: time-domain Monte Carlo, design A (spring capsule on string) vs B (spring in bridge)."""
import numpy as np, math, sys
g=9.81; GF=g/1000
DT=5e-4; M=0.200; W=M*g
STEP=12.5e-3; TOOTH=2.0; WOB=5e-3           # mm
K_SPR=1/(1/0.39+1/3.0); K_STIFF=3.0         # N/mm (spring+arm flex, pin-in)
ZETA=0.05; ACC=5000.0                       # mm/s^2
M_CAP=8.2e-3; MU_CAP=0.35; CLEAR=3.0        # capsule
def run(pad, setup, design, mode, direction, rng):
    mus, muk, visc, rip_a, rip_l = pad
    tilt, alpha, gain = setup
    k = K_SPR if mode=='static' else K_STIFF
    c = 2*ZETA*math.sqrt(k*1000*M)/1000      # N/(mm/s)
    if mode=='static': v_t, L = 0.1, None
    elif mode=='v10':  v_t, L = 10.0, 100.0
    else:              v_t, L = 100.0, 200.0
    x=0.0; v=0.0; xc=0.0; vc=0.0; stuck=True; t=0.0
    slip_t=None; Fs=[]; Ts=[]; Xc=[]
    sin_t=math.sin(math.radians(tilt))*direction
    while True:
        # command profile
        if mode=='static':
            vc=v_t
            if slip_t is not None and t>slip_t+1.0: break
            if t>60: break
        else:
            dist_left=L-xc
            if dist_left<=0: break
            vc=min(v_t, vc+ACC*DT, math.sqrt(max(2*ACC*dist_left,0)))
        xc+=vc*DT
        xs=math.floor(xc/STEP)*STEP + WOB*math.sin(2*math.pi*xc/TOOTH)
        ext=xs-x
        F=k*ext+c*(vc-v) if ext>0 else 0.0
        Fh=F*math.cos(math.radians(alpha))
        N=W - F*math.sin(math.radians(alpha))
        # surface ripple on mu
        r=1+rip_a*math.sin(2*math.pi*x/rip_l)
        cap=0.0
        if design=='A':
            cap=M_CAP*g*MU_CAP                           # slack-side capsule always dragging
            T=max(F,1e-6); sag=(M_CAP*g)*30*120/(T*150)  # taut-side capsule
            if sag>CLEAR: cap+=M_CAP*g*MU_CAP*min(1,(sag-CLEAR)/sag)
        grav=W*sin_t
        if stuck:
            if Fh-grav > mus*N*r + cap:
                stuck=False; slip_t=t if slip_t is None else slip_t
        if not stuck:
            mk=muk*(1+visc*abs(v)/100.0)*r
            a=(Fh-grav-mk*N-cap)/M/1000*1e6   # mm/s^2
            vn=v+a*DT
            if vn<=0 and Fh-grav<=mus*N*r+cap: vn=0.0; stuck=True
            v=vn; x+=v*DT
        Fs.append(F); Ts.append(t); Xc.append(xc); t+=DT
    Fs=np.array(Fs); Ts=np.array(Ts); Xc=np.array(Xc)
    # HX711 80 SPS: bin average + gain + noise + 60 Hz hum
    nb=int(round((1/80)/DT)); n=len(Fs)//nb
    hum=0.2*GF*np.sin(2*math.pi*60*Ts+rng.uniform(0,6.28))
    S=((Fs+hum)[:n*nb].reshape(n,nb).mean(1))*(1+gain)+rng.normal(0,0.15*GF,n)
    tb=Ts[:n*nb].reshape(n,nb).mean(1); xb=Xc[:n*nb].reshape(n,nb).mean(1)
    Wm=W*(1+rng.normal(0,0.02/200))
    if mode=='static':
        ma=np.convolve(S,np.ones(2)/2,'valid'); return ma.max()/Wm
    L_=L; sel=(xb>0.5*L_)&(xb<0.95*L_)
    return S[sel].mean()/Wm

def experiment(n_setups, seed=1):
    rng=np.random.default_rng(seed)
    res={d:{m:[] for m in ['static','v10','v100']} for d in 'AB'}
    for i in range(n_setups):
        muk=rng.uniform(0.08,0.45); mus=muk*rng.uniform(1.05,1.5)
        pad=(mus,muk,rng.uniform(0,0.05),rng.uniform(0.005,0.03),rng.uniform(5,40))
        setup=(rng.normal(0,0.1), rng.normal(0,0.38), rng.normal(0,0.0005))
        for d in 'AB':
            for m in ['static','v10','v100']:
                vals=[]
                for dirn in (+1,-1):
                    for rep in range(2):
                        sub=np.random.default_rng(rng.integers(1e9))
                        vals.append(run(pad,setup,d,m,dirn,sub))
                meas=np.mean(vals)
                true=mus if m=='static' else muk*(1+pad[2]*(10 if m=='v10' else 100)/100.0)
                res[d][m].append((pad[1], (meas-true)/true*100))
        print(f"  setup {i+1}/{n_setups} done", file=sys.stderr, flush=True)
    return res

if __name__=='__main__':
    n=int(sys.argv[1]) if len(sys.argv)>1 else 30
    res=experiment(n)
    names={'static':'정적 μs','v10':'동적 10 mm/s','v100':'동적 100 mm/s'}
    for d,label in [('A','현재 설계 (줄 위 캡슐)'),('B','수정안 (스프링을 링 쪽으로)')]:
        print(f"\n== {label} ==")
        for m in ['static','v10','v100']:
            arr=np.array(res[d][m]); e=arr[:,1]; lowmu=e[arr[:,0]<0.2]; himu=e[arr[:,0]>=0.2]
            print(f"  {names[m]:12s} 평균 {e.mean():+5.2f}%  95%오차 {np.percentile(abs(e),95):5.2f}%  "
                  f"| 스피드패드(μk<0.2) 평균 {lowmu.mean():+5.1f}%  컨트롤패드 평균 {himu.mean():+5.1f}%")
