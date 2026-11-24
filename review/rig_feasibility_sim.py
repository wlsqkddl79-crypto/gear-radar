import numpy as np
g=9.81
# ---- 1) stroke budget on a 490 x 420 pad ----
pad_long, sled_len, margin = 0.490, 0.060, 0.030   # m
usable = pad_long - sled_len - 2*margin
print(f"[1] usable stroke (long side) = {usable*1000:.0f} mm")
for v in [0.1, 0.3, 1.0, 2.0]:
    for a in [5, 10, 20]:
        acc = v*v/(2*a)
        steady = usable - 2*acc
        print(f"  v={v:>4} m/s a={a:>2} m/s^2 -> accel+decel {2*acc*1000:5.0f} mm, steady {steady*1000:5.0f} mm = {steady/v*1000:6.1f} ms"
              + ("" if steady>0 else "  <-- impossible"))

# ---- 2) dynamic sim: stage pulls sled via load cell (spring-damper) ----
def profile(t, v, a, usable, scurve):
    # returns stage velocity
    ta = v/a if not scurve else 2*v/a      # s-curve: peak accel a, avg a/2
    d_acc = v*ta/2
    tc = (usable - 2*d_acc)/v
    T = 2*ta + tc
    def acc_frac(tt):  # 0..1 velocity fraction during ramp
        x = np.clip(tt/ta,0,1)
        return x if not scurve else (1-np.cos(np.pi*x))/2
    vs = np.where(t<ta, v*acc_frac(t), np.where(t<ta+tc, v, v*(1-acc_frac(t-ta-tc))))
    vs = np.where(t>T, 0, vs)
    return vs, ta, tc, T

def sim(m=0.3, mu=0.318, v=2.0, a=20, k=2e5, zeta=0.02, scurve=False, fs=1000):
    N = m*g
    dt = 2e-6
    _,ta,tc,T = profile(np.zeros(1), v, a, usable, scurve)
    vs_all,_,_,_ = profile(np.arange(0, T+0.01, dt), v, a, usable, scurve)
    if tc<=0: return None
    n = int(T/dt)+1
    c = 2*zeta*np.sqrt(k*m)
    xs=x=vel=0.0
    F = np.empty(n)
    for i in range(n):
        vs = vs_all[i]
        Fc = k*(xs-x) + c*(vs-vel)
        Ff = mu*N*np.tanh(vel/1e-4)
        vel += (Fc-Ff)/m*dt
        x += vel*dt
        xs += vs*dt
        F[i]=Fc
    t = np.arange(n)*dt
    win = (t>ta) & (t<ta+tc)
    # sample at fs within window
    idx = np.where(win)[0][::int(round(1/(fs*dt)))]
    samp = F[idx]
    true = mu*N
    return dict(fn=np.sqrt(k/m)/2/np.pi, window_ms=tc*1000, n=len(samp),
                bias=(samp.mean()-true)/true*100, ripple=samp.std()/true*100,
                peak=np.abs(F).max()/true)

print("\n[2] 300 g sled, mu_k=0.318 (friction force = %.2f N)" % (0.318*0.3*g))
cases = [
 ("2 m/s rigid cell z.02, trapezoid 20",      dict(v=2, k=2e5, zeta=0.02, a=20)),
 ("2 m/s rigid cell z.02, S-curve pk40",      dict(v=2, k=2e5, zeta=0.02, a=40, scurve=True)),
 ("2 m/s rigid cell z.20, S-curve pk40",      dict(v=2, k=2e5, zeta=0.2, a=40, scurve=True)),
 ("2 m/s soft string, trapezoid 20",          dict(v=2, k=2e3, zeta=0.02, a=20)),
 ("1 m/s rigid cell z.20, S-curve pk20",      dict(v=1, k=2e5, zeta=0.2, a=20, scurve=True)),
 ("1 m/s rigid z.20, S-curve, HX711 80 SPS",  dict(v=1, k=2e5, zeta=0.2, a=20, scurve=True, fs=80)),
 ("0.1 m/s rigid z.02, trapezoid 5",          dict(v=0.1, k=2e5, zeta=0.02, a=5)),
]
for name, kw in cases:
    r = sim(**kw)
    if r is None: print(f'  {name:45s} -> no steady segment'); continue
    print(f"  {name:45s} fn={r['fn']:5.0f} Hz window={r['window_ms']:6.1f} ms samples={r['n']:4d} "
          f"bias={r['bias']:+6.1f}%  ripple(sd)={r['ripple']:6.1f}%  peak force={r['peak']:4.1f}x")

# ---- 3) pitching moment from pull height ----
print("\n[3] pull-point height -> front/rear foot load shift (300 g, mu .32, foot spacing 60 mm)")
F = 0.318*0.3*g; W=0.3*g
for h in [0, 5, 10, 20]:
    dN = F*h/1000/0.060
    print(f"  h={h:2d} mm: front foot unload {dN/W*100:4.1f}% of total weight")

# ---- 4) 5x5 grid cell vs sled footprint ----
cx, cy = 0.490/5, 0.420/5
print(f"\n[4] 5x5 cell = {cx*1000:.0f} x {cy*1000:.0f} mm; stroke inside cell at 0.1 m/s, a=2 m/s^2: "
      f"{(cx-sled_len-2*0.1**2/(2*2))*1000:.0f} mm steady ({(cx-sled_len-2*0.1**2/4)/0.1*1000:.0f} ms)")

# ---- 5) base grip plausibility ----
print("\n[5] base grip: required static mu of rubber on MDF")
for mass in [0.35, 0.45, 0.55]:
    print(f"  pad {mass*1000:.0f} g, measured 2.9 kgf -> mu_s = {2.9/mass:.1f}")

# ---- 6) uniformity CV vs measurement repeatability ----
rng = np.random.default_rng(1)
print("\n[6] 25 identical points, rig repeatability -> apparent CV")
for rep in [0.5, 1, 2, 3]:
    cvs=[ (s:=rng.normal(0.318,0.318*rep/100,25)).std(ddof=1)/s.mean()*100 for _ in range(2000)]
    print(f"  repeatability {rep}% -> apparent CV {np.mean(cvs):.1f}% (pad perfectly uniform)")

# ---- 7) rotating disc for 10 m/s tracking test ----
print("\n[7] rotating disc for tracking-speed test")
for r_ in [0.10, 0.15, 0.20]:
    rpm = 10/(2*np.pi*r_)*60
    print(f"  r={r_*100:.0f} cm: 10 m/s = {rpm:4.0f} rpm, centripetal {10**2/r_/g:4.0f} g")
