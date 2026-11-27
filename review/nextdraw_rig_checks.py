import numpy as np, math
g=9.81
W_sled=0.200*g
# ---------- A. spring capsule on the string: does it touch the pad? ----------
# capsule 14x14x40 PLA 100% with Ø8 bore for C166 spring, + spring + pin
vol = 14*14*40 - math.pi*4**2*40          # mm^3
m_cap = vol*1.24e-3 + 0.5 + 0.5           # g  (PLA 1.24 g/cc, spring ~0.5 g, pin ~0.5 g)
hook_h, half_h = 10.0, 7.0                # string at 10 mm, capsule half-height 7 mm
clear0 = hook_h - half_h                  # 3 mm clearance when string is perfectly straight
L, a = 150.0, 30.0                        # string length, capsule ~30 mm from ring post
b = L - a
print(f"[A] capsule mass ~{m_cap:.1f} g, static clearance to pad {clear0:.0f} mm")
for mu in [0.08, 0.15, 0.25, 0.33, 0.45]:
    T = mu*W_sled                          # string tension while sliding (N)
    sag = (m_cap*1e-3*g)*a*b/(T*L)         # mm, point load on taut string
    touch = "TOUCHES pad" if sag >= clear0 else "clear"
    print(f"   taut string, mu={mu:.2f} (T={T/g*1000:5.1f} gf): capsule sag {sag:5.2f} mm -> {touch}")
# slack string: 10 mm design slack over 150 mm -> sag of a slack line (parabola approx)
slack = 10.0
sag_slack = math.sqrt(3*L*slack/8)
print(f"   slack string (10 mm slack): natural sag ~{sag_slack:.0f} mm  (> {hook_h:.0f} mm hook height -> lies on pad)")
# drag of a capsule lying on the pad, as a fraction of measured force
print("   extra force if one capsule drags on the pad (PLA on cloth mu_c 0.2-0.5):")
for mu in [0.08, 0.15, 0.25, 0.33, 0.45]:
    F = mu*200
    lo, hi = m_cap*0.2, m_cap*0.5
    print(f"     pad mu={mu:.2f}: measured {F:5.1f} gf, capsule drag {lo:.1f}-{hi:.1f} gf = +{lo/F*100:4.1f}% .. +{hi/F*100:4.1f}%")

# ---------- B. post on the pen holder: bending under pull force ----------
# PLA post Ø16, length to load cell ~ (unknown) assume lever 40-60 mm from pen-holder clamp
E_pla = 3.0e9; d = 0.016; I = math.pi*d**4/64
print(f"\n[B] PLA post Ø16 bending (cantilever), tip deflection under max pull 135 gf:")
for Lp in [0.04, 0.06, 0.08]:
    F = 0.135*g
    defl = F*Lp**3/(3*E_pla*I)*1000
    print(f"   lever {Lp*1000:.0f} mm: {defl*1000:.1f} um  (post itself is stiff; real risk = pen-holder/servo arm compliance)")
# pen-holder moment: horizontal force at 10 mm + post length above holder
for Lp in [0.03, 0.05]:
    M = 0.135*g*Lp
    print(f"   moment on pen holder at {Lp*1000:.0f} mm lever: {M*1000:.1f} N*mm  (+ ~100 g hanging = {0.1*g*1000:.0f} mN vertical)")
# string height change if holder tilts
for tilt in [0.5,1,2]:
    dh = 50*math.sin(math.radians(tilt))
    err = (1-math.cos(math.atan(dh/150)))*100 + math.tan(math.atan(dh/150))*0.33*100
    print(f"   holder tilts {tilt}°  -> ring end moves {dh:.1f} mm vertically -> string angle error ≈ {err:.1f}% (mu~0.33)")

# ---------- C. time budget of one static pull ----------
k = 0.39  # N/mm
for mu_s in [0.10, 0.33, 0.45]:
    defl = mu_s*W_sled/k     # mm of spring compression before slip
    t = defl/0.1
    print(f"\n[C] static mu_s={mu_s}: spring takes {defl:.2f} mm -> {t:.0f} s at 0.1 mm/s" if mu_s==0.10 else
          f"[C] static mu_s={mu_s}: spring takes {defl:.2f} mm -> {t:.0f} s at 0.1 mm/s")
print("    (slack take-up must be done fast first; at 0.1 mm/s 10 mm slack alone = 100 s)")

# ---------- D. HX711 80 SPS vs static force ramp ----------
rate = 0.1*k/g*1000   # gf/s
print(f"\n[D] force ramp at 0.1 mm/s through C166: {rate:.1f} gf/s -> {rate/80:.3f} gf per HX711 sample (80 SPS): peak resolution fine")
