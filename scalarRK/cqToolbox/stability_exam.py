import math
from mpmath import mp, mpf, cos, sqrt as msqrt

mp.dps = 50  # 50-digit reference

def naive(a, b, gamma):
    return math.sqrt(a*a + b*b - 2*a*b*math.cos(gamma))

def stable(a, b, gamma):
    return math.sqrt((a - b)**2 + (2*math.sqrt(a*b)*math.sin(gamma/2))**2)
    #return math.hypot(a - b, 2*math.sqrt(a*b)*math.sin(gamma/2))

def test(a, b, gamma):
    return math.sqrt(((a*a+b*b)**2 - 4*a**2*b**2*math.cos(gamma)**2))/(a**2+b**2+2*a*b*math.cos(gamma))
    #return math.sqrt(a*a + b*b - a*b*math.sqrt(2*math.sqrt(1-math.sin(gamma/2)**2)))


def reference(a, b, gamma):
    a, b, g = mpf(a), mpf(b), mpf(gamma)
    return float(msqrt(a*a + b*b - 2*a*b*cos(g)))

a = b = 1.0
b = 1+1e-9
print(f"{'gamma':>10} {'naive':>22} {'test':>22} {'stable':>22} {'reference':>22}   rel.err naive rel.err test  rel.err stable")
for k in range(1, 11):
    gamma = 10.0**(-k)
    n, t, s, r = naive(a, b, gamma),test(a,b,gamma), stable(a, b, gamma), reference(a, b, gamma)
    en = abs(n - r) / r
    et = abs(t - r) / r
    es = abs(s - r) / r
    print(f"{gamma:>10.0e} {n:>22.15e} {t:>22.15e} {s:>22.15e} {r:>22.15e}   {en:>12.2e}  {et:>12.2e}  {es:>12.2e}")