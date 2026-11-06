import itertools, feasibility as f
f.V_PASS_IF_OK, f.V_PASS_IF_BAD = 0.95, 0.10
f.P_SOLVE = [0.90, 0.55, 0.20]
BUDGETS = [3, 7]
def avg(cs, cv, pol_for):
    t = 0
    for H in BUDGETS:
        f.H = H
        t += f.run(cs, cv, pol_for(H))/len(BUDGETS)
    return t
def rule(verify, me, mh, reserve):
    base = f.make_rule(verify, me, mh)
    def r(h,k,has,last,steps):
        a = base(h,k,has,last,steps)
        if a == 'solve' and has and steps < reserve: return 'stop'
        return a
    return r
for cs, cv in [(0.08,0.02),(0.12,0.03),(0.18,0.04)]:
    opt = avg(cs,cv,lambda H: None)
    sing = avg(cs,cv,lambda H: f.single); fix = avg(cs,cv,lambda H: f.fixed)
    best,bp=-9,None
    for v,me,mh,res in itertools.product([True,False],range(1,5),range(1,5),[0,2,3]):
        r = avg(cs,cv,lambda H: rule(v,me,mh,res))
        if r>best: best,bp=r,(v,me,mh,res)
    print(f"cs={cs} cv={cv} single={sing:.3f} fixed={fix:.3f} bestRule={best:.3f}{bp} OPT={opt:.3f} gap={100*(opt/best-1):.1f}%")
