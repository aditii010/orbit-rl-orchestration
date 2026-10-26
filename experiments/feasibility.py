"""Feasibility test: is there room for a learned policy over hand-made rules?
Toy world, solved exactly (expectimax over observable histories, Bayesian belief on hidden state).
Pre-registered rule: GO if optimal reward >= 1.10 * best hand rule reward."""
import itertools, sys
from functools import lru_cache

PRIOR = [0.4, 0.4, 0.2]            # easy, medium, hard
P_SOLVE = [0.90, 0.60, 0.25]       # chance a fresh solve is correct, per difficulty
DECAY = 0.85                       # each extra attempt is a bit less likely to be right
HINT = [0.70, 0.40, 0.15]          # P(hint says "looks easy" | difficulty)
V_PASS_IF_OK, V_PASS_IF_BAD = 0.90, 0.20   # noisy verifier
H = 6                              # max actions

def run(c_solve, c_verify, policy):
    """Exact expected reward of `policy` (policy=None -> optimal). Returns (value, policy_table)."""
    def p_ok(d, k): return P_SOLVE[d] * DECAY ** k

    def rec(belief, k, has, last, steps, hint, hist):
        # belief[d][c]; k = solves so far; last in {None,'pass','fail'}
        acts = ['stop'] + (['solve'] if steps > 0 else []) + (['verify'] if has and steps > 0 else [])
        def q(a):
            if a == 'stop':
                return sum(belief[d][1] for d in range(3)) if has else 0.0
            if a == 'solve':
                nb = [[0, 0] for _ in range(3)]
                for d in range(3):
                    m = belief[d][0] + belief[d][1]
                    nb[d][1] = m * p_ok(d, k); nb[d][0] = m * (1 - p_ok(d, k))
                return -c_solve + rec(nb, k + 1, True, None, steps - 1, hint, hist + ('solve',))
            # verify
            tot = -c_verify
            for obs in ('pass', 'fail'):
                nb = [[0, 0] for _ in range(3)]
                for d in range(3):
                    lik1 = V_PASS_IF_OK if obs == 'pass' else 1 - V_PASS_IF_OK
                    lik0 = V_PASS_IF_BAD if obs == 'pass' else 1 - V_PASS_IF_BAD
                    nb[d][1] = belief[d][1] * lik1; nb[d][0] = belief[d][0] * lik0
                pr = sum(nb[d][0] + nb[d][1] for d in range(3))
                if pr == 0: continue
                nb = [[x / pr for x in r] for r in nb]
                tot += pr * rec(nb, k, True, obs, steps - 1, hint, hist + ('verify:' + obs,))
            return tot
        if policy is None:
            return max(q(a) for a in acts)
        a = policy(hint, k, has, last, steps)
        return q(a if a in acts else 'stop')

    total = 0.0
    for hint in (1, 0):  # 1 = "looks easy"
        ph = sum(PRIOR[d] * (HINT[d] if hint else 1 - HINT[d]) for d in range(3))
        belief = [[PRIOR[d] * (HINT[d] if hint else 1 - HINT[d]) / ph, 0.0] for d in range(3)]
        # no candidate yet: put mass in c=0 slot (has=False so it is never read)
        total += ph * rec(belief, 0, False, None, H, hint, ())
    return total

def single(h,k,has,last,steps): return 'solve' if not has else 'stop'
def fixed(h,k,has,last,steps):
    if not has: return 'solve'
    if last is None: return 'verify'
    return 'stop'
def make_rule(verify, m_easy, m_hard):
    def rule(h,k,has,last,steps):
        m = m_easy if h else m_hard
        if not has: return 'solve'
        if verify:
            if last is None: return 'verify'
            if last == 'pass': return 'stop'
            return 'solve' if k < m else 'stop'
        return 'solve' if k < m else 'stop'
    return rule

if __name__ == '__main__':
    for cs, cv in [(0.10, 0.03), (0.15, 0.05), (0.25, 0.10), (0.35, 0.15)]:
        opt = run(cs, cv, None)
        s, f = run(cs, cv, single), run(cs, cv, fixed)
        best, bp = -9, None
        for v, me, mh in itertools.product([True, False], range(1, 5), range(1, 5)):
            r = run(cs, cv, make_rule(v, me, mh))
            if r > best: best, bp = r, (v, me, mh)
        print(f"c_solve={cs:.2f} c_verify={cv:.2f} | single={s:.3f} fixed={f:.3f} "
              f"bestRule={best:.3f}{bp} OPTIMAL={opt:.3f} | gap vs rule={100*(opt/best-1):.1f}%")
