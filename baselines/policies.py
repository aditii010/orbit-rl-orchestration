from env.orbit_env import A

SOLVE, RETRY, VERIFY = A["INVOKE_SOLVER"], A["RETRY_SOLVER"], A["INVOKE_VERIFIER"]
RESEARCH, SUMMARIZE, STOP = A["INVOKE_RESEARCHER"], A["INVOKE_SUMMARIZER"], A["STOP"]
MAX_STEPS = 8


def single_agent(obs, mask):
    return SOLVE if not obs[4] else STOP


def fixed_pipeline(obs, mask):
    n_res, n_ver, n_sum = (round(obs[8] * MAX_STEPS), round(obs[10] * MAX_STEPS),
                           round(obs[11] * MAX_STEPS))
    if n_res == 0:
        return RESEARCH
    if not obs[4]:
        return SOLVE
    if n_ver == 0:
        return VERIFY
    if n_sum == 0 and mask[SUMMARIZE]:
        return SUMMARIZE
    return STOP


def make_rule_router(research_on=(), verify=True, max_retries=2, skip_verify_conf=None,
                     min_money_for_retry=0.15, summarize=True, giveup_diff=None,
                     lowconf_retry=False):
    def rule(obs, mask):
        ttype = int(obs[:3].argmax())
        n_res = round(obs[8] * MAX_STEPS)
        n_solve = round(obs[9] * MAX_STEPS)
        n_ver = round(obs[10] * MAX_STEPS)
        passed, failed = obs[6] > 0.5, obs[7] > 0.5
        if not obs[4]:
            if ttype in research_on and n_res == 0:
                return RESEARCH
            return SOLVE
        hard = giveup_diff is not None and obs[3] >= giveup_diff
        retries_left = (not hard) and (n_solve - 1) < max_retries and obs[12] > min_money_for_retry
        trusted = skip_verify_conf is not None and obs[14] >= skip_verify_conf
        unchecked = not passed and not failed
        if lowconf_retry and unchecked and obs[14] < 0.5 and retries_left and mask[RETRY]:
            return RETRY
        if verify and not hard and unchecked and not trusted:
            return VERIFY
        if failed and retries_left and mask[RETRY]:
            return RETRY
        if summarize and mask[SUMMARIZE]:
            return SUMMARIZE
        return STOP
    return rule


def make_typed_rule(res_calls, verify, max_retries, summarize):
    def rule(obs, mask):
        t = int(obs[:3].argmax())
        n_res, n_solve = round(obs[8] * MAX_STEPS), round(obs[9] * MAX_STEPS)
        passed, failed = obs[6] > 0.5, obs[7] > 0.5
        if not obs[4]:
            return RESEARCH if n_res < res_calls[t] else SOLVE
        if verify[t] and not passed and not failed:
            return VERIFY
        if failed and (n_solve - 1) < max_retries and obs[12] > 0.05 and mask[RETRY]:
            return RETRY
        if summarize and mask[SUMMARIZE]:
            return SUMMARIZE
        return STOP
    return rule
