"""Print per-turn model actions, gate rejections, guard overrides and responses from an adapter output."""
import json, sys
d = json.load(open(sys.argv[1]))
only = sys.argv[2:] 
for r in d['results']:
    if only and not any(r['case_id'].startswith(o) for o in only):
        continue
    print('=====', r['case_id'], [t['decision'] for t in r['turns']])
    for n, t in enumerate(r['turns'], 1):
        evs = [a for a in r['action_audit'] if a.get('turn') == n]
        for a in evs:
            k = a.get('kind')
            if k == 'llm_step':
                args = a.get('arguments')
                print(f"  T{n} LLM {a['agent']}: {a['tool']} {json.dumps(args, ensure_ascii=False)[:160]}")
            elif k in ('gate', 'guard', 'fallback', 'override', 'veto', 'confirmation') or (a.get('error') and k != 'llm_step'):
                err = a.get('error') or {}
                print(f"  T{n} {k} {a['agent']}: {a['tool']} {json.dumps(a.get('arguments'), ensure_ascii=False)[:120]} err={err.get('code') if isinstance(err, dict) else err} res={json.dumps(a.get('result'), ensure_ascii=False)[:160]}")
            elif k == 'grounding' and a['tool'] == 'analyze':
                res = a['result']
                print(f"  T{n} grounding: cand={res['candidates']} uniq={res['unique_target']} req={res['requested_size']} orig={res['original_size_hints']} remedy={res['remedy_cue']} cancel={res['cancel_cue']} | {res['explain'][-2:]}")
        print(f"  T{n} => {t['decision']}: {t['response'][:300]!r}")
