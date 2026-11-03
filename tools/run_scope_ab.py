#!/usr/bin/env python3
"""A/B the scene checklist scope on the 140-image unfamiliar set.

Run it from the repository root:
    .venv/bin/python tools/run_scope_ab.py <out_dir> [images_per_scene]

EVAL_SET points at another image set with the same labels.jsonl layout.
The records carry the error type of every branch, because a batch that died of
rate limiting must never be read as a batch that found nothing.

Arms:
  split1  : current production - scene subjects split across 4 concurrent branches
  full1   : full scene checklist, one single call
  full4   : full scene checklist, 4 concurrent branches (each sees everything)

Every branch's error type is recorded, and an image whose branches all failed is
retried once with a single branch so the record is not silently empty.
"""
import difflib, json, os, subprocess as sp, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PY_BIN = str(BASE / '.venv' / 'bin' / 'python')
CHECKER = str(BASE)
SET = Path(os.environ.get('EVAL_SET',
                    '/Users/gjz/Documents/ChatGPT/复杂环境安全巡检/评测数据/施工安全图集_140张'))

ARMS = {
    'split1':  dict(scope='split',  groups=1),
    'full1':   dict(scope='full',   groups=1),
    'full4':   dict(scope='full',   groups=4),
    'hybrid1': dict(scope='hybrid', groups=1),
    'hybrid4': dict(scope='hybrid', groups=4),
    'family4':  dict(scope='family',  groups=4),
    'family1':  dict(scope='family',  groups=1),
    'famdims4': dict(scope='family_dims', groups=4),
}
# 上游在持续压测下会变慢（超时率从 8% 涨到 50%），图与图之间留一点间隔
PACE_SECONDS = float(os.environ.get('EVAL_PACE_SECONDS', '1.5'))
# 只跑需要的臂：EVAL_ARMS=hybrid1,hybrid4
WANTED = [name.strip() for name in os.environ.get('EVAL_ARMS', '').split(',') if name.strip()]


def run_branch(image, scene, g, total, scope):
    env = dict(os.environ, REQUEST_TIMEOUT_SECONDS='8', MODEL_MAX_ATTEMPTS='3',
               MODEL_RETRY_MIN_BUDGET='1.0', MODEL_CONNECT_TIMEOUT='2.0',
               IMAGE_DETAIL='low', MAX_IMAGE_EDGE='768',
               CHECKLIST_AUTO_MODE='subject', CHECKLIST_SUBJECT_GROUP=f'{g}/{total}',
               SCENE_CHECKLIST_SCOPE=scope, USE_GENERATED_RULES='0')
    cmd = [PY_BIN, f'{CHECKER}/src/safety_check.py', str(image), '--json-output', '--no-save']
    if scene:
        cmd += ['--scene', scene]
    t0 = time.perf_counter()
    p = sp.run(cmd, cwd=CHECKER, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    try:
        d = json.loads(p.stdout[p.stdout.index('{'):])
    except ValueError:
        return {'status': 'error', 'err': 'unparsable', 'wall': round(wall, 2),
                'msg': p.stdout[-120:] + p.stderr[-120:], 'usage': {}}
    err = d.get('error') or {}
    return {'status': d.get('status'), 'err': err.get('type') or '',
            'msg': str(err.get('message') or '')[:200], 'wall': round(wall, 2),
            'issues': d.get('issues') or [], 'usage': d.get('token_usage') or {}}


def merge(issues):
    merged, seen = [], set()
    for it in issues:
        if not isinstance(it, dict):
            continue
        key = ''.join(ch for ch in str(it.get('item') or '') if ch.isalnum())
        dup = next((o for o in merged if difflib.SequenceMatcher(
            None, key, ''.join(ch for ch in str(o.get('item') or '') if ch.isalnum())).ratio() >= 0.78), None)
        if dup is None:
            merged.append(it)
        elif float(it.get('confidence') or 0) > float(dup.get('confidence') or 0):
            dup.update(it)
    merged.sort(key=lambda it: ({'CLEAR': 0, 'SUSPECTED': 1, 'NOT_ASSESSABLE': 2}.get(
        str(it.get('status')).upper(), 3), -float(it.get('confidence') or 0)))
    return merged[:8]


def main():
    outdir = Path(sys.argv[1]); outdir.mkdir(parents=True, exist_ok=True)
    per_scene = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    path = outdir / 'records.jsonl'
    done = set()
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line); done.add((r['sample_id'], r['version']))
    fh = path.open('a', encoding='utf-8')
    labels = [json.loads(l) for l in (SET / 'labels.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    # 分层等距抽样：每个场景按间隔取，而不是取前 N 张。
    # 前 N 张会整段落在同一批子目录上（例如施工用电只取到"用电01-20"，
    # "补电"那十张一张都进不来），样本有偏，等距取才覆盖得住场景里的各种情况。
    picked = []
    by_scene: dict[str, list] = {}
    for lab in labels:
        by_scene.setdefault(lab['test_scene'], []).append(lab)
    for scene, items in by_scene.items():
        if per_scene >= len(items):
            picked.extend(items)
            continue
        stride = len(items) / per_scene
        for i in range(per_scene):
            picked.append(items[int(i * stride)])
    arms = len(WANTED) if WANTED else len(ARMS)
    print(f'取样 {len(picked)} 张（每场景 {per_scene} 张），共 {len(picked)*arms} 组', file=sys.stderr)
    for i, lab in enumerate(picked):
        image = SET / lab['image_path']; scene = lab['test_scene']
        for name, cfg in ARMS.items():
            if WANTED and name not in WANTED:
                continue
            if (lab['sample_id'], name) in done:
                continue
            groups, scope = cfg['groups'], cfg['scope']
            t0 = time.perf_counter()
            with ThreadPoolExecutor(groups) as ex:
                outs = list(ex.map(lambda g: run_branch(image, scene, g, groups, scope), range(groups)))
            wall = time.perf_counter() - t0
            fallback = ''
            if all(o['status'] != 'success' for o in outs):
                # 全路失败：再单路试一次，别让这条记录静默变空
                outs = [run_branch(image, scene, 0, 1, scope)]
                fallback = 'single_retry'
            # 余额耗尽要立刻停，否则后面的调用全是空转
            if any('insufficient_balance' in (o.get('err') or '') for o in outs):
                print('余额不足，停跑', file=sys.stderr)
                fh.close()
                raise SystemExit(3)
            issues = [i for o in outs if o['status'] == 'success' for i in o['issues']]
            payload = {'status': 'success' if issues else 'error',
                       'issues': merge(issues),
                       'scene': scene,
                       'model_detected_scene': '',
                       'branch_status': [o['status'] for o in outs],
                       'branch_errors': [o.get('err') or '' for o in outs],
                       'branch_messages': [o.get('msg') or '' for o in outs],
                       'branch_wall': [o['wall'] for o in outs],
                       'usage': {k: sum(int(o.get('usage', {}).get(k) or 0) for o in outs)
                                 for k in ('prompt_tokens', 'completion_tokens',
                                           'cache_hit_tokens', 'cache_miss_tokens')},
                       'fallback': fallback}
            rec = {'sample_id': lab['sample_id'], 'version': name, 'budget_seconds': 8.0,
                   'test_scene': scene, 'image_path': str(image),
                   'wall_seconds': round(wall, 4), 'result': payload, 'branches': groups,
                   'scope': scope}
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); fh.flush()
            errs = ','.join(e for e in payload['branch_errors'] if e)
            if PACE_SECONDS:
                time.sleep(PACE_SECONDS)
            print(f"[{i+1:>3}] {name:<7} {lab['sample_id']:<10} {wall:5.2f}s "
                  f"issues={len(payload['issues']):>2} {errs}{' FALLBACK' if fallback else ''}",
                  file=sys.stderr, flush=True)
    fh.close()


if __name__ == '__main__':
    main()
