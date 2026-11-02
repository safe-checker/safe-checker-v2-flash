#!/usr/bin/env python3
"""Score an A/B run from LLM-judged verdicts.

Usage:
    .venv/bin/python tools/score_arms.py <result_dir> [verdicts.json]

The verdicts must come from tools/judge_detections.py; a lexical comparison
would score every paraphrase as a miss.
"""
import json, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else '/tmp/safety_eval/results/final_merged')
verdict_file = Path(sys.argv[2]) if len(sys.argv) > 2 else src/'judge_verdicts.json'
recs = [json.loads(l) for l in (src/'records.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
verd = json.loads(verdict_file.read_text(encoding='utf-8'))

# status per (sample_id, version) from records
status = {(r['sample_id'], r['version']): r['result'].get('status') for r in recs}
wall   = defaultdict(list)
for r in recs:
    wall[r['version']].append(r['wall_seconds'])

judge_by = {}
for v in verd:
    judge_by[(v['sample_id'], v['version'])] = v

def summarize(version):
    rows = [v for v in verd if v['version'] == version]
    ok = [v for v in rows if status.get((v['sample_id'], version)) == 'success']
    hits = [v for v in ok if v['judge'].get('hit')]
    # recall over the arm's own sample set: samples that produced a successful report
    n_scenes = Counter(v['sample_id'] for v in ok)
    return {
        'version': version,
        'judged': len(rows),
        'success': len(ok),
        'complete_rate': len(ok)/len(rows) if rows else 0,
        'hit': len(hits),
        'hit_rate': len(hits)/len(ok) if ok else 0,
        'wall_med': statistics.median(wall[version]) if wall[version] else 0,
        'wall_p90': sorted(wall[version])[int(len(wall[version])*0.9)-1] if wall[version] else 0,
        'wall_max': max(wall[version]) if wall[version] else 0,
        'scene_hits': Counter(v['sample_id'].split('0')[0] for v in hits),
        'scene_total': Counter(v['sample_id'] for v in ok),
    }

versions = sorted({v['version'] for v in verd})
print('='*96)
print(f'最终验证：140 张陌生数据集 × {len(versions)} 臂（LLM 裁判，hit=真命中）')
print('='*96)
for ver in versions:
    s = summarize(ver)
    print(f"\n[{s['version']}]  成功 {s['success']}/{s['judged']} ({s['complete_rate']*100:.1f}%)   "
          f"命中 {s['hit']}/{s['success']} = {s['hit_rate']*100:.1f}%   "
          f"耗时 中位 {s['wall_med']:.2f}s / p90 {s['wall_p90']:.2f}s / max {s['wall_max']:.2f}s")

# paired: scene_conc4 vs auto_conc4 on same sample_ids
a, b = 'scene_conc4', 'auto_conc4'
if a in versions and b in versions:
    ja = {v['sample_id']: v for v in verd if v['version'] == a}
    jb = {v['sample_id']: v for v in verd if v['version'] == b}
    common = sorted(set(ja) & set(jb))
    both = only_a = only_b = neither = 0
    for s in common:
        ha = bool(ja[s]['judge'].get('hit')); hb = bool(jb[s]['judge'].get('hit'))
        if ha and hb: both += 1
        elif ha: only_a += 1
        elif hb: only_b += 1
        else: neither += 1
    n = len(common)
    print('\n' + '='*96)
    print(f'配对对比（同一张图，n={n}）')
    print('='*96)
    print(f'  两臂都命中 : {both}')
    print(f'  仅指定场景命中: {only_a}')
    print(f'  仅 AUTO 命中  : {only_b}')
    print(f'  两臂都漏检  : {neither}')
    print(f'  命中率  指定场景 {both+only_a}/{n} = {(both+only_a)/n*100:.1f}%   '
          f'AUTO {(both+only_b)}/{n} = {(both+only_b)/n*100:.1f}%   '
          f'Δ = {((both+only_a)-(both+only_b))/n*100:+.1f}pp')

    print('\n  分场景命中率（指定场景 / AUTO）:')
    scenes = defaultdict(lambda: [0, 0, 0])
    for s in common:
        sc = ja[s]['sample_id']
        sc = ''.join(ch for ch in sc if not ch.isdigit())
        ha = bool(ja[s]['judge'].get('hit')); hb = bool(jb[s]['judge'].get('hit'))
        scenes[sc][0] += ha; scenes[sc][1] += hb; scenes[sc][2] += 1
    for sc, (ha, hb, tot) in sorted(((k, v) for k, v in scenes.items()), key=lambda kv: -kv[1][2]):
        print(f'    {sc:<10} {ha}/{tot} = {ha/tot*100:5.1f}%   AUTO {hb}/{tot} = {hb/tot*100:5.1f}%   Δ{(ha-hb)/tot*100:+5.1f}pp')

# per-scene for the main arm
print('\n' + '='*96)
for ver in versions:
    if ver.endswith('_gen'): continue
    rows = [v for v in verd if v['version'] == ver]
    sc = defaultdict(lambda: [0,0])
    for v in rows:
        if status.get((v['sample_id'], ver)) != 'success': continue
        k = ''.join(ch for ch in v['sample_id'] if not ch.isdigit())
        sc[k][1] += 1
        if v['judge'].get('hit'): sc[k][0] += 1
    print(f'\n[{ver}] 分场景命中：')
    for k, (h, t) in sorted(sc.items(), key=lambda kv: -kv[1][1]):
        print(f'    {k:<10} {h}/{t} = {h/t*100:5.1f}%')
