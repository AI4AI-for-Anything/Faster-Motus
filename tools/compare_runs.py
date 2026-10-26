"""Compares two tools/fullrun.py runs (e.g. upstream and Faster-Motus) rank by rank: every iteration's losses,
learning rates and batch digest (bitwise), the numeric validation metrics, and the final parameter / ZeRO fp32 master /
Adam moment digests; and summarizes each run's host time per iteration (rank 0).

  python tools/compare_runs.py <run A dir> <run B dir> [--json out.json]

"bitwise over N ranks: True" requires both runs to cover the same steps and to have written their final state.
"""
import json
import statistics
import sys
from pathlib import Path


def load(run, rank):
    steps, vals, meta = {}, {}, {}
    for line in (Path(run) / f'rank{rank}.jsonl').read_text().splitlines():
        r = json.loads(line)
        if 'step' in r:
            steps[r['step']] = r
        elif 'val_step' in r:
            # numeric metrics only: 'visualization' is a PIL image recorded by str(), i.e. with its memory address
            vals[r['val_step']] = {k: v for k, v in r['val'].items() if v.startswith(('0x', '-0x'))}
        else:
            meta.update(r)
    state = Path(run) / f'state_rank{rank}.json'
    return steps, vals, meta, (json.loads(state.read_text()) if state.exists() else None)


def timing(steps, bpe):
    its = [(s, r['iter_s']) for s, r in sorted(steps.items()) if r['iter_s'] is not None]
    if not its:
        return {}
    # iteration s's batch is fetched just before it; the first batch of an epoch is fetched at steps k*bpe+1
    boundary = [t for s, t in its if bpe and (s - 1) % bpe == 0]
    steady = [t for s, t in its if s > 20 and not (bpe and (s - 1) % bpe == 0)]
    total = sum(t for _, t in its)
    return {'iterations': len(its), 'total_s': round(total, 1), 'mean_s': round(total / len(its), 4),
            'steady_median_s': round(statistics.median(steady), 4) if steady else None,
            'epoch_boundary_s': [round(t, 2) for t in boundary]}


def main():
    a, b = sys.argv[1], sys.argv[2]
    report = {'a': a, 'b': b, 'ranks': {}}
    world = len(list(Path(a).glob('rank*.jsonl')))
    all_ok = True
    for rank in range(world):
        sa, va, ma, ta = load(a, rank)
        sb, vb, mb, tb = load(b, rank)
        common = sorted(set(sa) & set(sb))
        first_loss = first_lr = first_batch = None
        for s in common:
            if first_batch is None and sa[s]['batch'] != sb[s]['batch']:
                first_batch = s
            if first_loss is None and sa[s]['loss'] != sb[s]['loss']:
                first_loss = s
            if first_lr is None and sa[s]['lr'] != sb[s]['lr']:
                first_lr = s
        val_same = {s: va[s] == vb.get(s) for s in va}
        st = {}
        if ta and tb:
            pa, pb = ta['params'], tb['params']
            st = {'params': len(pa), 'params_equal': sum(pa[k] == pb.get(k) for k in pa),
                  'master_equal': ta['master'] == tb['master'], 'adam_equal': ta['adam'] == tb['adam'],
                  'global_step': (ta['global_step'], tb['global_step'])}
        ok = (first_loss is None and first_lr is None and first_batch is None and all(val_same.values()) and
              bool(st) and st['params_equal'] == st['params'] and st['master_equal'] and st['adam_equal'] and
              len(common) == len(sa) == len(sb))
        all_ok &= ok
        report['ranks'][rank] = {
            'steps': (len(sa), len(sb), len(common)), 'first_loss_diff': first_loss, 'first_lr_diff': first_lr,
            'first_batch_diff': first_batch, 'val_equal': val_same, 'state': st, 'bitwise': ok,
            'batches_per_epoch': (ma.get('batches_per_epoch'), mb.get('batches_per_epoch')),
            'dataset_len': (ma.get('dataset_len'), mb.get('dataset_len')),
            'timing_a': timing(sa, ma.get('batches_per_epoch')), 'timing_b': timing(sb, mb.get('batches_per_epoch'))}
    report['bitwise_all_ranks'] = all_ok
    r0 = report['ranks'][0]
    print(f"bitwise over {world} ranks: {all_ok}")
    for rank, r in report['ranks'].items():
        print(f"rank {rank}: steps {r['steps']} first diff loss/lr/batch {r['first_loss_diff']}/{r['first_lr_diff']}/"
              f"{r['first_batch_diff']} val {r['val_equal']} state {r['state']}")
    print('batches/epoch', r0['batches_per_epoch'], 'dataset len', r0['dataset_len'])
    print('timing A', r0['timing_a'])
    print('timing B', r0['timing_b'])
    if '--json' in sys.argv:
        Path(sys.argv[sys.argv.index('--json') + 1]).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
