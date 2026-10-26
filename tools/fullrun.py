"""One rank of an observed Motus training run, for comparing two trees (or switch sets) bitwise.

Runs the tree's train/train.py exactly as scripts/train.sh does (same arguments, under torchrun), observed but not
changed:

- Rank r seeds Python, numpy and torch with BASE_SEED + r before train.py runs. Upstream never seeds, and torch may
  start every process from a different seed, so two unseeded runs never draw the same data, timesteps and noise.
- --deterministic: torch's deterministic algorithms (CUBLAS_WORKSPACE_CONFIG, cuDNN deterministic) and flash-attn's
  deterministic backward (upstream passes deterministic=False, whose backward accumulates with atomics).
- Every iteration, each rank appends its loss dict (exact float hex), learning rates, host time per iteration and the
  sha256 of the batch's actions and states to <out>/rank<r>.jsonl; validation metrics (rank 0, every val_interval)
  likewise.
- After the last iteration (the config's max_steps), each rank writes the sha256 of every model parameter (bf16, the
  same on all ranks), of its ZeRO fp32 master partitions and of their Adam moments to <out>/state_rank<r>.json.
  The final checkpoint (rank 0's accelerator.save_state) is skipped.

  python -m torch.distributed.run --nproc_per_node=8 ... tools/fullrun.py --tree T --out O [--deterministic] -- \
      --deepspeed configs/zero1.json --config <robotwin.yaml> --run_name robotwin --report_to tensorboard

tools/run_validation.sh launches it; tools/compare_runs.py compares two output directories.
"""
import argparse
import builtins
import hashlib
import json
import os
import random
import runpy
import sys
import time
from pathlib import Path

BASE_SEED = 67280421310721      # any fixed value; the validation runs used this one
TRAINER_CLASS = 'UniDiffuserTrainer'


def sha(t):
    import torch
    t = t.detach()
    if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()


def deterministic_flash_attn(torch):
    import flash_attn
    import flash_attn.flash_attn_interface as fai
    for name in ('flash_attn_func', 'flash_attn_varlen_func', 'flash_attn_qkvpacked_func', 'flash_attn_kvpacked_func',
                 'flash_attn_varlen_qkvpacked_func', 'flash_attn_varlen_kvpacked_func'):
        fn = getattr(fai, name, None)
        if fn is None:
            continue

        def make(fn):
            def wrapped(*args, **kwargs):
                if torch.are_deterministic_algorithms_enabled():
                    kwargs['deterministic'] = True
                return fn(*args, **kwargs)
            return wrapped
        w = make(fn)
        setattr(fai, name, w)
        if getattr(flash_attn, name, None) is fn:
            setattr(flash_attn, name, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('train_args', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    train_args = a.train_args[1:] if a.train_args[:1] == ['--'] else a.train_args
    rank = int(os.environ['RANK'])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / f'rank{rank}.jsonl', 'a', buffering=1)

    import numpy as np
    import torch
    if a.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        deterministic_flash_attn(torch)
    random.seed(BASE_SEED + rank)
    np.random.seed((BASE_SEED + rank) % 2**32)
    torch.manual_seed(BASE_SEED + rank)

    tree = Path(a.tree).resolve()
    entry = tree / 'train' / 'train.py'
    original_build = builtins.__build_class__
    last = {'t': None}

    def wrap(cls):
        real_step, real_train = cls.__dict__['train_step'], cls.__dict__['train']

        def train_step(trainer, batch):
            now = time.time()
            iter_s = None if last['t'] is None else now - last['t']
            last['t'] = now
            h = hashlib.sha256()
            for k in ('action_sequence', 'initial_state'):
                v = batch.get(k)
                if v is not None:
                    h.update(v.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
            metrics = real_step(trainer, batch)
            log.write(json.dumps({
                'step': trainer.global_step + 1, 'iter_s': iter_s, 'step_s': time.time() - now,
                'loss': {k: float(v).hex() for k, v in metrics.items()},
                'lr': [float(g['lr']).hex() for g in trainer.optimizer.param_groups],
                'batch': h.hexdigest()[:16]}) + '\n')
            return metrics

        def save_checkpoint(trainer, suffix=''):
            log.write(json.dumps({'skipped_checkpoint': trainer.global_step}) + '\n')

        def train(trainer, *args, **kwargs):
            g = real_train.__globals__
            real_eval = g['evaluate_model']

            def evaluate_model(*eargs, **ekw):
                m = real_eval(*eargs, **ekw)
                log.write(json.dumps({'val_step': trainer.global_step,
                                      'val': {k: (float(v).hex() if isinstance(v, (int, float)) or
                                                  torch.is_tensor(v) else str(v)) for k, v in m.items()}}) + '\n')
                return m
            g['evaluate_model'] = evaluate_model
            dl = trainer.train_dataloader
            log.write(json.dumps({'start': time.time(), 'batches_per_epoch': len(dl),
                                  'dataset_len': len(dl.dataset) if hasattr(dl, 'dataset') else None,
                                  'env': {k: v for k, v in os.environ.items() if k.startswith('MOTUS_OPT_')},
                                  'deterministic': a.deterministic}) + '\n')
            result = real_train(trainer, *args, **kwargs)
            log.write(json.dumps({'end': time.time(), 'global_step': trainer.global_step}) + '\n')
            dump_state(trainer)
            return result

        cls.train_step, cls.train, cls.save_checkpoint = train_step, train, save_checkpoint

    def dump_state(trainer):
        torch.cuda.synchronize()
        engine = trainer.accelerator.deepspeed_engine_wrapped.engine
        zopt = engine.optimizer
        module = engine.module
        state = {'global_step': trainer.global_step, 'params': {}, 'master': [], 'adam': []}
        for name, p in module.named_parameters():
            state['params'][name] = sha(p)
        for part in zopt.single_partition_of_fp32_groups:
            state['master'].append(sha(part))
            st = zopt.optimizer.state.get(part, {})
            state['adam'].append({k: (sha(v) if torch.is_tensor(v) and v.numel() > 1 else
                                      (float(v).hex() if torch.is_tensor(v) or isinstance(v, (int, float)) else str(v)))
                                  for k, v in sorted(st.items())})
        (out / f'state_rank{rank}.json').write_text(json.dumps(state, indent=0))
        log.write(json.dumps({'state_dumped': time.time()}) + '\n')

    def build_class(func, name, *bases, **kw):
        cls = original_build(func, name, *bases, **kw)
        if name == TRAINER_CLASS and func.__globals__.get('__file__') == str(entry):
            builtins.__build_class__ = original_build
            wrap(cls)
        return cls

    sys.path.insert(0, str(entry.parent))
    os.chdir(tree)
    sys.argv = [str(entry), *train_args]
    builtins.__build_class__ = build_class
    rc = 0
    try:
        runpy.run_path(str(entry), run_name='__main__')
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
    except BaseException:
        import traceback
        traceback.print_exc()
        rc = 1
    log.write(json.dumps({'exit': rc}) + '\n')
    log.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == '__main__':
    main()
