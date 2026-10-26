"""umt5 embeddings of a converted RoboTwin dataset with upstream's T5 code, shared by any number of GPU workers.

Each worker uses upstream's robotwin_converter.T5EmbeddingProcessor.process_meta_file (the Wan umt5-xxl encoder,
upstream's prompt handling and file format) on the one GPU the caller makes visible. Upstream's own multi-GPU wrapper
(process_t5_batch) is bypassed: it sets CUDA_VISIBLE_DEVICES=<n> and then selects cuda:<n>, which only works for
n = 0; and its T5 import needs the repository root on PYTHONPATH.

A worker only takes (subset, task)s whose conversion finished (<logs>/<subset>_<task>.done from
tools/convert_robotwin_full.py), claims a metas file by creating <claims>/<subset>__<task>__<id> (O_EXCL, works across
hosts on a shared file system), writes the embeddings under a temporary name and renames it when complete. Claims
older than 20 minutes without an output are taken over. Workers poll while the conversion runs and exit once
<logs>/CONVERT_DONE exists and nothing is left. About 2.2 s per episode file with two workers on one GPU.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/encode_t5_sharded.py --target /data/robotwin2 \\
      --wan /models/Wan2.2-TI2V-5B --claims /data/robotwin2_t5_claims --logs logs/convert --worker host0_g0_1
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

STALE = 1200

ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
ap.add_argument('--target', type=Path, required=True)
ap.add_argument('--wan', required=True, help='directory with models_t5_umt5-xxl-enc-bf16.pth and google/umt5-xxl')
ap.add_argument('--claims', type=Path, required=True)
ap.add_argument('--logs', type=Path, required=True)
ap.add_argument('--worker', required=True)
ap.add_argument('--tree', type=Path, default=Path(__file__).resolve().parents[1])
args = ap.parse_args()
wid = args.worker
sys.path.insert(0, str(args.tree / 'data/robotwin2/robotwin_data_convert'))
import robotwin_converter as rc  # noqa: E402

args.claims.mkdir(parents=True, exist_ok=True)
proc = rc.T5EmbeddingProcessor(args.wan, 512, 'cuda:0')


def pairs():
    """(claim name, metas file, umt5 file) of the converted (subset, task)s, as upstream's collect_meta_files_for_t5."""
    out = []
    for done in sorted(args.logs.glob('*.done')):
        subset, task = done.stem.split('_', 1)
        task_dir = args.target / subset / task
        for meta in sorted((task_dir / 'metas').glob('*.txt')):
            out.append((f'{subset}__{task}__{meta.stem}', meta, task_dir / 'umt5_wan' / f'{meta.stem}.pt'))
    return out


def claim(name):
    try:
        fd = os.open(args.claims / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, wid.encode())
    os.close(fd)
    return True


def steal(name):
    """Take over a claim whose worker left no output for STALE seconds (only one renamer succeeds)."""
    c = args.claims / name
    try:
        if time.time() - c.stat().st_mtime < STALE:
            return False
        os.rename(c, args.claims / f'{name}.stale.{wid}')
    except OSError:
        return False
    return claim(name)


def encode(meta, out):
    tmp = out.with_name(f'.{out.stem}.part.{wid}.pt')
    ok = proc.process_meta_file(str(meta), str(tmp))
    if ok:
        os.rename(tmp, out)
    return ok


done_n = failed = 0
rng = random.Random(wid)
t0 = time.time()
while True:
    todo = [p for p in pairs() if not p[2].exists()]
    rng.shuffle(todo)
    worked = False
    for name, meta, out in todo:
        if not claim(name) or out.exists():
            continue
        worked = True
        if encode(meta, out):
            done_n += 1
            if done_n % 50 == 0:
                print(f'{time.strftime("%H:%M:%S")} {wid}: {done_n} done, {failed} failed, '
                      f'{(time.time() - t0) / done_n:.2f} s/file', flush=True)
        else:
            failed += 1
            print(f'FAILED {meta}', flush=True)
    if worked:
        continue
    if (args.logs / 'CONVERT_DONE').exists():
        stolen = False
        for name, meta, out in todo:
            if not out.exists() and steal(name):
                stolen = True
                if encode(meta, out):
                    done_n += 1
                else:
                    failed += 1
                    print(f'FAILED {meta}', flush=True)
        if not stolen and not [p for p in pairs() if not p[2].exists() and not (args.claims / p[0]).exists()]:
            break
    time.sleep(30)
print(f'T5_WORKER_DONE {wid}: {done_n} done, {failed} failed', flush=True)
