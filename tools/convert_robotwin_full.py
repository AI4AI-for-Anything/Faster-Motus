"""Converts the full RoboTwin 2.0 aloha-agilex data with upstream's converter, in parallel.

Every episode of the aloha-agilex_clean_50.zip and aloha-agilex_randomized_500.zip archives of the 50 tasks (the files
data/robotwin2/robotwin_data_convert/download_robotwin_dataset.py downloads) is converted by
robotwin_converter.py with the values of its config.yml (fps 30, 320x360), one (subset, task) per converter run. The
converter numbers a task's episodes by their position in the task's sorted episode list, so per-task runs write
exactly what one whole-dataset run writes. Each job extracts its archive's data/ and instructions/ into the scratch
directory (about 9 GB per randomized task; /dev/shm is fast), converts into the target, and deletes the extracted copy.
The umt5 embeddings are a separate GPU step: tools/encode_t5_sharded.py.

  python tools/convert_robotwin_full.py --archives /data/robotwin2.0 --target /data/robotwin2 \\
      --scratch /dev/shm/robotwin_raw --logs logs/convert --jobs 24

<logs>/<subset>_<task>.done marks a finished job (a rerun skips it); <logs>/CONVERT_DONE marks the end. Episodes whose
joint positions exceed |4| are dropped by the converter itself (27,244 of 27,500 remain).
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SUBSETS = {'clean': 'aloha-agilex_clean_50', 'randomized': 'aloha-agilex_randomized_500'}
CONFIG = '''source_root: "{source}"
target_root: "{target}"
fps: 30
target_width: 320
target_height: 360
max_workers: 16
skip_existing: true
enable_t5_embeddings: false
wan_repo_path: "unused"
t5_max_length: 512
cuda_devices: ["0"]
log_level: "WARNING"
save_debug_frames: false
'''


def job(args, subset, task):
    done = args.logs / f'{subset}_{task}.done'
    if done.exists():
        return f'{subset}/{task} already done'
    stem = SUBSETS[subset]
    root = args.scratch / f'{subset}_{task}'
    shutil.rmtree(root, ignore_errors=True)
    dest = root / subset / task
    t0 = time.time()
    with zipfile.ZipFile(args.archives / task / f'{stem}.zip') as z:
        for name in z.namelist():
            parts = name.split('/')
            if len(parts) == 3 and parts[0] == stem and parts[1] in ('data', 'instructions') and parts[2]:
                out = dest / parts[1] / parts[2]
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as f, open(out, 'wb') as g:
                    shutil.copyfileobj(f, g, 1 << 22)
    n = len(list((dest / 'data').glob('*.hdf5')))
    t1 = time.time()
    cfg = root / 'config.yml'
    cfg.write_text(CONFIG.format(source=root, target=args.target))
    env = dict(os.environ, PYTHONPATH=str(args.tree), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2')
    with open(args.logs / f'{subset}_{task}.log', 'w') as log:
        r = subprocess.run([sys.executable, 'robotwin_converter.py', '--config', str(cfg)],
                           cwd=args.tree / 'data/robotwin2/robotwin_data_convert', env=env, stdout=log,
                           stderr=subprocess.STDOUT)
    shutil.rmtree(root, ignore_errors=True)
    out = args.target / subset / task
    made = len(list((out / 'metas').glob('*.txt'))) if (out / 'metas').exists() else 0
    msg = f'{subset}/{task}: {n} episodes, {made} converted, extract {t1 - t0:.0f} s, convert {time.time() - t1:.0f} s'
    if r.returncode == 0:
        done.write_text(msg + '\n')
    else:
        msg += f' FAILED rc={r.returncode}'
    return msg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--archives', type=Path, required=True, help='RoboTwin 2.0 root with <task>/aloha-agilex_*.zip')
    ap.add_argument('--target', type=Path, required=True, help="dataset directory (configs/robotwin.yaml's dataset_dir)")
    ap.add_argument('--scratch', type=Path, required=True)
    ap.add_argument('--logs', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=24)
    ap.add_argument('--tree', type=Path, default=Path(__file__).resolve().parents[1], help='tree with the converter')
    args = ap.parse_args()
    args.target, args.tree = args.target.resolve(), args.tree.resolve()
    tasks = sorted(p.name for p in args.archives.iterdir() if (p / f'{SUBSETS["clean"]}.zip').exists())
    assert len(tasks) == 50, f'{len(tasks)} tasks with RoboTwin 2.0 aloha-agilex archives in {args.archives}'
    for d in (args.target, args.scratch, args.logs):
        d.mkdir(parents=True, exist_ok=True)
    work = [('randomized', t) for t in tasks] + [('clean', t) for t in tasks]
    failed = 0
    with ThreadPoolExecutor(args.jobs) as pool:
        for msg in pool.map(lambda w: job(args, *w), work):
            failed += 'FAILED' in msg
            print(time.strftime('%H:%M:%S'), msg, flush=True)
    print('CONVERT_DONE' if not failed else f'CONVERT_FAILED {failed}', flush=True)
    if not failed:
        (args.logs / 'CONVERT_DONE').write_text('ok\n')


if __name__ == '__main__':
    main()
