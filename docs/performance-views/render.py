#!/usr/bin/env python3
"""Render the crimson README figures (the Faster-LLaVA style) from the recorded measurements. No GPU work.

  python docs/performance-views/render.py

Reads full-data-benchmark.json and history.json next to this file; writes 01-benchmark, 02-history and
../change-views/module-map as SVG and PNG.
"""
from pathlib import Path
import json
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'font.size': 10})
RED = '#8C0000'; MUTED = '#79696B'; INK = '#292122'; LIGHT = '#D7BDBF'; CARD = '#FAF5F5'; GRID = '#EADFE0'


def save(fig, path):
    for ext in ['svg', 'png']:
        p = path.with_suffix('.' + ext)
        fig.savefig(p, dpi=170, facecolor='white')
        if ext == 'svg':
            p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines()) + '\n')
    plt.close(fig)


def frame(ax, xlabel):
    ax.tick_params(length=0, labelsize=9, colors=MUTED, pad=10)
    for k in ['top', 'right', 'left']:
        ax.spines[k].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.grid(axis='x', color=GRID, lw=.7, zorder=0)
    ax.set_xlabel(xlabel, color=MUTED, labelpad=13)


def hero():
    d = json.loads((ROOT / 'full-data-benchmark.json').read_text())
    base, fast = d['headline']['baseline_s'], d['headline']['faster_s']
    fig = plt.figure(figsize=(9, 4.5))
    fig.text(.06, .90, 'Faster-Motus, optimized by AI agents', color=RED, weight='bold', size=19)
    fig.text(.06, .80, 'Motus · 8 GPUs · global batch 64 · full RoboTwin 2.0 (27,244 episodes)', color=MUTED, size=10)
    ax = fig.add_axes([.23, .24, .40, .46])
    ax.barh([1, 0], [base, fast], color=[LIGHT, RED], height=.24, zorder=3)
    ax.set_yticks([1, 0], ['Baseline', 'Faster-Motus'], color=MUTED)
    for y, v in zip([1, 0], [base, fast]):
        ax.text(v + .2, y, f'{v:.3f} s', va='center', color=INK, size=10, weight='bold')
    ax.set_xlim(0, 10.2); ax.set_ylim(-.6, 1.6); ax.set_xticks([0, 2, 4, 6, 8])
    frame(ax, 'Time per training iteration (s) ↓')
    fig.text(.76, .55, f"{d['headline']['speedup']:.2f}×", color=RED, size=33, weight='bold')
    fig.text(.76, .45, 'iteration speedup', color=MUTED, size=11)
    fig.text(.06, .055, 'Iterations 21–150 of each run. 1,100 deterministic iterations are bitwise identical to the baseline '
             'on all 8 ranks.', color=MUTED, size=9)
    save(fig, ROOT / '01-benchmark')


def history():
    rows = json.loads((ROOT / 'history.json').read_text())['rows']
    release = json.loads((ROOT / 'full-data-benchmark.json').read_text())['headline']['speedup']
    height = 7.0
    top = lambda inches: 1 - inches / height
    fig = plt.figure(figsize=(9, height), facecolor='white')
    fig.text(.09, top(.36), 'Faster-Motus', size=11, weight='bold', color=RED)
    fig.text(.09, top(.79), 'Agent Optimization History', size=21.5, weight='bold', color=INK)
    fig.text(.94, top(.65), f'{release:.2f}×', size=29, ha='right', weight='bold', color=RED)
    fig.text(.94, top(.88), 'released preset · full data', size=10.5, ha='right', color=MUTED)
    fig.text(.09, top(1.31), 'Cumulative speedup · 8 GPUs · benchmark fixture, 9 iterations per epoch', size=11, color=MUTED)
    ax = fig.add_axes([.09, top(4.19), .85, 2.61 / height])
    x = list(range(len(rows))); y = [v['speedup'] for v in rows]
    tags = [f"R{v['round']:02d}" for v in rows]
    ax.plot(x, y, color=RED, lw=2.3, zorder=2)
    for xi, v, row in zip(x, y, rows):
        ax.plot([xi], [v], marker='o', ms=6.5, zorder=3, color=RED, mfc=RED if row['check'] == 'full' else 'white', mew=1.8)
    for i, v in enumerate(y):
        ax.annotate(f'{v:.2f}×', (i, v), xytext=(0, 11 if i % 2 == 0 else -23), textcoords='offset points', ha='center',
                    size=12, color=INK, weight='bold')
    ax.set_xticks(x, tags); ax.set_yticks([6, 7, 8, 9, 10, 11]); ax.yaxis.set_major_formatter(lambda v, _: f'{v:.1f}×')
    ax.set_ylim(5.5, 11.3); ax.set_xlim(-.5, len(rows) - .5)
    ax.spines[['top', 'right', 'left']].set_visible(False); ax.spines['bottom'].set_color(GRID)
    ax.tick_params(length=0, labelsize=11.5, colors=MUTED, pad=9); ax.grid(axis='y', color=GRID, lw=.7, zorder=0); ax.set_axisbelow(True)
    for i, (row, tag) in enumerate(zip(rows, tags)):
        bx, by = .09 + (i % 2) * .44, top(5.18 + (i // 2) * .74)
        fig.add_artist(FancyBboxPatch((bx, by - .378 / height), .41, .585 / height, boxstyle='round,pad=0.004,rounding_size=0.007',
                                      transform=fig.transFigure, facecolor=CARD, edgecolor='none'))
        fig.text(bx + .014, by - .08 / height, tag, size=12, weight='bold', color=RED)
        fig.text(bx + .079, by - .08 / height, textwrap.fill(row['label'], width=24, break_long_words=False, break_on_hyphens=False),
                 size=11.5, color=INK, weight='bold', va='center', linespacing=1.35)
        if not row['in_release']:
            fig.text(bx + .40, by - .31 / height, 'off by default', size=9, color=MUTED, ha='right', va='center')
    fig.text(.09, top(6.75), 'Open marker: timing screen. Filled: full check (bitwise). Cumulative configurations, not isolated ablations.\n'
             'R02 and R04 remove the per-epoch loader restart, frequent only on this fixture; the release keeps R01 + R03.',
             size=9.5, color=MUTED, linespacing=1.4)
    save(fig, ROOT / '02-history')


def modulemap():
    fig = plt.figure(figsize=(9, 6.6), facecolor='white'); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(.055, .92, 'What changed in Faster-Motus', size=23, color=INK, weight='bold')
    cards = [('DeepSpeed gradient hooks', 'Used-parameter count cached per graph task\nO(n²) Python work per backward → O(n)',
              'MOTUS_OPT_DS_HOOK\nutils/fastpath.py', 'preset'),
             ('umt5 text embeddings', 'Memory-mapped embedding files\nParsed list kept per loader worker',
              'MOTUS_OPT_LANG_MMAP\ndata/robotwin2/robotwin_agilex_dataset.py', 'preset'),
             ('Data-loader workers', 'Workers kept across epochs, re-seeded\nSpeculative next epoch (not recommended)',
              'MOTUS_OPT_LOADER_REUSE / LOADER_SPEC\nutils/fastpath.py', 'off by default'),
             ('Training loop', 'Switches installed at import\nand before the first data iterator', 'train/train.py', '4 lines')]
    for i, (title, desc, path, count) in enumerate(cards):
        x, y = .055 + (i % 2) * .455, .49 - (i // 2) * .39
        ax.add_patch(FancyBboxPatch((x, y), .425, .34, boxstyle='round,pad=0.009,rounding_size=0.012', facecolor=CARD, edgecolor=GRID))
        ax.plot([x, x + .425], [y + .34, y + .34], color=RED, linewidth=2, solid_capstyle='round')
        ax.text(x + .014, y + .30, f'{i + 1:02d}', size=11, color=RED, weight='bold'); ax.text(x + .41, y + .30, count, size=10, color=MUTED, ha='right')
        ax.text(x + .014, y + .254, title, size=13, color=INK, weight='bold'); ax.text(x + .014, y + .198, desc, size=10.3, color=INK, va='top', linespacing=1.6)
        ax.text(x + .014, y + .028, path, size=9.5, color=MUTED, linespacing=1.5, va='bottom')
    save(fig, ROOT.parent / 'change-views/module-map')


if __name__ == '__main__':
    hero(); history(); modulemap()
