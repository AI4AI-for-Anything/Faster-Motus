#!/usr/bin/env python3
"""Regenerates the change views of Faster-Motus: every tracked file that differs from upstream Motus.

  git fetch https://github.com/thu-ml/Motus.git 88f6fe30c62af033874b0a2e48ad1cbdaf12cfdf   # once
  python3 docs/change-views/generate.py

Compares upstream's commit with the working tree and writes changes.json and review-table.md next to this file
(module-map.svg is drawn by docs/performance-views/render.py); those two outputs are not listed themselves. Needs only
Git and Python's standard library; never runs GPU work. The curated descriptions live here.
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

UPSTREAM = "88f6fe30c62af033874b0a2e48ad1cbdaf12cfdf"
OUTPUTS = {"docs/change-views/changes.json", "docs/change-views/review-table.md"}
PHASES = [  # the agent's code changes are the judged snapshot (docs/source-snapshot.json); the rest is the release
    {"id": "optimization", "label": "Agent optimization (judged snapshot)",
     "paths": ["data/robotwin2/robotwin_agilex_dataset.py", "train/train.py", "utils/fastpath.py"]},
    {"id": "release", "label": "Release: preset, validation tools, tests and documentation", "paths": None},
]

FILES = {
    "train/train.py": ("Training loop", "Imports utils/fastpath.py; calls apply_all() at import (DS_HOOK) and "
                       "before_train_loop() right before the first data iterator (loader switches)."),
    "utils/fastpath.py": ("Fast paths", "DS_HOOK: per-graph-task cache of DeepSpeed's used-parameter count. "
                          "LOADER_REUSE: persistent data-loader workers re-seeded like fresh ones. LOADER_SPEC: "
                          "speculative next-epoch worker pool. MOTUS_DIAG_LOADER: a diagnostic loader watchdog."),
    "data/robotwin2/robotwin_agilex_dataset.py": ("Data", "LANG_MMAP: memory-mapped umt5 embedding lists, cached "
                                                  "per worker; marks the dataset index-free for LOADER_SPEC."),
    ".harness/config.json": ("Preset", "The selected switches as an environment mapping: DS_HOOK + LANG_MMAP."),
    "scripts/faster-motus.env.sh": ("Preset", "Shell version of the preset; lists the two switches it leaves out."),
    ".gitignore": ("Repository", "Keeps the rendered charts despite upstream's *.png rule; ignores local runs."),
    "AGENTS.md": ("Agent workflow", "Rules for agents: upstream recipe, switches default off, bitwise checks, "
                  "LOADER_SPEC off, shared-GPU etiquette, what never to commit."),
    "skills/validate-faster-motus/SKILL.md": ("Agent workflow", "Checklist to compare Faster-Motus with upstream "
                                              "bitwise and in time per iteration on a new machine."),
    "README.md": ("Documentation", "Faster-Motus overview, results and switches; upstream README kept below."),
    "README.zh-CN.md": ("Documentation", "Chinese version of the README."),
    "docs/agent-environment.md": ("Documentation", "Validated stack, assets, full-data conversion, NCCL fix, "
                                  "upstream converter workarounds and troubleshooting."),
    "docs/full-data-results.md": ("Documentation", "Full-data setup, bitwise results, timing of every switch "
                                  "set, why LOADER_SPEC is off by default, and limits."),
    "docs/release-boundary.md": ("Documentation", "What this repository contains and what stays in the research "
                                 "harness."),
    "docs/source-snapshot.json": ("Documentation", "Per-file SHA-256 of the judged source snapshot."),
    "tests/test_ds_hook.py": ("Tests", "CPU: cached count equals DeepSpeed's per-backward maximum, per graph task "
                              "and parameter list."),
    "tests/test_lang_mmap.py": ("Tests", "CPU: mapped umt5 list equals torch.load's list bitwise; cached per worker."),
    "docs/performance-views/full-data-benchmark.json": ("Results", "Full-data timing and memory measurements behind "
                                                         "the README table and figures."),
    "docs/performance-views/history.json": ("Results", "The agent's rounds R01-R04 on the benchmark fixture, with "
                                            "their timing screens and the final full check."),
    "docs/performance-views/render.py": ("Results", "Draws the README figures and the module map from the two records "
                                         "(matplotlib, no GPU)."),
    "docs/performance-views/01-benchmark.svg": ("Results", "README headline figure: baseline vs Faster-Motus "
                                                     "time per iteration (PNG copy alongside)."),
    "docs/performance-views/02-history.svg": ("Results", "Cumulative speedup over the agent's rounds R01-R04 on the "
                                              "benchmark fixture (PNG copy alongside)."),
    "docs/change-views/module-map.svg": ("Documentation", "Cards summarizing the four code changes (drawn by "
                                         "render.py; PNG copy alongside)."),
    "tools/fullrun.py": ("Validation tools", "Runs train/train.py unchanged under torchrun with per-rank seeds and "
                         "optional determinism; records every step and the final state digests."),
    "tools/compare_runs.py": ("Validation tools", "Bitwise comparison of two recorded runs, rank by rank; timing "
                              "summary."),
    "tools/run_validation.sh": ("Validation tools", "Launches one recorded run with the config's paths, steps and "
                                "switches filled in."),
    "tools/mem_sample.sh": ("Validation tools", "Samples host memory while a run trains."),
    "tools/convert_robotwin_full.py": ("Data preparation", "Runs upstream's converter on all 100 RoboTwin 2.0 "
                                       "aloha-agilex archives in parallel."),
    "tools/encode_t5_sharded.py": ("Data preparation", "Upstream's umt5 encoder over the converted data, shared by "
                                   "any number of GPU workers."),
}
DEFAULTS = [  # (path prefix, module, description) for files without their own entry
    ("docs/performance-views/", "Results", "PNG copy of the SVG figure of the same name."),
    ("docs/change-views/module-map.png", "Documentation", "PNG copy of the module map."),
    ("docs/change-views/", "Documentation", "These change views and their generator."),
]
ORDER = ["Training loop", "Fast paths", "Data", "Preset", "Agent workflow", "Validation tools", "Data preparation",
         "Tests", "Results", "Documentation", "Repository"]


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True)


def describe(path):
    if path in FILES:
        return FILES[path]
    for prefix, module, text in DEFAULTS:
        if path.startswith(prefix):
            return module, text
    raise SystemExit(f"no description for {path}; add it to FILES in {__file__}")


def phase_of(path):
    return next(p["id"] for p in PHASES if p["paths"] is None or path in p["paths"])


def collect():
    status = dict(line.split("\t", 1)[::-1] for line in
                  git("diff", "--name-status", "--no-renames", UPSTREAM).splitlines())
    files = []
    for line in git("diff", "--numstat", "--no-renames", UPSTREAM).splitlines():
        added, removed, path = line.split("\t", 2)
        if path in OUTPUTS:
            continue
        module, text = describe(path)
        blob = git("hash-object", path).strip() if status[path] != "D" else None
        files.append({"phase": phase_of(path), "module": module, "path": path, "status": status[path],
                      "added": None if added == "-" else int(added),
                      "removed": None if removed == "-" else int(removed),
                      "description": text, "source_link": f"../../{path}", "blob_sha": blob})
    files.sort(key=lambda f: ([p["id"] for p in PHASES].index(f["phase"]), ORDER.index(f["module"]), f["path"]))
    return files


def table(files):
    out = ["# What changed in Faster-Motus", "",
           f"File-by-file review against upstream Motus `{UPSTREAM[:7]}`; this table and changes.json are not listed. "
           "The [module map](module-map.svg) summarizes the code changes. "
           "Regenerate this table with `python3 docs/change-views/generate.py` and the map with "
           "`python3 docs/performance-views/render.py`.", "",
           "**Attribution:** line counts describe the diff, not runtime benefit. The measured gains of the four "
           "switches are in [the optimization history](../performance-views/history.json) and "
           "[full-data results](../full-data-results.md).", ""]
    for phase in PHASES:
        rows = [f for f in files if f["phase"] == phase["id"]]
        add = sum(f["added"] or 0 for f in rows)
        rem = sum(f["removed"] or 0 for f in rows)
        out += [f"## {phase['label']}", "",
                f"**{len(rows)} changed paths · +{add} / −{rem} lines** (binary files not counted).", "",
                "| Module | File | Change | Purpose |", "|---|---|---:|---|"]
        for f in rows:
            change = f"{f['status']} binary" if f["added"] is None else f"{f['status']} +{f['added']} / −{f['removed']}"
            out.append(f"| {f['module']} | [{f['path']}]({f['source_link']}) | {change} | {f['description']} |")
        out.append("")
    out += ["A = added; M = modified. Upstream code is otherwise unchanged; every new code path is behind a "
            "`MOTUS_OPT_*` switch that defaults to off.", ""]
    return "\n".join(out)


def main():
    files = collect()
    (HERE / "changes.json").write_text(json.dumps({"schema_version": 2, "upstream": UPSTREAM, "phases": PHASES,
                                                   "files": files},
                                                  indent=2, ensure_ascii=False) + "\n")
    (HERE / "review-table.md").write_text(table(files))
    print(f"{len(files)} files in {len(PHASES)} phases")


if __name__ == "__main__":
    main()
