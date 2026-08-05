# Contributing to CellScope

Thanks for your interest. CellScope is research software built for a working
microscopy lab, and contributions — bug reports especially — are welcome from
anyone, whether or not you write Python.

Please note that everyone interacting in this project is expected to be
respectful and constructive. Reports of unacceptable behaviour can be sent to
george.dickinson@gmail.com.

## Getting help

If something is unclear, that is a documentation bug and worth reporting.

- **Questions about using CellScope** — open a
  [GitHub issue](https://github.com/gddickinson/cellscope/issues) with the
  `question` label. Please don't email; answering in the open helps the next
  person with the same problem.
- **General bioimage-analysis questions** (Cellpose behaviour, segmentation
  strategy, microscopy setup) are often better asked on the
  [image.sc forum](https://forum.image.sc/), which has a much larger community.

## Reporting a bug

Open an issue and include, as far as you can:

1. **What you ran** — the GUI and the button you pressed, or the exact command.
2. **What happened** versus what you expected.
3. **The recording** — modality (DIC / phase-contrast / multi-channel), pixel
   size in µm, frame count, and roughly how many cells per frame.
4. **`RUN_METADATA.json`** from the analysis run, if one was produced. Every
   run writes one, and it captures every parameter, the environment versions,
   the git commit, and the exact command to reproduce. Attaching it answers
   most of the questions we would otherwise have to ask.
5. **Your platform** — OS, whether you are on CUDA / MPS / CPU, and the output
   of `conda list cellpose torch`.

A recording that reproduces the problem is enormously helpful, but please only
share data you have the right to share. A single cropped frame is often enough.

## Suggesting a feature

Open an issue describing the biological or analysis question you are trying to
answer, not only the feature you have in mind. CellScope's pipeline is fairly
opinionated and there is often an existing route to the same result — and if
there isn't, the underlying question shapes the right design.

## Contributing code

### Setting up

CellScope needs **two conda environments**: `cellpose` (CP3, the GUIs, and the
analysis pipeline) and `cellpose4` (the cpsam ViT backbone, invoked by
subprocess). The install scripts create both:

```bash
git clone https://github.com/gddickinson/cellscope.git
cd cellscope
./install.sh          # install.bat on Windows
python download_models.py
```

Then, for development, install the package in editable mode inside the
`cellpose` env:

```bash
conda activate cellpose
pip install -e ".[dev]"
```

See the scope note at the top of `pyproject.toml` — the conda scripts remain
the canonical install path, and a plain `pip install cellscope` is not yet
supported.

### Running the tests

All tests run headless under `QT_QPA_PLATFORM=offscreen`:

```bash
conda run -n cellpose4 python scripts/test_focused_gui.py        # Phase A
conda run -n cellpose4 python scripts/test_comprehensive_gui.py  # Phases B-G
python scripts/aggregate_comprehensive_report.py                  # merge report
conda run -n cellpose4 python scripts/test_defaults_consistency.py
```

There are 107 checks across seven phases covering six GUIs. **Please run them
before opening a pull request**, and say in the PR which phases you ran. If a
change touches the detection or tracking pipeline, also run
`scripts/test_pipeline_regression.py` and report the resulting IoU / F1 against
the ground-truth recordings — a change that improves one recording and quietly
degrades another is the failure mode we most want to catch.

### House style

These conventions are enforced by habit rather than by a linter, but please
follow them — they exist so that both a human and a language model can navigate
the codebase months later.

- **Keep every file under 500 lines.** If a change would push a file over,
  split it into focused modules first.
- **Update `INTERFACE.md`.** It is the navigation map for the whole project:
  what each module contains, which functions live where, how the pieces
  connect. A structural change that doesn't update it is incomplete.
- **One source of truth for defaults.** Every pipeline parameter lives in
  `core/pipeline_defaults.py`, and every GUI and worker reads from there. Never
  hardcode a default in a GUI — that drift is the single most common bug class
  in this codebase and has bitten us at least three times.
- **Detection changes go through `core/unified_detection.py`.** Both the GUI
  and the command-line runner call it, which is what guarantees they produce
  identical output. Adding a detection path that bypasses it breaks that
  guarantee.
- Match the surrounding code's naming and comment density. Comment constraints
  the code can't express, not what the next line does.

### Pull requests

1. Branch from `main`.
2. Keep the change focused — one concern per PR.
3. Describe **what biological or analysis problem it solves**, not only what it
   changes.
4. Report the tests you ran and any benchmark movement.
5. Note any new dependency and why it's needed. Dependencies here are heavy
   (torch, cellpose, SAM) and each one raises the install cost for everyone.

## Validation data

Benchmarks quoted in the README come from ground-truth recordings that are not
all redistributable. If you need to validate against them, open an issue and we
can discuss what can be shared. Public benchmarks — for instance the Cell
Tracking Challenge `DIC-C2DH-HeLa` set used for the tracker comparison — are a
good target for any contribution that claims a tracking improvement.

## Citing

If CellScope contributes to published work, please cite it. `CITATION.cff` in
the repository root has the metadata, and GitHub's "Cite this repository"
button will format it for you.

## Licence

By contributing you agree that your contributions are licensed under the
project's [MIT Licence](LICENSE).
