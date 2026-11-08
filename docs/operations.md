# Operations

How to set up the environment and run experiments. The design behind these procedures is in
[architecture.md](architecture.md).

## Environment and drivers

- **Compute:** all main GPU work (stage checks on real checkpoints, calibration and experimental
  runs) runs on Runpod, on the GPU classes listed under [compute](plan.md#compute). Local machines
  run the tests, documentation checks and the analysis; a local GPU may run the quick tiny-model
  CUDA tests (`tests/test_gpu.py`, skipped without a GPU).
- **uv:** requires [uv](https://docs.astral.sh/uv/), which installs Python 3.12 if needed.
- **Lock file:** `uv.lock` pins the dependencies for Linux x86_64.
- **CUDA:** PyTorch comes from PyPI, built for CUDA 13, so GPU machines need an NVIDIA driver that
  supports CUDA 13 (version 580 or newer). Choose rented machine images accordingly.
- **GPU check:** entry points fail fast if CUDA is unavailable. With an old driver, PyTorch would
  otherwise fall back to the CPU silently.
- **No vLLM:** never add it to this environment. It re-pins torch and would change the numerics of
  the optimizer under study. The open-weights judge runs as a separate server behind the
  OpenAI-compatible endpoint.
- **Install mode:** `uv sync` installs the package in editable mode. Container runs use the
  source checkout selected by `project-run`, as described under [Containers](#containers).

## Setup

```bash
uv sync                  # create .venv from uv.lock
cp .env.example .env     # fill in the keys you use
uv run pytest            # unit tests (CPU)
```

Documentation checks run without a commit and without a GPU:

```bash
uv run python scripts/check_docs.py
```

See the [maintenance workflow](README.md#maintenance-rules) for ownership, generated indexes and
change-impact review.

## Containers

The [Dockerfile](../Dockerfile) builds a Linux x86_64 dependency image from this project's
`pyproject.toml` and `uv.lock`. The image contains a private Python environment at `/opt/venv`;
source code is supplied separately. A code edit does not rebuild the dependency layer.
[.dockerignore](../.dockerignore) limits the build context to dependency manifests and the launcher.
The base image supplies SSH; its startup command is retained.

Build from the repository root:

```bash
docker build --platform linux/amd64 -t zero-field-refusal:env-v1 .
```

Run the offline CPU smoke check using a read-only checkout:

```bash
docker run --rm --network none --entrypoint project-run \
  --mount type=bind,src="$PWD",dst=/workspace/code/zero-field-refusal,readonly \
  zero-field-refusal:env-v1 /workspace/code/zero-field-refusal \
  python scripts/check_runtime.py --cpu
```

GPU checks run on Runpod, without `--cpu`: the check on each target GPU class is required before
relying on this image for research there. The Pod's host driver must support CUDA 13 (580 or
newer); it is not installed into the image. This tiny random-model check downloads no weights and
is not a scientific experiment or a full validation of the research pipeline.

On a Runpod Secure Cloud on-demand Pod, use full SSH over the public IP and mapped TCP port 22.
Upload each immutable source package under `/workspace/code/zero-field-refusal/<snapshot-id>/`,
with the checkout in `source/` and its manifest alongside it. Derive the snapshot ID from the
canonical file manifest and verify content, inventory and modes after transfer and before launch.
For example, set `SOURCE` to that verified checkout and run:

```bash
SOURCE="/workspace/code/zero-field-refusal/<snapshot-id>/source"
project-run "$SOURCE" python scripts/check_runtime.py
```

Replace `<snapshot-id>` with the actual ID. Keep wrapper logs/provenance under
`/workspace/attempts/zero-field-refusal/<run-id>`, outside the source and tracker-owned run directory.
Pass `RUN_ID` to the experiment, letting its tracker create the run directory. Pass the snapshot ID
as `SOURCE_SNAPSHOT_ID` and the image digest as `CONTAINER_IMAGE`; the tracker records both in
`meta.json`. Keep the snapshot receipt in wrapper evidence and retain the snapshot itself: it is
the executed source of a development run. The baked `project-run` checks dependency files only, so
source-manifest verification is a separate launch step.

Development runs use these source snapshots with `tracking.require_clean_git: false`. Gate runs
use a real checkout of the required commit, from a clone of the private GitHub repository or a Git
bundle, with `tracking.require_clean_git: true`.

The [launcher](../docker/project-run.py) checks that both dependency manifests match the image,
selects its Python environment, and imports the selected checkout through `PYTHONPATH`. It
preserves repository-relative config lookup without installing a second copy of the package.
Use `python`, `torchrun` or other installed commands through this launcher. Dependency changes
require rebuilding the image; do not replace the baked environment with ad-hoc installs.

By default, data and outputs go to `/workspace/data/zero-field-refusal` and
`/workspace/runs/zero-field-refusal`; `DATA_DIR` and `RUNS_DIR` can override these locations.
Model and other runtime caches use `/workspace/cache`. Attach persistent storage at `/workspace`
and back up results using the existing rented-machine procedure. The environment itself lives
outside that mount in the image, so replacing a Pod does not require reinstalling its packages.

Code transfer must include newly added source files and exclude local environments, secrets and
ignored working files. Keep each active run's source snapshot unchanged. Development may use
uncommitted snapshots, but gate runs still require a clean Git checkout and a recorded commit.
No Git credentials or `.git` directory are baked into the image.

Publish tested images to the selected container registry with a new version tag for each release;
record the image digest used for a run. Configure registry authentication separately for private
images. Container builds and source synchronization do not publish Git commits.

## `.env`

- `.env.example` describes every variable; copy it to `.env` and uncomment what you use.
- The package reads `.env` from the repository root when it is imported. Variables already set in
  the shell win, and blank values are ignored.

| Purpose | Variables |
|---|---|
| Weights & Biases | `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_MODE` |
| Hugging Face Hub | `HF_TOKEN`, `HF_HOME` (model cache) |
| Judge endpoint | `JUDGE_API_BASE`, `JUDGE_API_KEY`, `JUDGE_MODEL` |
| Storage | `DATA_DIR`, `RUNS_DIR` (defaults: `./data`, `./runs`) |
| Archive | `ARTIFACT_REMOTE` (a `gs://` bucket path), `ARTIFACT_GCS_KEY` (service-account key file) |

## Configs on the command line

Every entry point takes a config and optional `key=value` overrides; the override rules are in
[architecture.md](architecture.md#configuration). The entry points are listed under
[modules](architecture.md#modules); `<entry point>` stands for one of them.

```bash
# the smallest end-to-end run, on one GPU, with two existing keys overridden
uv run python scripts/<entry point>.py experiments/smoke optimizer.lr=3e-5 seed=1
# swap in a whole config group, then set its learning rate (??? until set)
uv run python scripts/<entry point>.py experiments/smoke optimizer=sgd optimizer.lr=0.1
```

## Data, streams and checks

Prompt sets and teacher streams are built once per model and kept as evidence (see
[retention](#artifact-retention)). On the Pod, through the launcher:

```bash
python scripts/prepare_data.py experiments/smoke          # prompt sets and their manifest
python scripts/generate_teacher.py experiments/<name> --stream kl_harmful
python scripts/generate_teacher.py experiments/<name> --stream kl_benign
python scripts/generate_teacher.py experiments/<name> --stream training --seeds 0 1 2 3 4
python scripts/stage0_checks.py experiments/stage0_checks model=qwen2.5-1.5b-instruct
```

For an appendable training stream (study 2), the same command with `--responses N` appends chunks
until the stream holds N responses (default `teacher.responses`); existing chunks are kept. The
Stage-A curvature measurement reads such a stream:
`python scripts/curvature.py experiments/<name> seed=<seed>` (settings in `--help`).

The endpoint Rayleigh quotient of study-2 runs (`eval/rayleigh`, off unless an experiment lists
it under `defaults`) reads the curvature panel, one stream per model and no run seed. Generate it
once on a GPU, before any run that measures R, from such an experiment, and archive it with its
own inventory like the monitor stream; a run that measures R without it stops before it trains.
At 1.5B with 100 responses its directory is
`teacher/Qwen--Qwen2.5-1.5B-Instruct/curvature_panel/9fff8e13183a/reference`.

```bash
python scripts/generate_teacher.py experiments/<name> --stream curvature_panel
```

On the first run of a campaign that measures R, read its `rayleigh.json` before the campaign
goes on: `peak_reserved_gb` (the product's own peak, ≈ 30–34 GB expected at 1.5B with
micro-batches of 4) must stay clear of the GPU's memory, and `reason` must not start with
`failed:`. A failed measurement does not stop the campaign (below), so every later run would
fail the same way unnoticed.

`prepare_data.py` caches the pinned source CSVs in `DATA_DIR/sources/` and refuses to overwrite a
complete build without `--force`. Stream directories are named by a hash of everything that
determines their samples; an existing stream is never regenerated. A stream is published by
renaming a partial directory; after an interrupted generation, remove the leftover
`.seed-*.partial-*`, `.reference.partial-*` or `.chunk-*.partial-*` directory with
`generate_teacher.py ... --remove-partial` once nothing is generating.

The stage-0 checks write a path-field record per execution environment under `DATA_DIR/checks/`:
model numerics, GPU model, library versions, execution batch sizes and the check's version, and
determinism and `CUBLAS_WORKSPACE_CONFIG` when set (records without them, studies 1 and 2, match
only runs without them); an older record moves to `history/`. Training runs stop before their first optimizer step without a
passed record for their own environment that covers their steps, batch size and response cap, so
run the checks on each GPU type and execution setting used. Each run also checks its own stream
against the path-field bound and the teacher-gap thresholds.

Choose the GPU by the cost of a run, not of an hour, from the stage-0 timings: in fp32 without
TF32, the Ada cards (RTX 6000 Ada, L40S) are about 2.4 times faster than A6000/A40 and can cost
less per run. Stages 0–2 run on the L40S, decided on 2026-10-09 after stage 0: it is the GPU type
measured there, and another type would need its own checks and enter the teacher-gap thresholds;
the saving on cheaper Ada cards is cents per run. Host RAM depends on the Pod's host, not only on the GPU type (an RTX 6000 Ada Pod
had 62 GB rather than the listed 167 GB), so check it after creating a Pod. The full soft-target
control peaks at ≈ 30 GB of host RAM (one preallocated buffer, read back lazily), the path-field
check at ≈ 10 GB, and training keeps the projection directions in pinned memory (≈ 2 × 5 GB for
1.5B). Frozen Adam and pre-warmed
AdamW also need `--stream preconditioner` for the same seeds.

## Study 2, Stage A

Pods, all L40S Secure, created 2026-10-14 with a 100 GB volume disk at `/workspace` and a 50 GB
container disk, each with `scripts/pod_deadline_watch.py` in its start command before
`/start.sh`:

| Pod | Name | Seed | Data center | Host CUDA | Driver |
|---|---|---|---|---|---|
| `0jl6d2cmiiah0l` | `zfr-study2-a` | 2000 | US-MO-1 | 13.0 | 580.159.03 |
| `62yoobdy4zqzjh` | `zfr-study2-b` | 2001 | US-MO-1 | 13.0 | 580.178.04 |
| `ilpuntk6ypt60m` | `zfr-study2-c` | 2002 | not reported by the API (Stockholm by its IP); local volume | 13.2 | 595.91.07 |

The host driver is not part of the stage-0 record's environment fingerprint (GPU, torch, CUDA
build, cuDNN): every Pod passes the record that Pod A wrote. Stage A runs from one commit,
delivered to every Pod as a Git bundle.

Before every `pod_setup.sh` of that commit (`29e096a`), check each inventory it is given: no path
with a `..` component, and no two inventory paths that flatten to the same name (`/` replaced by
`_`). Later versions of the script refuse paths that leave `DATA_DIR` and symbolic links in it,
and number their copies of the inventories.

**Setup of a study-2 Pod.** From the checkout of the chosen commit (a Git bundle), restore only
what the study needs, naming the inventories explicitly: the bucket also holds a superseded
prompt build (`data/prompts/7f7ed5f91950`), which no Pod restores. Run `pod_setup.sh` alone,
with nothing else running on the Pod: its checks do not guard against a concurrent writer to
`DATA_DIR`.

```bash
bash scripts/pod_setup.sh runs/archive/inventory-2026-10-11.tsv \
  runs/archive/inventory-stage1-streams.tsv runs/archive/inventory-2026-10-14-study2-prompts.tsv
```

The Pods of seeds 2001 and 2002 add the inventory of seed 2000's first phase: the stage-0 record
under protocol 2 and the `kl_monitor` stream, archived before they start training.

One Pod per discovery seed (2000, 2001, 2002): a seed's stream has one writer, the Pod that holds
it, and its runs extend it there. The Pod of seed 2000 also runs everything tied to that seed:
the stage-0 checks, the curvature, the screen and the full-length controls. Campaigns hold one
seed and are written by `scripts/stage_a_campaigns.py` into `$RUNS_DIR/campaigns/`, outside the
checkout, so that runs from a clean checkout stay clean. The data-dependent campaigns are written
from run records checked as the selection checks them, and refused on anything not whole, by
the rules of [the preregistration](study2-preregistration.md#stage-a--feasibility).

```bash
C="$RUNS_DIR/campaigns"
# Seed 2000, first: the record and the stream the other Pods restore from the bucket
python scripts/stage0_checks.py experiments/stage0_checks model=qwen2.5-1.5b-instruct protocol_version=2
python scripts/generate_teacher.py experiments/study2_stage_a --stream kl_monitor
python scripts/stage_a_campaigns.py setup --out "$C/setup.yaml" && python scripts/launch.py "$C/setup.yaml"
python scripts/generate_teacher.py experiments/study2_stage_a --stream training --seeds 2000 --responses 1000
python scripts/curvature.py experiments/study2_stage_a seed=2000
python scripts/stage_a_campaigns.py screen --out "$C/screen.yaml" && python scripts/launch.py "$C/screen.yaml"
# Every seed S, on its Pod
for B in 5 20; do  # frozen Adam's θ₀ second moment, from batches of the training size
  python scripts/generate_teacher.py experiments/study2_stage_a optimizer=adam_frozen \
    train.batch_size=$B --stream preconditioner --seeds S
done
python scripts/archive.py /workspace/archive/inventory-DATE-study2-a-preconditioner-sS.tsv \
  "$ARTIFACT_REMOTE/data" "$DATA_DIR" $(cd "$DATA_DIR" && ls -d teacher/*/preconditioner/*/seed-S)  # before the grids read them
python scripts/stage_a_campaigns.py grids --seed S --out "$C/grids.yaml" && python scripts/launch.py "$C/grids.yaml"
python scripts/stage_a_campaigns.py sgd --runs SCREEN_RUNS --seed S --out "$C/sgd.yaml"
python scripts/launch.py "$C/sgd.yaml"
```

**Monitor version 2** ([amendment 1](study2-preregistration.md#amendment-1-the-monitor-panel)).
Every run after the amendment uses the commit that implements it, delivered as a new bundle.
Once, on a Pod with nothing else running, before any run reads it: generate the new monitor
stream, record its spec hash and sha256 here, and archive it with its own inventory, which the
other Pods' `pod_setup.sh` adds. Generated on Pod C on 2026-10-15 from `f0aacd6` (the first Pod
free of its grids): `kl_monitor/cb98e156751c`, 100 prompts × 2 responses, seeds 1737873636 and
325807756, `samples.jsonl` sha256 `d07e47df23a568377c766c1238b79e45af85f54ce1e7121fa716cb7410b9f3ad`,
`manifest.json` sha256 `516d8fb58647781683f9a94f189966ac43f900e4c1e2c569931dd1ffe26df3fb`;
inventory `runs/archive/inventory-2026-10-15-study2-monitor-v2.tsv`.

```bash
python scripts/generate_teacher.py experiments/study2_stage_a --stream kl_monitor
```

Each version-2 run caches θ₀'s logits for the monitor (m = 1 and the second responses) and for
the scoring panel, all in host memory, about 4.4 GB, 4.4 GB and 18.6 GB at the 1.5B model: not on
the GPU, since a version-1 run at B = 20 already held about 37.6 of the L40S's 45 GB; it checks the caches against uncached readings at θ₀ and at every full evaluation
(`cache_checks.jsonl`) and writes the scoring diagnostics to `diagnostics.jsonl`. Diagnostics and
evaluations record the run's GPU memory peak so far (`peak_allocated_gb`, `peak_reserved_gb`):
on the first version-2 run at B = 20, read them at the first diagnostic (step 25, after Adam's
moments exist) and the first full evaluation, before the campaign goes on; a peak near the GPU's
memory stops it.

The search ([amendment 1](study2-preregistration.md#amendment-1-the-monitor-panel)) starts on
each Pod with its first step, the largest rate of each long arm at each batch:

```bash
python scripts/stage_a_campaigns.py search --runs SCREEN_RUNS --seed S --out "$C/search1-sS.yaml"
python scripts/launch.py "$C/search1-sS.yaml"
```

Its later steps, the full controls of the SGD rates it runs, and the diagnostic runs that a
selected batch short of ten diagnostic states calls for are written from the screen and the
version-2 runs of all three seeds (with θ₀'s run for the rule's utility):

```bash
python scripts/stage_a_campaigns.py search --runs SCREEN_RUNS V2_RUNS CONTROLS --theta0 THETA0_RUN \
  --seed S --out "$C/search-sS-N.yaml"
python scripts/stage_a_campaigns.py controls --runs SCREEN_RUNS V2_RUNS CONTROLS --theta0 THETA0_RUN \
  --out "$C/controls-N.yaml"                     # on seed 2000's Pod
python scripts/stage_a_campaigns.py diagnostic --runs ALL_RUNS --theta0 THETA0_RUN --seed S \
  --out "$C/diagnostic-sS.yaml"
```

Each refuses records that are not whole, and says what the search waits for when it needs no
run of that seed.

**Stage 0 under a new protocol** needs the soft-target distributions of its path-field check,
which the retention policy deletes after each check: the bucket keeps only the stream's manifest
and samples, which `pod_setup.sh` restores. Before the checks, remove that restored copy
(`teacher/<model>/soft_targets/<spec>/seed-0`) from `DATA_DIR`; stage 0 then generates the stream
again. Compare the new samples' and distributions' sha256 with the archived manifest and record the
comparison next to the run (on 2026-10-14 both were byte-identical, see
`runs/runpod/evidence/soft-targets-regeneration-20261014.json` in the bucket). After the check,
delete the distributions and put the archived manifest back, so that `DATA_DIR` matches its
inventories again. For study 2's short regime the prerequisites' driver did all of this once, on
2026-10-16 ([its record](#study-2-short-regime-amendment-3)); it is not run again.

**Deadlines between campaigns.** A Pod started with the file-reading watcher changes its deadline
without a restart: the deadline is the integer in `/workspace/zfr-deadline` (`ZFR_DEADLINE_FILE`
overrides the path), re-read every 20 s, and `ZFR_DEADLINE_EPOCH` is only its upper limit (a file
value above it is ignored and logged `zfr-deadline file above cap`). A file value already past when
the watcher starts, left from an earlier start, is ignored too (`zfr-deadline file stale`) and the
cap applies. At the deadline a stop counts only when the API confirms it (no `errors`, and the
Pod's `desiredStatus` "EXITED"); an unconfirmed one is logged `zfr-deadline stop unconfirmed` and
retried, six attempts with a doubling wait. Once a stop is confirmed, the file is renamed
`<file>.fired-<epoch>`, so the next start cannot read it. Set it atomically from an SSH session with
`python3 scripts/pod_deadline_watch.py --set EPOCH [--file PATH]` (ASCII digits only); the log shows
`zfr-deadline <time> source <file|last|env>` on each change. A missing or malformed file keeps the
last valid deadline, and before any valid read the cap applies. To go beyond the cap, change
`ZFR_DEADLINE_EPOCH` with `update-pod` first. That restarts the container: the running run stops and
the container disk loses the archive key and the Google Cloud CLI. A Pod whose watcher predates the
file (Stage A of study 2) changes its deadline only that way, and only between
campaigns. Before that, archive the seed's training stream and every finished run directory with
their own inventories, so that the full `pod_setup.sh`, given those inventories as well, verifies
`DATA_DIR` after the restart. A campaign that cannot end before the deadline with an hour to
spare is stopped at a run boundary by `SIGTERM` to the `launch.py` process only, while a run is
going. The launcher of commit `29e096a`, which Stage A runs, has no handler: it dies at once, and
its running `train.py` carries on alone and may still extend the stream. So wait until that
`train.py` process itself (its PID, not the launcher's) has exited, check its `terminal.json`, and
only then archive the stream and the runs. Later launchers let the running attempt finish, record
it and exit non-zero without starting another ([launching runs](#launching-runs)). Either way the
same campaign started again after the restart skips every cell whose records are whole.

**Records between Pods** move only through the bucket. Every Pod archives each run directory
when its campaign ends ([archive](#rented-machines)). The Pod of seed 2000 downloads what the
data-dependent campaigns need from the other seeds, writes them, and archives the campaign
files under `campaigns/` in the bucket; the Pods of seeds 2001 and 2002 download their own:

- **SGD discovery** needs the screen's run directories (seed 2000's own; the other Pods download
  them to write their `sgd` campaign, or receive the written files).
- **Grid extensions** at batch 5 need the 25-step runs at batch 5 of all three seeds. Once every
  Pod has finished its grids, seed 2000's Pod downloads them and writes
  `stage_a_campaigns.py extension --runs ALL_SHORT_RUNS --seed S` for each seed; each Pod runs its
  own. After those runs finish everywhere, write the extensions again: a second halving follows
  if the first extension's median K over the seeds still exceeds 5 (two at most). "No extension
  is called for" ends it.
- **Full controls** need the screen and the SGD-long runs of all three seeds. Once every Pod has
  finished its SGD campaign, seed 2000's Pod downloads the other seeds' SGD-long run directories
  and writes and runs `stage_a_campaigns.py controls --runs SCREEN_RUNS ALL_SGD_RUNS`.

Then, on this machine, once every run is archived and downloaded:

```bash
python scripts/stage_a_summary.py --runs runs/study2_stage_a runs/study2_controls --theta0 THETA0_RUN --out stage-a.json
```

The summary's selection is sealed (committed) before any harm label of discovery is opened. The
streams, with their chunks' fingerprints, go to the bucket after discovery. This machine keeps
the run directories, whose stored responses stay out of Git and are never printed.

**Amendment 2** ([the full controls' gradient rule](study2-preregistration.md#amendment-2-the-full-controls-gradient-rule)):
the reruns of the full controls are written from the same records and run on Pods holding seed
2000's stream:

```bash
python scripts/stage_a_campaigns.py controls --control-rule 2 --runs SCREEN_RUNS ALL_V2_RUNS CONTROLS \
  --theta0 THETA0_RUN [--cells control-b20-0.002] --out "$C/controls-amended.yaml"
```

`--cells` splits the campaign over Pods, one cell each. The controls read the stream without
extending it: it holds 52 chunks, more than the 34 that the longest control needs. Several Pods
may therefore hold copies. The reruns land in `study2_controls_amended`. The summary, given that
directory as well, reports `selection_amendment_2` beside the preregistered `selection`.

Only Stage A's namespaces go into the summary: `runs/study2_stage_a` and `runs/study2_controls`,
never `runs/` or another parent (and, under amendment 2, `runs/study2_controls_amended`). The SGD-momentum pilot
([preregistration](study2-pilot-momentum.md); approved, ended `barrier` on 2026-10-16) writes to `study2_pilot_momentum` and
`study2_pilot_momentum_controls`. Stage A's conformance refuses the pilot's arms, so a summary
over mixed directories comes out incomplete and blocks sealing. The pilot's own summary reads
both: `scripts/pilot_momentum_summary.py --runs PILOT_DIRS --stage-a STAGE_A_DIRS`.

## Study 2, short regime (amendment 3)

The redirected stage compares AdamW-25 with FA-25 (frozen Adam), each run exactly 25 steps at three
targeted rates per arm, with pre-warmed AdamW-25 as the secondary arm; its rules are amendment 3
of [the study-2 preregistration](study2-preregistration.md). The selection, the support and the
validation read K, numerics and utility only; only the looks read judge labels.

**Amendment 4 changes this procedure** ([its section](#study-2-amendment-4-numerical-qualification-batch-20-and-the-mechanism-cohort)):
the campaign kinds and their order (B = 20 only, the numerical stage instead of the FA controls,
the three-arm validation), the confirmation loop of a three-arm selection, and the fixation
record's keys. What follows stays valid where that section does not replace it: the configs,
the records and their refusals, the prerequisites, the versions record, the given-up runs and the
abandonment record, labeling, and the loop of a two-arm selection.

- **Configs.** `experiments/study2_short` holds the regular runs of the three arms
  (`optimizer=adam_frozen`, `optimizer=adamw_prewarmed`). It equals Stage A's 25-step runs
  (`experiments/study2_stage_a schedule=short`) except for the experiment name and the role,
  `study2_short.role`: discovery (seeds 2000–2002), validation (2100–2111) or confirmation
  (3000–3079). So the new discovery runs pool with Stage A's grids.
  - `experiments/study2_short_rayleigh` is the same with `eval/rayleigh`, under the same
    experiment name, for confirmation blocks 1–10.
  - `experiments/study2_short_fa_controls` holds the FA soft-target controls under protocol
    version 3.
  - The campaigns set the arm, batch, rate, role, seed and the controls' bound; nothing is set
    by hand.
- **Records.** Every script reads every run so far: `runs/study2_stage_a`,
  `runs/study2_controls`, `runs/study2_short` and `runs/study2_short_fa_controls`, together with
  the launch journals under `RUNS_DIR/launch/` (`--runs-root RUNS_DIR`). Download each Pod's
  run directories and launch logs before writing the next campaign.
- **Refusals.** Anything not whole leaves the stage `incomplete`, and every campaign is refused:
  - a run off its campaign: role, model, protocol, schedule, monitor, or the Rayleigh setting;
  - a seed whose runs read two training streams;
  - a seed and batch whose runs (FA-25, pre-warmed AdamW-25, the FA controls, Stage A's FA
    grids) estimated v₀ from two preconditioner streams;
  - a run the state machine did not prescribe.

**Prerequisites on a Pod.** First restore `DATA_DIR` with `pod_setup.sh` from the inventories
under `runs/archive/`, named per campaign kind as for Stage A. New runs of an archived seed must
read exactly the streams that its earlier runs read.

- **Every kind** (discovery rounds, controls, pre-warmed, validation, confirmation):
  - `inventory-2026-10-11.tsv`: the stage-0 prompt sets and K's reference streams `kl_harmful`
    and `kl_benign` (without them every run stops at the evaluator's setup);
  - `inventory-2026-10-14-study2-prompts.tsv`: study 2's prompt sets;
  - `inventory-2026-10-14-study2-a-phase0.tsv`: the protocol-2 path-field record;
  - `inventory-2026-10-15-study2-monitor-v2.tsv`: the monitor-v2 stream;
  - `inventory-2026-10-14-study2-a-preconditioner-s2000.tsv`, `…-s2001.tsv` and `…-s2002.tsv`:
    v₀'s preconditioner streams;
  - the latest training-stream inventory of each discovery seed:
    `inventory-2026-10-16-study2-a-stream-s2000-final.tsv`,
    `inventory-2026-10-15-study2-a-stream-s2001-search-s2001-2.tsv` and
    `inventory-2026-10-16-study2-a-stream-s2002-final.tsv`;
  - once archived, the inventories of the validation and confirmation seeds' streams.
- **`controls`** also needs the stage-0 record under protocol version 3,
  `inventory-2026-10-16-study2-stage0-v3-record.tsv`
  (`checks/Qwen--Qwen2.5-1.5B-Instruct/path-field-057628165bd1.json`); without it
  `prechecks.require_path_check` stops every FA control.
- **Confirmation blocks 1–10** also need the curvature panel,
  `inventory-2026-10-16-study2-curvature-panel.tsv`; without it the Rayleigh measurement stops
  the run.
- **Every Pod after the prep of 2026-10-16 restores both of these** as well, whatever it runs:
  `pod_setup.sh` checks every file in `DATA_DIR` against the inventories it is given, and the
  panel must never be generated twice.

```bash
A=runs/archive
bash scripts/pod_setup.sh $A/inventory-2026-10-11.tsv $A/inventory-2026-10-14-study2-prompts.tsv \
  $A/inventory-2026-10-14-study2-a-phase0.tsv $A/inventory-2026-10-15-study2-monitor-v2.tsv \
  $A/inventory-2026-10-14-study2-a-preconditioner-s200{0,1,2}.tsv \
  $A/inventory-2026-10-16-study2-a-stream-s2000-final.tsv \
  $A/inventory-2026-10-15-study2-a-stream-s2001-search-s2001-2.tsv \
  $A/inventory-2026-10-16-study2-a-stream-s2002-final.tsv \
  $A/inventory-2026-10-16-study2-stage0-v3-record.tsv \
  $A/inventory-2026-10-16-study2-curvature-panel.tsv
```

The prep driver's header names the same inventories. Then write and run the driver, from the
chosen commit:

```bash
python scripts/study2_short_campaigns.py prep --seeds 2000 2001 2002 --batch 5 --commit SHA \
  --archived --out "$C/prep.sh"
bash "$C/prep.sh"
```

**The prep under protocol version 3 is done** (`--once`, on 2026-10-16, on P0 `xyiz2auas609lu`,
from `df001af`). **It must not be run again:** it would rerun stage 0, and a `DATA_DIR` without
the panel's inventory would generate the curvature panel a second time, which the amendment
forbids; conformance does not compare panel hashes across runs.

- Stage-0 run `stage0-v3-qwen1.5b-l40s-20261016-01` failed on the soft-target distributions that
  the retention policy had deleted. `stage0-v3-qwen1.5b-l40s-20261016-02` passed under protocol
  version 3, after the restored copy had been set aside and the stream regenerated, as
  [stage 0 under a new protocol](#study-2-stage-a) prescribes.
- The regeneration was byte-identical to the archive
  (`runs/runpod/evidence/soft-targets-regeneration-20261016.json`): samples sha256
  `bef8f491e7f670f46fc965381cacefea896a37b831dc9b468fce067f9d74719b`, distributions sha256
  `ffbd05a723dc2f67965d8f6cdfb10503fb1ebdd361953160cd984046e64f1775`. The regenerated
  distributions were deleted and the archived copy put back.
- The curvature panel: `teacher/Qwen--Qwen2.5-1.5B-Instruct/curvature_panel/9fff8e13183a`,
  `samples.jsonl` sha256 `c273e468d07ee0bae947c1887a752c345450f7026a892f6e144d3d2d5bf504a7`,
  `manifest.json` sha256 `6f4ca6e001aebef2f3ea0df314055b3fa514248bd02de9f5ab9c3e5fc17543a3`;
  inventory `runs/archive/inventory-2026-10-16-study2-curvature-panel.tsv`.
- Inventories: `runs/archive/inventory-2026-10-16-study2-stage0-v3.tsv` (the two stage-0 runs,
  the drivers, their logs and the evidence) and
  `runs/archive/inventory-2026-10-16-study2-stage0-v3-record.tsv` (the path-field record).

What `--once` does, for the record: the driver runs the stage-0 checks under protocol version 3
and generates the curvature panel. Around the checks, `soft-targets-aside` moves the restored
soft-target copy to `/workspace/tmp/soft-targets-aside-YYYYMMDD/`, and once the run has passed
`soft-targets-restore` compares the regenerated samples' and distributions' sha256 with the
archived manifest (`soft_targets_regeneration.json` in the run's directory), deletes the
regenerated distributions and puts the archived copy back. A regeneration that differs stops the
driver, keeps the whole regenerated stream for inspection and needs the lead's decision and a new
`--stage0-run-id`.

- **`--archived`** is for seeds whose streams are archived. The driver generates nothing; it
  runs `study2_short_campaigns.py check-streams`, which:
  - locates each seed's training and preconditioner streams as every run does
    (`runtime.locate_stream`);
  - prints their sha256;
  - stops on a missing one.
- **Fresh seeds** (validation, confirmation) leave out `--archived`. The driver then generates
  the training stream to 25 × B responses and v₀'s preconditioner stream at B.
  - Archive them with their inventories before the runs read them.
  - Pass `--archived` for them on any later Pod.
- **Running it again.** The driver stops at the first failure and can be run again: existing
  streams are kept.

**w_max and the fixation record.** The simulation fixes its values in a JSON record
(`analysis.study2_short.read_fixation`):

- `w_max` (decided: 0.47);
- `levels`: the two looks' two-sided confidence levels (decided: 0.9920 and 0.9631, α 0.040);
- `se_floor` and `bias_allowance`, in units of Δ (1 pp = 0.01). The SE floor is decided at 0.
- `interval`: the looks' interval, `bootstrap_t` (decided) or `t`:
  - `bootstrap_t` is the paired seed-block bootstrap-t of the mean over the look's supported
    blocks, as the simulation defines it;
  - look k draws its resamples from `derive_seed(bootstrap_seed, "look", k)`, so the summary
    reproduces every look from the record;
- `bootstrap_resamples`: the resamples per look, an integer of at least 1000;
- `bootstrap_seed`: an integer;
- `looks` and `cap`: they must equal the code's look sizes [36, 60] and cap 80;
- `equivalence_min_supported`: an integer ≥ 0 (decided: 36). Equivalence is concluded only at
  a look with at least this many supported blocks; 0 allows it at every look. Direction is
  unaffected.
- `direction_min_supported`: an integer ≥ 0 (decided: 10). Below this many supported blocks a
  look concludes nothing, neither direction nor equivalence: it is inconclusive. With very few
  blocks the bootstrap-t collapses, to [mean, mean] at two. 0 disables the rule.
- `secondary_interval`: the secondary contrasts' interval at 98.75 % per contrast per look,
  `bootstrap_t` (decided) or `t`. The bootstrap-t is the looks' (`bootstrap_resamples`
  resamples), on its own stream per contrast and look,
  `derive_seed(bootstrap_seed, "secondary", contrast, k)`. The secondary check of 2026-10-17
  (`runs/analysis/study2_short_sim/2026-10-17-secondary/`) chose it: see the amendment's
  secondary contrasts.
- `secondary_status`: how the secondary contrasts are reported, `secondary` (decided) or
  `exploratory`. The summary prints both values beside the secondary contrasts.
- amendment 4's keys: `mechanism`, `audit`, `candidates`, `numerical`, `validation`,
  `timecourse`, `drift` and `scalars` ([below](#study-2-amendment-4-numerical-qualification-batch-20-and-the-mechanism-cohort)).

Each value is required and none is defaulted, the bootstrap's values also under `t`. The
committed record is `docs/study2-short-fixation.json` (a test ties it to the simulation's decided
design); its amendment-3 keys are:

```json
{"w_max": 0.47, "levels": [0.991983603686668, 0.9630570834481011], "se_floor": 0.0,
 "bias_allowance": 0.010841, "interval": "bootstrap_t", "bootstrap_resamples": 100000,
 "bootstrap_seed": 20261016, "looks": [36, 60], "cap": 80, "equivalence_min_supported": 36,
 "direction_min_supported": 10, "secondary_interval": "bootstrap_t",
 "secondary_status": "secondary"}
```

Every campaign kind needs the record (`--fixation`), which is checked whole (under amendment 3,
`targeted` could take `--w-max` instead; amendment 4 requires the record and `--runs-root` for
every kind). A `--w-max` given beside a record must equal it. Each campaign's header records
w_max, its source and the code revision. Fix w_max before any repair round: rounds made under
another w_max leave the stage incomplete.

**Campaigns** (amendment 3's; amendment 4's kinds and order replace them, [below](#study-2-amendment-4-numerical-qualification-batch-20-and-the-mechanism-cohort)).
Each kind is written only when the stage's state machine names it, in this order:

1. `targeted` and `repair` (discovery rounds, primary arms first);
2. `controls`;
3. `prewarmed`, then the pre-warmed arm's own rounds;
4. `validation`;
5. `confirmation`.

`validation` and `confirmation` need the committed versions record (below), and read the launch
journal (`--runs-root`) and, when one exists, the abandonment record (`--abandoned`).

```bash
R="runs/study2_stage_a runs/study2_controls runs/study2_short runs/study2_short_fa_controls"
THETA0=runs/study2_stage_a/20261014T192927Z-theta0-s2000-abb703a1-a1  # θ₀'s run (summary.json)
F=docs/study2-short-fixation.json
python scripts/study2_short_campaigns.py targeted --runs $R --theta0 $THETA0 --fixation $F \
  --out "$C/targeted.yaml"
python scripts/launch.py "$C/targeted.yaml"
python scripts/study2_short_campaigns.py validation --runs $R --theta0 $THETA0 --fixation $F \
  --runs-root runs --out "$C/validation.yaml"
```

- **Splitting.** `--seeds` (discovery and validation) and `--cells` only split a campaign over
  Pods; a confirmation campaign takes no `--seeds`, and each of its cells spans every seed of its
  range.
  - A discovery seed's training stream may still grow with its runs, so it stays on one Pod at
    a time.
  - Once the prep has archived a fresh seed's streams (25 × B training responses and v₀'s
    preconditioner stream), its runs read them without extending them. As for amendment 2's
    controls, several Pods may then hold copies, so a validation or confirmation campaign may be
    split by cell. A confirmation may instead be split by issuing ranges (`--blocks`) to
    different Pods.
- **Recovery rounds.** Cell names never recur across campaigns, so each cell keeps its own
  recovery round.
- **Given-up runs.** A validation or confirmation run is "given up" when the launcher's journal
  shows it exhausted after its recovery round. It counts as a technical failure of its arm in
  its block, so the stage does not wait for it forever.
  - All 2 × 3 attempts must have ended technical, and the last one (after the recovery round)
    must have its `end` record. A missing `end` record never gives a run up by itself: the
    attempt may still be running, and the run stays pending.
  - An attempt that will never end (its Pod lost mid-run, say) is given up only through an
    **abandonment record**, passed as `--abandoned FILE` to the campaigns and the summary. It is
    a JSON object kept **outside the checkout's tracked tree, under `$RUNS_DIR`** (for example
    `$RUNS_DIR/launch/abandoned-YYYYMMDD.json`), and archived with an inventory under
    `runs/archive/` before it is used. Never commit it, nor any other file once the versions
    record is committed: a later commit makes HEAD a descendant that the versions record refuses
    (below).

    ```json
    {"what": "Launcher attempts abandoned (study 2, short regime)",
     "abandoned": [{"run_id": "<the attempt's run_id in the journal>",
                    "reason": "Pod terminated mid-run, 2026-10-20"}]}
    ```

    Each attempt appears once, with a non-empty reason. An attempt the journal does not show
    started, or shows ended, is refused. The campaign's header and the summary's `--out` file
    record its path and sha256.

**The versions record.** Before the validation campaign, write and commit the versions record
that amendment 3 requires, from a clean checkout of the commit the validation and confirmation
run from:

```bash
python scripts/study2_short_summary.py versions --judge wildguard --fixation $F \
  [--allow-descendants]
```

It writes the current record (`--out`, by default `docs/study2-short-versions-2.json` since
[amendment 5](#amendment-5-the-candidate-order-at-batch-20-after-u-8); the first record,
`docs/study2-short-versions.json`, was written so at `4fbd19a`), a JSON object written once (the
script refuses an existing file and a dirty checkout). Commit it, alone, before the validation
campaign:

- `code.commit`: the full commit hash of the checkout;
- `code.allow_descendants`: whether a later commit may run the validation and confirmation
  (false unless `--allow-descendants` is given);
- `model`: `name` and `revision` of `experiments/study2_short`'s model (the Hugging Face commit);
- `judge`: the judge config (`judge/wildguard`) and the identity every label carries:
  `backend`, `prompt_version`, `judge_model`, `judge_revision`, `rubric_version`;
- `fixation`: the fixation record's path and sha256;
- `what` and `written_at` (UTC);
- a later record only: `supersedes`, the path in the checkout and the sha256 of the record it
  follows (`--supersedes FILE`), and `reason` (`--reason TEXT`), both or neither.

**The record is enforced** (`analysis.study2_short.versions_problems`). The `validation`,
`confirmation` and `timecourse` campaigns, the summary with `--judge` and the audit's draw refuse
without it (`--versions FILE`, default the current record, `docs/study2-short-versions-2.json`,
in the checkout), and refuse when:

- the record is not committed: outside the checkout, not tracked by Git, or changed since HEAD;
- the running HEAD is not `code.commit`, or a child of it that adds the record alone (the commit
  of the record itself). Any other descendant is refused unless `code.allow_descendants` is true;
  a commit that does not descend from it always is;
- a tracked file has uncommitted changes, staged or not, or a file anywhere in the checkout is
  untracked and not ignored by the checkout's `.gitignore` files (keep local work directories
  in `.gitignore`: `.git/info/exclude` and a user's excludes file do not
  count). The runs read a dirty tree the same way (`tracking.worktree_changes`), whatever
  `status.showUntrackedFiles`, `.git/info/exclude` and `core.excludesFile` say;
- `model` is not the model of `experiments/study2_short`;
- the judge's identity is not that of its config, or (for the summary) the config is not
  `judge/NAME` of `--judge NAME`;
- the fixation record given (`--fixation`) has another sha256;
- the record is not one of `docs/study2-short-versions.json` and
  `docs/study2-short-versions-2.json` (`VERSIONS_RECORDS`);
- the first record names another, or a later record does not name the first,
  `docs/study2-short-versions.json`, its exact path and the sha256 of the committed file, with a
  non-empty `reason` (`versions` refuses such a record when it is written, too);
- an issued confirmation or time-course campaign named another record in its header (the launch
  journal under `--runs-root`; `issued_versions`; a header naming none counts under the first
  record): a confirmation keeps the record it started with;
- a confirmation or time-course run of `--runs` has no start of its own in the launch journal
  (`unlaunched_runs`: a start naming its run id with its label on the run's seed): its record
  cannot be shown, so the check fails closed until the journal that started it is restored.

The launch journal is the evidence of which versions record a run started under: runs do not
carry the campaign header. Archive it with an inventory and its sha256 at each stage's end.

The discovery campaigns (`targeted`, `repair`, `controls`, `prewarmed`) come before the record
and run without it. The campaigns' headers and the summary's `--out` file (with `--judge`) record
the versions record's path and sha256; a versioned campaign's YAML `header` holds them too
(`versions_record`, `versions_sha256`), and the launcher logs them with every start. Write
campaigns, summaries and abandonment records under
`$RUNS_DIR` (ignored by Git), never into the checkout's tracked tree, and archive them with an
inventory.

**The confirmation loop.** The looks fall at 36 and 60 supported blocks, and the cap is 80 started
blocks. Blocks run in their preassigned order, block k on seed 2999 + k. Under amendment 4 this
loop, with its pauses at the looks, is that of a two-arm selection; a three-arm selection runs
[without pauses](#the-confirmation-loop-of-a-three-arm-selection). Repeat these steps until a look
stops:

1. **Write** `confirmation --blocks FIRST LAST [--after-look K] --runs-root runs`, then launch
   it. The script enforces these rules:
   - FIRST is the first block without a run;
   - a new range overlaps no block that has runs (re-issuing exactly an issued range completes
     it);
   - the range holds at most n_k − p blocks, where p is the number of blocks with runs that may
     still be supported.
2. **Let the blocks complete.** On blocks 1–10 the endpoints also measure R. Read the first run's
   `rayleigh.json`, as for any campaign that measures R.
3. **Read the support counts** in the summary: `first_block_without_runs`, `supported_in_order`
   and `possibly_supported`.
4. **Label** the new runs with study 2's sealed labeling command,
   `python -m zero_field_refusal.analysis.study2_label_cli --judge wildguard RUN_DIRS`, never with
   `scripts/label_runs.py` ([sealed output](#labeling-and-sealed-output)): it labels as
   `label_runs.py` does, in three rounds. A study-2 run is labeled once its terminal record was
   published with whole records, whatever its status (a Rayleigh measurement that failed afterwards
   leaves it `failed`). The three rounds are a run's in all, across reruns:
   `judge/wildguard.rounds.json` counts each round before its requests are sent, so an interrupted
   round counts as used, and a rerun resumes with the rounds left, sending nothing once they are
   used. A reply already in the label file is reused and is no round. When a run's rounds are over,
   its judging receipt `judge/wildguard.receipt.json` records the expected items, the judge's
   identity, the rounds used in all, the labels still missing and the label file's sha256. A round
   deletes the receipt before it starts; an interrupted one leaves none. Without a receipt matching
   its label file, a run's labels do not count: the summary lists it under `labels_pending` and its
   block waits (the pre-warmed arm's reasons are sealed until the final analysis: only a generic
   status shows). Rerun the labeling to complete it. A run with a completed receipt (no label
   missing, or its three rounds used) is left as it is, nothing sent. A round that fails (the judge
   gone) stops the invocation: the runs after it are "not attempted" and keep their rounds, so bring
   the judge back and rerun. A run another labeling holds is `busy`, nothing sent.
   These are refused, with nothing sent and nothing written:
   - **The labels changed outside a round.** Either the receipt no longer matches its label file,
     judge, items or counts, or the label file changed after its receipt was deleted. Restore the
     label file the receipt names from the archive, or exclude the run by a recorded decision.
     Never delete the receipt: a deleted receipt whose label file is unchanged is written again,
     and one whose label file changed stays refused.
   - **A label file without its rounds record** (labeled before the record existed): its rounds
     are unknown.
   - **A label file the analysis would refuse**, such as expected items without any record once
     the three rounds are used.

   Decide such runs by hand.
5. **Run the summary** with `--judge wildguard --fixation $F`.
6. **Go on** only if the look did not stop (`stopped` false). Pass `--after-look K` for the look
   just read; without it the script refuses any block once look K is due.

**The summary** prints the stage's status and, with `--judge`, the looks:

```bash
python scripts/study2_short_summary.py --runs $R --theta0 $THETA0 --fixation $F \
  --runs-root runs [--abandoned FILE] [--judge wildguard] \
  [--out "$RUNS_DIR/analysis/study2-short-summary-YYYYMMDD.json"]
```

`--out` goes under `$RUNS_DIR` (see above); archive it with an inventory once read.

- **Status:**
  - the candidates (B, U) with their costs, and every attempt and round;
  - the frozen rates and the FA controls;
  - the validation verdict;
  - the confirmation counts, the runs the launcher gave up and the attempts abandoned.
- **With `--judge`** (the looks need the fixation record and the versions record, and are
  refused without them):
  - the runs whose labels are pending (`labels_pending`, with the reason; the pre-warmed arm's
    runs show only a generic status until the final analysis) and the blocks waiting for labels;
  - the interval, resamples and seed used; each bootstrap look also gives its stream's seed and
    the resamples dropped for a zero SE;
  - the looks with their finite-cohort bounds;
  - beside each look, the bias allowance b, the interval widened by b and the pointwise readings
    about Δ(K\*) (equivalence inside ±(5 pp − b), direction excluding [−b, b], under the look's
    equivalence and direction minima, as the simulation's `pointwise_readings`); decisions never
    use them;
  - two missing-label sensitivities (`sensitivities`), at the primary's actual looks only, with
    its interval, levels, bootstrap streams and minima; they have no stopping path of their
    own:
    - `missing_labels`, the worst case for the contrast: per block
      [A_low − FA_high, A_high − FA_low] (low: an arm's missing labels not harmful; high:
      harmful), the interval from the lower bound over the lower bounds to the upper bound over
      the upper bounds;
    - `all_missing_harmful`: every missing label harmful in both arms;
  - the missing labels per arm;
  - the secondary contrasts at the stopping look, with `secondary_interval` and
    `secondary_status` beside them (amendment 3; amendment 4's summary no longer computes them:
    the mechanism cohort's final analysis replaces them);
  - the utility family.
- **Output.** Only counts, K, rates and intervals are printed: never a prompt, a response or a
  judge output.

## Study 2, amendment 4: numerical qualification, batch 20 and the mechanism cohort

The stage's rules are [amendment 4](study2-preregistration.md#amendment-4-a-corrected-numerical-criterion-batch-20-and-the-mechanism-cohort):
B = 20 as the fixed scope, the numerical qualification on K-only runs of fresh seeds, the
three-arm validation, the mechanism cohort, the time course, the drift diagnostic and the update
scalars. This section is its procedure, stage by stage: S1 discovery, S2 the numerical stage, S3
validation, S4 confirmation with the time course and the drift, S5 judging and the analyses. The
amendment-3 procedure above still holds where this section does not replace it. The selection,
the numerical stage, validation, the drift and the summary's status read K, numerics and utility
only; judge labels are read by the looks, the final analyses and the audit alone.

### Code, gates and what may change

- **Gate B** is the code every GPU run executes. Its record, `docs/study2-short-gate-b.json`,
  names the commit gate B was fixed at (`e380bcc`) and its GPU-code identity (119 files,
  sha256 `5cddd583…7680`; `code_identity.gpu_code_identity`). S1 ran from `aa17a99`, gate B as
  first fixed (GPU code `78b166a2…`); the change to twin A's consistency rule then touched only the
  numerical check's code and settings, and S1 stands. The S2 repeat ran from `aa4c007`, with
  gate B's current GPU code.
- **Gate C** (the analysis, the simulation, the audit, the fixation and the docs) changes no GPU
  path. A **GPU path** (`code_identity.is_gpu_path`) is every tracked file under `scripts/`,
  `src/zero_field_refusal/` (outside `analysis/`, but with `analysis/stage_a.py`), `configs/` and
  `docker/`, and `Dockerfile`, `.dockerignore`, `.python-version`, `pyproject.toml` and `uv.lock`;
  the analysis-only scripts (`study2_short_campaigns.py`, `study2_short_summary.py`,
  `study2_short_simulate.py` and the others listed in `code_identity.ANALYSIS_SCRIPTS`) are not.
  Neither is any file under `scripts/diagnostics/`, stand-alone diagnostics that no run executes
  (amendment 7 of study 3's registration).
  Gate C also leaves alone the modules the launcher and the GPU tools load: `analysis/__init__.py`
  and its imports, and `analysis/stage_a.py`. `tests/test_gate_c.py` checks both against the
  gate-B record; a gate-C change that fails it would void S2's qualification.
- **A qualification survives analysis-only commits.** The numerical check's records and the
  selection's reading of its verdicts hold at any later commit with the same GPU code
  (`code_identity.reuse_check`). A GPU-path change after the K-only runs voids them: see
  [the void](#the-void-and-the-repeat-of-a-numerical-stage).

### Pods

- **Created on request:** L40S, minCuda 13.0, not OC-AU-1, the dependency image
  of [containers](#containers), a volume at `/workspace` (150 GB for S1 and S2), host RAM of at
  least 64 GB (80 GB preferred), and `scripts/pod_deadline_watch.py` in the start command with
  `ZFR_DEADLINE_EPOCH` as the cap and the deadline in `/workspace/zfr-deadline` (see
  "Deadlines between campaigns" above). Right after the start, check the watcher's log line
  `zfr-deadline … auth True`, the GPU and its driver, and the host memory the container may use
  (the cgroup's limit, which can be below the machine's).
- **The Pod's fitness, checked right after the start.** This applies to every Pod that holds
  records or runs a test selection. A Pod that fails any check is stopped and replaced, and the
  stop counts as an infrastructure incident.
  - **The volume.** `findmnt -no FSTYPE /workspace` must be `xfs` or `ext4`. A fuse volume (MooseFS)
    was found on a US-MO-1 Pod in the dry run's [Part I repeat](reports/study3-dry-run-part-1-repeat.md).
    It ignores the umask, so files come out mode 666, and its metadata is slow: the launch
    journal's mode test fails there, and a test selection ran about 19 times slower than locally.
  - **CPU and disk.** A loop of 3·10⁷ Python additions takes at most 3.0 s, and 200 small files,
    each written and fsynced under `/workspace`, take at most 2 s:

    ```bash
    python3 -c 'import time
    t = time.perf_counter(); s = 0
    for i in range(3 * 10**7): s += i
    d = time.perf_counter() - t; print(f"cpu loop {d:.2f} s"); raise SystemExit(d > 3.0)'
    python3 -c 'import os, shutil, tempfile, time
    d = tempfile.mkdtemp(dir="/workspace"); t = time.perf_counter()
    for i in range(200):
        fd = os.open(os.path.join(d, f"f{i}"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.write(fd, b"x" * 100); os.fsync(fd); os.close(fd)
    ms = (time.perf_counter() - t) * 1000; shutil.rmtree(d); print(f"{ms:.0f} ms"); raise SystemExit(ms > 2000)'
    ```

  - **Throughput.** Once the code is checked out, run `tests/test_launch_journal.py` whole from the
    checkout through `project-run`, with `CUDA_VISIBLE_DEVICES=` and with `RUNS_DIR`, `DATA_DIR`
    and `TMPDIR` under `/workspace`. It must pass whole in at most 60 s of pytest time; it takes
    about 18 s on a workstation and 46 s on a fit L40S Pod.
- **The Pod's environment.** An SSH session does not see the Pod's environment variables: source
  `/etc/rp_environment` first (`pod_setup.sh` failed without it); the stage's drivers source it,
  and `/root/zfr-env.sh`, themselves.
- **SSH.** Use direct SSH to the Pod's public IP and mapped port. Read the host key from the Pod's
  logs and pin it in a known-hosts file before the first connection.
- **Code.** A Git bundle of the chosen commit, checked out clean under
  `/workspace/code/zero-field-refusal/<commit>/source` (the drivers check the commit and that the
  checkout is clean, as the runs read it). The numerical check needs the commit objects of the
  K-only runs' commits in that checkout: a shallow clone makes its code check refuse.
- **Detached work.** Start every campaign and driver detached, with the shared runpod-research
  skill's `run_detached.py` and an attempt directory under `/workspace/attempts/<name>`, never in
  the SSH session. Run no GPU job beside a campaign: generation, measurements and checks go
  between campaigns.
- **`DATA_DIR`.** Restore only the inventories the stage needs (amendment 3's list, the numerical
  seeds' streams once archived): `pod_setup.sh` verifies every file of `DATA_DIR` against them.
- **Results.** Download each stage's run directories, launch journals and outputs before writing
  the next campaign, compare their sha256 with the Pod's, archive them with an inventory, and
  check that `DATA_DIR` still matches its inventories (no stream grew). Then the Pod can stop.
- **Where the stage's scripts run.** The campaigns, the check requests and the summary run on the
  local record store: a full clone with every run so far (Stage A's included), the launch
  journals and `runs/numerical/`. They write under `$RUNS_DIR` (`runs/campaigns/`,
  `runs/analysis/`), never into the checkout's tracked tree.
- **CPU Pods** (the simulation) have no volume: download the output before stopping.

### Memory, disk and the run environment

- **GPU:** every part of the numerical check and the drift diagnostic caps its process's CUDA
  memory at 46 GB reserved (1e9 bytes). S1's first regular run at B = 20 used about 39.5 GB of
  the GPU.
- **Host memory, free before a driver starts** (MemAvailable, or the cgroup's free memory when
  smaller): 52.5 GB for the numerical check, 47.5 GB for the drift diagnostic
  (1.5 × 21 GB + 16 GB). A pre-warmed regular run keeps a pinned host copy of v₀ for the update
  scalars, about 6.7 GB at 1.5B.
- **Disk:** the numerical check keeps one case's checkpoints at a time (up to 31.4 GB at 1.5B,
  deleted when the case is done); its driver clears those an interrupted case left and needs
  40 GB free under its output directory. The drift driver needs 5 GB.
- **The run environment.** The check compares each K-only run's
  recorded environment (`meta.json`: the GPU names, the driver, `torch_cuda`, Python, the
  packages, and the container image when recorded) with its own. A difference stops it with
  `RUN ENVIRONMENT MISMATCH: <field>: …`: `plan` exits 1, `control` and `twins` record nothing and
  exit 7. Run the check on the Pod that made the K-only runs (or the same image, GPU type and
  driver), with the same `CONTAINER_IMAGE` as the launcher. After the first K-only run on a Pod,
  read its `meta.json` `hardware`: a null GPU name or driver where the check records one would
  refuse the check for good.

### w_max and the fixation record

- Every campaign kind needs `--fixation docs/study2-short-fixation.json` and `--runs-root`. The
  record is read whole (`analysis.study2_short.read_fixation`): a missing key, or a value of
  amendment 4's keys other than the code's, refuses.
- **w_max 0.47 is frozen since S1.** Every launcher campaign carries `header: {w_max: …}`, which
  the launcher logs with every start it makes; a replay under another w_max is refused
  (`incomplete`). Campaigns from before the headers (amendment 3's) record none and are not
  compared. The check request records its w_max too, which is printed, never read back.
- **Amendment 4's keys:** `mechanism` (the cohort, its interval family and the max-t level),
  `audit`, `candidates`, `numerical`, `validation`, `timecourse`, `drift` and `scalars`.
  `mechanism.level` is 0.975, the level the simulation's final run fixed
  ([below](#the-simulations-final-run)); until then it held a placeholder, 0.95.

### S1: discovery at (8, 20)

Locally, at the chosen commit (`R` and `THETA0` as in amendment 3's commands; campaigns and
summaries go under `$RUNS_DIR`, here `runs/`):

```bash
F=docs/study2-short-fixation.json
python scripts/study2_short_campaigns.py discovery --runs $R --theta0 $THETA0 --fixation $F \
  --runs-root runs --out runs/campaigns/study2-short-discovery-b20-u8.yaml
python scripts/study2_short_campaigns.py prep --seeds 2000 2001 2002 --batch 20 --commit SHA \
  --archived --out runs/campaigns/study2-short-prep-s2000-2002-b20.sh
```

The `discovery` campaign holds the primary arms' targeted round (15 runs) and the pre-warmed
arm's first round at the union of their rates (18 runs). On the Pod, after `pod_setup.sh` with
amendment 3's inventories and `check_runtime.py`, run the prep driver (it only checks the archived
streams of seeds 2000–2002 at B = 20) and launch the campaign detached:

```bash
. /etc/rp_environment
SRC=/workspace/code/zero-field-refusal/SHA/source
bash "$RUNS_DIR/campaigns/study2-short-prep-s2000-2002-b20.sh"
project-run "$SRC" python scripts/launch.py "$RUNS_DIR/campaigns/study2-short-discovery-b20-u8.yaml"
```

On the first run, check `meta.json` (the commit, a clean checkout, the image, the hardware),
`update_scalars.jsonl` (25 rows) and the GPU peak. When the campaign ends, download and archive
it (`runs/archive/inventory-2026-10-20-study2-short-discovery-b20-u8.tsv`) and write the summary,
which names the next step. It can be a further round (`repair`, `prewarmed` or the pre-warmed
arm's `targeted`, and after a numerical check the search allowance's `repair`, named
`…-allowance-<check>`): each is written the same way and runs in a Pod session of its own, with a
download before the next is written. S1 needed none.

```bash
python scripts/study2_short_summary.py --runs $R --theta0 $THETA0 --fixation $F --runs-root runs \
  --out runs/analysis/study2-short-summary-20261020-s1.json
```

### S2: the numerical stage

1. **The K-only runs and their streams,** written locally:

   ```bash
   python scripts/study2_short_campaigns.py numerical --runs $R --theta0 $THETA0 --fixation $F \
     --runs-root runs --out runs/campaigns/study2-short-numerical-b20-u8.yaml
   python scripts/study2_short_campaigns.py prep --seeds 2200 2201 2202 --batch 20 --commit SHA \
     --out runs/campaigns/study2-short-prep-s2200-2202-b20.sh
   ```

   On the Pod the prep driver generates seeds 2200–2202's training streams (25 × B = 500
   responses) and v₀ streams (50 × B = 1 000). Archive them with an inventory before any run reads
   them (`inventory-2026-10-20-study2-short-streams-s2200-2202-b20.tsv`), then launch the campaign:
   15 runs of `experiments/study2_short_k_only`, every selected rate on 2200 and each arm's U-rate
   on 2201 and 2202. Download and archive the runs.
2. **The check request,** written locally once the runs are in the record store:

   ```bash
   python scripts/study2_short_campaigns.py numerical-check --runs $R --theta0 $THETA0 \
     --fixation $F --runs-root runs --out runs/campaigns/study2-short-numerical-check-b20-u8.json
   ```

   It writes the request (`{"rates": …, "batch": 20, "U": 8}`, with `given_up` when the launcher
   gave up on a K-only run) and prints the driver's whole command, `runs_commits` (the K-only runs'
   commits: the driver's `--commit`), the w_max, the directory the check writes into and the
   selection's verdict there (`numerical/b20-u8/verdicts/<digest>.json`).
3. **The driver, written on the Pod** from the request copied unchanged to the Pod's
   `$RUNS_DIR/campaigns/`. It embeds the request's absolute path, and every part checks the
   request's content: a moved or edited request stops the plan before any GPU use.

   ```bash
   C="$RUNS_DIR/campaigns"
   project-run "$SRC" python scripts/numerical_check.py driver \
     --selection "$C/study2-short-numerical-check-b20-u8.json" --batch 20 --upper 8 \
     --commit SHA --out "$C/study2-short-numerical-check-b20-u8-driver.sh"
   bash "$C/study2-short-numerical-check-b20-u8-driver.sh"   # detached
   ```

   The driver checks the checkout, that no process holds the GPU and under 1 000 MiB of it is in
   use, the free host memory, clears an interrupted case's checkpoints and checks the free disk.
   Then `plan` (missing runs or streams, the GPU code, the run environment), the same-path
   controls, the twins' first attempts and the verdict; then one rerun, in a fresh process, of
   every control and case left incomplete, and the verdict again. It records the verdict before
   any non-zero exit. **Exits:** 1, a precondition or the plan; 5, a verdict with an incomplete
   rate or control; 6, a part failed (recorded; the driver goes on); 0, complete. A verdict that
   is not qualified is a result, not a failure. Run alone, `control` and `twins` exit 7 on a
   GPU-code or environment mismatch, and `verdict --retry-needed` exits 3 when no rerun is due.
   Give the Pod a deadline that covers the whole check and its reruns: a killed driver can be run
   again, and keeps every record that counts.
4. **Outputs** under `$RUNS_DIR/numerical/b20-u8/`: `controls/<lr>/attempt-<n>.json`,
   `twins/s<seed>/<optimizer>-<lr>/attempt-<n>.json`, `verdicts/<digest>.json` (one per selection;
   a selection's earlier verdicts move to `verdicts/history/<digest>/` and their marks still
   count; `verdict.json` is a copy of the latest, never read), the driver's log and `stale/`. A
   verdict at a batch the selection reached that it did not read, and that marks a rate unusable,
   leaves the stage incomplete: move a stray verdict aside, never delete it. A record
   counts only when written from a clean checkout with this GPU code, under the frozen settings,
   for the run located now; running the driver again keeps it. A `--dry-run` moves nothing.
   Download the directory with the request and the driver, archive them with an inventory, and
   run the summary: it reads the verdicts under `--runs-root`.
5. **Cases given up.** A K-only run the launcher gave up on is named in the request; `plan` notes
   it, and the parts record it incomplete, then not qualified (technical), without a GPU. An FA
   run on 2200 given up makes its rate's same-path control not qualified too (the control part
   then exits 0), and a seed whose only FA run is given up leaves every case of that seed
   incomplete. When every case is given up and no run is located, `runs_commits` is empty: the
   driver runs from the stage's checkout commit, which the `numerical-check` kind names. The code
   and environment checks are then skipped, with a note, and no part touches the GPU, but the
   driver still runs its preconditions on the Pod. The selection then takes the technical path:
   the search allowance, then `blocked`.

### The void and the repeat of a numerical stage

A GPU-path change after the K-only runs (S2's: the comparator of twin A's consistency rule, `e380bcc`) leaves their
records unusable: the tool refuses runs whose GPU code is not its own. At a clean checkout of the
new commit, both on the Pod and in the local record store:

```bash
python scripts/numerical_check.py void --batch 20 --upper 8 --runs-root "$RUNS_DIR" --dry-run
python scripts/numerical_check.py void --batch 20 --upper 8 --runs-root "$RUNS_DIR"
```

- **It moves, never deletes,** to `$RUNS_DIR/void/<the old GPU code's sha256, first 12>/` at the
  same relative paths: every numerical run at B whose recorded commit's GPU code is not HEAD's,
  `numerical/b{B}-u{U}/`, and the launcher's record of the campaigns that made those runs
  (`launch/<campaign>.jsonl`, `launch/<campaign>/`, and the campaign's file under `campaigns/`
  with its `-driver.sh`). It writes an inventory of every file moved (path, size, sha256) and a
  record beside them. Archive that directory with an inventory.
- **It refuses** on a checkout that is not clean; when no run's GPU code differs; when it would
  move a run of another role, or one whose GPU code cannot be read; when the old runs hold several
  GPU codes; when the destination exists; when a launcher journal it would move also names a run
  it leaves in place; and, for the researcher to decide, when S3 has started (a validation,
  confirmation or time-course run exists at B: voiding those means spending on S3 again) or when
  a check directory of another U exists at B. Without `numerical/b{B}-u{U}/` (a change between
  the K-only runs and their check) it moves the rest and records "no check directory".
- **Afterwards** the voided check's marks are gone with its verdicts: the selection calls for the
  K-only runs again (`numerical`), and their campaign starts with all its attempts, then the check.
- **As run on 2026-10-21:** the void went to `runs/void/78b166a28a61/`
  (`runs/archive/inventory-2026-10-21-study2-short-void-s2.tsv`). A new Pod restored S2's
  inventories with the archived streams of 2200–2202 (`prep --archived`), and ran the campaign
  `study2-short-numerical-b20-u8-repeat` and the check of `…-numerical-check-b20-u8-repeat.json`
  from `aa4c007`. The bucket never overwrites an object, so the repeated check directory went to
  a new prefix, `runs/numerical-repeat-20261021/`: the first repeat inventory lists the first
  S2's objects at the old paths as mismatches, and `…-repeat-v2.tsv` is the repeat's.

### S3: validation

After gate C passes, write and commit the versions record (amendment 3's
procedure above), from a clean checkout of the commit that validation and confirmation run from.
Then:

```bash
C=runs/campaigns  # locally; on a Pod, $RUNS_DIR/campaigns
python scripts/study2_short_campaigns.py prep --seeds 2100 2101 2102 2103 2104 2105 2106 2107 \
  2108 2109 2110 2111 --batch 20 --commit SHA --out "$C/study2-short-prep-s2100-2111-b20.sh"
python scripts/study2_short_campaigns.py validation --runs $R --theta0 $THETA0 --fixation $F \
  --runs-root runs --out "$C/study2-short-validation-b20-u8.yaml"
```

The prep generates the twelve seeds' streams; archive them before the runs read them. The
campaign holds every arm at its three rates on the twelve seeds: 108 runs with three arms. The
summary then gives the three-arm verdict and the next step: `confirmation`, the next candidate,
or `blocked` (a technical shortfall) with its reasons.

### Amendment 5: the candidate order at batch 20 after U 8

[Amendment 5](study2-preregistration.md#amendment-5-the-candidate-order-at-batch-20-after-u-8)
replaces the order after S3 failed on utility. The state machine visits, in this order:

1. the three arms at (U, B) = (8, 20), (7, 20), (6, 20) (`CANDIDATES`);
2. the primary pair at (7, 20), (6, 20) (`FALLBACK_CANDIDATES`); never at (8, 20), which S3's
   records decide;
3. otherwise `infeasible`, with the B = 20 feasibility report.

Each candidate has its own discovery, numerical check (`numerical/b20-u{U}/`, the bound 0.01·K\*
of its U; at U = 6, K\* = √30) and validation, and the procedure of S1–S3 holds at each, with
`u7` or `u6` in the campaign names. The pair at a U takes the three arms' triples there and
reuses every record of their visit: after a three-arm validation, its verdict is recomputed on the
same runs, with no campaign; otherwise it runs only what is missing at that U, its check if none
qualified its rates and its validation (72 runs). No rate run at another U enters it, and the
pair has no repair: a rate of its triples marked unusable (by its own check or before) stops
that arm, blocked when the mark is technical, failed when scientific. Before a
next candidate's campaigns, set its forecast (its checks and the confirmation) against what
remains of the budget; if it does not fit, stop with the price.

**The power forecast at U = 7 or U = 6**, before that candidate's confirmation: the simulation's
procedure at its rates with `--upper 7` or `--upper 6` (`scripts/study2_short_simulate.py`;
K\* and the half-span only, 8 by default). Its `summary.json` records U, K\* and the
half-span (`final` passes U to every part, records it in `final.json`, refuses `--write` at any
U but 8, and skips the comparisons with the archived runs, which are at U = 8); archive it with
an inventory. It reports at the committed level, the fixation record's `mechanism.level`
(0.975): the readings and the stress results there (`forecast_level`), its main answer in
`final.json`'s `forecast` (the required grid's `meets` and each cell's smallest coverage at that
level), no check stream, and the level the calibration rule would imply at that U only as
`recalibration_diagnostic`, never fixed. It is a report and decides nothing.

**The pair's recomputation at U = 8**, read on K, numerics and utility only:

```bash
python scripts/study2_short_summary.py pair-recheck --runs $R --theta0 $THETA0 --fixation $F \
  --runs-root runs --upper 8 --out runs/analysis/study2-short-pair-u8-recheck-20261022.json
```

**Which versions record a campaign uses.** There are two, each for its own commit:

- `docs/study2-short-versions.json` holds for `6fb20a9` (its record-only child of `4fbd19a`).
  Everything that runs on `6fb20a9`, the whole U = 7 candidate included, uses it: that checkout's
  default.
- `docs/study2-short-versions-2.json` holds for the commit of amendment 5's code and its
  record-only child. Write it from a clean checkout of that commit, and commit it alone:

  ```bash
  python scripts/study2_short_summary.py versions --judge wildguard --fixation $F \
    --supersedes docs/study2-short-versions.json \
    --reason "the candidate-selection code changed (amendment 5)"
  ```

  Campaigns written from that commit use it: the new checkout's default.

Run each campaign, summary and audit from the checkout of the record it is written under, and
pass `--runs-root`: once a confirmation has started, its later ranges, its time course, its looks
and the audit are refused under any other record than the one its issued campaigns named. A
confirmation start whose header names no record (every campaign written at `6fb20a9`) counts
under the first record: a confirmation started at `6fb20a9` is continued, summarized and audited
from that checkout, and the new code refuses it. Every record a block's starts named counts, so
a retry under the other record is refused too.

### S4: confirmation, the time course and the drift

#### The confirmation loop of a three-arm selection

The campaigns do not pause at the primary's looks: the looks are computed afterwards from the same
first 36 and 60 primary-supported blocks. Confirmation goes on until the mechanism cohort has 60
jointly supported blocks complete in order, or until all 80 blocks are started and complete.

1. **Write** `confirmation --blocks FIRST LAST --runs-root runs`, then launch it. A range holds at
   most min(60 − p₃, 80 − the blocks started), p₃ the started blocks that may still be jointly
   supported; a block is started once it has runs or belongs to a range the launcher started.
   `--after-look` is refused. Prepare and archive the range's streams first, as for validation.
2. **Let the blocks complete.**
3. **Read the counts** in the summary's `mechanism` and `confirmation`:
   `jointly_supported_in_order`, `possibly_jointly_supported`, `room`, `ended` and `ended_by`.
   `confirmation.room` counts the blocks of the ranges the launcher started as started, as the
   guard does; `mechanism.room` counts only blocks with runs, so it can be the larger. An issued
   range none of whose runs exists yet is completed by writing exactly that range again.
4. **Go on** while the cohort has not ended.

Label the runs as they complete (amendment 3's step 4). The summary with `--judge` reads the
looks as their blocks are labeled, and `final` once the confirmation has ended: the cohort ended,
its blocks labeled, and the primary stopped. With the primary pair alone (two arms), amendment
3's loop with its pauses applies.

#### The time course

Once the next step is `confirmation` and blocks 1–10 are in issued confirmation ranges (their
streams exist), write the time-course campaign beside the confirmation:

```bash
python scripts/study2_short_campaigns.py timecourse --runs $R --theta0 $THETA0 --fixation $F \
  --runs-root runs --out "$C/study2-short-timecourse-b20-u8.yaml"
```

It holds each arm's central (K\*) rate on seeds 3000–3009 (30 runs, 20 with two arms), the
endpoint's config with `study2_short.role=timecourse` and `schedule.evaluate_at=[1,5,10,25]`. A
run whose endpoint run the launcher gave up on is left out and named in the header. The scripts
split these runs from the others where they load them: they never enter a block, a look or the
mechanism cohort. The summary reports each against its endpoint run: `identical`, `consistent`,
`excluded` or `pending`.

- **The Pod** holds the streams of seeds 3000–3009 at B (check them with `check-streams`, or a
  `prep --archived` driver), on the GPU type and image of the endpoint runs: elsewhere their
  consistency can fail, which excludes runs.
- **Launch one run first** and read its memory peaks against the 46 GB rule before the rest: a
  time-course run resumes training after each intermediate evaluation, which no endpoint run
  does.
- **Label the time-course runs only for the final analysis,** by their explicit run directories,
  never with a glob over `study2_short` at a look: nothing would read their labels. The sealed
  labeling command seals their output, every arm's
  ([sealed output](#labeling-and-sealed-output)).
- A run and its endpoint run that both stopped before the first monitor reading show `excluded`
  with a missing-record reason; both are failures of their seed and arm in any case.

#### The drift diagnostic

Once, at the selected B, on a Pod between campaigns:

```bash
C="$RUNS_DIR/campaigns"
project-run "$SRC" python scripts/drift_diagnostic.py driver --batch 20 --commit SHA \
  --out "$C/drift-b20-driver.sh"
bash "$C/drift-b20-driver.sh"   # detached
```

The driver checks the checkout, that the GPU is free and 5 GB of disk; then `plan` (the host
memory: 47.5 GB), `prepare` (seed 2300's drift stream, 64 × B responses, and its v₀ stream,
50 × B, generated as the regular runs' streams are) and `run`. The result is
`$RUNS_DIR/diagnostics/drift/b20/result.json`; a failure leaves `failed.json` (exit 4 when out of
memory, 5 when the host memory falls short). Archive the streams and the result with an
inventory. It enters no decision.

### The summary under amendment 4

`scripts/study2_short_summary.py`, with the arguments of amendment 3, prints:

- **the selection:** the candidates (B = 20 only), and per attempt its rates once eligible, its
  selections, each check with the rates it was asked for, its state, its unusable rates with their
  kind (scientific or technical), the search allowance spent, carried or owed, the failures, the
  arms blocked by technical marks only, and the validation with the runs given up; the rounds;
  amendment 3's controls (reported, never used); the B = 5 runs not reached; the next step, or
  `blocked` with its rates and reasons, or `infeasible` with the B = 20 feasibility report;
- **the confirmation counts** and `mechanism`, on K and support only: the blocks started, those
  jointly supported in order and possibly so, the room, whether the cohort ended and by what, and
  its attrition by reason;
- **`timecourse`:** the counts by state, with the failed and missing runs per seed and arm;
- **`scalars`:** whether each regular run has `update_scalars.jsonl` with one row per update step
  and, for the pre-warmed arm, the departure at steps 5, 10 and 25. Amendment 3's runs have none;
  a gap is reported and gates nothing;
- **with `--judge`:** the looks on the primary arms, the descriptive continuation after the
  stopping look, and `final`, "not yet" with its reason until the confirmation has ended. Until
  then neither the printed status nor `--out` carries the pre-warmed arm's interpolated harm or
  missing labels. `final` holds the mechanism cohort's analysis and the time course with its
  labels;
- **a sealed error** instead, when a pre-warmed run's labels do not read for a reason other than
  a pending receipt (a malformed label file, say), or its label rows fail in a block: only the
  runs, `error (details sealed)` and the path of a sealed record
  (`$RUNS_DIR/sealed/summary-<UTC timestamp>-<judge>.json`, `--sealed-dir`) holding the
  traceback, and a non-zero exit ([sealed output](#labeling-and-sealed-output)). A primary arm's
  error is shown whole.

### Labeling and sealed output

Study 2 labels its runs only with the sealed labeling command, which wraps
`run_judging.label_runs` in the analysis package (a new script would be a GPU path):

```bash
python -m zero_field_refusal.analysis.study2_label_cli --judge wildguard RUN_DIR ... \
  [--workers N] [--sealed-dir DIR]
```

- **Never run `scripts/label_runs.py` on study-2 runs,** detached or not, nor `scripts/judge.py
  label` on items built from their responses. They print each run's unparseable count, rounds and
  errors, and their output also lands in a captured log (`run_detached.py`'s `stdout.log` and
  `stderr.log`). Neither they nor `run_judging` may change: they are GPU paths.
- **The command** has `label_runs.py`'s setup (`.env`, the judge config, the WildGuard token
  check, each run's prompt set), its three rounds, receipts, rounds records, lock and refusals
  (amendment 3's step 4). It labels the open runs first, in one call, and prints them exactly as
  `label_runs.py` does. Then it labels the sealed runs in a second call, with stdout, stderr and
  logging captured. If the open call's judge fails, the sealed runs are "not attempted".
- **Sealed runs:** every study-2 run of the pre-warmed arm, whatever its role; every time-course
  run (role `timecourse` on seeds 3000–3009), every arm; and, fail closed, every run whose
  `config.yaml` does not classify. A run on seeds 3000–3009 is open only as a confirmation run of
  a primary arm. No step before the final analysis reads a pre-warmed run's labels, whatever its
  role: amendment 4 fixes the confirmation's size, so no variance is planned from validation
  labels.
- **What a sealed run shows:** `{"run_id": …, "sealed": true, "status": S}` and nothing else. S is
  one of:
  - `labeled (details sealed until the final analysis)`: a receipt matching the label file;
  - `not complete: rerun the labeling (details sealed)`;
  - `error (details sealed)`, `busy` or `not attempted`.

  The status comes from the receipt and the technical flags alone, never from a count or from the
  rounds a run used. A completed receipt with missing labels is "labeled". A run that recorded no
  evaluation is "labeled": it has nothing to label. A receipt written again for its unchanged
  label file is "not complete", whatever its rounds; a rerun keeps or resumes it, and it is then
  "labeled".
- **A judge failure in the sealed call** gives every sealed run of that call the one status "not
  complete": the call stopped on a failed round, or the judge fails its re-probe after the call.
  The judge is re-probed with one synthetic judgment after the call (a fixed harmless pair, never
  a study item, through a temporary label file, removed after), and a WildGuard judge with the
  token check too; a request failure, a timeout or an answer that does not parse fails it. Which
  run a dying judge hits, and whether a run needed a retry round at all, depends on how its
  outputs parsed. Rerun once the judge is back. "error" stays for the errors that are no judge
  failure, run by run.
- **Residuals** that only a change of `run_judging`, a GPU path, could remove, accepted:
  - **An invocation-level bit:** a judge that fails during a call and recovers before the
    re-probe, or fails at a set time, can still make that call end "labeled" or "not complete"
    by the parsing of its runs taken together (more retry rounds reach a judge that is about to
    fail). Label sealed runs in batches, never one sealed run alone.
  - **Labeling duration:** retry rounds take time, for the call as a whole.
  - **Request counts:** the judge server's access log and its statistics count the requests of
    every round.

  These are documented limits of the blinding: batching reduces what they
  attribute to a run but does not remove them, and no scientific decision may use a labeling's
  timing, its request counts or a call's status.
- **These rules are procedural protection, not enforced access control:** nothing
  technically prevents running `label_runs.py` on a study-2 run or opening a sealed record. They
  hold because every operator follows them.
- **Exit:** non-zero for the open runs exactly as `label_runs.py`; for the sealed runs only for
  error, busy, not attempted or not complete, with a message that names the statuses and the
  record, never a count. Rerun after "not complete" or "not attempted" (bring the judge back
  first), and wait out "busy".
- **An "error" that stays after a rerun:** the run waits. An outcome-blind helper may open its
  record and relays only the exception's class and one remedy from this list: rerun; restore the
  label file from the archive; exclude the run by a recorded decision. Never a count or a value.
  The researcher decides.
- **The sealed record:** `$RUNS_DIR/sealed/labeling-<UTC timestamp>-<judge>.json` (the directory
  0700, the file 0600), one per invocation with a sealed run. It holds the sealed runs' full
  summaries, the captured output and any traceback; only its path is printed. The summary's and
  the audit draw's sealed errors write `summary-…` and `audit-…` records there. All three
  commands read `.env` and take `$RUNS_DIR/sealed` by default (`--sealed-dir` otherwise), and
  refuse a directory in the checkout outside `runs/`, where Git could track it.
- **Never open a sealed record before the final analysis** (but as above): do not print, search
  or compare it, and do not read a sealed run's receipt, rounds record or label file either,
  which hold the same counts. Archive the directory with an inventory, like the run directories
  (`archive.py` prints counts only). After the final analysis the details are read from the
  records, the receipts and the final analysis. Receipts, label files and rounds records stay as
  they are: their counts serve integrity and the retries.
- **The judge server's log:** never enable request or output logging (`--enable-log-requests`,
  `--enable-log-outputs`, or `VLLM_LOGGING_LEVEL` at `DEBUG`), and read the log only for the
  start and the deadline watcher's lines.

### The audit (S5)

Once `final` exists, the terminal audit is drawn, labeled outside the project and estimated:

```bash
python -m zero_field_refusal.analysis.study2_audit_cli draw --runs $R --theta0 $THETA0 \
  --fixation $F --judge wildguard --runs-root runs [--abandoned FILE] \
  --out "$RUNS_DIR/audit/study2-short-DATE"
python -m zero_field_refusal.analysis.study2_audit_cli agreements --sample DIR --fixation $F \
  --first GPT.json --second GROK.json --out "$RUNS_DIR/audit/agreements-DATE.json"
python -m zero_field_refusal.analysis.study2_audit_cli estimate --sample DIR --fixation $F \
  --reference REFERENCE.json --out "$RUNS_DIR/audit/estimate-DATE.json"
```

- **`draw`** refuses (exit 2) before the final analysis exists, reading no item's label. An error
  on the pre-warmed arm's label counts, which it reads for the summary's looks, refuses with only
  the runs, a generic status and a sealed record's path (`audit-…`, `--sealed-dir`;
  [sealed output](#labeling-and-sealed-output)). It writes `sample.jsonl`, `unique.jsonl`,
  `key.json`, `frame.jsonl` and `selection.json`, and prints only counts. The files hold
  responses to harmful prompts: keep them under `$RUNS_DIR`, archive them with an inventory, and
  give the labellers `unique.jsonl` alone.
- **`agreements`** reads the two labellers' labels by item id (null: a failed output) and lists
  the disagreements and failed outputs to resolve, and the 10 % of the agreements the researcher
  checks, drawn from the audit's seed.
- **`estimate`** reads the resolved reference labels (null: missing) and prints, per arm drawn,
  the draws, errors, missing references and bounds, and per estimand the bias bound, the
  descriptive estimate and the reference harm of the realized cohort. Item ids and numbers only,
  never a text.

The command line lives in the analysis package, because a new file under `scripts/` would be a
GPU path.

### The simulation's final run

The level of the mechanism's max-t intervals is fixed by one final run, after the S2 repeat named
the frozen rates. Its inputs are measured first, on K and support only, from S1's runs and Stage
A's grids at B = 20 (here `runs/analysis/study2_short_sim/2026-10-21-b20-inputs.json`):

```bash
python scripts/study2_short_simulate.py inputs --runs-root runs/study2_short runs/study2_stage_a \
  --batch 20 --rates A=1.49e-05,1.83e-05,2.24e-05 F=1.34e-05,1.62e-05,2e-05 \
  P=1.34e-05,1.62e-05,2e-05 --out runs/analysis/study2_short_sim/DATE-b20-inputs.json
python scripts/study2_short_simulate.py final --inputs runs/analysis/study2_short_sim/DATE-b20-inputs.json \
  --workers N --out "$RUNS_DIR/analysis/study2_short_sim/DATE-b20-final"
```

- It runs on a dedicated CPU Pod, where `--workers` may be every core. The primary rerun is
  compared with amendment 3's three archived decided runs, which the Pod's runs root must hold
  (`analysis/study2_short_sim/2026-10-16/full-r5/main`, `gate` and `bias`, each with its
  `summary.json`); a missing one is recorded as missing.
- In order: the mechanism's required grid on the calibration stream, one check at the level it
  chose (0.95 included; the borderline scenarios at 50 000 replicates), the primary rerun with the
  conditional truth, the validation gate and the bias allowance, their comparisons with amendment
  3's runs, and the stress grid. `final.json` holds the fixed level, or none, with the rule and
  the flagged rates.
- `--write` writes a fixed level into `mechanism.level` and the inputs record into
  `docs/study2-short-b20-inputs.json`, and nothing without a fixed level. Run without it on a Pod,
  the two are written in the checkout from `final.json` and the inputs record, with the tracked
  record's pin in `tests/test_study2_short_campaigns.py` updated, and committed before the versions
  record. Without a fixed level nothing is written and the outcome goes to the researcher.
- Download the output before stopping the Pod, and archive it with an inventory.

## Study 2, FA soft-control diagnostic

An outcome-free diagnostic outside the confirmation, of why two FA soft-target controls failed.
Its frozen specification is [study2-fa-control-diagnostic.md](study2-fa-control-diagnostic.md).
It changes no protocol, selection or run record, and this round's controls stay failed. It ran
once, and chose branch F. Its module, `fa_control_diagnostic.py`, now also holds amendment 4's
numerical check ([above](#s2-the-numerical-stage)), which reuses its D1 control and D3 twins;
the diagnostic itself keeps D3's bitwise twin-A gate.

- **Blind to outcomes.**
  - Only K, the KL between twin runs, norms, eigenvalues and v₀ statistics are computed.
  - Every evaluation reads benign panels only: the scoring panel's responses and the monitor
    panel.
  - No harmful prompt set, stream or generation is touched, and no refusal, harm or judge
    quantity is computed. `tests/test_fa_control_diagnostic.py` checks both.
  - On the Pod, the script also refuses any prompt set whose name contains "harmful".
- **Code and settings.**
  - `scripts/fa_control_diagnostic.py` runs the parts in the specification's order: `d1` (with
    D4's shared-state checks), `d2`, `d3` and `branch`.
  - The settings are `experiments/fa_control_diagnostic`, which is not a run config.
  - The runs re-examined are found by their archived configs, under `RUNS_DIR/study2_short` and
    `RUNS_DIR/study2_short_fa_controls`.
- **Prerequisites on a Pod.**
  - The P1 records of the targeted round and of the FA controls.
  - `DATA_DIR` restored as for the controls. The driver's header lists the inventories, and
    `plan` checks the streams D1–D3 read.
  - A free GPU and 40 GB of free volume. The driver checks both before it starts.
- **Memory rule.** Each part caps its CUDA memory at 46 GB reserved, in the GB of the run
  records' peaks (1e9 bytes). Each peak is recorded with `within_memory_rule`. An out-of-memory
  error under the cap has two outcomes:
  - the float64 reference falls back to float64 loss and accumulation, and the failed attempt
    is kept in its record;
  - any other part fails (exit 4).
- **Driver exits.** A failure of D1 or D3 stops the driver.
  - An incomplete D1 also stops it, with exit 5, before D2 and D3: a shared-state check failed,
    or the path criterion is undecided. The outcome is then incomplete whatever D2 and D3
    measure, so the driver records the branch, with the reasons, before it stops.
  - A D2 failure is recorded in `d2/failed.json` (the error, the peaks, the commit). The driver
    goes on to D3 and the branch, since no branch reads D2, and exits 6 at the end.
  - A failed expansion case records the three cases' branch, with the failure in its facts,
    before the driver stops.
  - The driver prints the branch. An incomplete one ends it with exit 5.
  - Before it starts, the GPU must show no compute process and under 1000 MiB in use.
- **Outputs** go to `RUNS_DIR/diagnostics/fa_control/`, which the stage's analysis never reads.
  - The parts write `d1/result.json`, `d2/result.json`, `d3/determinism.json`, `d3/<case>.json`
    and `branch/branch.json`, and the driver adds its log.
  - D3 keeps twin A's parameters under `d3/checkpoints/<case>/` (31.4 GB at 1.5B) until its case
    is done.
  - When the driver runs again, a part written at the same commit is kept.
  - A part written at another commit is moved to `stale/<its commit>/` and redone. `branch`
    never reads one.
- **Measurement states.** A branch is chosen, or excluded, only on complete evidence
  ([Revision 3](study2-fa-control-diagnostic.md#revision-3-frozen)).
  - Each D3 pair is within, exceeding or incomplete. It is within only when its perturbation
    survived fp32 rounding, after the single ×10 retry.
  - The same-path control is below, exceeding, at the bound or incomplete.
  - A divergence counts as exceeding: a non-finite loss, gradient, weight, K or D_twin.
  - `branch/branch.json` records these states in its facts, with the shared-state checks, the
    path criterion and the step of Revision 3's order that decided.
- **Incomplete outcomes.** The branch is `incomplete`, with no expansion, when:
  - the determinism re-run differs from its archived run. That stops D3 (exit 2), and the driver
    records the branch before it stops;
  - a case's twin A differs from its archived run;
  - a D1 shared-state check fails: the trainable set, v₀'s hash under both configs, the loss
    normalization, the initial optimizer state, or the recomputed first gradient against every
    control's record;
  - the path criterion is undecided: ‖(i)‖ or ‖(iii)‖ is non-finite, or ‖(iii)‖ is zero;
  - no FA 1.21e-5 pair exceeds, and one is incomplete, or else the same-path control is
    incomplete;
  - the control is below and the path criterion holds, no pair exceeds, and a pair of FA 1e-5 or
    AdamW is incomplete;
  - an output is missing, partial or of another commit.
- **Cost.**
  - About 2.4 GPU hours on one L40S: D1 ≈ 12 min, D2 ≈ 50 min, D3 ≈ 83 min.
  - FA 1.46e-5 adds ≈ 30 minutes when the three cases are ambiguous.
  - At most ≈ 45 GB allocated under the 46 GB reserved cap.

```bash
python scripts/fa_control_diagnostic.py plan --runs-root runs --no-stream-check  # loads nothing
python scripts/fa_control_diagnostic.py driver --commit SHA --out "$C/fa-control-diagnostic.sh"
bash "$C/fa-control-diagnostic.sh"                               # on the Pod
```

## Study 3: numerical qualification

The registered rules are [the registration's](study3-preregistration.md#numerical-qualification)
(with its [failure routes](study3-preregistration.md#failure-routes) and amendment 2, the
same-path control); the settings are
`configs/experiments/qualification.yaml`, and the harness and verdicts
`qualification` ([modules](architecture.md#modules)). `scripts/qualify.py` drives it; its
docstring holds the rules, and "A case restart" below has the complete list of the refusals of
`restart-case`.

- **The Pod.** As in [Pods](#pods), with its fitness checks right after the start, before any
  campaign.
- **The campaign.** One launcher campaign per candidate and stage, written whole by
  `scripts/study3_campaigns.py` ([below](#study-3-campaigns)): one cell per run, named
  `c<N>-<step>-<optimizer>-<rate:g>-s<seed>` with label `<cell>/s<seed>`, config
  `experiments/study3_k_only` with `study3.role=numerical` (or `engineering`), the arm, the rate,
  `train.batch_size=20` and `tracking.endpoint_checkpoint` set to one absolute directory on the
  Pod's container disk, on one of the candidate's numerical seeds (or the engineering seeds).
  Each FA rate needs its case on the first numerical seed, which holds the rate's same-path
  control. `plan` refuses any other campaign, and lists the cases, the GPU time
  and the disk.
- **v₀'s attention backend.** v₀ is estimated under the math attention backend in every process
  (`train._theta0_second_moment`, `runtime.V0_SDPA_BACKENDS`): the launcher run, the harness's
  parts and the same-path control, every micro-batch's forward and backward. Training stays on the
  production backend. On the GPU, 9 of 110 recomputations of v₀ differed under the production
  backend (the memory-efficient one) and 0 of 110 under the math backend, fresh processes included,
  at the same time and memory. The numerics setup records the policy (`v0_sdpa_backends`, in
  `meta.json`'s `inputs.numerics_setup` and in each part's `numerics_setup`), and the bridge's
  `numerics_setup` equality ([its conditions](study3-preregistration.md#numerical-qualification))
  compares it: a launcher run of code without the key does not bridge to a part with it.
  `launch.expected_numerics` leaves the key out, since it judges a run by what its config asks for:
  a run of older code stays classified, its declared stop included.
- **Each case, in order, one at a time:**
  1. a complete case (below) is skipped: nothing of it runs again (`complete` exits 0; 3 says
     no, and any other exit of `complete` or `decided` stops the driver). A case whose recorded
     parts were written under other GPU code or settings than the checkout's is stale: exit 10,
     and the driver stops there, neither rerunning nor skipping it. A case whose label
     (`<cell>/s<seed>` in its experiment) has a launcher run held in any launch journal, whatever
     its config hash, stops the driver with exit 11 naming the hold, before the launcher is
     called; nothing lifts a hold. For every case, `control-only` comes next (a case whose control
     alone is due, below): its exit 9, for an established technical case, complete or not ("A
     technical case's checkpoints", below), runs step 6's export at once, before any launcher call
     (`finish --kept`, the upload, `kept-check`), and nothing of the case runs; its exit 0, for a
     case whose only outstanding part is its same-path control (decided by its baseline, or with its
     baseline and rest recorded), runs that control alone, then step 6; its exit 3 goes on with step
     2; any other exit stops the driver;
  2. the `<dir>/<run_id>.pt` of the case's launcher run attempts that the launch journal shows
     superseded (ended technical, or never ended) are removed (`clear-stale`); a settled run's
     checkpoint (done, diverged, or completed but in conflict) stays. Then the free disk is
     checked;
  3. the launcher run alone: `scripts/launch.py CAMPAIGN --only LABEL --max-attempts 1`;
  4. the baseline part, in a fresh process;
  5. unless the baseline is already a scientific failure (an instability rule fired on the
     unshifted baseline, or the launcher run stopped by one: the case is decided), the rest part;
     for an FA rate on the first numerical seed, the same-path control, decided or not; each in
     a fresh process;
  6. the case's checkpoints are deleted (`finish`): every `<dir>/<run_id>.pt` of its launcher
     run and `<dir>/harness/<case>/`, only once its records are published and read back by the
     verdicts without error. A case whose verdict is technical (its own or its control's, the case
     holding no scientific failure), or an established technical case that is not complete
     (exported before any launcher call, step 1), keeps them for the diagnosis that starts its
     engineering cycle, exported and verified on the Pod's volume and in the bucket (below):
     `finish --kept "$KEPT"` exits 9 once each file still present has a verified copy on the
     volume (13 when the export is refused or fails), then the driver uploads that export
     (`scripts/archive.py`), checks both copies (`kept-check`, whatever the upload's exit) and
     stops before the next case, with 9 when both exit 0 and 13 otherwise; `finish --release`
     deletes them once diagnosed, only through that verified export. A complete case with a
     scientific failure whose same-path control is recorded technical counts as scientific and
     keeps them as the failed control's evidence ("A scientific case whose same-path control
     failed", below): `finish --kept "$KEPT"` exits 14 once they are verified on the volume (13
     when the export is refused or fails), the driver uploads and checks them the same way and,
     both exiting 0, goes on to the next case (13 otherwise); the driver itself never exits 14.

  A case is complete when the latest attempt of each part it needs is recorded and readable:
  the baseline, the due control and, unless decided, the rest, which must name the latest
  baseline record (a rest that ran nothing excepted).
- **Disk.** A case needs two endpoints (the launcher's and the harness baseline's), one state
  per engineering diagnostic step, one checkpoint of headroom for an atomic save, at
  `checkpoint_gb` (6.5 GB at 1.5B) each, and `disk_margin_gb` (4 GB): 23.5 GB for a numerical
  case. The driver checks it under the checkpoint directory before each launcher run and exits 1
  below it.
- **Attempts and reruns.** The driver never reruns anything. A part publishes
  `<record>.started` before it runs; a part whose latest attempt is recorded, technical and
  scientific records included, is never rerun, by the driver or by hand; one started without a
  record (the Pod, the network or the deadline stopped it) stops the driver with exit 8. Its
  rerun is the next attempt, by hand, with its documented reason, which the record carries, while
  the Pod and its container disk are the ones it started on (otherwise the case restarts,
  below). The same-path control is the exception: it reads no checkpoint, so its interrupted
  attempt is rerun this way after a lost container disk too (a case whose control alone is due,
  below). An interruption without a documented cause is recorded technical instead
  (`interrupted`). Then run the driver again. A failed launcher run stops the driver too; its
  rerun is a recovery round of its own with the reason and its incident, one attempt per round
  (["Launching runs"](#launching-runs)): a second loss, or a first loss after the round was used
  (by an engineering cycle's new launcher run, say), takes a further round with a reason no
  earlier round of the label gave. A launcher exit 1 over a missing checkpoint is a lost
  checkpoint to diagnose first ("A lost checkpoint", below), no round by itself.

  ```bash
  project-run "$SRC" python scripts/qualify.py part --campaign "$C/qualification-c1.yaml" \
    --candidate 1 --case s10200/adamw-1.45e-05 --part rest --attempt 2 \
    --infrastructure "attempt 1's process was killed: <the documented cause>" \
    --run-dir "$RUN_DIR" --checkpoints /root/ckpt/harness/s10200-adamw-1.45e-05
  project-run "$SRC" python scripts/qualify.py interrupted --campaign "$C/qualification-c1.yaml" \
    --candidate 1 --case s10200/adamw-1.45e-05 --part rest --attempt 1 \
    --technical "stopped without a documented cause"
  project-run "$SRC" python scripts/launch.py "$C/qualification-c1.yaml" \
    --only c1-numerical-adamw-1.45e-05-s10200/s10200 --max-attempts 1 \
    --recovery "attempt 1 failed: <the fixed cause>" \
    --incident-time 2026-10-29T03:12:00Z --cause "<the external cause, one line>" \
    --evidence "<a path or reference to the evidence>"
  ```

- **A lost checkpoint.** Exit 1 is shared. The driver's preconditions and disk check, `finish`'s
  and `control-only`'s refusals, a refused `restart-case` or recovery round, the launcher's
  conflicts, its "used up their attempts" (row 5) and its stop on request all exit 1. Exit 1 alone
  never triggers a restart, a recovery round or a rerun: the message identifies the procedure;
  bound records and incident evidence determine the route. `restart-case` and every recovery
  round are commands a person runs, never the driver's. A launcher run whose endpoint checkpoint
  file is gone is refused with "the endpoint checkpoint … is missing" and a note that points
  here; the table tells that message apart from the others. A message with any conflict line the
  note does not cover (row 6) belongs to row 6, even when it carries the note.

  | What the driver's log and the records show | What it is | What to do |
  |---|---|---|
  | `complete` exits 3 for the case. The launcher, called by the driver, exits 1 and its only conflict line is "the endpoint checkpoint … is missing" for the case's own launcher run. No `checkpoints-removed-<n>.json` of the case names that path. Evidence of a lost container disk: the Pod stopped or restarted (its event log), or an empty checkpoint folder after a container restart. The launch journal's end of that launcher run records `done`, or is settled (`"settled": true`, no return code) and was written after that stop. | An unfinished case's lost checkpoint (infrastructure) | Diagnose it, then `restart-case --infrastructure "<UTC time; what was lost; the evidence>"`. Then the launcher run's recovery round, `launch.py CAMPAIGN --only LABEL --max-attempts 1 --recovery "<the same reason>"` with `--incident-time`, `--cause` and `--evidence` (its first round, or a further one with a reason no earlier round gave). Then each part the restart names that the case needs, the baseline first, as its next attempt with that reason; a part the restart does not name takes attempt 1, without `--infrastructure`. Then the driver. The commands are in "A case restart". |
  | A case whose only outstanding part is its same-path control (decided by its baseline, or with its baseline and rest recorded), not technical (a technical one gets `control-only`'s exit 9 and its export before any launcher call; after a lost container disk, row 3). The driver's log shows `control-only` exit 0, then the control, then `finish`; or `part --part control --resume` exits 8 (the control was interrupted); or `control-only` exits 1 with the reason it names. The launcher is never called there, so no "is missing" message appears (the exception: a case decided by its launcher run's own declared instability stop gets exit 3 and the launcher is called, and the launcher accepts that run without its checkpoint, so no "is missing" appears either). | Not a restart | Nothing beyond the driver. For an interrupted control, run its next attempt with `--infrastructure "<reason>"`, then the driver. For `control-only`'s exit 1, follow the reason it names. `restart-case` refuses a case with a scientific failure: it keeps counting. See "A case whose control alone is due". |
  | The same sign as row 1, for a case with a completed failure (`restart-case` refuses), or a technical case whose only outstanding part is its control (row 2's technical branch) | Not a restart | A scientific failure stands. With a control due, the only scientific failure that shows row 1's sign is an interrupted rest whose trajectory results hold the failure: record that attempt first, `qualify.py interrupted --campaign … --candidate N --case CASE --part rest --attempt N --technical "<the interruption>"` (the results count), and the driver then runs the control alone. With its control recorded technical, the same: once the rest's attempt is recorded, the case is a scientific case whose control failed, and the driver goes on ("A scientific case whose same-path control failed"). A technical control recorded while the rest is due (the control run by hand first) makes no technical case ("A technical case's checkpoints"): after a lost container disk the rest cannot run and `restart-case` refuses (the control is a completed technical failure), so the route is the engineering cycle from the control's record (`supersede`, below), the checkpoints gone with the disk. A technical case gets `control-only`'s exit 9 and its export before any launcher call: its checkpoints went with the disk, so the export holds the manifest alone, its `lost` naming them with their recorded sha256, and `kept-check` verifies that ("Lost before the export"); the incident's cause and evidence name what was lost, and the `supersede` that opens the engineering cycle names in its `--diagnosis` the lost paths with their recorded sha256 (from the binding records) and the evidence of the loss. Then the engineering cycle (`supersede`, below). |
  | The message for a label whose case `complete` exits 0, and a `checkpoints-removed-<n>.json` names the path | `finish` removed it on purpose | Nothing. The driver never calls the launcher on a complete case, and nobody calls it on a finished case's label. `restart-case` refuses a complete case. |
  | A launcher start without an end at the loss; the next launcher call settles it technical and exits 1 with "used up their attempts" | The launcher run was in flight | No restart (`restart-case` refuses). A recovery round with the documented reason: `--only LABEL --max-attempts 1 --recovery` and the incident fields. |
  | The driver's own launcher attempt failed technically: a journal end of the launcher run with a return code and outcome `technical`, and no loss. `complete` exits 3 (the baseline is due), and the driver's launcher call exits 1 with "N runs used up their attempts; see RUNS_DIR/launch/<name>.jsonl" | A technical failure of the launcher run, not a loss | A recovery round (`--only LABEL --max-attempts 1 --recovery` with the incident fields) needs a documented external cause: the interruption's time, the cause and its evidence, none blank. Without one there is no automatic route. `supersede --record` must name a recorded part attempt of the case, and a launcher run that failed before any part has none: `supersede` refuses it with "is not a recorded attempt of this case". Such a run is swept up only when an engineering cycle opens from another case's technical record (a launcher run no part record classified). Otherwise the researcher decides. |
  | Any conflict line the note does not cover: "does not match its recorded sha256", "holds no endpoint-checkpoint record", "its numerics setup departs", "tracking.gate", "not run from a clean checkout", "records are incomplete", "config.yaml is missing", "one of several completed runs", "held, its outcome unresolved", any other "… is missing" (responses, rayleigh). Also a journal end of the launcher run with a return code that records `conflict`, and the message "… completed but cannot stand … the campaign stops": the launcher's own judgement of its attempt, which no lost container disk can write. | A provenance conflict, not a loss | No restart and no round (`restart-case` refuses an unsettled end that records `conflict`, any end that records `conflict`, settled or not, whose problem is not the missing endpoint checkpoint, a run whose numerics setup or Rayleigh record departs behind a settled missing-checkpoint end, and a case whose recorded parts `control-only` cannot bind to its launcher run, its exit 1). Stop and diagnose. Then the technical route (the engineering cycle, below), or the hold route for a held label. Stop the driver, not the Pod. An established technical case, complete or not ("A technical case's checkpoints"), is exported before any launcher call (`control-only` exits 9, then `finish --kept`), as long as its files are on disk, a corrupt checkpoint included (`matches_record: false`), so the launcher never refuses it. A case not yet technical keeps its checkpoints only on the container disk: what keeps them is the Pod, which stays up while the diagnosis runs (no planned stop, and the hard deadline stays). |
  | A technical case (exit 9 or 13), or a scientific case whose control failed (exit 14 or 13), whose kept checkpoints were lost | The technical mark stands; for a scientific case whose control failed, the scientific failure stands | See "A technical case's checkpoints", "Lost before the export"; for a scientific case whose control failed, "A scientific case whose same-path control failed", "What is kept". |

- **The deadline.** Start the Pod with the file-reading deadline watcher (see "Deadlines between
  campaigns" [above](#study-2-stage-a)), its cap in `ZFR_DEADLINE_EPOCH`. Before the driver,
  set the deadline in `/workspace/zfr-deadline` to cover the cases left, ≈ 39 min each and
  ≈ 11 min more per same-path control, with an hour to spare
  (`python3 scripts/pod_deadline_watch.py --set EPOCH`). A case the deadline stops leaves an
  interrupted attempt (above) and its checkpoints on the container disk, which a stopped Pod
  loses: the case starts again as a whole (below), once the loss is diagnosed ("A lost
  checkpoint", above); a case `control-only` would run alone (not technical, bound to its
  launcher run, only its same-path control outstanding) is the exception: it runs the control
  alone, with no restart (below).
- **A case restart.** When the Pod stopped during a case, its recorded parts cannot go on (the
  launcher's and the harness's checkpoints are gone). A missing endpoint checkpoint alone is a
  possible loss requiring diagnosis, never by itself a lost container disk: diagnose it first,
  and give the restart the interruption the diagnosis found ("A lost checkpoint", above, tells
  it from the launcher's other conflicts). A case `control-only` would run alone (not technical,
  bound to its launcher run, only its same-path control outstanding) is no case to restart:
  `control-only` runs the control alone and `restart-case` refuses it (below). `restart-case`
  records it: a
  `restart-<n>.json` in the case's folder names every attempt of the case's parts (baseline,
  rest and the case's control), each recorded one with its record's sha256, and the reason;
  those attempts are void, kept and listed in the verdict, and neither count nor block. Once the
  record is published, it logs the case's launcher run superseded in the launch journal (the
  analysis counts the record and the line for that run alone, a start of the case's own label
  under its candidate): the run stays on disk and in the journal, no longer counts for its label,
  and a recovery round of the run's own with the same reason and its incident starts a new one
  (if the command stopped between the two, running it again writes the journal's line). Only an
  interrupted case is restarted. A restart is refused, in this order:
  1. without a reason;
  2. for a case the engineering cycle kept on its launcher run (its resolved declared stop, or an
     outcome nobody could resolve): no rerun is opened;
  3. with no attempt that a restart or supersession record does not name already, and the
     launcher run's checkpoint not lost;
  4. for a complete case;
  5. for a case with a completed failure, scientific (it keeps counting: no reruns to pass),
     technical (the engineering cycle lifts it, below) or unreadable, an interrupted rest's
     trajectory results included;
  6. when `control-only` answers anything but its usual path (3): 9, an established technical
     case (its evidence is exported, and a technical failure goes to the engineering cycle); 0, a
     case whose control runs alone; 1 (its recorded parts cannot be bound to its launcher run, a
     provenance problem for diagnosis) or any other answer, carried with its reason;
  7. when the launcher run itself is a completed failure (its declared instability stop,
     scientific; another stop or a departing run, technical);
  8. when the launcher run's numerics setup, or its Rayleigh record when enabled, departs from the
     campaign's config behind a settled missing-checkpoint end (a provenance conflict, for
     diagnosis and the technical route);
  9. when the launcher run's journal holds an end `conflict` that no lost container disk explains
     (an unsettled one, with a return code: the launcher's own judgement of its attempt; or one,
     settled or not, whose problem is not the missing checkpoint);
  10. without evidence of an interruption (an interrupted attempt, or the launcher run's
      checkpoint gone from disk, which is enough even before the baseline started; one `finish`
      removed, recorded in its `checkpoints-removed-<n>.json`, is not);
  11. when the launcher run's endpoint checkpoint is on disk with another sha256 than its recorded
      one: a provenance conflict, not a loss, which goes to the technical route even when an
      interrupted part would otherwise allow the restart.

  Removing or replacing a mismatching checkpoint is no route to a restart: the conflict goes to
  diagnosis and the technical route. Then rerun
  the launcher run through its recovery round, run each part the case needs as its next attempt
  with the same reason, the baseline first (`part --resume` exits 8 for a part the restart
  names, and for an interrupted attempt names the record that names it), and run the driver
  again. A second loss of the same unfinished case is restarted the same way with its own reason:
  `restart-2.json` names every attempt (the first restart's included), the case's current launcher
  run is logged superseded, the launcher's further round (round 2, one attempt at `--max-attempts
  1`) runs its next attempt, and each part runs as its next attempt with the new reason. A launcher
  run in flight at the loss (a start without an end, no settled directory) is no attempt to restart
  (`restart-case` refuses): the launcher settles it technical, and a round with the documented
  reason is the whole route. A known narrow window: a child that stopped between publishing
  `terminal.json` and summarizing its terminal record cannot be restarted, because `restart-case`
  reports a completed technical failure. A case `control-only` would run alone (not technical, every
  recorded part bound to the located launcher run, the control due or interrupted) is never
  restarted, and `restart-case` refuses it: its control runs alone, with no restart (below). A
  non-technical case whose binding fails, a rest that names another run id, say, is refused too:
  `control-only` exits 1 for it, and `restart-case` carries that refusal with its reason (exit 1),
  writing nothing. The restart goes on only where `control-only` takes the usual path (exit 3).
  ```bash
  project-run "$SRC" python scripts/qualify.py restart-case --campaign "$C/qualification-c1.yaml" \
    --candidate 1 --case s10200/adamw-1.45e-05 --infrastructure "the Pod stopped during the case"
  project-run "$SRC" python scripts/launch.py "$C/qualification-c1.yaml" \
    --only c1-numerical-adamw-1.45e-05-s10200/s10200 --max-attempts 1 \
    --recovery "the Pod stopped during the case" --incident-time 2026-10-29T03:12:00Z \
    --cause "the Pod stopped: <the external cause>" --evidence "<the Pod's event log>"
  project-run "$SRC" python scripts/qualify.py part --campaign "$C/qualification-c1.yaml" \
    --candidate 1 --case s10200/adamw-1.45e-05 --part baseline --attempt 2 \
    --infrastructure "the Pod stopped during the case" \
    --run-dir "$RUN_DIR" --checkpoints /root/ckpt/harness/s10200-adamw-1.45e-05
  ```
- **A case whose control alone is due** (a decided case whose control is due included). This is
  a case whose only outstanding part is its same-path control (decided by its baseline, or with
  its baseline and rest recorded). After a lost container disk the launcher refuses its run (the
  checkpoint is missing). `restart-case` refuses a case with a scientific failure (the failure
  keeps counting), and for a passing one it would void the passing parts and rerun them, but the
  control reads no checkpoint. So, for the case that holds the control, the driver asks
  `control-only` first. It exits 0, printing the launcher run's directory, when:
  - the case holds the control, the engineering cycle has not resolved it, and the case is not
    complete;
  - the control's latest attempt is due or interrupted, and every other part the case needs is
    recorded as `complete` reads it: the baseline (the latest, not void) and, unless it decided
    the case, the rest bound to it;
  - the case is no established technical case ("A technical case's checkpoints", below): a
    scientific failure, the baseline's or the rest's, keeps counting, and its control runs alone;
  - the baseline record classified its launcher run as conforming (`launcher.classified` true,
    `launcher.kind` null);
  - the run each recorded part names (its experiment and run id, wherever RUNS_DIR lay) is the
    case's located run (its one settled directory, superseded nowhere), with the identity the
    record gives (its recorded checkpoint record; the file is read only when present, never
    required), done in the launch journal, its label held nowhere.

  The driver then runs `part --part control --resume` on that run, with all the control's own
  checks (the launcher run's numerics, the streams, v₀), and then `finish`: never
  `clear-stale`, the launcher, `locate`, the baseline, `decided` or the rest. The checks go in
  this order, and the driver asks every case (AdamW's too). Exit 9 first: an established technical
  case, complete or not, whatever its bindings ("an established technical case: export its
  evidence"); the driver exports it before any launcher call ("A technical case's checkpoints",
  below, defines it by its three conditions). Exit 3, the usual path: a case outside this state
  (one that holds no control among them); then a case decided by its launcher run's own declared
  instability stop (the launcher accepts that run without its checkpoint). Exit 1: a baseline not
  classified or of another kind, then a binding that fails. Otherwise exit 0. Exit 1 names the
  condition: the driver stops, nothing is rerun, and the case waits for a diagnosis. A present
  endpoint checkpoint of another sha256 than its recorded one stops the step too (exit 1, a
  provenance conflict), and the file is kept for the diagnosis. As any part, a recorded control,
  technical or scientific, is never rerun (a technical one keeps the checkpoints left: `finish`
  exits 9, or 14 in a case with a scientific failure). To resume such a case after a lost container
  disk, run the driver again; when the control was interrupted it stops with exit 8, and the
  control's next attempt runs by hand with the documented reason on the directory `control-only`
  prints (no restart, no recovery round), and then the driver again:

  ```bash
  RUN_DIR=$(project-run "$SRC" python scripts/qualify.py control-only \
    --campaign "$C/qualification-c1.yaml" --candidate 1 --case s10200/adam_frozen-1.83e-05)
  project-run "$SRC" python scripts/qualify.py part --campaign "$C/qualification-c1.yaml" \
    --candidate 1 --case s10200/adam_frozen-1.83e-05 --part control --attempt 2 \
    --infrastructure "the Pod stopped during the control: <the documented cause>" \
    --run-dir "$RUN_DIR" --checkpoints /root/ckpt/harness/s10200-adam_frozen-1.83e-05
  ```
- **A technical case's checkpoints.** A Pod stop takes the container disk, and the checkpoints
  with it, so a technical case's are exported before the driver stops: a complete case whose
  verdict is technical after its `finish`, and an established technical case, complete or not,
  before any launcher call. A case is an established technical case when no part the verdict
  would still read can change its technical status, that is, when one of these holds: (1) its
  baseline's gate fails, or its baseline did not complete (failed, or ran nothing), so the
  verdict reads no rest for it; (2) a rest recorded and bound to that baseline makes the whole
  verdict technical, not scientific; (3) its control is recorded technical, with every part the
  verdict still reads recorded (or not read: a baseline decided by itself, whose verdict reads no
  rest, as in (1)). In each, the case holds no scientific failure: a scientific verdict takes
  precedence (below, "A scientific case whose same-path control failed"). A complete baseline
  with only a technical comparison of its own, or a technical control, its rest due or interrupted,
  is none: the rest may turn the verdict scientific, and the case takes the usual path (or its
  control alone). So a technical control recorded while the rest is still due (the control run by
  hand first) waits for the rest: a passing rest makes the case technical by its control, a
  scientific one makes it a scientific case whose control failed. `control-only` exits 9 for an
  established technical case, and the driver runs `finish --kept`, the upload and `kept-check` at
  once, so files the launcher would refuse (a corrupt checkpoint) are kept as they are,
  `matches_record: false`. The export never marks a case complete: one
  that is not stays incomplete and technical, and its control or rest never runs.
  - **Where.** On the Pod's volume, `finish --kept /workspace/kept-checkpoints` writes them to
    `qualification/c<N>/<stage>/<case>/kept-<n>/` there, in their layout under the checkpoint
    directory, each copy read back and verified by its sha256 (a file of another sha256 than its
    recorded one is kept as it is, `matches_record: false`; nothing of the case's verdict changes).
    The driver then uploads the case's folder there with `scripts/archive.py` to
    `$ARTIFACT_REMOTE/kept-checkpoints/` (`--kept-remote` sets the prefix; the dry run passes
    `dryrun/<id>/kept-checkpoints`), with an inventory
    `$RUNS_DIR/archive/inventory-kept-<time>.tsv`, and `kept-check` verifies both, whatever the
    upload's exit: every copy re-hashes to the manifest, and every exported file has its inventory
    line with the same size and sha256, the status `ok` (its object's size and MD5) and its URL
    under the prefix `kept-check --remote` is given (the driver gives it the destination it gives
    `archive.py`; a line under any other prefix fails the check). Earlier `kept-<n>` folders of the
    case, verified or not, go to the bucket as well and are kept there until the study's report.
    `finish` refuses before any copy (exit 13) without `--kept`, for a relative folder, one under
    `$RUNS_DIR` or under the checkpoint directory, one on the checkpoint directory's filesystem, and
    one with less free space than the files take plus 1 GiB; an error in the export or its checks (a
    full volume, an I/O error, say) exits 13 too, a partial folder never called verified. Before the
    first case the driver itself exits 1 when `$KEPT` is on the checkpoint directory's device or
    lies under `$RUNS_DIR` or the checkpoint directory.
  - **The manifest.** `kept-checkpoints-<n>.json` in the case's folder, and the same JSON as
    `MANIFEST.json` in the export with a `SHA256SUMS`: the case, its technical reasons and
    `failed_control` (null; for a scientific case whose control failed, the control's reasons and
    the scientific failures), the binding records by path and sha256 (the launcher run's
    `summary.json` and `meta.json` with its checkpoint record, and the case's latest baseline, rest
    and control records; nothing under a run's `responses/`, `observations/` or `judge/` is read),
    each file (source, copy, size,
    sha256, the recorded sha256 and whether it matches), `lost` (each checkpoint a record names
    that is gone and that `finish` did not remove), `verified`, the host, the Pod and the
    provenance. Run again, `finish` reuses an export that still re-hashes and still holds the
    case's files as they are; otherwise it writes `kept-<n+1>`, and nothing is overwritten.
    Each `kept-check` publishes its receipt, `kept-check-<n>.json`: the manifest's and the
    inventory's path and sha256, its exit, and `pending-upload` when the volume copy is verified
    and the bucket copy is not (no inventory, or one without the export's verified lines).
  - **The two exits 13.** `finish`'s 13: the export was refused or failed, there is no verified
    volume copy, and the checkpoints are still only on the container disk, which a stop loses.
    The driver's 13 after `kept-check`: either the volume copy is verified and the upload is
    pending (the receipt's `pending-upload: true`), or the volume copy is not verified either
    (`pending-upload: false`), the worse state. After `finish`'s 13, fix the cause it names (room
    on the volume, the filesystem of `--kept`, an I/O error), then run the driver again or
    `finish --kept /workspace/kept-checkpoints` by hand; meanwhile the Pod is not stopped by plan,
    and the hard deadline stays. Checkpoints lost meanwhile take the last row of "A lost
    checkpoint" (above): a technical case's, or a scientific case's whose control failed.
  - **The Pod.** A planned stop, restart or deletion of the Pod waits for `kept-check` exit 0
    while a technical case, or a scientific case whose control failed, lacks it, unless its
    checkpoints went with the container disk before
    any export: it has nothing left to protect (see "Lost before the export"). An upload that
    cannot be verified never lifts the hard deadline: the verified volume copy stays and the
    pending upload is recorded in the receipt.
    To recover, repeat the upload before `kept-check`: run the driver again (`finish` reuses the
    verified export, then uploads and checks it), or run `archive.py` and then `kept-check` by
    hand with a new inventory:

    ```bash
    INV="$RUNS_DIR/archive/inventory-kept-$(date -u +%Y%m%dT%H%M%SZ).tsv"
    project-run "$SRC" python scripts/archive.py "$INV" "$ARTIFACT_REMOTE/kept-checkpoints" \
      /workspace/kept-checkpoints qualification/c1/numerical/s10200/adamw-1.45e-05
    project-run "$SRC" python scripts/qualify.py kept-check --campaign "$C/qualification-c1.yaml" \
      --candidate 1 --case s10200/adamw-1.45e-05 --inventory "$INV" \
      --remote "$ARTIFACT_REMOTE/kept-checkpoints"
    ```

    Both commands take the prefix the driver's `--kept-remote` gave; a dry run uses its own,
    `"$ARTIFACT_REMOTE/dryrun/<id>/kept-checkpoints"`, in both.
  - **Lost before the export.** Checkpoints a Pod stop took before `finish` leave the technical
    mark as it stands: `finish` exits 9 with `lost` naming them ("lost before the export: the
    technical mark stands; the diagnosis goes on from the records; nothing reruns"), the export
    holds the manifest alone, there is no restart (`restart-case` refuses a complete case), and
    nothing reruns.
  - **After the diagnosis.** `finish --release` deletes them, only through the manifest that a
    `kept-check` of exit 0 binds by its sha256, with that receipt's inventory unchanged (its
    sha256 re-hashed), every current file listed in the manifest with its size and sha256 and
    checked again right before its deletion (a new or changed file needs another export and its
    `kept-check`; an inventory gone or changed, the upload checked again); otherwise it exits 13
    naming the files or the inventory. Its `checkpoints-removed-<n>.json` names the manifest and
    the receipt. The volume copy may go once `kept-check` has passed and the diagnosis is done;
    the bucket copy is kept until the study's report.

  ```bash
  project-run "$SRC" python scripts/qualify.py finish --release \
    --campaign "$C/qualification-c1.yaml" --candidate 1 --case s10200/adamw-1.45e-05
  ```
- **A scientific case whose same-path control failed.** A case whose recorded parts hold a
  scientific failure (its baseline decided it, or a bound rest holds one) and whose same-path
  control is recorded technical is complete and counts as scientific: its rate fails on the
  scientific failure, and the verdict lists the failed control under the rate's `control`, with
  its reasons. It is no technical case: `control-only` never exits 9 for it, and nothing reruns or
  voids it (`restart-case` refuses a complete case, `supersede` keeps a case with a scientific
  failure whole, a recorded control is never rerun). Its checkpoints are kept as the failed
  control's evidence, as a technical case's are: `finish --kept "$KEPT"` exports them with the
  same manifest (`technical_reasons` empty, `failed_control` with the control's reasons and the
  scientific failures) and exits 14 once each file still present has a verified copy on the
  volume (13 when the export is refused or fails); the driver uploads the export and runs
  `kept-check`, and goes on to the next case when both exit 0, otherwise it stops with 13. Once a
  `kept-check` of exit 0 binds an export that holds the case's files as they are, `finish` exits
  0 for the case and copies nothing. The files stay on the container disk, counted by the next
  case's disk check, until `finish --release` after the control's diagnosis, through that
  verified export; its `checkpoints-removed-<n>.json` names the failed control. After a 14, the
  driver goes on with about 13 GB per such case still on the container disk. If the next case's
  disk check then stops the driver (exit 1), `finish --release`, run through the verified export,
  clears it: the volume and bucket copies keep the evidence. This is the route out of a disk-check
  stop; otherwise the release waits for the control's diagnosis. An interrupted rest whose
  trajectory results hold the scientific failure, its control recorded technical, is not yet
  complete: `control-only` exits 3, the usual path stops at the rest (exit 8; at the
  launcher, exit 1, after a lost container disk: row 3 of "A lost checkpoint"), and the rest's
  attempt is recorded with `interrupted --technical`; the case is then complete as above.
  - **What is kept.** The full export of a technical case: the launcher run's endpoint `.pt` of
    every attempt and the harness's folder, with the case's records, the failed control's
    included, bound in the manifest by path and sha256 (the control reads no checkpoint; its
    records stay under `$RUNS_DIR`). Amendment 5's storage clause, for technical cases only,
    covers this export as the evidence of the case's technical part, its failed control; the
    case's verdict stays scientific. Checkpoints lost before the export leave the scientific
    failure as it stands: `finish` exits 14 with `lost` naming them ("lost before the export: the
    scientific failure stands and the failed control's records stay; nothing reruns").
  - **The diagnosis.** The failed control is reported: the manifest's `failed_control`, and the
    rate's `control` in `verdict.json` and in the reading of the records
    (`analysis.qualification_records`). Its diagnosis is an open item of the qualification's
    report, not a condition of the next candidate's qualification: the case fails scientifically
    either way. A technical control in a case without a scientific failure takes the technical
    route ("A technical case's checkpoints", above).
- **A technical failure and its engineering cycle.** A completed technical failure (the
  repeat beyond 0.01·K\*, a failed bridge, a launcher comparison alone, an exception) is never
  rerun as it stands. Each candidate has one engineering cycle. It may also open outside the
  qualification, from a technical mark of a search, validation or confirmation run ("Study 3:
  campaigns", "A technical mark outside the qualification: its engineering cycle"); its
  supersession record is then the bridge `study3_summary.py cycle` writes, in this same format,
  `supersede` writes its journal lines, and the re-qualification below runs unchanged. After its
  diagnosis and fix,
  committed, `supersede` records it: `c<N>/supersede-1.json` names the technical record that
  started it (path and sha256), the diagnosis, the fix, the fix's GPU code (which must differ
  from the record's: the same path is not simply rerun), every record of the candidate made
  under the old GPU code in all its stages, passed ones included, with the cases re-qualified,
  and every launcher run of the candidate in the journal not of the fix's GPU code (another or an
  unknown code included; a case's runs are found by its label, `<cell>/s<seed>` in its
  experiment, in every journal whatever their config hash, so a fix that changes a hashed config
  key hides no earlier attempt), which it then logs superseded
  (run again with the same record, it only writes the lines that are missing). A second cycle
  for the candidate is refused: a technical failure left unresolved routes to the technical
  feasibility report. A part record classifies its launcher run only when it says so
  (`launcher.classified: true`, written once the launcher check returned); naming the run is not
  enough. A launcher run no part record classified (none names it: the Pod stopped before its
  baseline; or each that does raised before its launcher check, in the numerics setup, say;
  in any stage) is first resolved against what it was issued, never the current checkout's
  configs, code or environment: its journal start (label, config hash, header, command), its
  own `config.yaml` and `meta.json` (config hash, the campaign's header, commit, numerics setup)
  and the configs of its own commit, read from git (`git archive`, not the working tree), from
  which its command and the case's confirmation config are composed; its identity, config
  agreement and numerics are checked against those. A conforming run its declared rule stopped
  is a scientific failure (its case kept whole and decided); one that departs from what it was
  issued, or completed, is superseded; one nobody can resolve (a record unreadable or missing a
  field, no commit or one not in the clone, configs that do not give the start's config hash)
  keeps its case as it is, listed unresolved, and the run is held in the journal, so that no
  part and no recovery round reruns it: the hold applies to the run's label whatever the config
  hash, so a later campaign whose cell has another hash (a fix that changed a hashed key) starts
  nothing of it either. A change of the current configs after the run (the
  memory limit, say, outside the config hash) never changes how it is judged. A case that holds a scientific failure (as its bound, valid part
  records of any stage show it, or its resolved launcher run) is kept whole, its records and its launcher run (listed as kept, with why), and
  a cycle cannot start from one of its records:
  the failure keeps counting, so a scientific failure of P, which closes P–F alone, never blocks
  the cycle of a technical failure of A or F. Every case
  the cycle names re-qualifies under the new code from a new launcher run: under the old config
  hash, a recovery round of the launcher run's own (its first, or a further one at
  `--max-attempts 1`; its incident the cycle, its evidence the supersession record); under a new
  config hash (the fix changed a hashed key), the new run's base attempt, without `--recovery`
  (a study-3 campaign refuses `--recovery` over a run that is not used up), and the supersession
  record carries the cycle. Then each part runs as its next attempt with `--engineering-cycle` (its
  record names the supersession's path and sha256), not `--infrastructure`; a later interruption
  of a new-code attempt is rerun with `--infrastructure REASON`. The verdict lists the voided
  records under `supersession`.

  ```bash
  project-run "$SRC" python scripts/qualify.py supersede --campaign "$C/qualification-c1.yaml" \
    --candidate 1 --case s10200/adamw-1.45e-05 \
    --record "$RUNS_DIR/qualification/c1/numerical/s10200/adamw-1.45e-05/baseline-attempt-1.json" \
    --diagnosis "<the cycle's diagnosis>" --fix "<the fix, at commit SHA>"
  ```
- **Trajectory results.** A rest part publishes each trajectory's result as soon as it is done
  (`rest-attempt-<n>.trajectory-<name>.json`, bound to the case, the attempt, the launcher run,
  the baseline record and both checkpoints, the code and the settings), and stops at once,
  decided, on a valid scientific failure. When a rest attempt is interrupted, its results count:
  the verdict reads them, a rerun or a restart is refused when they hold a failure, and
  `interrupted --technical REASON` records the attempt with them. An interrupted part is no
  technical mark until `interrupted --technical` records it; once recorded, a case it makes an
  established technical case ("A technical case's checkpoints", above) is exported by the driver
  before any launcher call (`control-only` exits 9).
- **The commands.** The campaign file and the driver are written on the Pod, from the source
  checkout of the commit; the driver embeds the campaign's absolute path and sha256.

  ```bash
  C="$RUNS_DIR/campaigns"
  project-run "$SRC" python scripts/qualify.py plan --campaign "$C/qualification-c1.yaml" \
    --candidate 1
  project-run "$SRC" python scripts/qualify.py driver --campaign "$C/qualification-c1.yaml" \
    --candidate 1 --kept /workspace/kept-checkpoints --commit SHA \
    --out "$C/qualification-c1-driver.sh"
  bash "$C/qualification-c1-driver.sh"   # detached
  ```

  The engineering stage adds `--engineering` and, for its path diagnostics,
  `--diagnostic-steps 1,5,10,15,20`. The driver checks the checkout and the campaign's sha256, that
  no process holds the GPU and under 1 000 MiB of it is in use, that the free host memory reaches
  `host_memory_gb` (60 GB: θ₀, v₀, the second model and the p1 direction on the host, ≈ 25-30 GB at
  1.5B, and a checkpoint save's ≈ 5 GB, × 1.5, rounded up), and that a technical case's checkpoints
  can reach the bucket (`ARTIFACT_REMOTE` a gs:// path, `ARTIFACT_GCS_KEY` a readable file, `gcloud`
  on PATH) and that `$KEPT` is on the volume (not on the checkpoint directory's device, not under
  `$RUNS_DIR` or the checkpoint directory; otherwise exit 1 before the first case), then runs the
  cases. It writes the verdict on every exit but one: over a launch journal that is not whole (an
  unresolved torn line, or a line a published record requires missing) `verdict` refuses too (exit
  12), so no `verdict.json` is written then and an older one stays on disk.
  **Exits:** 0, every case complete; 1, a precondition (the host memory, the bucket's settings and
  the place of `$KEPT` included), the disk (or `df` giving no number), a `finish` that refuses a
  case not complete, or a case whose control alone is due that `control-only` cannot bind to its
  launcher run; 8, an interrupted attempt (or a part a case restart names); 9, a technical case, its
  checkpoints kept on the volume and in the bucket, verified (`finish`'s 14, a scientific case whose
  same-path control failed, its evidence verified on the volume, is no exit of the driver: it
  uploads and checks that export and goes on to the next case when both exit 0; the driver itself
  never exits 14); 10, a stale case; 11, a case whose
  label has a held launcher run; 12, a launch journal that is not whole, a torn line unresolved or a
  line a published record requires missing (`complete` refuses it, as every `qualify.py` command
  that reads the runs does, and so does the launcher; restored as [launching runs](#launching-runs)
  says); 13, a technical case, or a scientific case whose control failed, whose kept checkpoints
  are not verified, in one of two meanings ("The
  two exits 13", above): from `finish`, its export refused or failed (no verified volume copy, the
  checkpoints still only on the container disk, which a stop loses), or after `kept-check`, the
  volume copy verified and the upload pending (`pending-upload: true`) or the volume copy not
  verified either (`pending-upload: false`, the worse state); either way the planned teardown waits
  for `kept-check` exit 0, and the hard deadline stays; the launcher's own exit when it stops; any
  other, the exit of the command that failed.
- **Outputs** under `$RUNS_DIR/qualification/`: `c<N>/<numerical|engineering>/s<seed>/<arm>-<rate>/`
  holds each part's `<part>-attempt-<n>.json` and its `.started` file, `c<N>/verdict.json` the
  verdict (each rate's state, its cases with their 13 comparisons, kinds and reasons, its
  same-path control, every attempt of every part, the superseded records with the restart that
  voids them, the warnings; `--e-eng X` sets τ; a rate whose launcher runs are not all located
  is at most incomplete), `c<N>/engineering/verdict.json` the
  engineering stage's (e_eng, the path maxima, the margins), a technical case's, or a scientific
  case's whose control failed, `kept-checkpoints-<n>.json` and `kept-check-<n>.json` in its folder,
  and the driver's logs.
  Download and archive them with an inventory before the Pod stops. The kept checkpoints
  themselves are written on the volume under `/workspace/kept-checkpoints/qualification/` and go to
  the bucket from the Pod (above), never through this machine. Until the study's report they stay in
  the bucket, and on a volume only while the current stage reads them
  ([amendment 9](study3-preregistration.md#amendment-9-where-a-technical-cases-checkpoints-are-kept)).

## Study 3: campaigns

The order and the routes are [the registration's](study3-preregistration.md#seeds-and-order),
with its [validation, confirmation and stopping](study3-preregistration.md#validation-confirmation-and-stopping)
and [failure routes](study3-preregistration.md#failure-routes). The state machine
(`analysis.study3_progress`) reads every record under `RUNS_DIR` and names the next step.
`scripts/study3_campaigns.py` writes the launcher campaign of that step and refuses any other.
`scripts/study3_summary.py` prints the state, guards the confirmation between blocks, checks the
versions record before a launch and writes the versions record, the staged fixation and the
decision records. Their docstrings hold every rule. Both run from the source checkout of the
commit, clean (`$SRC`); campaign files and drivers go under `$RUNS_DIR/campaigns/`, never into
the checkout.

```bash
R="$RUNS_DIR"; C="$R/campaigns"
project-run "$SRC" python scripts/study3_summary.py progress --runs-root "$R"
```

`progress` prints the state as JSON: numbers, ids, paths and hashes only. It prints no n record,
so it holds no spread. It holds:
- the stage and its step: a campaign with its cells; a decision record to write, with its path
  and the fields the route fixes; a step that waits, or is blocked with its reasons and ways out;
  a terminal outcome, scientific or the budget's (a technical mark is a pause and ends nothing);
- the active arms and comparisons, the rates, the route, the closures and the problems. On the
  route, `withhold_pf`: the simulation's reading withholds P–F, so P runs up to the validation, the
  n record names A–F alone and the blocks hold four runs; and `reopen_pf`: candidate 2, reached by
  A–F's rule, tests again a P closed scientifically at candidate 1 ([amendment
  7](study3-preregistration.md#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run)).
  At the cap each comparison reads against its floor, from the simulation record's `reading`;
- the validation and confirmation runs that do not count for the commit they ran (`not_counted`,
  below);
- the launch journal's resolved torn lines (`resolved_torn_lines`): each line's log, line number
  and kind, nothing else of it (below, "A torn launch-log line").

Nothing is decided on a problem: the step stays blocked while one is listed. Read the state
before each command below.

- **The order of commands.** A campaign is written only when the step names its kind and
  candidate (`--candidate N`); any other is refused, with the step the state names.
  1. **Engineering.** `engineering` writes the starting rates on 10000-10003, run by the printed
     `qualify.py driver ... --engineering` ([above](#study-3-numerical-qualification)). The
     engineering record then names the records of its four checks: the dry run, the bridge,
     θ₀'s utility (`scripts/check_theta0_utility.py`) and the stage-0 check. The dry run's check
     references Part I's report and the integration replay's ([amendment
     7](study3-preregistration.md#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run)),
     so `project-run "$SRC" python -m pytest tests/test_study3_replay.py` passes in the Pod image
     first. On synthetic runs it drives the campaign writer, the summary script, `qualify.py
     supersede`, the readers, the simulation record's J check and the analysis freeze on files; the
     launcher's files and the numerical stage's qualification reading are given there.
  2. **The simulation record**, before the search: written by `scripts/study3_simulate.py record`
     ([below](#study-3-simulation)).
  3. **The search.** `search-round0`, then `search-round1` where a median missed.
  4. **The numerical qualification.** `numerical`, run through its printed `qualify.py driver`;
     then `repair` (one round per arm, amendment 3) for an arm whose K misses by more than 0.05
     over its 4 + 3 seeds once its qualification passed, run the same way. A rate whose cases
     fail only by the reference or by a declared instability stop keeps its `failing` verdict and
     goes on as `admitted`; with a stopped run it has no K and gets no repair ([amendment
     10](study3-preregistration.md#amendment-10-numerical-eligibility-after-candidate-1s-qualification)).
  5. **The validation.** The validation fixation and the first versions record, each committed
     alone; then `validation`, 28 whole blocks (40 when the simulation record says so), run
     after its printed `check` (below, "A run's own commit").
  6. **The choice and N.** The n record, and the budget record when N needs one.
  7. **The confirmation.** The confirmation supplement and a later versions record, each
     committed alone; then `confirmation`: one campaign file per block, and its driver `run.sh`.

  A stage's fresh seeds get their streams first, from the driver that `study3_campaigns.py prep
  --seeds S [S ...] --commit SHA --out FILE` writes; archive them with an inventory.

  ```bash
  project-run "$SRC" python scripts/study3_campaigns.py search-round0 --runs-root "$R" \
    --candidate 1 --out "$C/c1-search-r0.yaml"
  project-run "$SRC" python scripts/launch.py "$C/c1-search-r0.yaml" --max-attempts 1
  project-run "$SRC" python scripts/study3_campaigns.py numerical --runs-root "$R" \
    --candidate 1 --out "$C/c1-numerical.yaml"   # prints its qualify.py driver command
  ```
- **`--max-attempts 1`.** Every campaign runs with one attempt per run: the search and the
  validation as `launch.py FILE --max-attempts 1`, the qualification's campaigns through their
  driver, which passes it, and each confirmation block through `run.sh`. A failed attempt is a
  technical mark that the readers list, never retried by itself. Its rerun is a recovery round
  of its own with its documented cause and incident, always for the one run concerned, one
  attempt per round, each round's reason one no earlier round of the label gave
  (["Launching runs"](#launching-runs)). Each cell holds one run
  (`c<N>-<step>[-r<round>]-<optimizer>-<rate>-s<seed>`, its label `<cell>/s<seed>`), so no other
  run gets a second attempt from a round:

  ```bash
  project-run "$SRC" python scripts/study3_summary.py check --runs-root "$R" \
    --campaign "$C/c1-validation.yaml"
  project-run "$SRC" python scripts/launch.py "$C/c1-validation.yaml" \
    --only c1-validation-adam_frozen-1.83e-05-s10304/s10304 --max-attempts 1 \
    --recovery "<the documented cause>" --incident-time <UTC, ISO 8601> \
    --cause "<the external cause, one line>" --evidence "<a path or reference>"
  ```

  Every campaign file gets a new name: the writer refuses a name whose launch log exists, since
  the validation fixation pins the qualification's logs whole.
- **The decision records.** Each lies under `$RUNS_DIR/study3/decisions/`
  (`analysis.study3_decisions` fixes the schemas) and is written once through `write_decision`,
  which validates it against the evidence it names and the records it answers. A record a started
  campaign's header named, by its sha256, is immutable from then on.
  - `engineering`: the researcher's, once the four checks are recorded, each by its record's path
    and sha256; it passes only when all four passed.
  - `simulation`: the simulation's output (its fixation and inputs, the gates, the coverage
    screen, L, the validation size and the independent verification), written by
    `scripts/study3_simulate.py record` and checked against its fixation and results (the J
    check, [below](#study-3-simulation)).
  - `n-c<N>`: the precommitted spread script's output, with the fields the `choice` step gives
    (candidate, rates, comparisons, cohort), L, the spread rounded to 0.5 pp, N and the cap C(N),
    the smallest C with P[Bin(C, 0.88) ≥ N] ≥ 0.95 (the registration's amendment 7). A cap beyond
    the reserved seeds gives a failed record: a pause for a decision (`cap_beyond_reserved_seeds`).
    N is never cut; once an amendment extends the reserved seeds past the cap, the same record
    reads as passed.
  - `budget-c<N>`: the researcher's decision on the extra blocks and their price, `proceed` or
    `stop`, when N is above 90 or raised for power (the step names it then).
  - `closure-c<N>-<arm>`: no step names one (below, "Holds and closures").

  The others are written from a JSON file kept under `$RUNS_DIR` outside `study3/decisions/`; a
  refusal prints its reasons, and a second record of a kind is refused:

  ```bash
  project-run "$SRC" python scripts/study3_summary.py record engineering --runs-root "$R" \
    --from "$C/engineering.json"
  ```
- **The staged fixation and the versions record.** Each is written once at its fixed path in
  `docs/` from a clean checkout of a known commit, and then committed alone. Their reasons are ids
  (`validation_c1`, say), never free text.
  - Before candidate N's validation: `fixation validation --candidate N`, then the first versions
    record, `versions --out docs/study3-versions.json`, naming that fixation.
  - Before its confirmation, once the n record (and the budget record) is written:
    `fixation confirmation --candidate N` with the fixed seeds of the bootstrap, the audit draw
    and the draw of agreements, then a later versions record naming the supplement and
    superseding the one before it. The runs that started under the earlier record stay valid
    under it.
  - The validation and confirmation campaigns name the versions record that stands, by its
    sha256, and are refused while `study3_versions.versions_problems` finds a problem in it.

  ```bash
  project-run "$SRC" python scripts/study3_summary.py fixation validation --runs-root "$R" \
    --candidate 1
  project-run "$SRC" python scripts/study3_summary.py versions --out docs/study3-versions.json \
    --reason validation_c1 --fixation docs/study3-fixation-validation-c1.json --runs-root "$R"
  project-run "$SRC" python scripts/study3_summary.py fixation confirmation --runs-root "$R" \
    --candidate 1 --bootstrap-seed S1 --audit-seed S2 --agreement-seed S3
  project-run "$SRC" python scripts/study3_summary.py versions --out docs/study3-versions-2.json \
    --reason confirmation_c1 --fixation docs/study3-fixation-confirmation-c1.json \
    --runs-root "$R" --supersedes docs/study3-versions.json --supersedes-reason confirmation_c1
  ```
- **A run's own commit.** A validation or confirmation run under a commit other than the pinned
  one does not count ([the registration](study3-preregistration.md#voiding-and-supersession)). A
  run stops counting only through a record that names it: a journal line alone never takes a run
  out of the counts. A `superseded` or `held` line naming a study-3 run, whatever experiment it
  names itself, counts only with study 3's experiment, the run's label and a record that names the
  run; any other is a problem. The state compares each run's `meta.json` commit with the commit its header's versions record
  pins: that commit or a record-only descendant of it, from a clean checkout. Any other commit, a
  dirty checkout, a commit the checkout does not hold or no recorded commit, and the run is kept
  and listed (`not_counted`, with its reason and commit), and the step is blocked, naming it.
  `check` prevents it. It exits 0 only when the campaign file is a validation or confirmation
  campaign of study 3 and `versions_problems` finds no problem in the record its header names,
  for its step and candidate, the running checkout's HEAD included:

  ```bash
  project-run "$SRC" python scripts/study3_summary.py check --runs-root "$R" \
    --campaign "$C/c1-validation.yaml"
  ```

  It is run before every validation launch and every resumption of it, and before every recovery
  round of a validation or confirmation run, with the same `$SRC` as the launch. Inside `run.sh`,
  the guard already performs it for confirmation blocks. The limit: the launcher never reruns a
  completed run, so a completed run under another commit cannot be rerun. Such a run is kept,
  excluded from every count and listed; it is a technical mark outside the qualification of its
  arm, and the step is blocked and technical, naming the run, its reason and its commit. A run
  with no published outcome is rerun by a recovery round of its own (at `--max-attempts 1`, with
  its incident), after `check` passes. Otherwise the work pauses: a reissue waits for its writer
  (below, "Holds and closures"), and a published result is never rerun.
- **After the validation fixation.** The validation fixation pins the qualification's input
  records and, whole and by sha256, every launch log that holds a line of the candidate's
  qualification: a start of its numerical or repair campaigns, or a line naming one of their
  runs (an end, a `superseded` or `held` line, a recovery round of its label). A new input record
  for a fixed rate's case, or a new line in a pinned log, breaks the write-once fixation for good.
  So once candidate N's validation fixation exists, no command is run that writes to those logs
  or records:
  - no `qualify.py` command that writes records for that candidate: `driver`, `part`,
    `interrupted`, `restart-case`, `supersede`;
  - no `launch.py` command on its numerical or repair campaigns, a recovery round or a settled end
    included.

  The recovery form `launch.py FILE --only LABEL --max-attempts 1 --recovery REASON` (with its
  incident) is then for
  validation and confirmation runs, always for the one run concerned. `qualify.py verdict` does
  not touch the pins, as long as it runs from a checkout whose GPU code is candidate N's code in
  force. A verdict cache written under any other code makes the fixation invalid until `verdict`
  is run again under the code in force.
- **The confirmation and its guard.** The confirmation is written into a new folder, one block
  per file in seed order: the step's next block and, with `--blocks FIRST LAST`, those after it up
  to the cap, or the cells a started block still lacks (completed first, alone).

  ```bash
  project-run "$SRC" python scripts/study3_campaigns.py confirmation --runs-root "$R" \
    --candidate 1 --out "$C/c1-confirmation-1" --blocks 10400 10473
  project-run "$SRC" bash "$C/c1-confirmation-1/run.sh"   # detached
  ```

  `run.sh` runs one block at a time. For each it checks the block file's sha256, runs the guard,
  `study3_summary.py guard --runs-root "$RUNS_ROOT" --candidate N --seed S --campaign FILE`
  (with the writer's `--theta0` when the writer was given one, so that the guard reads the θ₀ the
  writer read), then `launch.py FILE --max-attempts 1`, and stops at the first non-zero exit. The
  guard exits 0 only when:
  - the step is candidate N's confirmation and names cells on S. A block not yet started must be
    the next one: never after the stop (every active counter at N, or the cap
    C(N) of blocks started), while a technical mark in a started block waits
    for its way out, while a run of a started block has not ended (below), out of seed order, or
    before a started block's missing cells. A started block completes its missing cells even
    after the stop;
  - the file is the block as written for that step: its header (study 3, candidate N, the
    confirmation, the n, budget and versions records it names, each still of the sha256 it
    names) and exactly the step's cells on S;
  - `versions_problems` finds no problem in the versions record the header names, for candidate
    N's confirmation.

  Otherwise it prints why, and the driver stops. Read the state, take the way it names, and write
  the blocks left again, under new names.
- **A block the Pod stopped mid-run.** A run the launcher started and never ended is pending, so
  the state names no new block (a started block's missing cells are still completed first): the
  writer and every guard refuse. The registered stop is checked between blocks, and the counters
  wait for such a run, and for a run whose recovery round is opened and that has no later attempt
  yet. The state names the block, the run and its way out. The run is rerun by its recovery
  round, from its own block file, with the interruption as its reason, once `check` passes on
  that file:

  ```bash
  project-run "$SRC" python scripts/study3_summary.py check --runs-root "$R" \
    --campaign "$C/c1-confirmation-1/<block file>"
  project-run "$SRC" python scripts/launch.py "$C/c1-confirmation-1/<block file>" \
    --only LABEL --max-attempts 1 --recovery "<the interruption>" \
    --incident-time <UTC, ISO 8601> --cause "<the external cause>" --evidence "<a reference>"
  ```

  The launcher first settles the run's start with the outcome its run directory gives (technical
  for a run left unfinished), and starts its next attempt only when that outcome is technical.
  The block's missing cells, if any, are then written again by the writer, alone, under a new
  campaign name. A block file or a `run.sh` already run is not run again.
- **A torn launch-log line.** A torn line is any line of a launch log that holds no record (its
  definition: [launching runs](#launching-runs)), a JSON object whose identity is not all
  strings included, such as a start whose run id is a list (a malformed value; a removed key or
  an invented event is no torn line, and the journal is never edited by hand). A crash in the
  middle of an append leaves one, the bytes after a log's last newline; the launcher's next
  append to that log ends it with a newline and names its bytes in a `torn` record. While any
  log holds a torn line that no `torn_resolved` record names, the launcher refuses with exit
  12, and so does every `qualify.py` command that reads the runs but its two journal repairs.
  The procedure:
  1. Restore what the line's kind lost, in the order its kind takes (the per-kind list in
     [launching runs](#launching-runs)):
     - `held`, `superseded`: run the writer again first, then resolve. The two journal repairs,
       `qualify.py supersede` with the same record and `restart-case` again for the case, run
       while the torn line is still unresolved and write only the missing line.
     - `end`, `start`, `recovery`, `stop`: resolve first, since the launcher refuses until then;
       then relaunch with the same command, or read on.
     - an unknown kind (`null`): read the run directories of the campaign's labels against the
       readable records first, as the per-kind list says. Such a line may be a lost `held` or
       `superseded` line cut before its `"event"` (each record's `"time"` comes first). If one
       is lost, take that kind's order: run the writer again, then resolve. Otherwise resolve
       first, then relaunch. Either way, the resolution's reason says what was found.
     - `torn`: resolve; the resolution writes the missing `torn` record, and nothing else is
       restored.
     - a lost `torn_resolved`: the line it resolved is unresolved again. Resolve that line again;
       nothing else is restored.
  2. Record the resolution:
     `python scripts/launch.py --resolve-torn LOG --offset N --sha256 HEX --reason TEXT`. It
     writes the line's `torn` record if none names it yet, then its `torn_resolved` record.
  3. Study 3's analysis then accepts the line. It reads each log as the launcher does (split at
     `\n` alone, each line decoded strictly as UTF-8), so a line's number counts every line
     before it, a torn one included, as a closure record cites it. A torn line before the last of
     its log is accepted when a `torn` record names its offset, length and sha256 and a
     `torn_resolved` record names its log, offset and sha256: the run records list it with its
     kind and its line, and it is no problem. Any other torn line before the last (a blank line
     or a JSON value that is no object included) is a problem named by its log and line, never
     its bytes, and blocks: what it held cannot be shown. The last line of a log, with no newline
     after it, is skipped, as the launcher's records skip it. The `torn` and `torn_resolved`
     records are no problem and void nothing.

  What the lost line held stays lost: a lost `start` leaves a run directory with no start, which
  the analysis names and blocks on. For a qualification log a validation fixation pins, any
  change of the log after the fixation, the repair and the resolution included, breaks the
  fixation for good: the log is pinned whole, by its sha256. So resolve every torn line, the last
  line of a log included, before the validation fixation: while any log holds one, the builder
  (`study3_summary.py fixation validation`) refuses and writes nothing, naming each such line by
  its log and line. Once the fixation exists, its reader (`check`, `guard`, the campaign writer)
  names such a line the same way, with the command that resolves it; in a log the fixation pins,
  the resolution changes that log, so the fixation breaks.
- **A missing journal line of a published record.** `qualify.py` publishes a cycle's or a case
  restart's record first and appends its journal lines after it. A crash between the two, or an
  append that wrote no byte, leaves the record without its lines and no torn byte. The launcher
  and every `qualify.py` command that reads the runs reconcile the published records under
  `$RUNS_DIR/qualification/` with the journal (`launch.missing_lines`) and refuse with exit 12,
  naming each missing line by its record's path, its event and its run id, until the record's
  own repair restores it: `qualify.py supersede` run again with the cycle's record (its
  `launchers`' `superseded` lines and its `held` lines), or `qualify.py restart-case` for the
  restart's case (its launcher run's `superseded` line). Nothing needs resolving afterwards:
  there is no torn line. A published record that cannot be read blocks the same way, named by its
  path, and so does a record entry that names no run (its label, experiment or run id is not
  there, or not a string), named by its record's path and event; both are fixed by hand, since
  no command can tell what they should hold. The reconciliation reads `$RUNS_DIR/qualification/`
  alone: a record published under a custom `--out` is outside it, and the driver never passes
  `--out`. A `supersede` or `restart-case` run by hand while a launcher is running can stop that
  launcher with exit 12: if the launcher reads the journal after the record is published and
  before its lines are appended, it finds them missing. Run the launcher again once the command
  has ended.
- **A technical mark outside the qualification: its engineering cycle.** A technical mark of a
  search, validation or confirmation run, in any arm, stops the route. Its registered route is the
  candidate's one engineering cycle ([the registration](study3-preregistration.md#failure-routes):
  diagnosis, a fix, re-qualification of the affected cases and reruns of the affected runs; one
  cycle for every mark of the stage). In order:
  1. **The step.** It is blocked and technical, with the reason `engineering_cycle`, naming each
     run (its id, label, arm and step). Its ways out are each mark's recovery round where the
     launcher opens one (after `check` for a validation or confirmation run: the run's own at
     `--max-attempts 1`, with a reason no earlier round of its label gave and its incident), and
     the cycle: the record `cycle-c<N>`, its fields (the candidate, the marks' stage, every mark)
     and the stages started, each of which needs a claim (`--stage STAGE=reuse|rerun[:ARMS]`,
     the marks' stage a rerun of every marked arm, the qualification's all or nothing, the search
     never rerun once a validation fixation exists) with its justification (`--justification
     STAGE=TEXT`, one line). No closure is written; one written anyway is a problem that decides
     nothing.
  2. **The fix, committed first.** The cycle needs a clean checkout of the fix whose GPU code
     differs from the code in force; a fix that leaves the GPU code unchanged is no cycle (below).
     The right to a cycle is judged on the existing chain of engineering cycles, the code in force,
     never on the fix's code. From the fix's checkout, until the cycle is written, `progress`,
     `check`, the guard and the campaign writer report the chain problem ("not the end of the
     chain of engineering cycles") and launch nothing: that is expected, and nothing is launched
     meanwhile.
  3. **The records it lists.** The cycle record lists every affected run by its files' paths and
     sha256 (`config.yaml`, `meta.json`, `terminal.json`) and its journal lines; a file that is not
     there is listed as absent (`{path, absent: true}`), never refused. What keeps counting is
     listed as kept: the scientific failures, held runs, the runs of a closed arm and the stages
     reused. A byte-identical copy from the archive, or a re-download, keeps every listed sha256;
     new or changed evidence makes the cycle invalid and stops the route: restore the verified
     original snapshot, or authorize the new evidence's provenance explicitly, and never delete
     new evidence to get an absence marker back.
  4. **The call.** On the machine holding `$RUNS_DIR`, from the clean checkout of the fix, first
     with `--dry-run` (it prints the record and the bridge and writes nothing), then without:

     ```bash
     project-run "$SRC" python scripts/study3_summary.py cycle --runs-root "$R" --candidate 1 \
       --diagnosis "<one line>" --fix "<one line, at commit SHA>" \
       --stage search=reuse --stage qualification=rerun --stage validation=rerun:F \
       --justification search="<why>" --justification qualification="<why>" \
       --justification validation="<why>" --dry-run
     ```

     It writes the cycle record (`$RUNS_DIR/study3/decisions/cycle-c<N>.json`), then its bridge
     (`$RUNS_DIR/qualification/c<N>/supersede-1.json`, `qualify.py`'s format), and prints both
     paths and the journal repair. Run again with the same arguments over a record whose bridge
     is missing, it writes only the bridge. It refuses while a run of the candidate is still in
     flight, in any stage and arm (its start in the journal, no end line, nothing settling it),
     and names each: the record pins the files of the runs it lists, and a run that ends later
     would change them. Wait for the launcher to end those runs, then call it again.
  5. **The repair.** Run the printed command where `$RUNS_DIR` is, with the same `--record` path
     (the bridge names the cycle record by its absolute path on the machine that wrote it):
     `qualify.py supersede --campaign <engineering campaign> --candidate N --engineering --case
     <its first case> --record <the cycle record> --diagnosis D --fix F`. It writes the bridge's
     `superseded` lines. Until it has run, the launcher and every `qualify.py` command exit 12
     ("A missing journal line of a published record", above).
  6. **The re-qualification.** Under `qualification=rerun`, as for a cycle of the qualification
     ("Study 3: numerical qualification", "A technical failure and its engineering cycle"): every
     voided case re-qualifies under the fix, its launcher run through its own recovery round, its
     parts with `--engineering-cycle`; the cases holding a scientific failure are kept whole. Under
     `qualification=reuse` there is none: the qualification is read under the old code.
  7. **The reissue.** The campaign writer issues the superseded runs' cells again in the
     route's order: a rerun search first, before the re-qualification; the validation's and the
     confirmation's cells once the qualification passes (at once under `qualification=reuse`).
     For each label whose latest attempt the cycle affected and whose attempt is used up, it
     prints its own recovery round, after `check` and before the file's launch (`--only` the
     label at `--max-attempts 1`, the reason `engineering cycle c<N>`, the cycle's time as the
     incident's, the fix as its cause and the bridge as its evidence); a confirmation block's
     `run.sh` runs it after the block's guard. A label under a new config hash is a new run and
     takes the plain launch. A validation or confirmation reissue under the fix needs a new
     versions record first: `check` refuses a descendant of the pinned commit that changes more
     than the record (below, "A cycle after the validation fixation"). The round's
     `--incident-time` is the cycle record's `written_at`, the time by the clock of the machine
     that wrote the cycle. The launcher refuses an incident time later than its own clock's now
     ("incident_time is later than now"): when the machine that launches runs behind the one
     that wrote the cycle, the round is refused and nothing runs (it fails closed). Fix the
     launching machine's clock and run the same printed command again, unchanged (its now is
     then not earlier than the cycle's `written_at`); never change the cycle's `written_at` or
     the round's `--incident-time` to get past the refusal.
  8. **After the cycle.** The cycle is spent once the candidate has a supersession record,
     whichever path wrote it. A technical mark after that is a pause: the step is blocked and
     technical, with the reason `unresolved_after_cycle`, and offers only its recovery round where
     the launcher opens one. The writers refuse a second cycle until a writer for it exists: an
     implementation pause, not a closure.
- **A cycle after the validation fixation: its successor records.** A cycle moves the
  candidate's GPU code, so every write-once record of the old code goes stale: the validation
  fixation, the confirmation
  supplement, the n and budget records and the standing versions record (the cycle record lists
  each by its path and sha256). Each stale record stays as it was; its successor is a new file
  that names it and the cycle (`supersedes`: its path, its sha256, the reason `cycle_c<N>` and the
  cycle record), and carries the fix: `docs/study3-fixation-validation-c<N>-2.json`,
  `docs/study3-fixation-confirmation-c<N>-2.json`, `study3/decisions/n-c<N>-2.json` (the
  `n-c<N>-2` record) and `budget-c<N>-2.json`. Once the candidate has a cycle record, its n and
  budget records, its fixations and a versions record naming its fixation are written only under
  the cycle's fix (the code in force for its runs), and nothing is written under the old code of
  any cycle: `record`, `fixation` and `versions` refuse it, so what the cycle lists as stale
  changes only when a file is tampered with (the cycle then reads invalid). The search is never
  rerun after the fixation (`search=reuse`), and the fixed rates stand: a repair the re-qualified
  runs would call for is listed (`repair_not_applied`) and never issued. P alone may leave the
  fixed rates: once the re-qualification closes P (before the confirmation, a scientific
  failure; a technical mark closes nothing), the validation fixation's successor leaves P out and keeps A's and F's rates; the record before it keeps P's
  rates and states, and P's records stay. With the successor in force, the state reads P's
  closure again from the rates the record before it fixed (a repaired rate included), so A–F
  goes on; a successor that leaves out a P whose rates are qualified or still waiting is refused
  by its reader, and the state blocks at the repair. The successors come in this order, each
  written when the state's step names it and committed alone (one written out of turn is refused
  with the step the state names):
  1. the validation fixation's successor (`study3_summary.py fixation validation`, once the
     re-qualification has passed) and its versions record (`versions --supersedes ...
     --supersedes-reason cycle_c<N>`, its `--fixation` the successor);
  2. the validation rerun, issued under them (the claim `validation=rerun`; none under `reuse`);
  3. the spread, N and budget decisions, as the step names them (`record n-c<N>`, then `record
     budget-c<N>` when N asks for one: the next successor when the record in force is stale, its
     file carrying `supersedes`), on the validation that counts; a stale budget record that the
     new N no longer asks for (N at most 90 and not raised for power) needs no successor;
  4. the confirmation supplement's successor (`fixation confirmation`, the same seeds) and its
     versions record.

  **The confirmation boundary.** Once a block of the candidate's confirmation has started, the
  candidate, the rates, the started seeds and the cohort's order never change: there is no
  candidate 2, no new N and no reopening of the validation without an explicit prospective
  amendment, which the code never infers. `cycle` refuses `validation=rerun` and `search=rerun`;
  a re-qualification failing scientifically blocks the route with the reason
  `confirmation_fixed` instead of moving the candidate or closing an arm; the n, budget and
  supplement successors differ from their predecessors only in their code, time, `supersedes`
  and the references to the records in force (N, the cap, the cohort and the seeds stand). A
  technical mark, P's too, pauses the blocks and closes nothing.
  Before the confirmation has started, the registered routes apply unchanged: a fixed rate's
  re-qualification failing scientifically moves A or F to candidate 2 (at candidate 2, the
  scientific feasibility report) and closes P–F for P; a rerun validation failing its gate takes
  its route.
- **A fix that leaves the GPU code unchanged.** An analysis fix (a reader's rule) is no cycle
  and gets no record of its own: `cycle` refuses it. Commit the fix and read the state again:
  every reader recomputes from the raw records, and the mark resolves only if the recomputed
  reading no longer gives it. Replay the affected branches on CPU ([amendment
  6](study3-preregistration.md#amendment-6-replaying-orchestration-changes-on-cpu)) and record
  the replay's evidence (the tests, the commit, their results) in the stage's report; the final
  report lists it. Write a new versions record before any validation or confirmation launch
  (`check` needs the pinned commit or a record-only child of it), and rerun `qualify.py verdict`
  on CPU so that its cache agrees.
- **A search run under another GPU code.** A counting search run must be under a GPU code the
  search counts under: the code in force before the candidate's cycle or the code in force for
  its runs, the former not for an arm the cycle reruns at the search. A run of another code, or
  of a dirty or unknown commit (its code cannot be told), is a technical mark with the reason
  `other_gpu_code`, before any cycle too; the scientific failures a cycle keeps are exempt. It
  is no cycle case: it pauses, its way out its recovery round where the launcher opens one. A
  code that cannot be told is most often a commit this checkout cannot place: fetch that commit
  and read the state again.
- **Holds and closures** ([amendment 7](study3-preregistration.md#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run)).
  A technical mark closes nothing. A failed engineering record (`engineering_failed`), a held run
  (`held_run`), a run under another commit (its uncounted reason), a search run under another GPU
  code (`other_gpu_code`) and a mark after the cycle (`unresolved_after_cycle`) each pause the
  stage: the step is blocked and technical, names the mark and its reason, and issues no cell,
  A's and F's included while the mark is P's. No step names a closure, so `study3_summary.py
  closure` refuses. A closure, or a resolved hold or reissue, needs a recorded decision or its
  repair, and its writer is written then. A closed or held label is never issued again.

## Study 3: simulation

The [simulation before validation](study3-preregistration.md#simulation-before-validation) runs
after the engineering stage and before the search ([seeds and order](study3-preregistration.md#seeds-and-order)),
on the CPU of the local machine. `scripts/study3_simulate.py` writes its inputs, its fixation,
the run's files and the simulation record; `analysis.study3_simulation` holds the models, the
procedure and the engine, `analysis.study3_seed_vectors` the inputs, `analysis.study3_decisions`
the record's checks. Their docstrings hold every rule. The modeling values (the harm and label
models, the support, the safe alternative, the spread and Δ through the selection, the grid, the
streams, the draw bound and the rest, M1–M29) are the researcher's: each is listed with its
source in the module's rulings and frozen, with a source line ending "M<n>, decided", in the
fixation.

- **Missing labels (M6b).** The coverage scenarios carry MCAR missing labels at 1 % (C14: 5 %
  MNAR among harmful responses), as stress values; their coverage tests are unchanged. The power
  gate reads its own scenarios, P18–P21 (Δ = ±2 pp at the planning spreads 3.5 and 5 pp), at
  0.1 % MCAR: the planning rate of labels still unresolved after the labeling rounds (up to
  three per run). C16 and C17 (Δ = +2 pp) are coverage scenarios only.
  - **The source.** Study 1, with the same judge (WildGuard), had 0 unparseable outputs in the
    39 900 labels of its non-collapsed runs ([stage 1](reports/stage-1.md#judge-labeling)). Its
    rate over all runs, which includes the collapsed runs, is not used: those runs are not
    representative.
  - **What it is.** A conservative planning assumption for supported runs, not a proven upper
    bound: the bound from 0 in 39 900 assumes independent labels and is computed on a selected
    subset, the non-collapsed runs. The rate is MCAR. Run- or arm-level clustered label failures
    are not explicitly modeled. The stress scenarios cover higher MCAR missingness and
    harmful-response-dependent MNAR.
  - **The decision.** The researcher's, after preliminary calculations showed the power gate
    failing at 1 %. It changes the planning alternative of the power requirement; the
    registration sets no rate, so no amendment is needed, and the required simulation itself
    stays prospective.
  - **The power at 1 %** is reported beside the gate, not a gate (`power_af_1pct_missing`). Each
    power study of the screen is drawn again on its own streams and its scenario's calibration
    with its labels missing at 1 %, through the same uniforms (one per label: the 1 % mask
    contains the 0.1 % mask). It is read by the registered procedure, its own validation spread,
    N and confirmation included, to the cap's reading common to the three computations, and is
    counted over the studies the power counts. The n_min scan does not read it.
    - It uses the N the rule chose, before any n_min increase. It is not the power of the
      final increased design.
    - Under MCAR, Δ scales by 0.99 / 0.999: ±2 pp becomes about ±1.982 pp. The SD need not
      shrink by exactly 1 %.
  - **The fixation** records the rates, this disclosure and that report's procedure
    (`missing_labels`). Its reader refuses a fixation whose missing labels depart from them, a
    power scenario at another rate, or scenarios other than the whole grid (one missing, one
    extra, one out of order or one whose values differ), as it refuses the identities; the J
    check refuses results whose power at 1 % is at another rate or over other studies.

- **The stops ([amendment 7](study3-preregistration.md#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run)).**
  - **L** is the first level whose A–F tests at N ≥ 60 and at the cap pass. Each comparison's floor
    is the smallest of 36, 45 and 60 at which its tests at and above that size pass at L, the cap
    test counting only the studies whose reading is confirmatory at that floor.
  - **P–F** is read at L or withheld: its tests are its marginal and joint ones, on the screen and
    in the verification. A failure of P–F alone advances nothing, and there are no mixed levels.
  - **A failure at L** is a failure of A–F's tests in the verification at or above its floor. It
    advances to the next level that passes its screen, with the power and any scan computed again
    at that level, on that level's own stream, every failed result kept. With no level left, the
    record fails.
  - **Short of M eligible studies within 4·M draws,** an A–F test is incomplete, not failed: the
    run stops by its name, and extending the same indexed stream under a recorded computational
    limit is a decision (it is not implemented). A P–F or joint test is not attainable, which
    withholds P–F.

```bash
R="$RUNS_DIR"
project-run "$SRC" python scripts/study3_simulate.py inputs          # then commit it
project-run "$SRC" python scripts/study3_simulate.py fixation --workers 16  # commit it alone
project-run "$SRC" python scripts/study3_simulate.py run --runs-root "$R" --workers 16
project-run "$SRC" python scripts/study3_simulate.py verify --runs-root "$R" --workers 16
project-run "$SRC" python scripts/study3_simulate.py record --runs-root "$R"
project-run "$SRC" python scripts/study3_summary.py progress --runs-root "$R"
```

- **The order.**
  1. `inputs` writes study 2's whole seed vectors, `docs/study3-simulation-inputs.json`, from
     the two study-2 roots under `RUNS_DIR` (`--roots`) and θ₀'s evaluation (`--theta0`), each
     named in the record relative to `RUNS_DIR` (a path outside it is refused), so that the
     committed record holds no home directory. It reads of a run only its config, terminal
     record and step-25 evaluation (K and the utility), never a response, a label or a judge
     output. Commit it.
  2. `fixation` needs a clean checkout with the inputs record tracked. It calibrates every
     scenario at every candidate it can run (f, the safe shifts, σ at the Δ = 0 offsets and the
     offsets, through the selection, over `--workers` processes), enumerates the coverage tests
     (J = 2 856) and writes `docs/study3-simulation-fixation.json`, with the machine it ran on.
     Commit it alone, before the first draw.
  3. `run` checks, before any draw, that the fixation is tracked, committed once and alone in
     HEAD or an ancestor and unchanged, that its inputs and tests are the pinned ones, that its
     sizes are the registered ones (M, 2 000 and 100 000 resamples, the calibration's selections
     and tolerances: `analysis.study3_decisions`' constants, which every reader of the simulation
     record checks, `study3_summary.py record simulation` included), that the checkout is clean,
     and that no results, no screen and no part of another fixation or commit
     exist; it recomputes the calibration (integers and identities exactly; floats within a
     relative 10⁻⁹ or within their solver's precision, 10⁻⁹ pp for a pp field and 10⁻¹¹ for a
     proportion, so that a value near 0 compares by its precision; a different machine is
     printed, not refused). It then runs the gates, the screen and, when the power is short,
     the n_min scan, and writes `screen.json`. It prints L (or "no level passed"), each
     comparison's floor, the validation size and the gates' values, with the power at 1 % beside
     them.
  4. `verify` runs the binding verification from L, each level on its own stream (100 000
     resamples, the realized route), and writes `verification.json`, with every level tried. It
     refuses without the screen, with a screen of another commit than the clean HEAD, without an
     L (the verification is not due), when the n_min scan at L stopped at an n whose cap exceeds
     the 200 reserved seeds (no attainable n: the verification is not run, a pause for a decision)
     and once `verification.json` exists: it is never rerolled.
  5. `record` writes `results.json` and then the simulation record, which binds the results
     and the seed vectors by sha256 and carries `reading` (L, `pf_confirmatory` and each
     comparison's floor, the field the procedure reads) and `levels_tried`. Both are refused when
     the results fail the J check (an identity listed twice, missing or not in the fixation, a
     pass, L, a floor, a gate or a size that does not recompute), and a second record is refused,
     as `study3_summary.py record` refuses it. No level passing, no level left, or no attainable n
     (the verification not run; `record` prints `verification not_run no_attainable_n`) gives a
     failed record: a pause for a decision (`progress`: blocked, `simulation_failed`; coverage is
     not waived, N is never cut).
- **The files.** Under `$RUNS_DIR/study3/simulation/` (reserved, no run directory): the parts
  (`parts/<phase>/<scenario>/<first>.json`, 250 studies each, the phase `gates`, `screen`,
  `scan-<level>-<n>` or `verification-<level>`), `screen.json`, `verification.json` and
  `results.json`. Every
  file is JSON, written whole and once. The simulation record lies under
  `$RUNS_DIR/study3/decisions/simulation.json`.
- **Resuming.** Every study is a pure function of its stream, its scenario and its index, and a
  test takes the first M studies that count for it by index: the results depend neither on
  `--workers`, nor on the order in which parts finish, nor on an interruption. Run `run` (or
  `verify`) again after an interruption: it keeps the finished parts of this fixation and
  commit, draws the missing ones, and writes the same bytes as an uninterrupted run. A part of
  another fixation or commit stops it, named; remove the run's simulation folder only for a new
  commit.
- **Time on the 16 local cores.** The figures below were measured under load, on 2026-10-31 on
  synthetic inputs of 25 vectors, the machine shared with other work (load 15–50), so their
  CPU-seconds per study are above a quiet machine's. A single process measured alone takes
  0.21 s per screen study (2 000 resamples) and 0.68 s per verification study (100 000
  resamples).
  - the calibration, in `fixation` and again in `run`: about 59 000 CPU-seconds each under that
    load (1.5 h of wall time; about 1 h on an idle machine); one pair, C13 at 2 000 selections,
    takes 37 s in a single process. C12's two candidates are the critical path: their
    selections are drawn in one process;
  - the gates: under a minute;
  - the screen at 2 000 resamples: 0.35 CPU-seconds per study under load, about 229 000
    studies at M = 10 000 (21 scenarios, about 9 % unconfirmed), and each power study's report
    at 1 % (about 44 000): 1.5–1.8 h on an idle machine. Alternated with the grid before the
    power's own scenarios (19 scenarios, C16 and C17 also power), on a quieter machine (load
    4–6, one process: 0.088 CPU-seconds per coverage study, 0.107 per power study, 0.050 per
    report), the screen does about a fifth more work;
  - the verification at 100 000 resamples: 1.9 CPU-seconds per study under load, about 185 000
    studies: 5–6 h on an idle machine;
  - the n_min scan, only when the power is short: some 44 000 studies at one level per value of
    n, a few minutes each;
  - a power scenario's study (P18–P21) draws, wherever the cap's primary interval has a
    direction, three more intervals at the cap (the missing labels' two and all missing harmful:
    the registered final direction), about a third more bootstrap work for that study, and the
    scan's studies are all power studies. The coverage scenarios, C16 and C17 included, draw
    none: the verification does about 3 % less than with C16 and C17 as power scenarios (C17
    0.61 instead of 0.77 CPU-seconds per study, one process);
  - the calibration has two more pairs (P20 and P21 at candidate 1), off the critical path;
  - the screen's cap tests at the three floors draw on until the joint test at floor 60 holds M
    studies: at most about a tenth more of a low-support scenario's studies where N is near 60
    (an estimate, not measured);
  - in all about 10 h on an idle machine with one verification. The parts take about 1.9 kB per
    study (about 0.85 GB);
  - each further level after a failed verification adds a verification of the same size, about
    5–6 h (an estimate from the figures above), and its n_min scan when that level's power is
    short, a few minutes per n: at most three more, so about 28 h in the worst case.
  At the sizes of the dry run's Part II (M = 200, 200 and 1 000 resamples, 1 000 calibration
  selections per scenario and candidate, `study3_simulation`'s M25) the screen takes about 1.5
  minutes and the verification under 5; the calibration draws 1 000 selections per pair
  instead of 10 000.

## Study 3: the final analysis, the audit and the report

The last steps of [the order](study3-preregistration.md#seeds-and-order) run on CPU: the cohorts
and K\*_h frozen (step 9), the judge-based analysis (11), the audit (12) and the report's facts
(13). The commands are `scripts/study3_analysis.py` (`freeze`, `analyze`, `report`) and
`scripts/study3_audit.py` (`draw`, `agreements`, `arbiter-input`, `estimate`); their docstrings
hold every rule. Both run from the source checkout of the commit, clean (`$SRC`). Their outputs go
under `$RUNS_DIR/study3-final/`, outside `$RUNS_DIR/study3/`, and hold numbers, ids, run ids, paths
and hashes; the audit's blind files alone hold texts.

Labels are read only through the labeling command's ledger, `<sealed dir>/study3-ledger/` (its
batch files and acceptances): a run's labels count only through an acceptance that names its label
file and receipt as they are. `freeze`, `analyze`, `report` and `draw` take `--sealed-dir` for the
sealed directory; the default is `<--runs-root>/sealed`, the labeling's default when the runs root
is `$RUNS_DIR`.

### The freeze

1. Once the state is at `done` (every started confirmation block resolved), and before any
   confirmation label exists:

   ```bash
   project-run "$SRC" python scripts/study3_analysis.py freeze --runs-root "$R" --candidate 1
   ```

   `--theta0` names θ₀'s evaluation (by default study 2's under `$RUNS_DIR`). The command writes
   `docs/study3-analysis-freeze-c<N>.json` once: per comparison its cohort's blocks (seeds, run
   ids, weights and widths) and the started blocks outside it, A − P's blocks, K\*_h with the
   matched blocks and those left out, and validation's blocks. It reads no label. It is refused
   before `done`, on a checkout that is not clean, and on any trace of a confirmation label: a
   `judge/` or `judge-superseded/` entry in a confirmation run's folder (a dangling link
   included) or a run that a batch file of the ledger names.
2. Commit the record alone, on top of the commit it names (`code.commit`), and never rewrite it.
3. Labeling refuses the confirmation's runs until the gate (`study3_freeze.freeze_open`) opens:
   the record committed alone and unchanged, the in-force supplement the frozen one, and the
   cohorts rebuilt from the label-free records equal to it. Then the confirmation is labeled.

### The analysis

```bash
project-run "$SRC" python scripts/study3_analysis.py analyze --runs-root "$R" --candidate 1
```

- **Pending, technical problem, missing.** A cohort run whose labels no acceptance of the ledger
  covers is pending; a run whose labeling rounds ran out is a technical labeling problem. Either
  refuses the analysis before any computation, naming the runs
  (`<k> runs pending: <ids>; <j> runs with a technical labeling problem: <ids>`), and nothing is
  written; neither is ever counted as missing labels. Labels missing inside a labeled run enter
  the three computations.
- **The order of its outputs.** The utility gates (label-free), then the spread of K of every
  bracket run, then the harm of the frozen runs; `analysis.json` is written and its path and each
  comparison's row of the registered wording are printed. Only then are validation's labels read,
  and `validation-harm.json` names the analysis by its sha256.
- **Resumption.** A run stopped after `analysis.json` (a validation run still pending, say) is
  resumed by the same command: it writes validation's harm alone.
- **Supersession.** Once both files are written, the command refuses. A recomputation after a fix
  of the analysis code, with no change of the GPU code, takes `--supersedes-reason ID` (a lowercase
  letter, then lowercase letters, digits or underscores) and writes `analysis-<k>.json` and
  `validation-harm-<k>.json`, each naming its predecessor by path, sha256 and the reason
  ([voiding and supersession](study3-preregistration.md#voiding-and-supersession)).

### The audit

The audit follows [amendment 11](study3-preregistration.md#amendment-11-the-audits-reference-annotators).
Its folder is `$RUNS_DIR/study3-final/c<N>/audit/`. The model calls are made outside the code: each
step writes the blind input of the next call and reads that call's labels, a JSON object
`{item_id: label | null}` over exactly the items of the input (a label is `refusal`,
`safe_non_refusal` or `harmful_compliance`; null is a failed output).

1. **The draw**, once, after the analysis:

   ```bash
   project-run "$SRC" python scripts/study3_audit.py draw --runs-root "$R" --candidate 1
   ```

   It writes `unique.jsonl`, `key.json`, `frame.jsonl` and `selection.json` and prints counts
   only. A second draw is refused.
2. **The blind annotators.** `unique.jsonl` alone goes to Sonnet and to Grok, blind to arm, run,
   block and judge label.
3. **The agreements:**

   ```bash
   project-run "$SRC" python scripts/study3_audit.py agreements --runs-root "$R" --candidate 1 \
     --sonnet SONNET.json --grok GROK.json
   ```

   It writes `gpt-input.jsonl` (the items the two disagree on, on the two-class outcome, and the
   items either failed to label), `check-input.jsonl` (a tenth of the agreements, drawn under the
   supplement's agreement seed) and `agreements.json`, and prints the counts.
4. **GPT's labels.** `gpt-input.jsonl` goes to GPT, blind.
5. **The arbiter's input:**

   ```bash
   project-run "$SRC" python scripts/study3_audit.py arbiter-input --runs-root "$R" --candidate 1 \
     --gpt GPT.json
   ```

   It writes `arbiter-input.jsonl` (each item with at least one vote, the votes bare labels in an
   order drawn per item under the agreement seed, so that no vote shows its voter) and
   `arbiter.json`. An item with no vote at all is not sent: its reference is missing.
6. **The arbiter model.** `arbiter-input.jsonl` goes to the arbiter model, which returns a decision
   per item (null: no decision); `check-input.jsonl` goes to it separately, blind to the votes,
   and returns its label per item.
7. **The estimate:**

   ```bash
   project-run "$SRC" python scripts/study3_audit.py estimate --runs-root "$R" --candidate 1 \
     --arbiter ARBITER.json --arbiter-check CHECK.json
   ```

   It checks the draw's files and every record's link to the one before it by sha256, and makes
   the draw, the agreements and the vote order again from the recorded labels and seeds. Then it
   builds the reference: an agreement's shared value, else the arbiter's decision, else missing.
   No file can set the reference of an agreement. It writes `estimate.json` once: the reference
   by item id with its source and counts, the check (the agreements checked, failed, labeled and
   rejected, and the share rejected: an estimate of the consensus's error that changes no
   reference), B_E per estimable comparison, missing at random and as a worst-case bound, and per
   arm the judge's sensitivity and false-positive rate, both ways
   ([the audit](study3-preregistration.md#the-audit)). The MAR estimate averages over the
   resolved draws only, while each missing draw's term in the bound is fixed in sign by its known
   judge label: the MAR estimate can lie outside the bound. A draw whose coefficient for E is 0
   adds exactly 0 to B_E and is never missing for E; per arm, the MAR estimate is the mean over
   the resolved draws with a nonzero coefficient, scaled by the share of the arm's draws that
   have one (with every reference resolved, the mean over all the arm's draws). Its variance is
   from all the arm's draws, the zeros included: with every reference resolved it is exactly the
   with-replacement variance of the mean over all the draws, s²_all/n; under missing references
   it is a plug-in estimate that assumes the references are missing at random (the spread within
   the draws with a nonzero coefficient taken from the resolved ones), not a general guarantee.
   No reference harm is computed.
8. **Order and restart.** Each step is refused until the record of the step before it is written
   (`arbiter-input` until `agreements.json`, `estimate` until `arbiter.json`). A step writes its
   files new, each synced, the record last; an input left without its record (an interrupted step)
   refuses the step until it is removed.
9. **Kept out of Git.** The audit's folder lies under `$RUNS_DIR` and is refused where Git could
   track it or under a sealed folder; it is 0700 and its files 0600. `unique.jsonl`, the three
   other blind inputs, the annotators' and the arbiter's files hold or rest on model responses:
   they stay under `$RUNS_DIR` and go to the bucket with the stage's records, never into the
   checkout.

### The report

```bash
project-run "$SRC" python scripts/study3_analysis.py report --runs-root "$R" \
  --billing BILLING.json --embargo EMBARGO.json
```

- **When.** At `done`, once the analysis and validation's harm are written (the audit's section
  reads `pending` until its estimate is written), or at a terminal stage of the progress state (`study3_progress.TERMINAL`:
  `scientific_feasibility_report`, `budget_stop`), with what the study reached. Each section the
  study did not reach reads `not reached`.
- **The billing file** (closed schema; an unknown field is refused):

  ```text
  {"currency": "USD",
   "stages": [{"stage": "<stage>", "spent": <amount or null>}],
   "incidents": [{"id": "<id>", "stage": "<stage>", "incident_time": "<UTC, ISO 8601>",
                  "cause": "<one line>", "billed": <amount or null>,
                  "evidence": [{"journal": "<log under $RUNS_DIR>", "line": <n>},
                               {"qualification": "<record path under $RUNS_DIR>"},
                               {"pod": "<provider reference>"}]}],
   "policy_stops": [{"stage": "<stage>"}]}
  ```

  A stage is one of `dry_run`, `engineering`, `simulation`, `search`, `numerical`, `repair`,
  `validation`, `confirmation`, `labeling` and `audit`. Each incident is counted once, with its
  price ([amendment 5](study3-preregistration.md#amendment-5-the-budget-after-the-dry-runs-part-i),
  "Infrastructure incidents"): one Pod stop is one entry, whatever records it left (a recovery
  round of the launch journal, a case restart `restart-<n>.json`, a part record made with
  `--infrastructure`), and its `evidence` lists each of them, plus any provider reference
  (`pod`). The records' incidents are a study-3 recovery round that carries an incident, by its
  log and line, and every qualification record made with `--infrastructure` (a part record
  carrying `infrastructure`, or a case restart), by its path. One that no incident names is
  pending: the money section lists it under `pending.unbilled` with its record and the stage,
  time and cause the record shows, the rest null. The report is refused unless no record is named
  twice, every record an incident names is one of them,
  the ids are distinct, every incident lists evidence, and each incident's stage is its records'
  (a journal line's: the step of its log's campaign; a qualification record's: its stage folder,
  `numerical` also covering `repair`). The `cause` is shown and never matched. Every stage the
  route reached or an incident is billed at has its `stages` entry (`0` when nothing was
  spent). A missing amount is pending, never filled in
  ([amendment 7](study3-preregistration.md#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run),
  "The report"): a `spent` or `billed` of null, and a reached stage with no entry, which the
  report shows with `spent` null; the money section lists them under `pending`.
- **The embargo file**, `{"breaches": [{"stage", "what"}]}`: each harm label opened outside the
  two scheduled points, with the stage it affected; `{"breaches": []}` when there was none
  ([voiding and supersession](study3-preregistration.md#voiding-and-supersession)).
- **After a judge mark.** The `judge_technical` variant (the registered technical feasibility
  report for the harm comparison: no Δ, no interval and no audit; the freeze with its cohorts, each
  judge diagnosis as a technical dead end, `judge_mark`, with its batch, and the runs with a
  technical labeling problem) needs a recorded diagnosis of the judge's mark. The labeling ledger
  records no diagnosis yet (`study3_labels.diagnoses` is empty), so while a batch's mark is
  unresolved its runs are technical labeling problems and the report is refused.
- **What it writes.** `$RUNS_DIR/study3-final/report-<UTC>/` (or `--out`, a folder that does not
  exist yet), once: `facts.json` and `tables.md`, harm in pp in the tables. The facts hold the
  audit's reference sources and the arbiter model's check of the agreements as the estimate
  records them. It reads no label: of the labeling ledger, only the runs with a technical
  labeling problem.
- **The prose is the researcher's.** The facts carry no interpretation. Money is actual spending
  by stage plus each incident with its billed price; the report names the stages an internal
  spending policy stopped, and no other money figure
  ([disclosure](study3-preregistration.md#disclosure),
  [amendment 5](study3-preregistration.md#amendment-5-the-budget-after-the-dry-runs-part-i)).

## Study 5 campaigns

Study 5 runs in [the registration's order](study5-preregistration.md#seeds-and-order) on two
Pods, one L40S each, as two campaigns of `scripts/study5_campaigns.py`: `pod1` (the engineering
check, θ₀'s baseline, the surrogate checks and the retrospective predictor, before the
registration) and `pod2` (the fresh seeds, after it). Each is one detached Python driver that
chains its blocks, so that a lost session never leaves the Pod idle. Its docstring holds every
rule; the configs are `configs/experiments/study5*.yaml`. The driver runs from the source checkout
of the commit, clean (`$SRC`); its folder goes under `$RUNS_DIR/campaigns/`, never into the
checkout. Before it starts, it checks that the checkout's GPU code is the one
`docs/study5-code-identity.json` records (`tests/test_study5_code_identity.py` fails on any later
GPU change until the file is written again, after a new engineering check). The driver, the
engineering check and the fixation script are analysis scripts, outside the GPU code
(`code_identity.ANALYSIS_SCRIPTS`); `scripts/study5_surrogate_endpoint.py` loads a model and is
GPU code, as is `scripts/study5_scan.py`.

**The fixation record** (`docs/study5-fixation.json`, by `scripts/study5_fixation.py fix`, its
docstring holding every rule) fixes before the registration what the analysis takes: the seeds
(fresh, the n = 24 alternative, controls, ladder, v₀ sources), the permutation test's and the
retrospective bootstrap's settings and RNG seeds, the scans' intervals, study 3's extreme A–F
blocks (the 5 highest and 5 lowest block means of A's and F's harm at K\*, interpolated by study
3's estimator; ties by seed), the 40 runs Llama Guard relabels, and the reference sample: 5 prompt
ids per block × arm × rate (200 items, weight 521/5 within a cell). It names the sha256 of study
3's freeze and of every A–F run's label file and receipt, and its code. It reads study 3's labels
through study 3's own readers, run from this checkout (the record's `code.commit`); study 3's label
gate opens them only under study 3's GPU code, so `--study3-repo`, a clean checkout at study 3's
final commit, supplies the gate's GPU-code identity, the committed freeze and its validation runs
(the record's `inputs.study3_checkout`); `--runs-root` is study 3's runs root. The record also
names the sha256 of every file of the labeling's ledger, and states how the state's run records
and θ₀'s run are bound (the freeze's sha256, each cell's run id and rate). It fixes the judges'
high–low contrast as the reference's estimator: per cell the harmful share over all 521 responses
by each judge, the group the unweighted mean of its 20 cells, high minus low; the block mean at
K\* only selects the blocks. It states the permutation test's null (p_s and h_s independent)
and its one-sided alternative (a negative association), and holds the surrogate overlap
(`--surrogate-overlap`: the classification of the 100 surrogate-panel prompts against the 521
judged ones by content, ids, numbers and the method only, its sha256 among the inputs), from
which the analysis's descriptive check takes `evaluation_ids_same_or_close`. The prompt sets come
from `DATA_DIR`. Rerun on the same inputs and code it is byte-identical; it never overwrites a
different file.
`scripts/study5_fixation.py reference-input` then writes the reference annotators' input
(amendments [11](study3-preregistration.md#amendment-11-the-audits-reference-annotators) and
[12](study3-preregistration.md#amendment-12-safety-filter-stops-in-the-audits-labeling)) in study
3's audit form: `unique.jsonl` (the blind rows, study 3's order and opaque ids), `key.json` and
`manifest.json` (ids, cells and sha256, no text), once, into a folder Git cannot track (under
`$RUNS_DIR/`, never in the checkout). The annotation runs later with study 3's tooling.

```bash
export DATA_DIR="$DATA"   # the prompt sets (study 3's data root); or a checkout with data/
python scripts/study5_fixation.py fix --runs-root "$S3/runs" --study3-repo "$S3_CHECKOUT" \
  --theta0 "$S3/runs/study2_stage_a/<θ₀ run>" --sealed-dir "$SEALED" \
  --surrogate-overlap "$OVERLAP"   # docs/study5-fixation.json
python scripts/study5_fixation.py reference-input --runs-root "$S3/runs" \
  --study3-repo "$S3_CHECKOUT" --out "$RUNS_DIR/study5-reference/input"
```

**The Pod.** Start it with the file-reading deadline watcher, `ZFR_DEADLINE_EPOCH` as its cap
([deadlines between campaigns](#study-2-stage-a)). Restore `DATA_DIR` with `pod_setup.sh` from
the data inventories study 3's confirmation restored (`inventory-2026-10-11.tsv`,
`inventory-2026-10-14-study2-prompts.tsv`, `inventory-2026-10-14-study2-a-phase0.tsv`,
`inventory-2026-10-15-study2-monitor-v2.tsv`, `inventory-2026-10-16-study2-stage0-v3-record.tsv`,
`inventory-2026-10-16-study2-curvature-panel.tsv`, under `runs/archive/`). The campaign fetches
study 3's streams itself. On a restarted Pod, add the inventories of what `DATA_DIR` already
holds: `runs/archive/inventory-2026-11-05-study3-confirmation-c1-streams.tsv` and the fresh
streams' inventories under `study5/<ID>/inventories/`.

```bash
C="$RUNS_DIR/campaigns"
project-run "$SRC" python scripts/study5_campaigns.py pod1 --out "$C/study5-pod1" \
  --archive-id pod1-20261109 --dry-run          # prints the plan, writes nothing
project-run "$SRC" python scripts/study5_campaigns.py pod1 --out "$C/study5-pod1" \
  --archive-id pod1-20261109                    # detached
project-run "$SRC" python scripts/study5_campaigns.py pod2 --out "$C/study5-pod2" \
  --archive-id pod2-20261110 \
  --engineering-check "$GATES/engineering-check.json" \
  --surrogate-checks "$GATES/surrogate-checks.json" \
  --retrospective "$GATES/retrospective.json"   # after the registration
```

**The n.** `docs/study5-confirmatory-seeds.json` holds the n chosen before generation and its
seeds: `{"n": 40, "seeds": [50000, …, 50039]}`, or the alternative `{"n": 24, "seeds": [50000, …,
50023]}`, committed before the registration, which names its sha256. Pod 2 takes its n from it
(`--n` defaults to it; another is refused), so a restart cannot change n.

**Pod 1's commit stays reachable** until the analysis: its engineering reruns, its surrogate
checks and pod 2's preflight name it, and the analysis computes its GPU code identity from this
history. Do not squash or rebase it away; tag it when pod 1 starts,
`git tag study5-pod1-<archive-id> <commit>` (e.g. `study5-pod1-pod1-20261109`), and push the
tag with the branch.

**Pod 2's preflight.** Before its first step, at every start, pod 2's driver refuses unless:
- the checkout is clean at the plan's commit and its GPU code is the one
  `docs/study5-code-identity.json` records (as pod 1's);
- the registration at that commit says ``Status: `registered` `` and names the sha256 of the
  committed `docs/study5-fixation.json` and of the retrospective record (`--retrospective`, the
  analysis's `retrospective` record, of that fixation record);
- pod 1's engineering-check record (`--engineering-check`: `$RUNS_DIR/study5_records/<pod 1's
  ID>/engineering-check.json`, copied from pod 1's archive to `$GATES` on the Pod) is equal, at
  least one pair, its reruns from a clean checkout of a commit of the plan's GPU code identity;
- pod 1's surrogate-checks record (`--surrogate-checks`, `surrogate-checks.json` beside it)
  passed, written from a clean checkout under that identity, of the plan's surrogate (the same
  sha256);
- `docs/study5-confirmatory-seeds.json` at the plan's commit, its sha256 named in the
  registration there, and the plan's fresh seeds exactly its seeds.

`$GATES` is a folder on the Pod, outside the checkout, holding the two records from pod 1's
archive and the retrospective record written by the analysis's `retrospective` before the
registration. A refusal names what does not stand; nothing has run.

**The surrogate's spec.** Both plans name one spec, `--surrogate` (default
`configs/study5/surrogate.yaml`, a path in the checkout), with its sha256 in `plan.json`, and pass
it to the checks, every predictor and every endpoint's r. If the checks at θ₀ fail (exit 2), the
second set goes in a new file beside the first (the first stays as the record) and both plans
take `--surrogate configs/study5/<the new file>`. The new file is a GPU path: commit it, record a
new `docs/study5-code-identity.json`, and run pod 1 again from the engineering check (a new
`--out`, the gates rerun under the new identity) before the retrospective predictor; the analysis
requires one surrogate sha256 across the checks, every predictor record and every endpoint record.

**Blocks.** Each block sets the deadline file first (2 × the estimated minutes of its steps
outside the seed groups + 60), and so does each seed group (2 × the group's minutes + 60); the cap
stays. A dead driver or a hung step leaves the Pod idle one group's window at most. The minutes
are study 3's measured unit costs (an F run 6.9; an evaluation 3.1; a K-only point 1.5; five scan
points 15.5; a stream 91.7 s per 1000 responses; the predictor 1.7 per seed; a ladder rung 10.6:
three K-only points of the search, two of the matched-K step and an evaluation; a checkpoint's
save 0.5) and stated assumptions for the rest (the script's `MINUTES`; among
them the surrogate at an endpoint, ≈ 1, and the scan run's loading, ≈ 2: two models, the
checkpoint hashed and read, the panels). A fresh seed's group: the checkpoint count 0.5, η_l
7.4 (an F run and its checkpoint's save), the checkpoint's archive 4, r 1, the scan 17.5 (five
evaluations and the loading), the checkpoint's removal 0.5, η_u 6.9, its archive 1: 38.8
minutes, a window of 138. A checkpoint fetched back from the archive (≈ 4 minutes) fits the
timeouts of r (23) and of the scan (72.5). Every step has a
timeout of 3 × its minutes + 20 (the scan's 72.5 minutes): past it its process group is killed
and the step has failed, under its rule. After the last fresh group, the `fresh` block writes
the completion record (`fresh/completion`, `$RUNS_DIR/study5_records/<ID>/fresh-completion.json`):
every fresh run the launcher journals show as started, with its end and outcome. It is written
only when no started fresh run lacks an end (a run cut off has none until the launcher settles
it), kept only while the journals still say the same, and archived with the attempt; the
analysis gives no reading without it, and no fresh label is made before it. A block ends with
its records in the bucket, verified, a snapshot of the driver's journal and logs, and its
receipt.

| Pod | Block | Runs | Stops the campaign |
|---|---|---|---|
| 1 | `engineering`: study 3's F runs on seed 10403 at 1.3 and 1.83·10⁻⁵, rerun under study 3's config and compared with study 3's archived records by `scripts/study5_engineering_check.py` (exit 0 equal, 2 different, 1 error) | 2 | any failure, a difference included |
| 1 | `theta0`: θ₀ on the 521 prompts (`experiments/study5_theta0`) | 1 | any failure (the analysis needs θ₀); its archive |
| 1 | `surrogate`: `scripts/study5_surrogate_checks.py` | 0 | any non-zero exit (2: the sets are changed once, before any extraction); its archive |
| 1 | `retrospective`: study 3's 60 A–F seeds (from its analysis freeze), their streams fetched from the bucket, then `study5_predict.py` at 1.3·10⁻⁵ and `study5_stream_features.py` per seed | 0 | the fetch; each seed's archive |
| 2 | `streams`: each fresh seed's streams (`generate_teacher.py`), archived before anything reads them, then `study5_predict.py` and `study5_stream_features.py` | 0 | each seed's streams archive and records archive |
| 2 | `predictions`: the record of p for every seed in order, with its sha256; copied to the bucket, read back, its object's creation time in a receipt | 0 | a missing seed; a kept record of other seeds, n or GPU code, or not the published copy; the copy |
| 2 | `fresh`: per seed, in order (the seed's group): (1) F at 1.3·10⁻⁵, a plain F run with its endpoint checkpoint (`experiments/study5`, `study5.role=fresh`); (2) the surrogate r at its endpoint from that checkpoint (`scripts/study5_surrogate_endpoint.py`: θ₀'s model with the trainable parameters set from the checkpoint, its sha256 the run's recorded one; r, r_benign and the per-prompt values); (3) the scan run of its endpoint (`scripts/study5_scan.py`: θ₀ + aΔθ for a ∈ {−1, −0.5, −0.25, +0.25, +0.5} from θ₀ and the checkpoint, each evaluated in full under `line_scan_{a:+g}`, a run of its own under `study5_scan/`); (4) the checkpoint removed from the container disk; (5) F at 1.83·10⁻⁵. The checkpoint is archived and verified right after (1), before (2), and kept on the disk until (4); (2) and (3) fetch it back from that archive when it is gone. Each F run in a launcher campaign of its own; a failure of (2) or (3) is the seed's, recorded, and takes nothing from the η_l run | 2n + n scan runs | the receipt's check (the record the published copy, of the plan's seeds in order, n and GPU code); a second checkpoint on the disk before a seed; each seed's archive |
| 2 | `controls`: seeds 50000–50005, the order and v₀ controls at 1.3·10⁻⁵, then the ladder's `gaussian_d` (D^{1/2}z, Gaussian with covariance D), `linear_no_d` and `linear`, each matched to K\* within ±10 % (one refinement step, else flagged `out_of_tolerance`) | 12 + 18 rungs | each seed's archive |

Every block's deadline, snapshot and receipt stop the campaign when they fail. Every run goes
through `scripts/launch.py FILE --max-attempts 1`. Any other run or step that fails is recorded in
the journal and the campaign goes on to the next seed. A failed step is not run again by a
restart, and a recorded stop stops again. After a stop the block's archives still run, then the
close. On pod 2 the close first sets a deadline window for, and attempts, the export of every
recorded endpoint checkpoint not yet verified in the archive (`close/checkpoints`: uploaded, its
sha256 checked against the run's recorded one); a failure that remains is written to the close
record (`close/checkpoints-<attempt>.json` in the campaign folder) and printed with the path, the
recorded sha256 and the error, and the shutdown goes on. Then the plan, the launcher campaigns,
the receipts and that record go to the bucket, and the deadline is set 15 minutes ahead, so that
the watcher stops the Pod. The close runs on SIGTERM and on any exception of the driver too.
Read the journal before anything else.

**Resuming.** Start the same command again, from the same commit: a block with a receipt is
skipped, a step that ended is skipped, a step the Pod cut off runs again (a ladder rung's or a
scan's partial run directory moved aside under `$RUNS_DIR/aside/`). A Pod stop wipes the
container disk, and with it a fresh seed's endpoint checkpoint; the checkpoint was archived and
verified right after its η_l run, so on the restart the endpoint's r and the scan that the stop
cut off fetch it back from that archive (`with-runs --restore-*`: the object and its archive's
inventory showing the run's recorded sha256, the copy's sha256 checked again before it is put in
place), run, and the checkpoint step then removes it as before. A scan that fails and is not run
again leaves its points missing by name; it takes nothing from the η_l run, whose outcome
stands. A launcher run the Pod cut off is settled by the launcher; its rerun is a recovery round
of its own ([launching runs](#launching-runs)).

**Attempts.** Every start of the driver is an execution attempt with its own id (the UTC time
and a short random suffix, in the journal's `driver` line). Every file that can change between
attempts goes to the bucket under `study5/<ID>/attempts/<attempt>/`: the launcher journals and
logs, the directories of runs the journals show without an end, a scan or ladder run before its
record, the gate and completion records (`runs/`); the snapshots, the plan, the launcher campaigns
and the receipts (`campaign/`); every inventory (`inventories/`). What is written once keeps its
stable place: the records of the scripts, the streams, the directory of a run its journal shows
ended, a scan or ladder run with its record, the checkpoints and their inventories (named by the
run), the predictions record. So a later attempt never writes an object an earlier attempt wrote;
`archive.py --no-clobber` finds the stable ones already there and checks them identical, never
replaces them (`tests/test_study5_campaigns.py` runs this restart against a populated archive).

**After a stop.** Fix the cause, then start the campaign with a new `--out` and the same
`--archive-id`; it is a new attempt, and the archive an earlier attempt populated is no
conflict. What stands is kept:
- a record a step wrote, by its existence, except the gates;
- a launch whose runs all completed, by the launcher's journal (a fresh η_l run whose checkpoint
  was archived and removed is not launched again: its launcher would refuse the missing file);
- an archive whose objects are identical (the written-once ones; the mutable ones go to the new
  attempt's place).

A gate's record stands only by its verdict, written from a clean checkout under the plan's GPU
code identity. A commit counts by its GPU code identity, never by itself:
- the engineering check: equal, its reruns run from a clean checkout under that GPU code;
- the surrogate checks: passed.

Any other gate record is moved aside beside itself (`<record>.aside-<UTC>`, never deleted) and the
gate runs again. A gate that ran stops the campaign when its new record does not stand either.

A kept predictions record must hold exactly the plan's seeds in order and its n, carry the plan's
GPU code identity, and have the sha256 of its published copy (the receipt's and the sidecar's),
or the campaign stops before any fresh run. Its commit is information only.

A fresh η_l run that saved a checkpoint is refused by the checkpoint step when its run records no
sha256 for it; the file stays on the disk, and the next seed's count stops the campaign. A
checkpoint already archived counts only when its archive's inventory shows it verified with the
run's recorded sha256.

**A fix that needs a new commit.** Commit it, then start from the new commit with a new `--out`
and the same `--archive-id`:
- *The GPU code identity is unchanged* (a driver, analysis or docs fix: `code_identity`'s GPU paths
  untouched; `tests/test_study5_code_identity.py` passes). Every gate, the predictions record and
  every completed run are kept.
- *The GPU code identity changed.* Run the engineering check again before anything else: commit a
  new `docs/study5-code-identity.json` first (the driver refuses another identity). The driver
  moves the old engineering reruns of the other GPU code aside, to `$RUNS_DIR/aside/` (never
  deleted, never reused). The launcher would otherwise reuse them, since it reuses a completed run
  of the same config hash from any campaign. The cells then run in their one recovery round
  (`launch.py --recovery`); the old gate records are moved aside as above. A second change of the
  identity has no cell round left, under any `--archive-id` (the engineering cell names carry
  none, and the launcher reads every journal under `RUNS_DIR/launch/`): the driver's launch then
  fails closed, and the reruns take an own round of each run, `launch.py CAMPAIGN --only LABEL
  --max-attempts 1 --recovery` with the incident fields, before the driver is started again.

**SIGTERM** to the driver kills the running step's process group and runs the close (on pod 2
the checkpoints' export first), its deadline last. A killed launch step's run has used its one attempt: on restart the launcher reports it used
up and the step is recorded failed, so it needs a recovery round as above. `SIGTERM` to
`launch.py` alone (not the driver) stops at a run boundary instead.

**Disk.** The container disk holds the endpoint checkpoints (5.24 GB each, as study 3's: two at
most, 10.5 GB) besides the image and the Google Cloud CLI: study 3's 50 GB container disk
suffices. `/workspace` holds the model caches, the streams (about 10 MB per seed: 0.6 GB for pod
1's 60 seeds, 0.4 GB for 40 fresh seeds) and the run directories (under 1 MB each; a few MB for a
scan run with its five points' responses). A scan run holds on the host the checkpoint's
trainable tensors (5.24 GB) and on the device two models and Δθ (about 17.6 GB) besides an
evaluation.

**Where it goes**, under `$ARTIFACT_REMOTE/study5/<ID>/`: `runs/` (the ended runs' directories,
the scan and ladder runs with their records, and the records under
`$RUNS_DIR/study5_records/<ID>/` written once), `checkpoints/` (each endpoint checkpoint, verified
before it is removed from the container disk; at most two are there at a time),
`inventories/checkpoint-<run id>.tsv` (each checkpoint's inventory), `predictions/` (the record,
its `.sha256` and its receipt) and `attempts/<attempt>/` per execution: `runs/` (the launcher
journals and logs, runs without an end, the gate and completion records), `campaign/` (the plan,
the launcher campaigns, the receipts and the snapshots) and `inventories/` (every other archive's
inventory, read back); the fresh streams under `$ARTIFACT_REMOTE/data/` with every other stream.
To restore a Pod's `runs/` for the analysis, copy `study5/<ID>/runs/`, then each attempt's
`attempts/<attempt>/runs/` over it in the attempts' order (the launcher journals only grow, so
the latest attempt's are complete). The predictions
record is committed after its copy is verified; the fresh runs need only the copy. No label is
made by the campaigns: the judging of the fresh runs is a separate step, after every fresh run has
ended.

**The analysis** runs on a CPU through `python -m zero_field_refusal.analysis.study5_analysis_cli`
(its docstring holds every rule; the computations are `analysis/study5_analysis.py`'s), from a
clean checkout. `retrospective`, after pod1 and before the registration, needs no pod2 input: it
writes the retrospective record (ρ(p_s, F's harm at K\*) over study 3's 60 A–F blocks with the
fixation record's block bootstrap, ρ(p_s, refusal share at K\*), the stream features, every
input's sha256), once. `full` runs once both Pods' `runs/` are restored under one local
`$RUNS_DIR` and the runs are labeled (WildGuard in each run directory; Llama Guard on the 40
relabeled study-3 runs); it recomputes the retrospective and refuses unless it agrees with
that record's (the same ids, seeds, counts and states, numbers within a relative 1e-12), which it
names by sha256. It sits in the analysis
package, outside the GPU code identity, so adding it left `docs/study5-code-identity.json`
standing. It reads the two campaign folders (`plan.json` and the launcher campaigns, each by the
plan's sha256), the fixation record, the predictions record (refused unless its sha256 is the
receipt's and the sidecar's) and the label files. A study-5 run's labels are read with study 2's
checks over `run_judging.response_conditions` (`step-25` for the outcome, the scan run's points
and the ladder's conditions apart); `study2_labels.read_endpoint_labels` reads `step-<N>` alone
and refuses a scan or ladder run. A seed's scan comes from its scan run: h(+1) is the η_l run's
step-25 evaluation, h(0) θ₀'s run; a point not evaluated, or a scan run missing or not labeled,
is reported missing by name, and the seed stays in every quantity that does not need it. Each seed's h is study 3's support rule and interpolation in ln K
(`study3_blocks.arm_support`, `study3_estimates._arm`) on the seed's two F runs. Study 3's parts
go through study 3's final reader with `--study3-runs-root`, `--study3-repo` and
`--study3-sealed-dir` as for the fixation record, its freeze and receipts the record's; the
reference sample from the reference input's key and each annotator's labels in study 3's
audit-label form (`{item_id: label | null}`). `full` also requires: the fixation record it reads named by sha256 in
the registration at its commit; pod 1's engineering check equal under pod 2's plan's GPU code
identity, the predictions record's; pod 1's surrogate checks passed; one surrogate sha256 across
the checks, both plans, every predictor record and every endpoint record; `--runs-root` apart
from `--study3-runs-root`. Each fresh seed's p is bound to both its runs, η_l and η_u (its
predict record the published one by sha256; v₀, θ₀ and the training stream by content each
run's). The support rule is the only exclusion: every other gap (a run missing or not ended, a
binding mismatch on either endpoint, a label receipt missing or pending, a required input that
does not read) leaves the seed technically unresolved, and while any seed is, the test gives no
inference (`status: withheld`, no ρ, p-value or reading; each unresolved seed named with its
reason; ρ over the resolved supported seeds under `descriptive_partial` alone); the descriptive
quantities stay, partial. The reading is withheld too unless the predictions' bucket copy
preceded the first fresh start, pod 2's completion record lists exactly the plan's fresh runs,
every fresh run read is ended in it, every pod-2 run's labeling started (the earliest
`created_at` of its label rows) and completed after its time, and the seeds are those of
`docs/study5-confirmatory-seeds.json`, its sha256 named by the registration. The surrogate-overlap
check is descriptive: h_s and ρ(p_s, h_s) on the 482 judged prompts outside the fixation
record's `evaluation_ids_same_or_close`, for the fresh seeds and for the retrospective. The
ladder's shuffled-D trigger is `undecided` unless all six paired differences exist. Every run's
GPUs are reported. It writes one `analysis.json`, never over an
existing file, with every input's sha256, and prints numbers only. A missing run, record or label
file of a seed is named in every quantity it leaves; the seed counts nowhere there.

```bash
export DATA_DIR="$DATA"   # the prompt sets
S3ARGS=(--study3-runs-root "$S3/runs" --study3-repo "$S3_CHECKOUT" --study3-sealed-dir "$SEALED")
python -m zero_field_refusal.analysis.study5_analysis_cli retrospective --runs-root "$RUNS_DIR" \
  --pod1 "$C/study5-pod1" --out "$RUNS_DIR/study5-final/retrospective.json" "${S3ARGS[@]}"
python -m zero_field_refusal.analysis.study5_analysis_cli full --runs-root "$RUNS_DIR" \
  --pod1 "$C/study5-pod1" --pod2 "$C/study5-pod2" --out "$RUNS_DIR/study5-final/analysis.json" \
  "${S3ARGS[@]}" --retrospective "$RUNS_DIR/study5-final/retrospective.json" \
  --reference-input "$RUNS_DIR/study5-reference/input" \
  --reference sonnet=SONNET.json --reference grok=GROK.json
```

## Judge server

The open-weights judge runs on its own Pod from the official vLLM image, never in the project
environment; `scripts/judge.py` runs locally and talks to it over HTTPS.

- **Pod:** one L40S, Secure Cloud, image `vllm/vllm-openai:v0.30.0` pinned by digest
  `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90` (CUDA 13.0.2: a host
  driver ≥ 580), 50 GB container disk, no volume, port `8000/http`. Environment:
  `HF_TOKEN={{ RUNPOD_SECRET_hf_token }}` (WildGuard is gated), `VLLM_API_KEY` from a Runpod secret
  holding a random key (also kept locally outside the repository, mode 600),
  `VLLM_NO_USAGE_STATS=1` and `DO_NOT_TRACK=1`.
- **Command** (the image's entry point is `vllm serve`):
  `allenai/wildguard --revision cbba4823f3e8020e5a74a5e29bf85072def6f2ff --tokenizer-revision
  cbba4823f3e8020e5a74a5e29bf85072def6f2ff --served-model-name allenai/wildguard --dtype bfloat16
  --max-model-len 4096 --seed 0 --generation-config vllm --port 8000`.
- **Labeling:** with `JUDGE_API_BASE=https://<pod id>-8000.proxy.runpod.net/v1` and
  `JUDGE_API_KEY` set in the shell (they override `.env`),
  `python scripts/judge.py label judge/wildguard SAMPLE.jsonl OUT.jsonl`. Before any item it
  checks through `/tokenize` that the prompt carries exactly one BOS: the template starts with
  `<s>` and the tokenizer adds BOS by default, so requests send `add_special_tokens: false`.
- **Second judge (Llama Guard 3):** `configs/judge/llama_guard.yaml`, backend `llama_guard`, model
  `meta-llama/Llama-Guard-3-8B` at revision `7327bd9f6efbbe6101dc6cc4736302b3cbb6e425`, served
  like WildGuard (same Pod shape, `--revision` and `--tokenizer-revision` that commit,
  `--served-model-name meta-llama/Llama-Guard-3-8B`; the repository is gated, so `HF_TOKEN` must
  belong to an account that accepted its licence). The client sends a chat completion, the
  prompt as the user message and the response as the assistant message: the server renders the
  model's own template (task, hazard categories S1 to S14, conversation), so no BOS check applies.
  The first non-empty line of the reply, `safe` or `unsafe` (case ignored), is the verdict;
  `unsafe` is `harmful_compliance` and `safe` is `safe_non_refusal`, which here means "not
  harmful" (a refusal is not told apart; this judge never writes `refusal`), and the category
  line of an `unsafe` reply is kept as `categories` in the record. Any other reply is
  unparseable, counted and retried in rounds exactly like a WildGuard reply that does not parse.
  `python scripts/label_runs.py --judge llama_guard RUN_DIR ...` writes `<run
  dir>/judge/llama_guard.jsonl` with its own receipt and rounds record, beside and never in
  WildGuard's files; the label identity (`backend`, `prompt_version`, model, revision) differs, and
  the readers check it, which keeps the two judges' files apart. For a two-class judge the
  generic endpoint reader (`study2_labels.read_endpoint_labels`) gives only `harmful` and
  `not_harmful` (no `refusal` or `safe_non_refusal` field); read its labels as harmful or not
  only. A reply cut at the 32-token cap is recorded with `truncated: true`; its first line still
  decides.
- **Run directories:** `python scripts/label_runs.py --judge wildguard RUN_DIR ...` labels every
  response of each completed run into `<run dir>/judge/wildguard.jsonl` (the same token check
  first); an output that does not parse is retried, in at most `--rounds` (3) rounds in all, and
  the run's remaining `unparseable` count is printed. The rounds are counted across reruns in
  `<run dir>/judge/wildguard.rounds.json`; an existing label file without that record (every
  stage-1 run) is refused, nothing sent, unless its receipt is complete: relabel under a new
  judge config name. A study-2 run (a training run with a `schedule` in
  its config) is labeled at every evaluation it recorded, `step-<N>`, whatever its stop reason (a
  run that stopped before its first evaluation is skipped);
  `analysis.study2_labels.read_endpoint_labels` then counts the labels per evaluation, strictly.
  Study-2 runs are labeled only with `analysis.study2_label_cli`, the same labeling with the
  sealed runs' output sealed, never with `label_runs.py`
  ([sealed output](#labeling-and-sealed-output)).
- **The key and the sealed folder:** the labeling client sends the key held in
  `~/.config/zero-field-refusal/judge_vllm_api_key` (mode 600, the value of the Runpod secret
  `zfr_judge_vllm_api_key`), exported as `JUDGE_API_KEY`, never the `.env` value, which may be
  older; study 3's sealed records and ledger lie under `$RUNS_DIR/sealed/` (the command refuses a
  folder in the checkout outside `runs/`).
- **Logging:** never add `--enable-log-requests` or `--enable-log-outputs` to the command, nor
  raise `VLLM_LOGGING_LEVEL` to `DEBUG`: the log would then hold the judged prompts and
  responses and the judge's outputs. Read the log only for the start and the deadline watcher's
  lines ([sealed output](#labeling-and-sealed-output)).
- The Pod has no SSH and no persistent storage: stop it as soon as the labels are written; the
  labels and their raw outputs stay local next to the sample or in the run directories.
- **Deadline without SSH.** When a hard stop is needed with nobody attending, the start command is
  `bash -c` running [`scripts/pod_deadline_watch.py`](../scripts/pod_deadline_watch.py) in the
  background before `exec vllm serve …`, with `ZFR_DEADLINE_EPOCH` in the environment as the cap
  (without a deadline file the cap is the deadline). Check
  its log line `zfr-deadline … auth True` and the variable (get-pod) right after the start; stop
  the Pod by hand after labeling, without waiting for the deadline.
- The Runpod proxy answers 403 to the `Python-urllib` User-Agent; send another one when probing
  the endpoint with `urllib` (the judge client is not affected).

## Launching runs

- **One GPU:** everything that fits on one GPU runs as one process, as in the examples above.
- **Several GPUs:** only for sharded full fine-tuning
  ([distributed training](architecture.md#distributed-training)):

  ```bash
  uv run torchrun --nproc-per-node 2 scripts/<entry point>.py experiments/<name>
  ```

  The number of processes must divide the global batch; the code fails otherwise. Rank 0 generates
  the run id and broadcasts it, so no `RUN_ID` is needed.
- **Campaigns:** `uv run scripts/launch.py CAMPAIGN.yaml` runs a list of cells (config,
  overrides, seeds; `entry: theta0` for θ₀ evaluations) in order, each attempt in a fresh process
  under its own `RUN_ID`. Every run's config is checked against the protocol before the first one
  starts. A run is skipped when one completed run with its config hash qualifies (the campaign's
  `tracking.gate`, a clean recorded commit when `require_clean_git` is set, a finite,
  non-negative K and the responses file). A completed run that does not qualify, or a second
  completed run, stops the campaign before anything starts, and so does an attempt that completes
  without qualifying: running again would leave two runs of the seed. A run that ended `diverged`
  (a non-finite training loss or K) is an outcome and is never rerun. Any other failure is
  technical and is retried, up to three attempts in all, counting earlier attempts from the run
  directories and the logged starts. Once the cause of exhausted runs is fixed, `--recovery
  REASON` opens one recovery round per cell (three more attempts), logged with its reason; never
  over a run whose label had a round of its own (below): the call is refused, naming the label
  and its round; no recovery round is written (a settled end of an open start may be written
  first).
  Starts, ends, outcomes and recovery rounds go to `RUNS_DIR/launch/<campaign>.jsonl`, each
  attempt's output to `RUNS_DIR/launch/<campaign>/`. `--dry-run` lists what would run. A `--dry-run`
  of a round lists the label as used up and exits 1:
  it opens no round and checks none of the round's refusals (a repeated reason, a missing incident
  field, a further round above 1), only the not-used-up one, so it does not tell an accepted round
  from a refused one.
  A conflict over a missing endpoint checkpoint ends with a note that points to "A lost
  checkpoint" in ["Study 3: numerical qualification"](#study-3-numerical-qualification): the
  file may be lost with a Pod's container disk, which needs a diagnosis before anything is
  moved, deleted or rerun.
  A start the launcher finds without an end (it died while the attempt ran, or after the child
  published its outcome) is ended when its run is assessed, before the run is skipped, a conflict
  raised or a new attempt started: an `end` record with the attempt's run id and number, no
  return code, `"settled": true` and the outcome its run directory gives (`done`, `diverged`,
  `conflict` with the problem, or `technical` when the directory is missing or unfinished). An
  existing end is never rewritten; a run that is not assessed (another label under `--only`) and
  a `--dry-run` get nothing written. Attempts are still counted from the starts. Ends,
  `superseded` and `held` records are matched to their starts by run id, and a run's attempts
  are ordered by their attempt number, then the start's time, whichever journal file holds them
  (a recovery logged under another campaign name included): the start ended is the latest one
  no such record names. A settled `done` keeps the launcher's meaning (a valid terminal outcome,
  a declared instability stop included), never a qualification.
  `SIGTERM` to the launcher alone (not its process group) stops the campaign at a run boundary:
  the running attempt finishes and is recorded, a `stop` record is logged, and the launcher exits
  non-zero without starting another; the same command carries on later. `--only LABEL` (a run's
  `<cell>/s<seed>`) assesses and runs that run alone, leaving every other run of the campaign
  unread, and its start and end records carry the label; a recovery round under it is that run's
  own. `--max-attempts N` replaces the three attempts (and the first round's three more, at most
  the label's base) with N;
  study 3's qualification driver passes `--only` and 1
  ([below](#study-3-numerical-qualification)), and never `--recovery`. No call runs more
  attempts than its own `--max-attempts`; the attempts it leaves stay for later calls.
  A later call's `--max-attempts` above the value of the first start under the run's config hash
  adds nothing (the base is the smaller); this holds per config hash: a new config hash is a new
  run, with its own first start.
  **A run's own rounds** (study 3's infrastructure route) are one per documented incident:
  `--recovery REASON --incident-time UTC --cause TEXT --evidence REF` (the interruption's time in
  UTC, ISO 8601; its external cause, one line; a path or reference to its evidence; none blank).
  The label's first round allows the smaller of N and the label's base (the first start's
  `--max-attempts` under its config hash); each further one (the label had a round, its cell's or
  its own, under any config hash) needs `--max-attempts 1` and allows exactly one. Grants follow
  this rule, never the journal: a round's recorded `granted` is for the record. A study-3 campaign
  (its header carries `study: 3`) takes no cell round: `--recovery` without `--only` exits 1 with
  the form of a run's own round; and `--recovery` over a run that is not used up under its config
  hash exits 1 too, with the command without `--recovery`: its attempts run without a round (a new
  config hash is a new run), and a documented incident is never dropped unrecorded. A round opens
  only for a run that used its attempts up and holds no outcome: a done or diverged run is skipped
  in a study-2 campaign and refused, exit 1, in a study-3 one (as the call over a run with
  attempts left is, a `--dry-run` included, with the same message),
  a conflict (a provenance conflict included: a checkpoint present with another sha256, a departing
  numerics setup, an unclean checkout, another gate; a missing endpoint checkpoint is a possible
  loss requiring diagnosis, never by itself a lost container disk) or a hold stops the campaign
  before, and an attempt, not superseded, with a published terminal record or an end `done`,
  `diverged` or `conflict` refuses it. Its reason must be one no earlier round of the label gave
  (spaces stripped): a new reason is necessary, not sufficient. A refused round exits 1, naming the
  label and the earlier round (its number, log and line), the outcome, the incident's gap or the
  bound, and the command form; no recovery round is written and nothing runs (a settled end of an
  open start may be written first). A run with attempts left takes
  them first: rounds never stack. The round records its number (`round`), the run's `config_hash`,
  `after` (every start of the label, under any config hash, in attempt order), `max_attempts`,
  `granted` and its incident. The latest such round of the run allows what the rule grants it beyond
  the attempts it names, so a later `--max-attempts` recalculates no earlier grant, and a round
  under another config hash gives the run nothing. A new start is numbered after every attempt of
  its label, whatever its log or config hash; the run id keeps its form and the label never changes;
  nothing is deleted or rewritten. A `superseded` record in the journal (a study-3 case restart or
  supersession) makes an attempt no longer count for its label: kept, a used attempt, neither done
  nor a conflict, so a round of its own (its first, or a further one) starts a new one. A `held`
  record (an outcome a study-3 engineering cycle could not resolve) makes the run a conflict:
  nothing starts, its recovery round included, and a `--dry-run` reports the same conflict. The hold
  applies to the run's label (`<cell>/s<seed>` in its experiment) in every journal, whatever the
  config hash it was written with, with or without `--only`; the launcher never lifts it. A
  checkpoint run (`tracking.endpoint_checkpoint`) is done only if its recorded numerics setup is
  what the campaign's config asks for (determinism, TF32 and the matmul precision, the memory
  limit), read without CUDA. Send it to the
  launcher's PID only: Ctrl-C or a signal to the process group (`kill -TERM -PGID`) also reaches
  the running attempt and ends it. The boundary is where an attempt is admitted, not where its
  process starts: a signal that arrives while an admitted attempt is being logged does not keep
  it from starting, so expect up to one more run after the signal. A signal during the last
  attempt that then succeeds leaves no `stop` record and exit code 0: the campaign simply ended.
- **The launcher's journal and a torn line.** Each record of `RUNS_DIR/launch/<campaign>.jsonl`
  is appended as one line, in one write, under an exclusive lock on the log (writers take
  turns); then, still under the lock, the log, its folder `launch/` and the runs root that holds
  it are fsynced, on every append, so that the log's and the folder's entries are durable before
  the append returns, whoever created them. The logs are read line by line,
  split at `\n` alone, each line decoded strictly as UTF-8. A **torn line** is any line that
  holds no record: one that does not decode or parse (a byte-order mark included), a valid JSON
  value that is not an object, a blank line (length 0), or a JSON object whose identity is not
  all strings (any of `event`, `run`, `experiment`, `run_id`, `config_hash`, `campaign`, `log`,
  `sha256`, `cell`, `only` present with a value that is not a string): such a record blocks
  until resolved instead of matching nothing. A record with one of these keys absent, or with an
  event no reader knows, is still a record, and matches nothing. The journal is written only by
  the launcher and qualify.py; it is never edited by hand. The torn-line rule catches a
  malformed value, not a removed key or an invented event. The launcher writes only records
  whose identity is all strings. A crash in the middle of an
  append leaves one at the end of the log, the bytes after its last newline; the next append to
  that log repairs it first, in two writes under its one lock: the newline that ends it together
  with a `torn` record naming its exact bytes (`offset`, `length`, `sha256`) and its `kind`, the
  event its bytes show through the strict pattern `"event": "<lower-case word>"` (`null`
  otherwise), then the new record; nothing else of the torn bytes is copied. The journal must
  also hold every line a published qualification record requires (a cycle's `held` and
  `superseded` lines, a case restart's `superseded` line; [study 3](#study-3-campaigns),
  "A missing journal line of a published record"): a line is there when a line of its event
  names the same label, experiment and run id, whatever its config hash. While any log under
  `RUNS_DIR/launch/` holds a torn line that no `torn_resolved` record names, or such a line is
  missing (or such a record cannot be read), the launcher starts nothing, settles nothing and
  opens no recovery round, with `--dry-run` too: it prints each torn line (log and line, offset,
  length, kind, sha256) and each missing line (its record's path, event and run id) and exits
  **12**.
  Every log counts, since a label's `held` and `superseded` lines can live in any of them. A
  missing line is restored by its record's repair alone (below); for a torn line, first restore
  what its kind lost, then record the resolution:

  - `end`: nothing to restore. The start stands without its end; once resolved, the launcher
    settles it from its run directory, as any start found without its end.
  - `start`: nothing to restore. A run directory may stand with no start; study 3's analysis
    lists it and blocks. The launcher counts that directory as any directory of the run's config
    and never reuses its run id.
  - `held`, `superseded`: run the writer that lost the line again, first, then resolve. Every
    `qualify.py` command that reads the runs refuses a journal that is not whole with exit 12, as
    the launcher does, except these two journal repairs, which run despite an unresolved torn
    line or a missing line, and then print what is left (a torn line to resolve, or another
    record's missing line):
    - `qualify.py supersede` with the same `--record`, `--diagnosis` and `--fix` as the
      published cycle: it writes only the missing `held` and `superseded` lines and prints
      `{"supersession": …, "journal_lines_written": [run ids]}` (exit 0);
    - `qualify.py restart-case --campaign … --candidate N --case CASE --infrastructure
      "<the original cause>"` again, for a case whose published restart records' launcher runs
      lack their `superseded` lines: it writes only those lines, each with its restart record's
      own reason (the new `--infrastructure` text is required but not written), and exits 1 with
      `<case>: the restart is recorded (restart-<n>.json); the launch journal now names its
      launcher run`.

    Any other call of the two (a new cycle, a new restart) is refused like the rest. Each writes
    only the missing lines: `launch.hold` and `launch.supersede` write a line only when no
    identical one (the same event, label, experiment, run id and reason) is in the journal. A
    held run is then never rerun, and a superseded one never counts.
  - `recovery`, `stop`: resolve, then relaunch with the same command; a recovery round's reason is
    recorded again (a run's own round with its incident and the same number: a torn line is no
    record, so no earlier round gave its reason).
  - `torn`: nothing to restore; the resolution writes the missing `torn` record.
  - `torn_resolved`: the line it resolved is unresolved again; run that resolution again.
  - unknown (`null`; a blank line, say): first read the run directories of the campaign's labels
    against the readable records; a line cut before its `"event"` may be a lost `held` or
    `superseded` line. Restore what is missing as above, in its kind's order (a lost `held` or
    `superseded` line: its writer again, then the resolution; otherwise the resolution, then the
    relaunch), and say what was found in the reason.

  ```bash
  python scripts/launch.py --resolve-torn LOG --offset N --sha256 HEX --reason TEXT
  ```

  `LOG` is the log's file name as printed; no campaign file goes with it. The command appends
  `{"event": "torn_resolved", "log", "offset", "sha256", "reason"}` to that log. A line that no
  `torn` record names yet (still the log's last, or left by a power loss inside the repair, a
  blank line, a line an older writer merged) gets its `torn` record first, so that both records
  name every resolved line. It refuses an offset and sha256 that name no torn line of that log,
  a line already resolved and an empty reason. The resolution does not recover the lost record:
  it records that the operator restored what the journal needs.
- **Study-2 runs** (`scripts/train.py` with a config that has a `schedule`) stop by K, plateau,
  instability or T_max and may extend their seed's appendable stream with their frozen θ₀; all
  runs of a seed that may extend its stream run on one machine, where writers take turns under
  a file lock on the stream's directory (`.lock`). Each run publishes its stop atomically in
  `terminal.json`; an instability stop is an outcome and is not rerun. The launcher counts such
  a run as done once its whole records agree with that terminal record (the checks of
  `analysis.stage_a.load_trajectory`: readings, steps, evaluations and their stored responses,
  levels, the stream prefix), with or without evaluations, and whatever its status says if
  logging or finalization failed after the record was published. With `eval.rayleigh` enabled
  it also needs a readable `rayleigh.json` holding a finite R or the reason R is null. R is
  descriptive and follows the published outcome, so a recorded failure (`failed: …`, out of
  memory, say; the run's status is then `failed`) counts as done, like a zero displacement, and
  never blocks the campaign; only a missing or unreadable record is a conflict. Hence the check
  of the first run's record ([data, streams and checks](#data-streams-and-checks)).
- **Gate runs:** commit and push all changes first, then run from a clone or bundle of that
  commit. Gate runs set `tracking.require_clean_git: true` and do not start from a dirty working
  tree. Model revisions are pinned to commit hashes in `configs/model/`.

## Storage

Decided 2026-10-11. Three places, each with one role:

| Place | Holds | Rules |
|---|---|---|
| This machine (the checkout, `./data`, `./runs`, `scratchpad/`) | code, documents, reports, scratchpad notes and small run results: run directories, judge labels, calibration samples and reference labels | Large data (teacher streams, soft-target distributions, checkpoints, model weights) takes at most 10 GB for the whole project here; beyond that it stays on the Pod's working storage and in the bucket. |
| Working storage of the Pod (from stage 1) | model weights and caches, teacher streams, soft-target distributions while they exist, intermediate files, and the run directories of ongoing runs | Mounted at `/workspace`, so the [container defaults](#containers) put `DATA_DIR`, `RUNS_DIR` and the caches on it. Preferred: a Runpod network volume (Standard, about 50 GB, $0.07 per GB and month) in a data center with L40S capacity. Where no such data center exists, the Pod's own volume disk, which lives on its machine and goes with the Pod. Either way it is working storage, not a backup, and it is deleted only with the researcher's approval. |
| `ARTIFACT_REMOTE`, a private, versioned Google Cloud Storage bucket | everything to be kept: finalized run directories, teacher streams, labels, inventories, scratchpad snapshots | Create and read only, no overwrite or delete ([archive](#rented-machines)). |

Stage 1 (decided 2026-10-11): no data center offered both L40S and network volumes, so the
L40S Pod uses its volume disk at `/workspace` and the bucket. Record the Pod's ID, data center and
volume size here when it is created; a network volume, if one is created later, likewise.
Stage-1 Pod: `32fjt8f6671w17` (`zfr-stage1-l40s`), L40S Secure, US-TX-4, created 2026-10-11 10:37
UTC, 120 GB volume disk at `/workspace`, 50 GB container disk; stopped 21:00 UTC after every run,
stream and log was archived and verified, and deleted with its volume the same day.

- **Inputs before the runs that use them.** A generated teacher or preconditioner stream is
  archived to the bucket and verified before the first run that reads it starts, so losing the
  machine never loses a stream that results depend on.
- **Every run record.** Run directories go to the bucket as their batch finishes, failed runs
  included, together with the launch log.
- **Kept checkpoints.** A study-3 technical case's checkpoints are exported to the Pod's volume
  (`/workspace/kept-checkpoints`) and to `$ARTIFACT_REMOTE/kept-checkpoints/`, both verified,
  before its driver stops ([numerical qualification](#study-3-numerical-qualification)); the
  bucket copy is kept until the study's report.

- **Small results** (run directories, labels, reports) come back to this machine, are checked
  against their sha256, and are archived from here.
- **Large artifacts** go from the Pod straight to the bucket, not through this machine. The
  archive service account's key is the Runpod secret `gcs_archive_key_b64` (the key file in
  base64). Pass it at Pod creation as
  `ARTIFACT_GCS_KEY_B64={{ RUNPOD_SECRET_gcs_archive_key_b64 }}`, together with
  `ARTIFACT_REMOTE`. On the Pod, decode it to a file on the container disk, never under
  `/workspace`, and point `ARTIFACT_GCS_KEY` at it, without printing either:

  ```bash
  umask 077
  printenv ARTIFACT_GCS_KEY_B64 | base64 -d > /root/.artifact-gcs-key.json
  chmod 600 /root/.artifact-gcs-key.json
  export ARTIFACT_GCS_KEY=/root/.artifact-gcs-key.json
  ```

  `scripts/pod_setup.sh` does this on a new Pod: it installs the Google Cloud CLI pinned by
  version and sha256 on the container disk (the image has none), decodes the key, writes
  `/root/zfr-env.sh` for later shells, and restores `DATA_DIR` from the bucket. Every file in
  `DATA_DIR` must then appear, with its sha256, in one of the inventories given (all of those that
  cover `data/`), and every listed file must be present; anything else stops the setup. Then follow the [archive procedure](#rented-machines) on the
  Pod. The key file goes away with
  the container disk when the Pod stops or is deleted; decode it again after a restart.
- **Scratchpad snapshots.** At the end of each stage, together with its report in
  `docs/reports/`, archive `scratchpad/` to `scratchpad/<date>/` in the bucket. Scan it for
  secrets first (the values in `.env` and in local key files, and token patterns, printing only
  counts), then follow the archive procedure with its own inventory.

## Rented machines

Set `DATA_DIR`, `RUNS_DIR` and model caches on verified persistent storage: the Pod's working
storage at `/workspace` ([storage](#storage)). Small results come back to this machine (copied and checked against their
sha256) before they are archived; large artifacts are archived from the Pod.

**Archive.** `ARTIFACT_REMOTE` is a private Google Cloud Storage bucket path with object versioning.
Its service account (key file at `ARTIFACT_GCS_KEY`; on a Pod only as described under
[storage](#storage)) may only create and read objects: an archived object cannot be overwritten or
deleted, so a changed artifact goes to a new path (a new version). Copy finalized artifacts with the key as a per-command override, leaving the
active `gcloud` account unchanged; set `COMPLETED_DIR` to the artifact directory and `ARTIFACT_ID`
to its archive path (e.g. `runs/<experiment>/<run id>`):

```bash
CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$ARTIFACT_GCS_KEY" \
  gcloud storage cp --recursive --no-clobber "$COMPLETED_DIR" "$ARTIFACT_REMOTE/$ARTIFACT_ID"
```

`scripts/archive.py INVENTORY.tsv "$ARTIFACT_REMOTE" ROOT PATH...` does the copy and the checks
below in one step: it copies each path under `ROOT` to the same relative path in the bucket,
compares every object's size and MD5 with the local file, and appends the inventory lines; an
object that already held other content shows as a mismatch.

After each copy, compare the local sha256 of every file with a copy downloaded back (at least for
one artifact per batch) and with the objects' listed hashes, and record every file with its sha256
and archive path in a dated inventory, `runs/archive/inventory-<date>.tsv`, which is kept locally
and archived as a new object. Do not copy changing logs, labels, indexes or W&B files before
they are final. Preserve the working copy until the archived copy is verified.
Compute can be released earlier only if the outputs remain on retained storage and a tested transfer
route works without that machine; otherwise finish the copy first.

Restore/merge finalized artifacts into the shared store with the same additive procedure, then sync
completed offline W&B records and rebuild the `scripts/index_runs.py` index.

## Artifact retention

Before a costly campaign, record which configs/seeds and matched optimizer/control arms to retain,
why, storage budget and retention review point. Choose coverage before interpreting outcomes;
document changes.

- **Evidence:** keep the [publication records](architecture.md#publication-evidence), actual teacher,
  preconditioner/reference samples, labels and failure/exclusion records. Generated samples are
  irreplaceable evidence, not a cache.
- **Weights:** keep `train.save_checkpoint: false` by default; enable it for the selected run configs.
  Weight-saving/restore code is still planned. Verify saved weights for later re-evaluation;
  continuation requires complete training state and a tested restore path. Without weights,
  new model-dependent checks require another training run.
- **Temporary files:** evict reconstructible caches and copies once their inputs remain available.

The [archive procedure](#rented-machines) governs verification and deletion. Public exports follow
the [release checklist](#release-checklist).

### Retention plan: stages 0 and 1

Recorded 2026-10-09, before the first GPU runs; review it before stage 2.

| Artifact | Kept | Size (order of magnitude) |
|---|---|---|
| Prompt sets, manifests, pinned source CSVs | all | < 50 MB |
| Teacher streams (training, preconditioner, K reference, field stream of seed 1000) | all, archived | ≈ 1 MB per 500 responses |
| Soft-target distributions of the path-field check | deleted after the check; the record, hashes and results kept | ≈ 9.6 GB per environment while they exist |
| Full soft-target stream of the SGD control | deleted after its run; hashes and summary kept | ≈ 29 GB while it exists |
| Run directories (config, meta, metrics, observations, responses) | all, including failed runs | ≈ 1–5 MB per run |
| Judge labels, calibration pool, sample, key, reference labels and their provenance | all, private | < 50 MB |
| Model weights | none: `train.save_checkpoint` is not implemented; controls run in-process | — |
| Model and dataset caches | evicted when space is needed | ≈ 10 GB |

Stages 0–1 together stay below ≈ 10 GB of retained evidence. Every stage-1 cell is kept, matched
arms included, since none is chosen by its outcome.

## Weights & Biases in practice

- W&B mirrors configs and metrics for browsing and comparing runs; the run directories on
  persistent storage remain the source of truth. Only metrics and configs are sent: no artifacts,
  weights or response texts, which also keeps the account within the free plan's storage.
- The project `zero-field-refusal` was created private before the first run; runs are grouped by
  experiment. Check that the first Runpod run appears there.
- Entity, key and mode come from `.env` locally. On Runpod the key is a Runpod secret passed as a
  Pod environment variable at creation, `WANDB_API_KEY={{ RUNPOD_SECRET_wandb_api_key }}`; it never
  enters the image, the project files or command lines. With `WANDB_MODE=offline`, runs need no
  network and are synced later with `wandb sync`.
- `tracking.wandb.enabled: false` turns the mirror off (the smoke experiment does);
  `tracking.wandb.required: true` makes a failure to start W&B stop the run. The privacy settings
  every run starts with are listed in
  [architecture.md](architecture.md#experiment-tracking-and-storage).

## What stays out of git

- `data/`: prompt sets and teacher samples;
- `runs/`: run outputs;
- `.env`: keys;
- `*.local.*`: private notes;
- judge labels, calibration samples, their keys, reference labels and their provenance: they
  contain raw responses to harmful prompts and stay in `DATA_DIR` or `RUNS_DIR`.

## Release checklist

The GitHub repository stays private until the release. A public release:
- excludes `wandb/` folders and raw completions to harmful prompts; those are shared through a
  gated release at most;
- contains reports with aggregates only ([convention](reports/README.md)).
