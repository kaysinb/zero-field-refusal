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

**Setup of a study-2 Pod.** From the checkout of the reviewed commit (a Git bundle), restore only
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
reviewed commit:

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
  untracked and not ignored by the checkout's `.gitignore` files (keep local work directories,
  such as `.superpowers/`, in `.gitignore`: `.git/info/exclude` and a user's excludes file do not
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
  names the commit gate B was reviewed at (`e380bcc`) and its GPU-code identity (119 files,
  sha256 `5cddd583…7680`; `code_identity.gpu_code_identity`). S1 ran from `aa17a99`, gate B as
  first approved (review fa15; GPU code `78b166a2…`); the fa21 change then touched only the
  numerical check's code and settings, and S1 stands. The S2 repeat ran from `aa4c007`, with
  gate B's current GPU code.
- **Gate C** (the analysis, the simulation, the audit, the fixation and the docs) changes no GPU
  path. A **GPU path** (`code_identity.is_gpu_path`) is every tracked file under `scripts/`,
  `src/zero_field_refusal/` (outside `analysis/`, but with `analysis/stage_a.py`), `configs/` and
  `docker/`, and `Dockerfile`, `.dockerignore`, `.python-version`, `pyproject.toml` and `uv.lock`;
  the analysis-only scripts (`study2_short_campaigns.py`, `study2_short_summary.py`,
  `study2_short_simulate.py` and the others listed in `code_identity.ANALYSIS_SCRIPTS`) are not.
  Gate C also leaves alone the modules the launcher and the GPU tools load: `analysis/__init__.py`
  and its imports, and `analysis/stage_a.py`. `tests/test_gate_c.py` checks both against the
  gate-B record; a gate-C change that fails it would void S2's qualification.
- **A qualification survives analysis-only commits.** The numerical check's records and the
  selection's reading of its verdicts hold at any later commit with the same GPU code
  (`code_identity.reuse_check`). A GPU-path change after the K-only runs voids them: see
  [the void](#the-void-and-the-repeat-of-a-numerical-stage).

### Pods

- **Created by the coordinator on request:** L40S, minCuda 13.0, not OC-AU-1, the dependency image
  of [containers](#containers), a volume at `/workspace` (150 GB for S1 and S2), host RAM of at
  least 64 GB (80 GB preferred), and `scripts/pod_deadline_watch.py` in the start command with
  `ZFR_DEADLINE_EPOCH` as the cap and the deadline in `/workspace/zfr-deadline` (see
  "Deadlines between campaigns" above). Right after the start, check the watcher's log line
  `zfr-deadline … auth True`, the GPU and its driver, and the host memory the container may use
  (the cgroup's limit, which can be below the machine's).
- **The Pod's environment.** An SSH session does not see the Pod's environment variables: source
  `/etc/rp_environment` first (`pod_setup.sh` failed without it); the stage's drivers source it,
  and `/root/zfr-env.sh`, themselves.
- **SSH.** Use direct SSH to the Pod's public IP and mapped port. Read the host key from the Pod's
  logs and pin it in a known-hosts file before the first connection.
- **Code.** A Git bundle of the reviewed commit, checked out clean under
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
- **The run environment** (the final review of gate B, I3). The check compares each K-only run's
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

Locally, at the reviewed commit (`R` and `THETA0` as in amendment 3's commands; campaigns and
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

A GPU-path change after the K-only runs (S2's: the fa21 comparator, `e380bcc`) leaves their
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
  it leaves in place; and, for the coordinator to decide, when S3 has started (a validation,
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

After the second reviewer approves gate C, write and commit the versions record (amendment 3's
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

Study 2 labels its runs only with the sealed labeling command (review fa27), which wraps
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

  These are documented limits of the blinding (review fa28): batching reduces what they
  attribute to a run but does not remove them, and no scientific decision may use a labeling's
  timing, its request counts or a call's status.
- **These rules are procedural protection, not enforced access control** (review fa28): nothing
  technically prevents running `label_runs.py` on a study-2 run or opening a sealed record. They
  hold because every operator follows them.
- **Exit:** non-zero for the open runs exactly as `label_runs.py`; for the sealed runs only for
  error, busy, not attempted or not complete, with a message that names the statuses and the
  record, never a count. Rerun after "not complete" or "not attempted" (bring the judge back
  first), and wait out "busy".
- **An "error" that stays after a rerun:** the run waits. An outcome-blind helper may open its
  record and relays only the exception's class and one remedy from this list: rerun; restore the
  label file from the archive; exclude the run by a recorded decision. Never a count or a value.
  The coordinator decides.
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
  record. Without a fixed level nothing is written and the outcome goes to the second reviewer.
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
docstring has every rule.

- **The campaign.** One launcher campaign per candidate and stage, written whole by
  `scripts/study3_campaigns.py` ([below](#study-3-campaigns)): one cell per run, named
  `c<N>-<step>-<optimizer>-<rate:g>-s<seed>` with label `<cell>/s<seed>`, config
  `experiments/study3_k_only` with `study3.role=numerical` (or `engineering`), the arm, the rate,
  `train.batch_size=20` and `tracking.endpoint_checkpoint` set to one absolute directory on the
  Pod's container disk, on one of the candidate's numerical seeds (or the engineering seeds).
  Each FA rate needs its case on the first numerical seed, which holds the rate's same-path
  control. `plan` refuses any other campaign, and lists the cases, the GPU time
  and the disk.
- **Each case, in order, one at a time:**
  1. a complete case (below) is skipped: nothing of it runs again (`complete` exits 0; 3 says
     no, and any other exit of `complete` or `decided` stops the driver). A case whose recorded
     parts were written under other GPU code or settings than the checkout's is stale: exit 10,
     and the driver stops there, neither rerunning nor skipping it. A case whose label
     (`<cell>/s<seed>` in its experiment) has a launcher run held in any launch journal, whatever
     its config hash, stops the driver with exit 11 naming the hold, before the launcher is
     called; nothing lifts a hold;
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
     verdicts without error. A case whose verdict is technical (its own or its control's) keeps
     them for the diagnosis that starts its engineering cycle: `finish` exits 9 and the driver
     stops before the next case; `finish --release` deletes them once diagnosed.

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
  below); an interruption without a documented cause is recorded technical instead
  (`interrupted`). Then run the driver again. A failed launcher run stops the driver too; its
  rerun is the launcher's recovery round with the reason.

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
    --recovery "attempt 1 failed: <the fixed cause>"
  ```

- **The deadline.** Start the Pod with the file-reading deadline watcher (see "Deadlines between
  campaigns" [above](#study-2-stage-a)), its cap in `ZFR_DEADLINE_EPOCH`. Before the driver,
  set the deadline in `/workspace/zfr-deadline` to cover the cases left, ≈ 39 min each and
  ≈ 11 min more per same-path control, with an hour to spare
  (`python3 scripts/pod_deadline_watch.py --set EPOCH`). A case the deadline stops leaves an
  interrupted attempt (above) and its checkpoints on the container disk, which a stopped Pod
  loses: the case starts again as a whole (below).
- **A case restart.** When the Pod stopped during a case, its recorded parts cannot go on (the
  launcher's and the harness's checkpoints are gone). `restart-case` records it: a
  `restart-<n>.json` in the case's folder names every attempt of the case's parts (baseline,
  rest and the case's control), each recorded one with its record's sha256, and the reason;
  those attempts are void, kept and listed in the verdict, and neither count nor block. Once the
  record is published, it logs the case's launcher run superseded in the launch journal: the run
  stays on disk and in the journal, no longer counts for its label, and the launcher's recovery
  round with the same reason starts a new one (if the command stopped between the two, running
  it again writes the journal's line). Only an interrupted case is restarted: a restart is
  refused without a reason, for a complete case, without evidence of an interruption (an
  interrupted attempt, or the launcher run's checkpoint gone from disk, which is enough even
  before the baseline started; one `finish` removed, recorded in its `checkpoints-removed-<n>.json`,
  is not), when the launcher run
  itself is a completed failure (its declared instability stop, scientific; another stop,
  technical), and for a case with a completed failure, scientific (it keeps counting: no reruns
  to pass) or technical (the engineering cycle lifts it, below), an interrupted rest's
  trajectory results included. Then rerun the launcher run through its recovery round, run each
  part the case needs as its next attempt with the same reason, the baseline first
  (`part --resume` exits 8 for a part the restart names), and run the driver again.
  ```bash
  project-run "$SRC" python scripts/qualify.py restart-case --campaign "$C/qualification-c1.yaml" \
    --candidate 1 --case s10200/adamw-1.45e-05 --infrastructure "the Pod stopped during the case"
  project-run "$SRC" python scripts/launch.py "$C/qualification-c1.yaml" \
    --only c1-numerical-adamw-1.45e-05-s10200/s10200 --max-attempts 1 \
    --recovery "the Pod stopped during the case"
  project-run "$SRC" python scripts/qualify.py part --campaign "$C/qualification-c1.yaml" \
    --candidate 1 --case s10200/adamw-1.45e-05 --part baseline --attempt 2 \
    --infrastructure "the Pod stopped during the case" \
    --run-dir "$RUN_DIR" --checkpoints /root/ckpt/harness/s10200-adamw-1.45e-05
  ```
- **A technical failure and its engineering cycle.** A completed technical failure (the
  repeat beyond 0.01·K\*, a failed bridge, a launcher comparison alone, an exception) is never
  rerun as it stands. Each candidate has one engineering cycle. After its diagnosis and fix,
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
  the cycle names re-qualifies under the new code: the launcher's recovery round, then each part
  as its next attempt with `--engineering-cycle` (its record names the supersession's path and
  sha256), not `--infrastructure`; a later interruption of a new-code attempt is rerun with
  `--infrastructure REASON`. The verdict lists the voided records under `supersession`.

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
  `interrupted --technical REASON` records the attempt with them.
- **The commands.** The campaign file and the driver are written on the Pod, from the source
  checkout of the commit; the driver embeds the campaign's absolute path and sha256.

  ```bash
  C="$RUNS_DIR/campaigns"
  project-run "$SRC" python scripts/qualify.py plan --campaign "$C/qualification-c1.yaml" \
    --candidate 1
  project-run "$SRC" python scripts/qualify.py driver --campaign "$C/qualification-c1.yaml" \
    --candidate 1 --commit SHA --out "$C/qualification-c1-driver.sh"
  bash "$C/qualification-c1-driver.sh"   # detached
  ```

  The engineering stage adds `--engineering` and, for its path diagnostics,
  `--diagnostic-steps 1,5,10,15,20`. The driver checks the checkout and the campaign's sha256,
  that no process holds the GPU and under 1 000 MiB of it is in use, and that the free host
  memory reaches `host_memory_gb` (60 GB: θ₀, v₀, the second model and the p1 direction on the
  host, ≈ 25-30 GB at 1.5B, and a checkpoint save's ≈ 5 GB, × 1.5, rounded up), then runs the
  cases. It writes the verdict on every exit. **Exits:** 0, every case complete; 1, a
  precondition (the host memory included), the disk (or `df` giving no number) or a `finish`
  that refuses a case not complete; 8, an interrupted attempt (or a part a case restart names);
  9, a technical case; 10, a stale case; 11, a case whose label has a held launcher run; the
  launcher's own exit when it stops; any other, the exit of the command that failed.
- **Outputs** under `$RUNS_DIR/qualification/`: `c<N>/<numerical|engineering>/s<seed>/<arm>-<rate>/`
  holds each part's `<part>-attempt-<n>.json` and its `.started` file, `c<N>/verdict.json` the
  verdict (each rate's state, its cases with their 13 comparisons, kinds and reasons, its
  same-path control, every attempt of every part, the superseded records with the restart that
  voids them, the warnings; `--e-eng X` sets τ; a rate whose launcher runs are not all located
  is at most incomplete), `c<N>/engineering/verdict.json` the
  engineering stage's (e_eng, the path maxima, the margins), and the driver's logs. Download and
  archive them with an inventory before the Pod stops.

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
  a terminal outcome;
- the active arms and comparisons, the rates, the route, the closures and the problems;
- the validation and confirmation runs that do not count for the commit they ran (`not_counted`,
  below).

Nothing is decided on a problem: the step stays blocked while one is listed. Read the state
before each command below.

- **The order of commands.** A campaign is written only when the step names its kind and
  candidate (`--candidate N`); any other is refused, with the step the state names.
  1. **Engineering.** `engineering` writes the starting rates on 10000-10003, run by the printed
     `qualify.py driver ... --engineering` ([above](#study-3-numerical-qualification)). The
     engineering record then names the records of its four checks: the dry run, the bridge,
     θ₀'s utility (`scripts/check_theta0_utility.py`) and the stage-0 check.
  2. **The simulation record**, before the search.
  3. **The search.** `search-round0`, then `search-round1` where a median missed.
  4. **The numerical qualification.** `numerical`, run through its printed `qualify.py driver`;
     then `repair` (one round per arm, amendment 3) for an arm whose K misses by more than 0.05
     over its 4 + 3 seeds once its qualification passed, run the same way.
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
  with its documented cause, always for the one run concerned. The launcher grants a round once
  per cell, and each cell holds one run (`c<N>-<step>[-r<round>]-<optimizer>-<rate>-s<seed>`, its
  label `<cell>/s<seed>`), so no other run gets a second attempt from it:

  ```bash
  project-run "$SRC" python scripts/study3_summary.py check --runs-root "$R" \
    --campaign "$C/c1-validation.yaml"
  project-run "$SRC" python scripts/launch.py "$C/c1-validation.yaml" \
    --only c1-validation-adam_frozen-1.83e-05-s10304/s10304 --max-attempts 1 \
    --recovery "<the documented cause>"
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
    screen, L, the validation size and the independent verification).
  - `n-c<N>`: the precommitted spread script's output, with the fields the `choice` step gives
    (candidate, rates, comparisons, cohort), L, the spread rounded to 0.5 pp, N and the cap.
  - `budget-c<N>`: the researcher's decision on the extra blocks and their price, `proceed` or
    `stop`, when N is above 90 or raised for power (the step names it then).
  - `closure-c<N>-<arm>`: written from the step (below).

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
  one does not count ([the registration](study3-preregistration.md#voiding-and-supersession)).
  The state compares each run's `meta.json` commit with the commit its header's versions record
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
  completed run, so a completed run under another commit cannot be rerun, and the state blocks on
  it. Its way out is a decision the researcher records (`supersession_needed`); a run with no
  published outcome is rerun by its recovery round, after `check` passes.
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

  The recovery form `launch.py FILE --only LABEL --recovery REASON --max-attempts 1` is then for
  validation and confirmation runs, always for the one run concerned. `qualify.py verdict` does
  not touch the pins, as long as it runs from a checkout whose GPU code is candidate N's code in
  force. A verdict cache written under any other code makes the fixation invalid until `verdict`
  is run again under the code in force.
- **The confirmation and its guard.** The confirmation is written into a new folder, one block
  per file in seed order: the step's next block and, with `--blocks FIRST LAST`, those after it up
  to the cap, or the cells a started block still lacks (completed first, alone).

  ```bash
  project-run "$SRC" python scripts/study3_campaigns.py confirmation --runs-root "$R" \
    --candidate 1 --out "$C/c1-confirmation-1" --blocks 10400 10469
  project-run "$SRC" bash "$C/c1-confirmation-1/run.sh"   # detached
  ```

  `run.sh` runs one block at a time. For each it checks the block file's sha256, runs the guard,
  `study3_summary.py guard --runs-root "$RUNS_ROOT" --candidate N --seed S --campaign FILE`
  (with the writer's `--theta0` when the writer was given one, so that the guard reads the θ₀ the
  writer read), then `launch.py FILE --max-attempts 1`, and stops at the first non-zero exit. The
  guard exits 0 only when:
  - the step is candidate N's confirmation and names cells on S. A block not yet started must be
    the next one: never after the stop (every active counter at N, or the cap
    N + max(10, ⌈0.12·N⌉) of blocks started), while a technical mark in a started block waits
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
    --only LABEL --recovery "<the interruption>" --max-attempts 1
  ```

  The launcher first settles the run's start with the outcome its run directory gives (technical
  for a run left unfinished), and starts its next attempt only when that outcome is technical.
  The block's missing cells, if any, are then written again by the writer, alone, under a new
  campaign name. A block file or a `run.sh` already run is not run again.
- **A torn launch-log line** (a residual risk). The launcher appends each journal line with its
  newline, and does not first end a line a crash cut short. A crash mid-append leaves the last
  line cut short, which every reader skips, as the launcher does. If another append follows, the
  cut line is no longer the last: it is corrupt, and the record appended after it is joined to it
  and lost too. The analysis then blocks, naming the log and the line, since a start it held
  cannot be shown. For a qualification log the validation fixation pins, any edit of the log
  breaks the fixation for good. The repair belongs to a later change of the launcher: a fresh
  line when the log does not end with a newline.
- **Holds and closures.** Two kinds of run are unresolved technical marks: a run a candidate's
  engineering cycle held (an outcome nobody could resolve, [above](#study-3-numerical-qualification)),
  and a technical run outside the qualification that no recovery round reruns. The step names
  each: for a held run, the closure record to write; for a technical run, its ways out (a recovery
  round, where the launcher would open one, and its closure record). The researcher decides which
  way out is taken. A closure record states the terminal technical outcome. A closure of A or F
  ends the study with the technical feasibility report, and candidate 2 is not taken; a closure of
  P closes P–F, and A–F goes on. It is written from the step, with exactly the fields the step
  names; `--run-id` chooses the run when the step names closures of two runs of the arm:

  ```bash
  project-run "$SRC" python scripts/study3_summary.py closure --runs-root "$R" --candidate 1 \
    --arm F
  ```

  The hold and the failure's evidence stay as they are, and a closed or held label is never
  issued again.

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
- **Logging:** never add `--enable-log-requests` or `--enable-log-outputs` to the command, nor
  raise `VLLM_LOGGING_LEVEL` to `DEBUG`: the log would then hold the judged prompts and
  responses and the judge's outputs. Read the log only for the start and the deadline watcher's
  lines ([sealed output](#labeling-and-sealed-output)).
- The Pod has no SSH and no persistent storage: stop it as soon as the labels are written; the
  labels and their raw outputs stay local next to the sample or in the run directories.
- **Deadline without SSH.** When a hard stop is needed (the user away), the start command is
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
  REASON` opens one recovery round per cell (three more attempts), logged with its reason.
  Starts, ends, outcomes and recovery rounds go to `RUNS_DIR/launch/<campaign>.jsonl`, each
  attempt's output to `RUNS_DIR/launch/<campaign>/`. `--dry-run` lists what would run.
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
  own. `--max-attempts N` replaces the three attempts (and the round's three more) with N; study
  3's qualification driver passes `--only` and 1 ([below](#study-3-numerical-qualification)).
  A `superseded` record in the journal (a study-3 case restart or supersession) makes an attempt
  no longer count for its label: kept, a used attempt, neither done nor a conflict, so its
  recovery round starts a new one. A `held` record (an outcome a study-3 engineering cycle could
  not resolve) makes the run a conflict: nothing starts, its recovery round included, and a
  `--dry-run` reports the same conflict. The hold applies to the run's label (`<cell>/s<seed>` in
  its experiment) in every journal, whatever the config hash it was written with, with or
  without `--only`; the launcher never lifts it. A checkpoint run (`tracking.endpoint_checkpoint`) is done
  only if its recorded numerics setup is what the campaign's config asks for (determinism, TF32
  and the matmul precision, the memory limit), read without CUDA.
  Send it to the
  launcher's PID only: Ctrl-C or a signal to the process group (`kill -TERM -PGID`) also reaches
  the running attempt and ends it. The boundary is where an attempt is admitted, not where its
  process starts: a signal that arrives while an admitted attempt is being logged does not keep
  it from starting, so expect up to one more run after the signal. A signal during the last
  attempt that then succeeds leaves no `stop` record and exit code 0: the campaign simply ended.
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
| Working storage of the Pod (from stage 1) | model weights and caches, teacher streams, soft-target distributions while they exist, intermediate files, and the run directories of ongoing runs | Mounted at `/workspace`, so the [container defaults](#containers) put `DATA_DIR`, `RUNS_DIR` and the caches on it. Preferred: a Runpod network volume (Standard, about 50 GB, $0.07 per GB and month) in a data center with L40S capacity. Where no such data center exists, the Pod's own volume disk, which lives on its machine and goes with the Pod. Either way it is working storage, not a backup, and it is deleted only with the user's approval. |
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
