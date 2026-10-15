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

`prepare_data.py` caches the pinned source CSVs in `DATA_DIR/sources/` and refuses to overwrite a
complete build without `--force`. Stream directories are named by a hash of everything that
determines their samples; an existing stream is never regenerated. A stream is published by
renaming a partial directory; after an interrupted generation, remove the leftover
`.seed-*.partial-*`, `.reference.partial-*` or `.chunk-*.partial-*` directory with
`generate_teacher.py ... --remove-partial` once nothing is generating.

The stage-0 checks write a path-field record per execution environment under `DATA_DIR/checks/`:
model numerics, GPU model, library versions, execution batch sizes and the check's version; an
older record moves to `history/`. Training runs stop before their first optimizer step without a
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
On Pod A, once, before any run reads it: generate the new monitor stream, record its spec hash
and sha256 here, and archive it with its own inventory, which the other Pods' `pod_setup.sh`
adds.

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

Its later steps, and the full controls of the SGD rates it runs, are written from the version-2
runs of all three seeds once each step has resolved.

**Stage 0 under a new protocol** needs the soft-target distributions of its path-field check,
which the retention policy deletes after each check: the bucket keeps only the stream's manifest
and samples, which `pod_setup.sh` restores. Before the checks, remove that restored copy
(`teacher/<model>/soft_targets/<spec>/seed-0`) from `DATA_DIR`; stage 0 then generates the stream
again. Compare the new samples' and distributions' sha256 with the archived manifest and record the
comparison next to the run (on 2026-10-14 both were byte-identical, see
`runs/runpod/evidence/soft-targets-regeneration-20261014.json` in the bucket). After the check,
delete the distributions and put the archived manifest back, so that `DATA_DIR` matches its
inventories again.

**Deadlines between campaigns.** A Pod's deadline changes only through its environment
(`ZFR_DEADLINE_EPOCH`, with `update-pod`), which restarts the container: the running run stops and
the container disk loses the archive key and the Google Cloud CLI. Change it only between
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
  first); rerun it until no output is unparseable.
- The Pod has no SSH and no persistent storage: stop it as soon as the labels are written; the
  labels and their raw outputs stay local next to the sample or in the run directories.
- **Deadline without SSH.** When a hard stop is needed (the user away), the start command is
  `bash -c` running [`scripts/pod_deadline_watch.py`](../scripts/pod_deadline_watch.py) in the
  background before `exec vllm serve …`, with `ZFR_DEADLINE_EPOCH` in the environment. Check
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
  `SIGTERM` to the launcher alone (not its process group) stops the campaign at a run boundary:
  the running attempt finishes and is recorded, a `stop` record is logged, and the launcher exits
  non-zero without starting another; the same command carries on later. Send it to the
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
  logging or finalization failed after the record was published.
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
