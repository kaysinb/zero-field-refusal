"""Study 3's numerical qualification on a Pod: the plan, a case's parts, the driver, the verdict.

    python scripts/qualify.py plan --campaign FILE --candidate N [--engineering]
        [--diagnostic-steps T1,T2,...]
    python scripts/qualify.py driver --campaign FILE --candidate N [--engineering]
        [--diagnostic-steps T1,T2,...] --kept DIR [--kept-remote PREFIX] --commit SHA
        --out FILE [--dry-run]
    python scripts/qualify.py part --campaign FILE --candidate N [--engineering] --case NAME
        --part baseline|rest|control
        (--resume | --attempt N [--infrastructure REASON | --engineering-cycle])
        --run-dir DIR --checkpoints DIR [--diagnostic-steps T1,T2,...] [--device D]
    python scripts/qualify.py complete|decided|control-only|clear-stale|locate
        --campaign FILE --candidate N [--engineering] --case NAME
    python scripts/qualify.py finish [--kept DIR | --release] --campaign FILE --candidate N
        [--engineering] --case NAME
    python scripts/qualify.py kept-check --campaign FILE --candidate N [--engineering]
        --case NAME --inventory FILE --remote PREFIX
    python scripts/qualify.py interrupted --campaign FILE --candidate N [--engineering]
        --case NAME --part baseline|rest|control --attempt N --technical REASON
    python scripts/qualify.py restart-case --campaign FILE --candidate N [--engineering]
        --case NAME --infrastructure REASON
    python scripts/qualify.py supersede --campaign FILE --candidate N [--engineering]
        --case NAME --record PATH --diagnosis TEXT --fix TEXT
    python scripts/qualify.py verdict --campaign FILE --candidate N [--engineering] [--e-eng X]

Every command also takes ``--runs-root DIR`` (RUNS_DIR) and ``--out DIR`` (the part records,
default RUNS_DIR/qualification/). The rules are docs/study3-preregistration.md, "Numerical
qualification", "Failure routes", "Voiding and supersession" and "Amendment 2"; the harness, the
records and the verdicts are ``zero_field_refusal.qualification``'s, and the settings the tracked
configs/experiments/qualification.yaml (never another file).

**The cases** (``campaign_cases``) are the runs of a launcher campaign (``launch.load_campaign``),
one case per run: its seed, arm and rate as its config has them, for ``--candidate`` (numerical,
or with ``--engineering`` the engineering stage). Each run must be a K-only training run
(``eval.k_only``) of the role (``study3.role``: numerical, or engineering), at B 20 and 25 steps,
on the candidate's numerical seeds (or the engineering seeds), with its endpoint checkpoint
(``tracking.endpoint_checkpoint``) in one absolute directory for all; an FA rate (the control
arm) of a numerical campaign needs its case on the candidate's first numerical seed, which holds
the rate's same-path control (amendment 2). ``plan`` lists the cases, the expected GPU time
(``MINUTES``: a launcher run ≈ 7 min, a baseline part ≈ 7 min, a rest part ≈ 25 min, a
same-path control ≈ 11 min), the disk need and the host memory, and exits 1 on a campaign
outside this.

**A case, in order** (one at a time, on one Pod; the driver):

1. ``complete`` exits 0 when the case is done (below): then nothing of it runs again; 3 when it
   is not. ``complete`` and ``decided`` say no with exit 3 alone: any other non-zero exit is a
   failure, and stops the driver. First, both check the case's recorded parts (the latest
   attempt of each) against this checkout's GPU code (``code_identity.gpu_code_identity``) and
   the tracked settings: a record of other code or settings makes the case ``stale``, exit 10
   (``STALE``), and the driver stops there: a stale case is never rerun by it, and never skipped
   (its re-qualification is a decision of its own, "Voiding and supersession"). Then, before the
   launcher can be reached: a launcher run of the case's label (its experiment and
   ``<cell>/s<seed>``) held in any launch journal, whatever its config hash (``launch.hold``: an
   outcome the engineering cycle could not resolve), exits 11 (``HELD``) naming the run and the
   hold's reason, and the driver stops there; nothing here lifts a hold. For the case that
   holds its rate's same-path control, ``control-only`` follows ``complete``'s no (below, "A
   case whose control alone is outstanding"): its exit 0 prints the launcher run's directory,
   and the driver runs ``part --part control --resume`` on it alone, then step 7; its exit 3 is
   the usual path, steps 2 to 7; any other exit stops the driver.
2. ``clear-stale`` removes ``<dir>/<run_id>.pt`` (and its partial file) of every attempt of the
   case's launcher run that the launch journal shows superseded (``superseded_run_ids``: ended
   technical, or not ended, without a settled directory); the checkpoint of a settled run (done,
   diverged, or completed but in conflict, kept for the diagnosis) stays.
3. The free disk of the checkpoint directory against the need (``disk_need_gb``: two endpoints,
   the launcher's and the harness baseline's, ``checkpoint_gb`` each, one state per diagnostic
   step, one checkpoint of headroom for an atomic save, and ``disk_margin_gb``); below it the
   driver exits 1.
4. ``python scripts/launch.py CAMPAIGN --only LABEL --max-attempts 1``: the launcher run alone,
   one attempt; a failed one stops the driver, and is attempted again only through a recovery
   round of its own, by hand: ``--recovery REASON`` on that command with its documented incident
   (``--incident-time``, ``--cause``, ``--evidence``), one attempt per round, each round with a
   reason no earlier round of the label gave (``launch``, "Recovery rounds"). The driver never
   passes ``--recovery``.
5. ``locate`` prints the launcher run's directory (its settled run: done, or diverged).
6. ``part --part baseline --resume``; then, unless ``decided`` (exit 0: the baseline record is
   already a scientific failure, a declared instability rule fired on the unshifted baseline or
   the launcher run was one; ``qualification.case_verdict`` of the baseline alone), ``part
   --part rest --resume``: a decided case is not measured further, its rate fails whatever the
   rest shows. For an FA rate on the first numerical seed, ``part --part control --resume``
   either way (the control reads no checkpoint).
7. ``finish`` deletes the case's checkpoints (every ``<dir>/<run_id>.pt`` of its launcher run's
   attempts and the harness's directory ``<dir>/harness/<case>/``), only once its records are
   published and valid; it exits 1 and keeps them for a case that is not complete. A case whose
   verdict is technical (its own, or its same-path control's: ``technical_reasons``) keeps them
   for the diagnosis, "Failure routes", and ``finish --kept DIR`` exports them to the Pod's
   volume (below, "A technical case's kept checkpoints"): it exits 9 (``TECHNICAL_CASE``) once
   every file still present has a verified copy there, 13 (``NOT_EXPORTED``) when the export was
   refused or failed (no verified volume copy: the checkpoints are still only on the container
   disk, which a stop loses). On 9 the driver uploads the export to the bucket
   (``scripts/archive.py``) and, whatever the upload's exit, checks both copies (``kept-check``),
   and stops there, before the next case: with 9 when both exit 0, with 13 otherwise (the
   receipt's ``pending-upload`` true: the volume copy verified, the upload pending; false: the
   volume copy not verified either, the worse state). ``finish --release`` deletes them once the
   diagnosis is done, only through that verified export.

A case is **complete** when the latest attempt of each of its parts is recorded, is that part's
of that case, and the verdicts read it without error (``case_verdict``, and ``control_verdict``
for its control): the baseline, the due control and, unless decided, the rest, which counts only
when it names the latest baseline record's sha256 (``baseline_record``; a rest that read no
baseline, having run nothing or failed before, excepted). Records are published atomically and
durably (``qualification.publish``).

**A technical case's kept checkpoints** ("Failure routes": a technical case keeps its
checkpoints for the diagnosis). The Pod's container disk goes with a Pod stop, so ``finish
--kept DIR`` (the driver's ``--kept``: on a Pod ``/workspace/kept-checkpoints``, the volume)
copies the case's files, exactly those ``finish --release`` would delete (``case_files``), to
``DIR/qualification/c<N>/<stage>/<case>/kept-<n>/`` in the layout they have under the checkpoint
directory: each written under a partial name, fsynced, renamed and its folder fsynced, the source
hashed while read and the copy read back, verified when the two sha256 agree. A present
checkpoint whose sha256 is not its recorded one is exported as it is (``matches_record`` false):
the export changes neither the technical mark nor any verdict. Refused before any copy, exit 13
and nothing called verified: no ``--kept``, a relative DIR, DIR under the runs root or the
records' folder (its copies would join the small results archived from there) or under the
checkpoint directory, DIR on the checkpoint directory's filesystem (``same_filesystem``:
``st_dev``, the container disk), or less free space under DIR than the files take with a margin
of 1 GiB (``KEPT_MARGIN``); an ``OSError`` anywhere in the export, these checks included (a
folder, a ``stat``, the free space, a copy, a manifest, ``SHA256SUMS``), exits 13 too, naming the
error, a partial ``kept-<n>`` left without a case manifest and never called verified. The manifest,
``kept-checkpoints-<n>.json`` in the case's folder and the same JSON in the export as
``MANIFEST.json`` with a ``SHA256SUMS``, holds the case, its stage, candidate and
``technical_reasons``; the binding records (``binding_records``: the located launcher run's
``summary.json`` and ``meta.json`` by path and sha256 with its ``endpoint_checkpoint`` record, and
the case's latest baseline, rest and control records by path, file sha256 and ``record_sha256``;
nothing else of the run's directory is opened or listed); each file (its source path, its copy's
path relative to DIR, size, sha256, the sha256 the binding records give that path or null, whether
it matches, whether it is a partial file); ``lost``, every checkpoint those records name that is not
on disk and that ``finish`` did not remove on purpose (``released``), with its recorded sha256 and
the records naming it; ``verified``, the host, the Pod, the time and the provenance. Run again
over an export whose copies re-hash to its manifest under DIR, the case's files listed in it as
they are now, nothing is copied (exit 9); otherwise ``kept-<n+1>`` is written, and nothing is ever
overwritten. Checkpoints lost before the export leave the technical mark as it stands: ``finish``
exits 9 when every file still present is verified, ``lost`` names what was lost, and it says
"lost before the export: the technical mark stands; the diagnosis goes on from the records;
nothing reruns" (``restart-case`` refuses a complete case). ``kept-check --inventory FILE
--remote PREFIX`` (the driver's: the REMOTE_PREFIX it gives ``archive.py``) reads the case's
latest manifest, re-hashes every copy under its DIR (with ``MANIFEST.json`` and ``SHA256SUMS``),
and reads ``scripts/archive.py``'s inventory: every line's URL lies under ``PREFIX/``, and every
exported file needs its line (path relative to DIR, size, sha256, md5, URL, status) with the
exported size and sha256, the status ``ok`` (the object's size and MD5 are the file's) and the URL
``PREFIX/<path>``. It exits 0 when all hold, 13 naming what fails (each line under another
prefix included), and publishes its receipt ``kept-check-<n>.json`` in the case's folder: the
manifest's and the inventory's path and sha256, the prefix, the exit, what fails, and
``pending-upload``
true for a volume copy verified whose bucket copy is not (no inventory, or one without the
export's verified lines). ``finish --release`` of a technical case deletes only through the
manifest of a receipt of exit 0 that names its sha256 (``bound_export``), the inventory that
receipt names still of the receipt's sha256, when every current file is listed in the manifest
with its size and sha256, each file checked again right before its deletion; otherwise it exits
13 naming the files or the inventory (a file new or changed since the export needs another
export and its ``kept-check``; an inventory gone or changed, the upload checked again). Its
``checkpoints-removed-<n>.json`` names that manifest and that receipt by path and sha256. A kept
checkpoint is not one ``finish`` removed (``released``): its absence is never evidence of an
interruption. The planned teardown of the Pod waits for ``kept-check`` exit 0; an upload that
cannot be verified leaves the volume copy kept and the receipt ``pending-upload``, and never
lifts the hard deadline.

**A case whose control alone is outstanding** (``control-only``; "Failure routes", amendment 2):
a case whose only outstanding part is its same-path control, decided by its baseline or with its
baseline and rest recorded. A Pod stop then loses the container disk, the launcher run's
checkpoint with it: the launcher refuses that run (its checkpoint is missing), and
``restart-case`` refuses a case with a scientific failure (it keeps counting) or voids passing
parts, although the control reads no checkpoint. ``control-only`` first answers as
``complete`` and ``decided`` (a journal that is not whole, exit 12; stale, 10; held, 11), then
exits 0, printing the launcher run's directory and nothing else, when the case holds its rate's
same-path control, the engineering cycle has not resolved it (``resolution_of``), it is not
complete (``case_state``), the control's latest attempt is due or interrupted, every other part
the case needs is recorded as ``complete`` reads it (``case_state`` of the case without its
control: the latest baseline record, recorded and not void, and, unless the baseline decided
the case, a rest bound to it; the verdicts read them), the case is not technical
(``technical_reasons`` of those parts: a scientific failure, the baseline's or the rest's, keeps
counting, and its control runs alone), and its recorded parts bind it to its launcher run
without the launcher (``control_binding``): the baseline record classified the run as
conforming (``launcher.classified`` true and ``launcher.kind`` null), and each record, the
baseline's and the rest's, names (``launcher_run``) the case's located run (``located``; by its
experiment and run id, wherever RUNS_DIR lay), superseded nowhere, with its identity, which it
must name (``q.launcher_identity``: the recorded checkpoint record); the run is done in the
launch journal (its end, settled or not), no ``held`` line names its label, and the run's
endpoint checkpoint, read only when present and never required, is not on disk with another
sha256 than its recorded one (``checkpoint_conflict``, as ``restart-case`` reads it: a
provenance conflict, which stops the step and is kept for the diagnosis). The driver then runs
``part --part control --resume`` on that run, with every check of the control (``run_part``: the
launcher run against the case and this process's numerics, its streams and v₀), and neither
``clear-stale``, the disk check, the launcher, ``locate``, the baseline, ``decided`` nor the
rest; ``finish`` follows. The checks go in this order. Exit 3 (``NO``), the usual path: a case
whose control is not outstanding or with another part outstanding; then a case its own verdict
marks technical, whatever its bindings, before any binding check, so that its ``finish``
exports its checkpoints; then a case decided by its launcher run's own declared instability
stop (a run the launcher accepts without its checkpoint). Exit 1, naming the condition: a
baseline record not classified or of another non-null kind, then any binding that fails; the
driver stops there, and nothing is rerun. Otherwise exit 0. The control's attempts are a part's
(below): ``--resume`` runs attempt 1, or exits 8 for an interrupted one, whose rerun is by hand
(``--attempt N --infrastructure REASON``) whatever happened to the container disk, as the
control reads no checkpoint; a recorded control, of any kind, is never rerun. A decided case is
never restarted: its control runs alone.

**Attempts of a part** (``part``; "Failure routes": an infrastructure interruption is rerun under a
new attempt id, a technical or scientific record never). Before ``run_part`` the part publishes
``<record>.started``; an attempt that has it and no record was interrupted. ``--resume`` (the
driver's call) runs attempt 1 of a part without any attempt, does nothing for a part whose latest
attempt is recorded (whatever its kind), and exits 8 for one whose latest attempt was interrupted.
``--attempt N`` runs attempt 1 of a part without any attempt, or the attempt after an interrupted
one, which needs ``--infrastructure REASON``, the documented interruption, and its record carries it
from its one publish (``infrastructure``: the reason, the earlier attempts and the case's restart
records); a part whose latest attempt is recorded, whatever its kind, is refused. ``interrupted
--technical REASON`` records an interrupted attempt, unexplained, as a technical part record
carrying the reason ("an unexplained incomplete record is technical"); it refuses an attempt with a
record. A rest attempt publishes each trajectory's result as soon as it is done (``qualification``,
"Trajectory results"): when its attempt is interrupted, the verdict reads them as its record
(``rest_from_trajectories``), ``interrupted`` records them in the part's record, and ``part
--attempt N`` refuses to rerun an attempt whose results hold a failure, scientific or technical
(they keep counting). The part needs a clean checkout of a commit, sets the numerics up
(``runtime.configure_numerics``) before any CUDA work and builds the environment from the run's
campaign config; the survival retry and the float64 fallback stay inside the part.

**A case restart** (``restart-case --infrastructure REASON``; "Voiding and supersession": an
infrastructure-interrupted attempt is void and kept, its rerun a new attempt id). A Pod stop during
a case loses the container disk, with the launcher run's checkpoint and the harness's: the case
cannot go on from its recorded parts. A missing endpoint checkpoint alone is a possible loss
requiring diagnosis, never by itself a lost container disk: the restart's reason documents the
interruption the diagnosis found. Exit 1 is shared (the
launcher's conflicts, this command's refusals, ``control-only``, ``finish``, the driver's
preconditions), so it never triggers a restart by itself: docs/operations.md, "A lost checkpoint",
tells the launcher's messages apart. ``restart-case`` publishes (atomically, from a clean checkout)
``restart-<n>.json`` in the case's folder, naming every attempt of the case's parts (the baseline,
the rest and, for the case that holds it, the same-path control): each recorded one with its
record's sha256, status and kind, each interrupted one by its number; the reason; and the case's
launcher run. Then, the record published, it logs that run superseded in the launch journal
(``launch.supersede``): the run is kept, no longer counts for its label (``assess``, ``located``),
and a recovery round of the run's own with the same reason (its first, or a further one at
``--max-attempts 1``, one attempt per documented incident: ``launch``, "Recovery rounds") starts a
new run; run again once every attempt is named, the command only completes that journal line. A
second loss in the same case is restarted the same way: ``restart-<n+1>.json`` names every attempt,
the earlier restarts' included, and the case's current launcher run is logged superseded; then the
launcher's further round with the new reason, and each part as its next attempt. A launcher run in
flight at the loss (a start without an end, no settled directory) is no attempt to restart: the
launcher settles it technical, and a round of its own with the documented reason is the whole route.
Only an interrupted case is restarted: it is refused without a reason, when every attempt is named
already, for a complete case, without evidence of an interruption (an interrupted attempt, or the
launcher run's checkpoint gone from disk, unless ``finish`` removed it on purpose: its
``checkpoints-removed-<n>.json`` records), when the launcher run itself is a completed failure (its
declared instability stop, scientific, or another stop or a departing run, technical:
``launcher_problems`` without the comparison with this process's environment), when any attempt is a
completed failure (``attempt_failures``): a scientific one, which keeps counting ("no reruns to
pass"), a technical one (the engineering cycle lifts it: below), or an unreadable record; an
interrupted rest attempt counts by its trajectory results; when the launcher run's journal holds
an end ``conflict`` that no lost container disk explains (``own_conflict_end``: an unsettled one,
the launcher's own judgement of its attempt, or one whose problem is not the missing checkpoint);
and when the launcher run's endpoint checkpoint is on disk with another sha256 than its recorded one
(``checkpoint_conflict``): a provenance conflict, not a loss, which goes to the technical route even
when an interrupted attempt would otherwise allow the restart. A recorded attempt that passes so far
(the baseline, a control) depends on the lost run or checkpoint and is named with the others. The
attempts it names are void: a part whose recorded attempts are all named is ``voided``, so the case
is not complete; ``part --resume`` exits 8 for it, and ``part --attempt N`` takes its next attempt
with ``--infrastructure REASON`` (its record names the restart records); a rest waits for the
baseline's new attempt. The named records are kept, listed, and voided in the verdict (with the
reason, under ``restart``).

**Supersession** (``supersede --record PATH --diagnosis TEXT --fix TEXT``; "Failure routes": one
engineering cycle per candidate; "Voiding and supersession"). A completed technical failure stops
counting only through the candidate's supersession record, ``c<N>/supersede-1.json``: it names the
technical record that started the cycle (its path and sha256), the diagnosis and the fix, the GPU
code of the fix (this clean checkout's) and the old one, every record of the candidate made under
the old GPU code in all its stage folders, passed ones included, with the cases re-qualified, and
every launcher run of the candidate in the launch journal not of the fix's GPU code (the runs its
records name and those of the campaign's cases, a case with no part attempt yet included, found by
the case's label, its experiment and ``<cell>/s<seed>``, in every journal whatever their config
hash, so that a fix that changes a hashed config key hides none; another or an unknown code
included, since a re-qualification under the fix could not use it), which it then logs superseded; a
case whose run is superseded is re-qualified, whether or not it had a part record. A case that holds
a scientific failure, as its bound, valid part records read with the verdicts show it (another
stage's included; a launcher run's declared stop counts only when such a record classified it
scientific), is kept whole: none of its records (its parts, its control, its trajectory results) nor
its launcher run is voided, it is listed as kept with why, and the driver's stale check passes it;
the failure keeps counting (no reruns to pass). A record classified its launcher run only when it
says so (``launcher.classified``, true once ``launcher_problems`` returned): naming the run is not
enough. A launcher run no part record classified (none names it: the Pod stopped before its
baseline, say; or each that does raised before its check returned, in the numerics setup or the
check itself; a campaign case's or another stage's, as its records name the case) is first resolved
against what it was issued and its own provenance, never this checkout's configs, code or
environment (``resolve_launcher_run``): its journal start (label, config hash, header, command), its
saved config and meta.json (config hash, the campaign's header, commit, numerics setup) and the
configs of its own commit, read from git (``configs_at``), from which its command and the case's
confirmation config are composed; its identity, config agreement and numerics are checked against
those. Conforming and stopped by its declared instability rule, a scientific failure (the case is
kept whole and decided, and no part runs for it); departing from what it was issued on readable
records, or completed, superseded as any run; unresolvable (records unreadable, a needed field
missing, no commit or one not in this clone, configs that do not compose or do not give the start's
config hash), the case is kept as it is, listed unresolved, and the run held in the journal
(``launch.hold``), so that neither a part nor the launcher's recovery round reruns it. The verdict
counts a resolved outcome as the case's (``resolved_verdict``), beside the verdicts of any records
that did not classify the run. The cycle is refused a second time for the candidate (a technical
failure left unresolved after it routes to the technical feasibility report), for a record that is
not a completed technical failure of the case or that belongs to a case the cycle keeps whole (by
its records or its launcher run), and when the fix's GPU code is the old one (the same path is not
simply rerun); run again with the same record, it only writes the journal's lines that are missing.
Each case it names re-qualifies under the new code from a new launcher run and new attempts of its
parts. Under the old config hash the new run comes from a recovery round of the run's own (its
first, or a further one at ``--max-attempts 1``, its incident the cycle, its evidence the
supersession record); under a new config hash (the fix changed a hashed key) it is the new run's
base attempt, without ``--recovery``, which a study-3 campaign refuses over a run that is not used
up, and the supersession record carries the cycle. Of the parts, the attempt after one the cycle
named carries ``--engineering-cycle`` (its record: ``engineering_cycle``, the supersession's path
and sha256) instead of ``--infrastructure``; a later interruption of a new-code attempt is rerun
with ``--infrastructure REASON``, as any other. The verdict lists the records it voids under
``supersession``.

``verdict`` writes RUNS_DIR/qualification/c{N}/verdict.json (with ``--engineering``,
``c{N}/engineering/verdict.json`` with ``engineering_summary``): per rate of the campaign its
``rate_verdict`` under this checkout's GPU code and the tracked settings, with its cases' and its
same-path control's verdicts (states, kinds, reasons, the 13 comparisons, warnings with
``--e-eng``), every case record of the candidate on disk shown (one of no campaign rate under
``unassigned``), every attempt of every part (``attempts``: recorded or interrupted, with each
record's sha256, status, kind and the reason of the restart that names it, as ``restart``, or of the
supersession, as ``supersession``), and prints the states. A record a case restart or supersession
names is listed under ``superseded`` with that reason and passed to ``rate_verdict`` as voided (its
record sha256 → the reason): it neither counts nor blocks. A rest whose latest attempt was
interrupted is read from its trajectory results. A record under a later attempt that no restart
names (which these commands never write) is not hidden by it: it is listed under ``superseded`` and
counted against its rate as a case of its own (a control's as a superseded control of its rate). The
launcher runs located now (one settled directory per run) are the expected ones; a rate one of whose
launcher runs is not located is at most ``incomplete``, with the reason and a warning.

**A launch journal that is not whole** (``launch``, "The journal"). Every command but ``plan``
and ``driver``, which read no runs root, first reads ``launch.torn_lines`` and
``launch.missing_lines``: while a launch log holds a torn line that no ``torn_resolved`` record
names, or a published cycle or restart record under RUNS_DIR/qualification/ requires a
``held`` or ``superseded`` line that no log holds (a crash between the record and its lines,
or an append that wrote no byte; an unreadable record blocks too), it prints the launcher's
refusal (each torn line's log and line, offset, length, kind and sha256; each missing line's
record path, event and run id), acts on nothing and exits 12 (``launch.TORN_JOURNAL``). A torn
line is any line of a launch log that holds no record, a JSON object whose identity is not all
strings included (``launch.well_formed``). Two calls, decided before anything acts, are journal
repairs and go on: ``supersede`` run again with its published
record (``cycle_to_repair``: it writes only the journal's missing ``held`` and ``superseded``
lines) and ``restart-case`` for a case whose published restart records lack their
``superseded`` lines (``restart_lines_to_repair``: it writes only those lines, each with its
record's reason, and exits 1 naming the restart records). Each then prints what it leaves: a
torn line's resolution (``scripts/launch.py --resolve-torn``), or another record's repair, is
next. Any other call of the two, a new cycle or a new restart, is refused as the rest.

``driver`` writes the bash driver for a Pod from the source checkout of SHA (``--kept DIR``, an
absolute folder outside the checkpoint directory, is each case's ``finish --kept``; the export goes
to ``$ARTIFACT_REMOTE/<--kept-remote>``, by default ``kept-checkpoints``): it checks the checkout
and the campaign file's sha256, that nothing else uses the GPU, the free host memory
(``host_memory_gb`` of the settings, ``drift.host_memory``: θ₀, v₀, the second model and the p1
direction on the host and a checkpoint save's transient), and that a technical case's kept
checkpoints can reach the bucket (``ARTIFACT_REMOTE`` a gs:// path, ``ARTIFACT_GCS_KEY`` a readable
file, ``gcloud`` on PATH), then runs each case in order as above, and
writes the verdict on every exit, but one: over a launch journal that is not whole (an
unresolved torn line, or a line a published record requires missing) ``verdict`` refuses too
(exit 12), so no verdict.json is written then and an older one stays on disk.
Exits: 0 when every case is complete; 1 a precondition (the host
memory and the bucket's settings included), the disk (``df``'s answer must be a number), a
``finish`` that refuses a case not complete, or a case whose control alone is outstanding that
``control-only`` cannot bind to its launcher run; 8 an interrupted attempt (or a part a case
restart or supersession names); 9 a technical case, its checkpoints kept on the volume and in
the bucket, verified; 10 a stale case; 11 a case whose label has a held launcher run; 12 a torn
line in a launch journal, or a line a published record requires missing from it (``complete``'s
refusal, or the launcher's); 13 a technical case whose kept checkpoints are not verified: from
``finish``, its export refused or failed (no verified volume copy, the checkpoints still only on
the container disk, which a stop loses), or after ``kept-check`` (run whatever ``archive.py``'s
exit), its receipt's ``pending-upload`` true (the volume copy verified, the upload pending) or
false (the volume copy not verified either, the worse state); either way the planned teardown
waits for ``kept-check`` exit 0, and the hard deadline stays; the launcher's own when it stops;
any other, the failed command's. Before the first case the driver also exits 1 when ``$KEPT`` is
on the checkpoint directory's device (``stat -L -c %d``) or lies, resolved, under ``$RUNS_DIR``
or the checkpoint directory.
The driver never reruns a part or the launcher run: a rerun is a command of its own, with its
reason."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, NoReturn

from omegaconf import DictConfig, OmegaConf

from zero_field_refusal import code_identity, runtime
from zero_field_refusal import fa_control_diagnostic as fcd
from zero_field_refusal import qualification as q
from zero_field_refusal.analysis.qualification_verdicts import REST_MISSING
from zero_field_refusal.config import config_hash, load_config
from zero_field_refusal.launch import (
    TORN_JOURNAL,
    Run,
    TornJournalError,
    assess,
    attempt_key,
    expected_numerics,
    held_runs,
    hold,
    is_missing_checkpoint,
    journal_records,
    label_starts,
    launch_records,
    load_campaign,
    missing_lines,
    supersede,
    superseded_runs,
    torn_lines,
)
from zero_field_refusal.paths import CONFIG_ROOT, REPO_ROOT, load_env, runs_root
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.schedule import INSTABILITY
from zero_field_refusal.tensor_hash import file_sha256 as checkpoint_sha256
from zero_field_refusal.tracking import CLEAN_CHECKOUT, git_state

SETTINGS = CONFIG_ROOT / "experiments" / "qualification.yaml"  # the tracked settings, by path
RUN = 'project-run "$SRC" python scripts/qualify.py'
# A case restart or an engineering cycle's supersession record, in the case's folder
VOIDING_FILE = re.compile(r"(restart|supersede)-(\d+)\.json")
LAUNCH = 'project-run "$SRC" python scripts/launch.py'
NO = 3  # ``complete``, ``decided``, ``control-only``: no (any other non-zero exit fails)
INTERRUPTED = 8  # ``part --resume``: the latest attempt was started and left no record
TECHNICAL_CASE = 9  # ``finish``: the case's verdict is technical; its checkpoints are kept
STALE = 10  # ``complete``, ``decided``, ``control-only``: records of other code or settings
HELD = 11  # ``complete``, ``decided``, ``control-only``: the label has a held launcher run
# A technical case's checkpoints lack a verified export (the launcher's exits are 0, 1, 2 and 12):
# ``finish``'s, its export refused or failed (no verified volume copy); ``kept-check``'s and the
# driver's after it, the upload pending (``pending-upload`` true) or the volume copy not verified
# either (false)
NOT_EXPORTED = 13
# The Pod's environment and the source checkouts (the driver's preamble)
POD_ENVIRONMENT = (". /etc/rp_environment", ". /root/zfr-env.sh")
SOURCES = "/workspace/code/zero-field-refusal"
GPU_USED_MIB = 1000  # the driver starts only below this much GPU memory in use
# Whether the free host memory (``drift.host_memory``: MemAvailable, or the cgroup's free memory
# when smaller) reaches the settings' ``host_memory_gb`` (exit 0); run by the driver through
# project-run. No $, backtick, backslash or double quote inside.
HOST_FREE = ("import sys; from zero_field_refusal.drift import host_memory; m = host_memory(); "
             "print('free host memory (GB):', m); "
             "sys.exit(0 if (m['available_gb'] or 0) >= float(sys.argv[1]) else 1)")  # fmt: skip
# The expected GPU minutes on one L40S at 1.5B: a launcher run (≈ 7.1 min, "Budget and timing"),
# the baseline part (one trajectory with its setup), the rest (six) and a same-path control
MINUTES = {"launcher": 7, "baseline": 7, "rest": 25, "control": 11}
ROLES = {False: "numerical", True: "engineering"}


def comment(text: str) -> list[str]:
    return textwrap.wrap(text, 100, initial_indent="# ", subsequent_indent="# ",
                         break_long_words=False, break_on_hyphens=False)  # fmt: skip


def load_settings():
    return load_config(SETTINGS)


@dataclass(frozen=True)
class Planned:
    """A case of the campaign: its launcher run, the directory of that run's endpoint
    checkpoint, and whether it holds its rate's same-path control."""

    run: Run
    case: q.QualificationCase
    endpoint: Path
    control: bool

    @property
    def harness(self) -> Path:
        """The harness's checkpoints of the case (``run_part``'s ``checkpoints``)."""
        return self.endpoint / "harness" / self.case.name.replace("/", "-")


def campaign_cases(
    path: Path, scope: q.QualificationScope, candidate: int, engineering: bool
) -> tuple[str, list[Planned], list[str]]:
    """The campaign's name, its cases in order and why it is not a qualification campaign of the
    candidate (module docstring, "The cases"; empty when it is)."""
    name, runs = load_campaign(path)
    problems: list[str] = []
    try:
        seeds = scope.engineering_seeds if engineering else scope.seeds(candidate)
        first = None if engineering else scope.seeds(candidate)[0]
    except q.QualificationError as exc:
        return name, [], [str(exc)]
    planned, directories = [], set()
    for run in runs:
        cfg = load_run_config(run.config, run.overrides)
        case = q.QualificationCase(int(cfg.seed), str(cfg.optimizer.name), float(cfg.optimizer.lr),
                                   candidate, engineering, batch=int(cfg.train.batch_size),
                                   steps=int(cfg.train.steps))  # fmt: skip
        where = OmegaConf.select(cfg, "tracking.endpoint_checkpoint", default=None)
        role = OmegaConf.select(cfg, "study3.role", default=None)
        mine = []
        if run.entry != "train" or OmegaConf.select(cfg, "eval.k_only", default=None) is not True:
            mine.append("not a K-only training run (eval.k_only)")
        if role != ROLES[engineering]:
            mine.append(f"study3.role is {role!r}, not {ROLES[engineering]!r}")
        if (case.batch, case.steps) != (q.BATCH, q.STEPS):
            mine.append(f"B {case.batch} and {case.steps} steps, not B {q.BATCH} and {q.STEPS}")
        if case.seed not in seeds:
            mine.append(f"seed {case.seed} is not among the {ROLES[engineering]} seeds "
                        f"{list(seeds)}")  # fmt: skip
        if not (isinstance(where, str) and Path(where).is_absolute()):
            mine.append(f"tracking.endpoint_checkpoint is {where!r}, not an absolute directory")
        else:
            directories.add(Path(where))
        problems += [f"{run.label}: {p}" for p in mine]
        control = case.optimizer == scope.control_arm and case.seed == first
        planned.append(Planned(run, case, Path(where or "."), control))
    if len(directories) > 1:
        problems.append(f"the endpoint checkpoints go to several directories: "
                        f"{sorted(map(str, directories))}")  # fmt: skip
    names = [p.case.name for p in planned]
    problems += [f"two runs are case {n}" for n in sorted({n for n in names if names.count(n) > 1})]
    if not engineering:
        held = {p.case.lr for p in planned if p.control}
        for lr in sorted({p.case.lr for p in planned if p.case.optimizer == scope.control_arm}):
            if lr not in held:
                problems.append(f"{scope.control_arm} at {lr:g} has no case on seed {first}, "
                                "which holds its same-path control")  # fmt: skip
    return name, planned, problems


def disk_need_gb(settings, diagnostic_steps) -> float:
    """The free space a case needs (module docstring, step 3), in GB of 1e9 bytes."""
    checkpoints = 2 + len(list(diagnostic_steps)) + 1
    return checkpoints * float(settings.checkpoint_gb) + float(settings.disk_margin_gb)


def need_kb(gb: float) -> int:
    """``gb`` (1e9 bytes) in the 1024-byte blocks of ``df -Pk``, rounded up."""
    return math.ceil(gb * 1e9 / 1024)


def minutes(planned: list[Planned]) -> int:
    per_case = MINUTES["launcher"] + MINUTES["baseline"] + MINUTES["rest"]
    return per_case * len(planned) + MINUTES["control"] * sum(p.control for p in planned)


def parse_steps(text: str | None) -> list[int]:
    return sorted({int(t) for t in text.split(",")}) if text else []


# ------------------------------------------------------------------ the records of a case


@dataclass(frozen=True)
class PartState:
    """A part's latest attempt on disk: ``due`` (none), ``recorded`` (with its record),
    ``interrupted`` (started, no record), ``voided`` (every recorded attempt named by the case's
    restart or supersession records) or ``invalid`` (a record that is unreadable, or not that
    part's of that case). ``attempts`` maps every attempt to recorded or interrupted."""

    state: str
    attempt: int
    record: dict | None
    attempts: dict[int, str]


def attempt_files(folder: Path, part: str) -> dict[int, str]:
    """Every attempt of a part in a case's folder: recorded (its record exists) or interrupted
    (only its ``.started`` file does), by attempt number."""
    pattern = re.compile(rf"{re.escape(part)}-attempt-(\d+)\.json(\.started)?")
    seen: dict[int, set[str]] = {}
    for file in folder.glob(f"{part}-attempt-*") if folder.is_dir() else []:
        match = pattern.fullmatch(file.name)
        if match:
            seen.setdefault(int(match[1]), set()).add("started" if match[2] else "record")
    return {n: "recorded" if "record" in kinds else "interrupted"
            for n, kinds in sorted(seen.items())}  # fmt: skip


def case_folder(out: Path, case: q.QualificationCase) -> Path:
    """The folder of a case's attempt records and restart records."""
    return (out / case.attempt_path("baseline", 1)).parent


def file_sha256(path: Path) -> str | None:
    """The sha256 of a record file's bytes (``record_sha256`` of a published record), or None."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def case_name_of(folder: Path) -> str:
    """The case a folder holds (``s<seed>/<arm>-<rate>``)."""
    return f"{folder.parent.name}/{folder.name}"


def stage_folder(folder: Path) -> Path:
    """The candidate's stage folder (``c<N>/<stage>/``) above a case's folder."""
    return folder.parent.parent


def candidate_folder(folder: Path) -> Path:
    """The candidate's folder (``c<N>/``) above a case's folder: it holds the supersession."""
    return folder.parent.parent.parent


def voiding_records(folder: Path, kind: str | None = None) -> list[tuple[Path, dict | None]]:
    """The records that void attempts of the case in ``folder``: its restart records
    (``restart-<n>.json``, in the folder) and the candidate's supersession record
    (``supersede-<n>.json``, in the candidate's folder, ``c<N>/``); ``kind``: one of them alone.
    In order, each with its record."""
    found = []
    for where, wanted in ((folder, "restart"), (candidate_folder(folder), "supersede")):
        if kind not in (None, wanted) or not where.is_dir():
            continue
        found += [(wanted, int(m[2]), f) for f in where.glob(f"{wanted}-*.json")
                  if (m := VOIDING_FILE.fullmatch(f.name))]  # fmt: skip
    return [(f, _read(f)) for _, _, f in sorted(found)]


def restart_records(folder: Path) -> list[tuple[Path, dict | None]]:
    return voiding_records(folder, "restart")


def void_reason(path: Path, record: dict | None) -> str:
    """Why the attempts a voiding record names are void: a restart's reason, or the
    supersession's diagnosis and fix."""
    record = record or {}
    if path.name.startswith("supersede-"):
        return (f"superseded by an engineering cycle ({path.name}): {record.get('diagnosis')}; "
                f"fix: {record.get('fix')}")  # fmt: skip
    return str(record.get("reason"))


def _entries(folder: Path, kind: str | None = None):
    """(voiding record path, its record, an attempt entry of this case) for every attempt the
    voiding records name (a supersession's entries name their case)."""
    case, stage = case_name_of(folder), stage_folder(folder).name
    for path, record in voiding_records(folder, kind):
        for entry in (record or {}).get("attempts") or []:
            if (isinstance(entry, dict) and entry.get("case", case) == case
                    and entry.get("stage", stage) == stage):  # fmt: skip
                yield path, record, entry


def named_records(folder: Path, kind: str | None = None) -> dict[str, str]:
    """The attempt records the case's restart and supersession records name: each record
    file's sha256 → why it is void (module docstring, "A case restart", "Supersession")."""
    named: dict[str, str] = {}
    for path, record, entry in _entries(folder, kind):
        if isinstance(entry.get("sha256"), str):
            named.setdefault(entry["sha256"], void_reason(path, record))
    return named


def named_attempts(folder: Path, kind: str | None = None) -> set[tuple[str, int]]:
    """Every (part, attempt) a restart or supersession record names, recorded or interrupted."""
    return {(e.get("part"), e.get("attempt")) for _, _, e in _entries(folder, kind)}


def cycle_of(folder: Path, part: str, attempt: int) -> tuple[Path, dict | None] | None:
    """The supersession record that names this attempt of this part of the case (its latest),
    or None: the rerun after it carries ``--engineering-cycle``; a later interruption of a
    new-code attempt is rerun with ``--infrastructure`` as any other."""
    for path, record, entry in _entries(folder, "supersede"):
        if entry.get("part") == part and entry.get("attempt") == attempt:
            return path, record
    return None


def trajectory_results(folder: Path, attempt: int) -> list[dict]:
    """The trajectory results a rest attempt published (``qualification``, "Trajectory
    results")."""
    found = [_read(f) for f in sorted(folder.glob(f"rest-attempt-{attempt}.trajectory-*.json"))]
    return [r for r in found if r is not None]


def interrupted_rest(folder: Path, attempt: int) -> dict | None:
    """An interrupted rest attempt as a record, from its trajectory results
    (``rest_from_trajectories``), or None without any."""
    return q.rest_from_trajectories(trajectory_results(folder, attempt))


def part_state(out: Path, case: q.QualificationCase, part: str) -> PartState:
    folder = case_folder(out, case)
    attempts = attempt_files(folder, part)
    if not attempts:
        return PartState("due", 1, None, attempts)
    latest = max(attempts)
    if attempts[latest] == "interrupted":
        return PartState("interrupted", latest, None, attempts)
    named = named_records(folder)
    if all(file_sha256(folder / f"{part}-attempt-{n}.json") in named
           for n, s in attempts.items() if s == "recorded"):  # fmt: skip
        return PartState("voided", latest, None, attempts)
    record = _read(out / case.attempt_path(part, latest))
    if record is None or record.get("part") != part or record.get("case") != case.record():
        return PartState("invalid", latest, None, attempts)
    return PartState("recorded", latest, record, attempts)


def _read(path: Path) -> dict | None:
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def case_state(out: Path, p: Planned, scope: q.QualificationScope) -> tuple[bool, bool, str]:
    """Whether the case is complete, whether its baseline decided it (scientific), and why
    (module docstring, "complete"). A case the engineering cycle kept on its launcher run's
    resolved declared stop is complete and decided; one it kept unresolved is not complete."""
    resolution = resolution_of(out, p.case)
    if resolution is not None:
        if resolution["outcome"] == q.SCIENTIFIC:
            return (
                True,
                True,
                (
                    "decided by its launcher run's declared stop, resolved in the "
                    f"engineering cycle: {'; '.join(resolution['reasons'])}"
                ),
            )
        return False, False, ("its launcher run's outcome is unresolved (the engineering "
                              f"cycle): {'; '.join(resolution['reasons'])}")  # fmt: skip
    baseline = part_state(out, p.case, "baseline")
    if baseline.state != "recorded":
        return False, False, f"baseline: {baseline.state} (attempt {baseline.attempt})"
    try:
        decided = q.case_verdict(baseline.record, None, scope)["kind"] == q.SCIENTIFIC
    except Exception as exc:  # a record the verdict cannot read is not a valid one
        return False, False, f"the baseline record cannot be read: {type(exc).__name__}: {exc}"
    needed = ([] if decided else ["rest"]) + (["control"] if p.control else [])
    states = {part: part_state(out, p.case, part) for part in needed}
    for part, state in states.items():
        if state.state != "recorded":
            return False, decided, f"{part}: {state.state} (attempt {state.attempt})"
    rest = states["rest"].record if "rest" in states else None
    # A rest that read no baseline (it ran nothing, or failed before reading one) binds none:
    # the verdict holds it technical
    read_none = rest is not None and (rest.get("status") == "not_run" or (
        rest.get("status") == "failed" and rest.get("baseline_record") is None))  # fmt: skip
    if (rest is not None and not read_none
            and (rest.get("baseline_record") or {}).get("sha256")
            != q.record_sha256(baseline.record)):  # fmt: skip
        return False, decided, ("the rest record is not bound to the latest baseline record: "
                                "the case is not complete")  # fmt: skip
    try:
        q.case_verdict(baseline.record, rest, scope)
        if p.control:
            q.control_verdict(states["control"].record, scope, p.case.candidate)
    except Exception as exc:
        return False, decided, f"the records cannot be read: {type(exc).__name__}: {exc}"
    return True, decided, "decided by the baseline: a scientific failure" if decided else "complete"


def stale_records(
    out: Path, p: Planned, code: dict[str, Any] | None, policy: dict[str, Any] | None
) -> list[str]:
    """The case's recorded parts (the latest attempt of each) written under other GPU code or
    settings than this checkout's (``code``: ``code_identity.gpu_code_identity``; ``policy``:
    the tracked settings' record): such a case is ``stale`` (module docstring, step 1). A
    record the candidate's engineering cycle kept (a scientific failure) is not."""
    stale = []
    kept = kept_records(case_folder(out, p.case))  # the engineering cycle's: they keep counting
    for part in ["baseline", "rest", *(["control"] if p.control else [])]:
        state = part_state(out, p.case, part)
        if state.state != "recorded" or file_sha256(
                out / p.case.attempt_path(part, state.attempt)) in kept:  # fmt: skip
            continue
        provenance = state.record.get("provenance") or {}
        if code is None or (provenance.get("gpu_code") or {}).get("sha256") != code.get("sha256"):
            stale.append(f"{part} attempt {state.attempt}: other GPU code")
        elif provenance.get("settings") != policy:
            stale.append(f"{part} attempt {state.attempt}: other settings")
    return stale


def technical_reasons(out: Path, p: Planned, scope: q.QualificationScope) -> list[str]:
    """Why a complete case's verdict is technical (its own, or its same-path control's), or
    nothing: such a case keeps its checkpoints for the diagnosis ("Failure routes"). A case
    decided by its launcher run's resolved stop has none."""
    if resolution_of(out, p.case) is not None:
        return []
    baseline = part_state(out, p.case, "baseline").record
    rest = part_state(out, p.case, "rest").record
    verdict = q.case_verdict(baseline, rest, scope)
    reasons = list(verdict["reasons"]) if verdict["kind"] == q.TECHNICAL else []
    if p.control:
        control = q.control_verdict(part_state(out, p.case, "control").record, scope,
                                    p.case.candidate)  # fmt: skip
        if control["kind"] == q.TECHNICAL:
            reasons += [f"control: {r}" for r in control["reasons"]]
    return reasons


# --------------------------------------------------------------- the launcher run's files


def journal_run_ids(run: Run, root: Path) -> list[str]:
    """The run ids of every logged attempt of the run (launch journal, start records), in
    attempt order (``launch.label_starts``: the attempt number, then the start's time, never the
    journal file's name)."""
    return list(dict.fromkeys(r["run_id"] for r in label_starts(run, launch_records(root))))


def label_run_ids(run: Run, root: Path) -> list[str]:
    """The run ids of every logged attempt of the run's label (its experiment and
    ``<cell>/s<seed>``), whatever their config hash, in attempt order (``launch.attempt_key``),
    across all journals: how an engineering cycle finds a case's launcher runs, so that a fix
    that changes a hashed config key does not hide an attempt issued before it."""
    starts = [r for r in launch_records(root)
              if r.get("event") == "start" and r.get("run") == run.label
              and r.get("experiment") == run.experiment
              and isinstance(r.get("run_id"), str)]  # fmt: skip
    return list(dict.fromkeys(r["run_id"] for r in sorted(starts, key=attempt_key)))


def label_superseded(run: Run, root: Path) -> set[str]:
    """The run ids of the run's label (its experiment and ``<cell>/s<seed>``) that the launch
    journal names superseded, whatever their config hash, as ``launch.supersede`` judges a line
    it would write: the one check of the commands that decide whether a ``superseded`` line is
    missing before writing it (``restart_lines_to_repair``, ``log_superseded``,
    ``repair_cycle_journal``). What a campaign under the current config hash counts stays
    ``launch.superseded_runs``."""
    return {r["run_id"] for r in launch_records(root)
            if r.get("event") == "superseded" and r.get("run") == run.label
            and r.get("experiment") == run.experiment
            and isinstance(r.get("run_id"), str)}  # fmt: skip


def settled(run: Run, root: Path) -> Path | None:
    """The run's settled launcher directory (``launch.assess``: done, or diverged), or None."""
    state = assess(run, root, launch_records(root))
    return state.done or state.diverged


def remove_checkpoints(p: Planned, run_ids: list[str]) -> list[str]:
    removed = []
    for run_id in run_ids:
        for path in (p.endpoint / f"{run_id}.pt", p.endpoint / f".{run_id}.pt.partial"):
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    return removed


def located(run: Run, root: Path) -> Path | None:
    """For the verdict, after the checkpoints are gone (``assess`` would call such a run a
    conflict): the one directory of the run's config whose outcome is settled, or None. A run a
    case restart or supersession named superseded in the journal is not one."""
    found = []
    superseded = superseded_runs(run, launch_records(root))
    for run_dir in sorted((root / run.experiment).glob("*")):
        meta = _read(run_dir / "meta.json")
        if meta is None or meta.get("config_hash") != run.config_hash:
            continue
        if run_dir.name in superseded:
            continue
        if meta.get("status") in ("completed", "diverged") or (run_dir / "terminal.json").is_file():
            found.append(run_dir)
    return found[0] if len(found) == 1 else None


def control_binding(
    p: Planned, records: Mapping[str, dict], root: Path
) -> tuple[Path | None, list[str]]:
    """Whether the recorded parts of a case whose control alone is outstanding bind it to its
    launcher run without the launcher (module docstring, "A case whose control alone is
    outstanding"); ``records``: the baseline record, and the rest record when there is one. The
    baseline record classified the run as conforming (``launcher.classified`` true,
    ``launcher.kind`` null); each record names (``launcher_run``, by its experiment and run id,
    wherever RUNS_DIR lay) the case's located run (``located``), superseded nowhere, with the
    record's identity, which it must name (``q.launcher_identity``: the recorded checkpoint
    record); the run is done in the launch journal (its end, settled or not), its label held
    nowhere, and its endpoint checkpoint, when on disk, has its recorded sha256
    (``checkpoint_conflict``: read only when present, never required). The located run, and what
    fails."""
    problems = []
    launcher = records["baseline"].get("launcher") or {}
    if launcher.get("classified") is not True:
        problems.append("the baseline record did not classify its launcher run "
                        "(launcher.classified is not true)")  # fmt: skip
    elif launcher.get("kind") is not None:
        problems.append(f"the baseline record's launcher check did not pass (launcher.kind is "
                        f"{launcher.get('kind')!r}, not null)")  # fmt: skip
    journal = launch_records(root)
    superseded = label_superseded(p.run, root)
    run_dir = located(p.run, root)
    if run_dir is None:
        problems.append("no located launcher run (one settled directory of its config hash "
                        "that no superseded line names)")  # fmt: skip
    for part_name, record in records.items():
        named = record.get("launcher_run")
        if isinstance(named, str) and Path(named).name in superseded:
            problems.append(f"the launcher run the {part_name} record names, {Path(named).name}, "
                            "is superseded in the launch journal")  # fmt: skip
        if run_dir is None:
            continue
        # <experiment>/<run id>, as ``fa_control_diagnostic.run_identity`` names a run
        if not isinstance(named, str) or Path(named).parts[-2:] != run_dir.parts[-2:]:
            problems.append(f"the {part_name} record names the launcher run {named!r}, not the "
                            f"located run {run_dir}")  # fmt: skip
        identity = (record.get("launcher") or {}).get("identity")
        if not isinstance(identity, dict):
            problems.append(f"the {part_name} record names no launcher identity "
                            f"(launcher.identity is {identity!r})")  # fmt: skip
        elif q.launcher_identity(run_dir) != identity:
            problems.append(f"the located run {run_dir.name}'s identity is not the one the "
                            f"{part_name} record names")  # fmt: skip
    if run_dir is not None:
        conflict = checkpoint_conflict(run_dir)  # as ``restart-case`` reads it
        if conflict is not None:
            summary = _read(run_dir / "summary.json") or {}
            recorded = (summary.get("endpoint_checkpoint") or {}).get("sha256")
            problems.append(f"the launcher run {run_dir.name}'s endpoint checkpoint {conflict} is "
                            f"on disk with sha256 {checkpoint_sha256(conflict)}, not its recorded "
                            f"{recorded}: a provenance conflict; the file is kept, and the case "
                            "goes to diagnosis")  # fmt: skip
        ends = [r.get("outcome") for r in journal
                if r.get("event") == "end" and r.get("run_id") == run_dir.name]  # fmt: skip
        if not ends or any(outcome != "done" for outcome in ends):
            problems.append(f"the launch journal's end of {run_dir.name} is not done: {ends}")
    held = held_runs(p.run, journal)
    if held:
        problems.append(f"its label {p.run.label} has a held launcher run: {held}")
    return run_dir, problems


def control_only(p: Planned, out: Path, root: Path, scope: q.QualificationScope) -> Path:
    """The launcher run of a case whose only outstanding part is its same-path control (module
    docstring, "A case whose control alone is outstanding": decided by its baseline, or with its
    baseline and rest recorded), not technical: the driver then runs that control on it, and
    neither the launcher nor any other part. In this order: exits 3 (``NO``), the driver's usual
    path, for a case not in that state, then for a case its own verdict marks technical
    (``technical_reasons``, whatever its bindings: ``finish`` exports its checkpoints), then for
    a case decided by its launcher run's own declared instability stop; exits 1, naming why, for
    a case whose recorded parts cannot be bound to the launcher run (``control_binding``: a
    baseline not classified or of another kind included)."""
    name = p.case.name

    def no(why: str) -> NoReturn:
        print(f"{name}: {why}: the usual path", file=sys.stderr)
        sys.exit(NO)

    if not p.control:
        no("it holds no same-path control")
    resolution = resolution_of(out, p.case)
    if resolution is not None:
        no(f"the engineering cycle kept the case on its launcher run {resolution['run_id']} "
           f"({resolution['outcome']})")  # fmt: skip
    complete, _, why = case_state(out, p, scope)
    if complete:
        no(f"{why}: complete, nothing is outstanding")
    control = part_state(out, p.case, "control")
    if control.state not in ("due", "interrupted"):
        no(f"its control's attempt {control.attempt} is {control.state}, neither due nor "
           "interrupted")  # fmt: skip
    # The other parts, as ``complete`` reads them: the baseline and, unless it decided, the rest
    others = dataclasses.replace(p, control=False)
    recorded, decided, why = case_state(out, others, scope)
    if not recorded:
        no(f"{why}: {'' if decided else 'not decided, and '}a part besides its control is "
           "outstanding")  # fmt: skip
    # Technical by its own verdict, whatever its bindings: its control, and the export of its
    # checkpoints by ``finish``, on the usual path
    technical = technical_reasons(out, others, scope)
    if technical:
        no(f"a technical case ({'; '.join(technical)}), whose finish exports its checkpoints")
    baseline = part_state(out, p.case, "baseline").record or {}
    launcher = baseline.get("launcher") or {}
    if launcher.get("classified") is True and launcher.get("kind") == q.SCIENTIFIC:
        no("decided by its launcher run's own declared instability stop, a run the launcher "
           "accepts without its checkpoint")  # fmt: skip
    records = {"baseline": baseline}
    if not decided:
        records["rest"] = part_state(out, p.case, "rest").record or {}
    run_dir, problems = control_binding(p, records, root)
    if problems or run_dir is None:
        sys.exit(f"{name}: its same-path control alone is outstanding, but its recorded parts "
                 f"cannot be bound to its launcher run without the launcher: "
                 f"{'; '.join(problems)}. The driver stops; nothing is rerun")  # fmt: skip
    return run_dir


# ---------------------------------------------------- a technical case's kept checkpoints

KEPT_MANIFEST = re.compile(r"kept-checkpoints-(\d+)\.json")  # an export's manifest (case folder)
KEPT_RECEIPT = re.compile(r"kept-check-(\d+)\.json")  # ``kept-check``'s receipt (case folder)
KEPT_FOLDER = re.compile(r"kept-(\d+)")  # an export's folder on the volume
KEPT_REMOTE = "kept-checkpoints"  # the driver's --kept-remote by default, under ARTIFACT_REMOTE
UPLOADED = "ok"  # scripts/archive.py's status of an object whose size and MD5 match its file
KEPT_MARGIN = 1 << 30  # the free space an export needs beyond its files (1 GiB)
LOST_BEFORE_EXPORT = ("lost before the export: the technical mark stands; the diagnosis goes on "
                      "from the records; nothing reruns")  # fmt: skip


def case_files(p: Planned, root: Path) -> list[Path]:
    """The case's checkpoint files on disk, exactly those ``finish --release`` deletes: the
    ``<dir>/<run_id>.pt`` and ``.<run_id>.pt.partial`` of every attempt of its launcher run
    (``journal_run_ids``) and every file under its harness's folder ``<dir>/harness/<case>/``."""
    found = [path for run_id in journal_run_ids(p.run, root)
             for path in (p.endpoint / f"{run_id}.pt", p.endpoint / f".{run_id}.pt.partial")
             if path.is_file()]  # fmt: skip
    if p.harness.is_dir():
        found += sorted(path for path in p.harness.rglob("*") if path.is_file())
    return found


def numbered(folder: Path, pattern: re.Pattern[str]) -> list[tuple[int, Path]]:
    """The entries of ``folder`` whose name ``pattern`` gives a number, in its order."""
    if not folder.is_dir():
        return []
    return sorted((int(m[1]), f) for f in folder.iterdir() if (m := pattern.fullmatch(f.name)))


def same_filesystem(a: Path, b: Path) -> bool:
    """Whether two paths lie on one filesystem: ``os.stat(...).st_dev`` of each, or of the nearest
    folder above it that exists (a checkpoint directory gone with the container disk)."""

    def device(path: Path) -> int:
        path = Path(path).absolute()
        while not path.exists() and path != path.parent:
            path = path.parent
        return os.stat(path).st_dev

    return device(a) == device(b)


def free_bytes(folder: Path) -> int:
    return shutil.disk_usage(folder).free


def fsync_folder(folder: Path) -> None:
    """A folder's entries made durable: a rename or a new entry in it survives a crash."""
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def make_folders(folder: Path) -> None:
    """``folder`` and each missing folder above it, every new entry made durable."""
    missing = []
    while not folder.exists():
        missing.append(folder)
        folder = folder.parent
    for new in reversed(missing):
        new.mkdir()
        fsync_folder(new.parent)


def copy_durably(source: Path, target: Path) -> tuple[int, str]:
    """``source`` copied to ``target``, never over a file: written under a partial name, fsynced,
    renamed, its folder fsynced. The source's size and sha256, hashed while it is read."""
    if target.exists():
        raise FileExistsError(f"{target} exists: an export overwrites nothing")
    make_folders(target.parent)
    partial = target.with_name(f".{target.name}.copying")
    digest, size = hashlib.sha256(), 0
    with source.open("rb") as read, partial.open("xb") as write:
        while chunk := read.read(1 << 24):
            digest.update(chunk)
            size += len(chunk)
            write.write(chunk)
        write.flush()
        os.fsync(write.fileno())
    os.replace(partial, target)
    fsync_folder(target.parent)
    return size, digest.hexdigest()


def write_durably(path: Path, data: bytes) -> None:
    """A new file written whole and durably (a partial name, fsync, rename, its folder fsynced)."""
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("xb") as write:
        write.write(data)
        write.flush()
        os.fsync(write.fileno())
    os.replace(partial, path)
    fsync_folder(path.parent)


def matches_export(path: Path, entry: Mapping[str, Any]) -> bool:
    """Whether a file holds the bytes an export's manifest lists for ``entry``: its size, then its
    sha256."""
    try:
        return path.stat().st_size == entry["size"] and checkpoint_sha256(path) == entry["sha256"]
    except OSError:
        return False


def sums_bytes(manifest: Mapping[str, Any], manifest_sha256: str | None) -> bytes:
    """An export's ``SHA256SUMS`` (``sha256sum -c`` in its folder): each copy's sha256 and path
    relative to the export's folder, then ``MANIFEST.json``'s."""
    export = PurePosixPath(manifest["export"])
    lines = [f"{entry['sha256']}  {PurePosixPath(entry['copy']).relative_to(export)}"
             for entry in manifest["files"]]  # fmt: skip
    return "".join(f"{line}\n" for line in [*lines, f"{manifest_sha256}  MANIFEST.json"]).encode()


def binding_records(
    p: Planned, out: Path, root: Path
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """What an export's manifest binds, its binding records, read by path and sha256 alone: the
    case's located launcher run's ``summary.json`` and ``meta.json`` (path, sha256) with the
    summary's ``endpoint_checkpoint`` record, and the case's latest baseline, rest and control
    records (path, file sha256, ``record_sha256``). Nothing else of the run's directory is opened
    or listed: its responses/, observations/ and judge/ least of all. With them, every checkpoint
    those records name, by path: each record that names it, with the sha256 it gives."""
    named: dict[str, list[dict[str, Any]]] = {}

    def name(record: Path, checkpoint: Any) -> None:
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("path"), str):
            named.setdefault(checkpoint["path"], []).append(
                {"record": str(record), "sha256": checkpoint.get("sha256")})  # fmt: skip

    def bound(path: Path) -> tuple[dict[str, Any], dict | None]:
        try:
            data = path.read_bytes()
        except OSError:
            return {"path": str(path), "sha256": None}, None
        try:
            record = json.loads(data)
        except ValueError:
            record = None
        found = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
        return found, record if isinstance(record, dict) else None

    launcher = None
    run_dir = located(p.run, root)
    if run_dir is not None:
        summary, record = bound(run_dir / "summary.json")
        checkpoint = (record or {}).get("endpoint_checkpoint")
        launcher = {"run_dir": str(run_dir), "summary": summary,
                    "meta": bound(run_dir / "meta.json")[0],
                    "endpoint_checkpoint": checkpoint}  # fmt: skip
        name(run_dir / "summary.json", checkpoint)
    parts: dict[str, Any] = {}
    for part in ["baseline", "rest", *(["control"] if p.control else [])]:
        state = part_state(out, p.case, part)
        if state.state != "recorded":
            parts[part] = None
            continue
        path = out / p.case.attempt_path(part, state.attempt)
        parts[part] = {"attempt": state.attempt, "path": str(path), "sha256": file_sha256(path),
                       "record_sha256": q.record_sha256(state.record)}  # fmt: skip
        if part == "baseline":
            name(path, state.record.get("checkpoint"))
        launcher_check = state.record.get("launcher")
        identity = launcher_check.get("identity") if isinstance(launcher_check, dict) else None
        name(path, identity.get("checkpoint") if isinstance(identity, dict) else None)
    return {"launcher": launcher, "parts": parts}, named


def kept_refusal(p: Planned, out: Path, root: Path, kept: Path | None) -> str | None:
    """Why an export to ``kept`` could not keep the case's checkpoints (module docstring, "A
    technical case's kept checkpoints"), or None."""
    if kept is None:
        return "no --kept DIR (the folder on the Pod's volume: /workspace/kept-checkpoints)"
    if not kept.is_absolute():
        return f"--kept {kept} is not an absolute path"
    if not kept.is_dir():
        return f"--kept {kept} is not an existing folder"
    where = kept.resolve()
    for label, other, why in (
        ("the runs root", root, "its copies would join the small results downloaded and "
         "archived from there"),
        ("the records' folder", out, "its copies would join the records"),
        ("the checkpoint directory", p.endpoint, "it goes with the container disk"),
    ):  # fmt: skip
        if where.is_relative_to(other.resolve()):
            return f"--kept {kept} lies under {label} {other}: {why}"
    if same_filesystem(kept, p.endpoint):
        return (f"--kept {kept} is on the filesystem of the checkpoint directory {p.endpoint} (the "
                "container disk, lost with it)")  # fmt: skip
    return None


def export_problems(manifest: Mapping[str, Any], manifest_sha256: str | None) -> list[str]:
    """What of an export on the volume does not re-hash to its manifest: a manifest not verified,
    a copy missing or other than its exported bytes (size, sha256), a ``MANIFEST.json`` other than
    the case folder's manifest, a ``SHA256SUMS`` other than the manifest gives."""
    problems = []
    if manifest.get("verified") is not True:
        problems.append("the manifest is not verified (a copy did not read back as its source)")
    try:
        kept, export = Path(manifest["kept"]), Path(manifest["export"])
        for entry in manifest["files"]:
            copy = kept / entry["copy"]
            if not copy.is_file():
                problems.append(f"{copy}: missing on the volume")
            elif not matches_export(copy, entry):
                problems.append(f"{copy}: not its exported bytes (size or sha256)")
        if file_sha256(kept / export / "MANIFEST.json") != manifest_sha256:
            problems.append(f"{kept / export / 'MANIFEST.json'}: not the case's manifest")
        sums = hashlib.sha256(sums_bytes(manifest, manifest_sha256)).hexdigest()
        if file_sha256(kept / export / "SHA256SUMS") != sums:
            problems.append(f"{kept / export / 'SHA256SUMS'}: not the manifest's sums")
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"the manifest cannot be read: {type(exc).__name__}: {exc}")
    return problems


def source_problems(manifest: Mapping[str, Any], files: list[Path]) -> list[str]:
    """The case's files an export does not hold as they are now: one it does not list (new since
    the export), or one whose size or sha256 is not the exported one (changed since)."""
    entries = {entry.get("source"): entry for entry in manifest.get("files") or []
               if isinstance(entry, dict)}  # fmt: skip
    problems = []
    for path in files:
        entry = entries.get(str(path))
        if entry is None:
            problems.append(f"{path}: not in the export (new since)")
        elif not matches_export(path, entry):
            problems.append(f"{path}: not its exported bytes (changed since)")
    return problems


def reusable(path: Path, kept: Path, files: list[Path]) -> dict | None:
    """The manifest at ``path`` when its export can stand for a new one: made under ``kept``,
    listing exactly the case's files, its copies re-hashing to it (``export_problems``) and the
    files still its exported bytes (``source_problems``); otherwise None."""
    manifest = _read(path)
    if manifest is None or manifest.get("kept") != str(kept):
        return None
    listed = {entry.get("source") for entry in manifest.get("files") or []
              if isinstance(entry, dict)}  # fmt: skip
    if listed != {str(file) for file in files} or export_problems(manifest, file_sha256(path)):
        return None
    return None if source_problems(manifest, files) else manifest


def write_export(
    p: Planned, out: Path, root: Path, technical: list[str], kept: Path, files: list[Path],
    export: Path, number: int,
) -> dict[str, Any]:  # fmt: skip
    """The export itself (``export_kept``): the new folder ``export``, each file copied and read
    back, ``MANIFEST.json`` and ``SHA256SUMS`` in it, and last the case's manifest
    ``kept-checkpoints-<number>.json``. Its manifest; an ``OSError`` anywhere is the caller's."""
    folder = case_folder(out, p.case)
    binding, named = binding_records(p, out, root)
    make_folders(export.parent)
    export.mkdir()  # a new folder: an export overwrites nothing
    fsync_folder(export.parent)
    entries = []
    for source in files:
        target = export / source.relative_to(p.endpoint)
        size, sha = copy_durably(source, target)
        copy_sha = checkpoint_sha256(target)  # the copy read back
        records = named.get(str(source), [])
        entries.append({
            "source": str(source), "copy": target.relative_to(kept).as_posix(), "size": size,
            "sha256": sha, "copy_sha256": copy_sha, "verified": copy_sha == sha,
            "recorded_sha256": records[0]["sha256"] if records else None,
            "matches_record": all(r["sha256"] == sha for r in records) if records else None,
            "named_by": records, "partial": source.name.endswith(".pt.partial")})  # fmt: skip
    gone = released(folder)  # removed on purpose by finish: never lost
    lost = [{"path": path, "recorded_sha256": records[0]["sha256"], "named_by": records}
            for path, records in sorted(named.items())
            if path not in gone and not Path(path).is_file()]  # fmt: skip
    manifest = {
        "case": p.case.record(), "stage": ROLES[p.case.engineering],
        "candidate": p.case.candidate, "technical_reasons": technical, "kept": str(kept),
        "export": export.relative_to(kept).as_posix(), "number": number, "binding": binding,
        "files": entries, "lost": lost, "verified": all(e["verified"] for e in entries),
        "host": socket.gethostname(), "pod_id": q.pod_id(),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provenance": q.qualification_provenance(load_settings()),
    }  # fmt: skip
    q.publish(export / "MANIFEST.json", manifest)
    write_durably(export / "SHA256SUMS", sums_bytes(manifest, q.record_sha256(manifest)))
    q.publish(folder / f"kept-checkpoints-{number}.json", manifest)  # last: the case's record
    return manifest


def export_kept(p: Planned, out: Path, root: Path, technical: list[str], kept: Path | None) -> int:
    """``_export_kept``, its every ``OSError`` exit 13 with the error named (the refusal and space
    checks' included: an unreadable folder, a failing ``stat``): no verified volume copy, the
    checkpoints still only on the container disk."""
    try:
        return _export_kept(p, out, root, technical, kept)
    except OSError as exc:
        print(f"{p.case.name}: the export could not be checked or made: {exc}. Nothing is called "
              "verified; the checkpoints stay where they are, still only on the container disk; "
              "fix the cause, then finish --kept again", file=sys.stderr)  # fmt: skip
        return NOT_EXPORTED


def _export_kept(p: Planned, out: Path, root: Path, technical: list[str], kept: Path | None) -> int:
    """A technical case's checkpoints (``case_files``) exported to the Pod's volume (module
    docstring, "A technical case's kept checkpoints"): each copied durably to
    ``<kept>/qualification/c<N>/<stage>/<case>/kept-<n>/`` in the layout it has under the
    checkpoint directory, read back and verified by its sha256, then the manifest, in the export
    (``MANIFEST.json``, with ``SHA256SUMS``) and last in the case's folder
    (``kept-checkpoints-<n>.json``). Refused before any copy (``kept_refusal``, or less free space
    than the files take with ``KEPT_MARGIN``); an earlier manifest whose copies re-hash to it under
    ``kept``, the case's files as it lists them, is reused: nothing is copied. An ``OSError``
    anywhere in the export (``write_export``) leaves the partial ``kept-<n>`` without a case
    manifest, never called verified. The exit: 9 (``TECHNICAL_CASE``) when every file still
    present has a verified copy, 13 (``NOT_EXPORTED``) otherwise: refused, failed or not verified,
    the checkpoints still only on the container disk."""
    name, folder = p.case.name, case_folder(out, p.case)

    def refused(why: str) -> int:
        print(f"{name}: the export is refused and nothing is copied: {why}. The checkpoints stay "
              "where they are, still only on the container disk, and nothing is called verified",
              file=sys.stderr)  # fmt: skip
        return NOT_EXPORTED

    def lost_note(lost: list[dict[str, Any]]) -> None:
        if lost:
            print(f"{name}: {LOST_BEFORE_EXPORT}: {[entry['path'] for entry in lost]}",
                  file=sys.stderr)  # fmt: skip

    refusal = kept_refusal(p, out, root, kept)
    if refusal is not None:
        return refused(refusal)
    files, manifests = case_files(p, root), numbered(folder, KEPT_MANIFEST)
    reused = reusable(manifests[-1][1], kept, files) if manifests else None
    if reused is not None:  # nothing is copied
        lost_note(reused.get("lost") or [])
        print(json.dumps({"case": name, "manifest": str(manifests[-1][1]), "reused": True,
                          "files": len(files)}))  # fmt: skip
        return TECHNICAL_CASE
    need, free = sum(path.stat().st_size for path in files) + KEPT_MARGIN, free_bytes(kept)
    if free < need:
        return refused(f"{free} bytes are free under {kept}, below the {need} its files take "
                       f"with a margin of {KEPT_MARGIN} bytes")  # fmt: skip
    parent = kept / "qualification" / folder.relative_to(out)
    number = 1 + max([n for n, _ in manifests] + [n for n, _ in numbered(parent, KEPT_FOLDER)],
                     default=0)  # fmt: skip
    export = parent / f"kept-{number}"
    try:
        manifest = write_export(p, out, root, technical, kept, files, export, number)
    except OSError as exc:  # no case manifest: nothing of this export is called verified
        print(f"{name}: the export to {export} failed: {exc}. No manifest is published in the "
              "case's folder and nothing is called verified; the checkpoints stay where they "
              "are, still only on the container disk, and finish --kept again exports to a new "
              "folder", file=sys.stderr)  # fmt: skip
        return NOT_EXPORTED
    path = folder / f"kept-checkpoints-{number}.json"
    entries, lost = manifest["files"], manifest["lost"]
    if not manifest["verified"]:
        bad = [entry["source"] for entry in entries if not entry["verified"]]
        print(f"{name}: a copy did not read back as its source's bytes, so nothing of {export} is "
              f"called verified: {bad}. The checkpoints stay where they are, still only on the "
              f"container disk; finish --kept again writes kept-{number + 1}",
              file=sys.stderr)  # fmt: skip
        return NOT_EXPORTED
    lost_note(lost)
    print(json.dumps({"case": name, "manifest": str(path), "export": str(export),
                      "files": len(entries),
                      "lost": [entry["path"] for entry in lost]}))  # fmt: skip
    return TECHNICAL_CASE


def inventory_problems(
    manifest: Mapping[str, Any], manifest_size: int, manifest_sha256: str, inventory: Path,
    remote: str,
) -> list[str]:  # fmt: skip
    """What scripts/archive.py's inventory does not show verified in the bucket under ``remote``
    (its REMOTE_PREFIX): every line's URL lies under ``remote/``, and each exported file (the
    copies, ``MANIFEST.json``, ``SHA256SUMS``) needs its line (the latest of its path, relative to
    the volume folder) with the exported size and sha256, the status ``ok`` (the object's size
    and MD5 are the file's) and the URL ``remote/<its path>``."""
    try:
        export = manifest["export"]
        expected = {entry["copy"]: (entry["size"], entry["sha256"]) for entry in manifest["files"]}
        sums = sums_bytes(manifest, manifest_sha256)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"the manifest cannot be read: {type(exc).__name__}: {exc}"]
    expected[f"{export}/MANIFEST.json"] = (manifest_size, manifest_sha256)
    expected[f"{export}/SHA256SUMS"] = (len(sums), hashlib.sha256(sums).hexdigest())
    try:
        rows = inventory.read_text().splitlines()
    except OSError as exc:
        return [f"the inventory {inventory} cannot be read: {exc}"]
    rows = [cells for cells in (row.split("\t") for row in rows)
            if len(cells) == 6 and cells[0] != "path"]  # fmt: skip
    lines = {cells[0]: cells for cells in rows}
    problems = [f"{cells[0]}: archived at {cells[4]}, not under {remote}/" for cells in rows
                if not cells[4].startswith(f"{remote}/")]  # fmt: skip
    for rel, (size, sha) in expected.items():
        cells = lines.get(rel)
        if cells is None:
            problems.append(f"{rel}: no line in the inventory {inventory}")
        elif (cells[1], cells[2]) != (str(size), sha):
            problems.append(f"{rel}: the inventory's size and sha256 ({cells[1]}, {cells[2]}) "
                            f"are not the exported {size}, {sha}")  # fmt: skip
        elif cells[5] != UPLOADED:
            problems.append(f"{rel}: not verified in the bucket ({cells[5]})")
        elif cells[4].startswith(f"{remote}/") and cells[4] != f"{remote}/{rel}":
            problems.append(f"{rel}: archived at {cells[4]}, not at {remote}/{rel}")
    return problems


def kept_check(p: Planned, out: Path, inventory: Path, remote: str) -> int:
    """Whether the case's latest export is kept, verified, on the Pod's volume and in the bucket
    (module docstring, "A technical case's kept checkpoints"): its copies re-hash to its manifest
    (``export_problems``) and archive.py's ``inventory`` shows every exported file verified
    (``inventory_problems``). Reads only; then publishes its receipt, ``kept-check-<n>.json`` in
    the case's folder: the manifest's and the inventory's path and sha256, the exit, what fails,
    and ``pending-upload`` for a volume copy verified whose bucket copy is not. 0, or 13
    (``NOT_EXPORTED``)."""
    name, folder = p.case.name, case_folder(out, p.case)
    manifests = numbered(folder, KEPT_MANIFEST)
    if not manifests:
        print(f"{name}: no export of its checkpoints (no kept-checkpoints-<n>.json in {folder}): "
              "finish --kept DIR exports them", file=sys.stderr)  # fmt: skip
        return NOT_EXPORTED
    path = manifests[-1][1]
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    manifest = _read(path)
    if manifest is None:
        volume, bucket = [f"{path}: unreadable"], []
    else:
        volume = export_problems(manifest, sha)
        bucket = inventory_problems(manifest, len(data), sha, inventory, remote)
    code = NOT_EXPORTED if volume or bucket else 0
    pending = not volume and bool(bucket)
    number = 1 + max((n for n, _ in numbered(folder, KEPT_RECEIPT)), default=0)
    receipt = folder / f"kept-check-{number}.json"
    teardown = None
    if code:
        teardown = "planned teardown waits for kept-check; the hard deadline stands"
        if pending:
            teardown = f"the volume copy is verified and kept, the upload pending: {teardown}"
    q.publish(receipt, {
        "case": p.case.record(), "manifest": {"path": str(path), "sha256": sha},
        "inventory": {"path": str(inventory.absolute()), "sha256": file_sha256(inventory)},
        "remote": remote, "exit": code,
        "problems": volume + bucket, "pending-upload": pending, "teardown": teardown,
        "host": socket.gethostname(), "pod_id": q.pod_id(),
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds")})  # fmt: skip
    if volume:
        print(f"{name}: its export on the volume is not verified: {volume}"
              + (f"; in the bucket: {bucket}" if bucket else "") + f". {teardown}",
              file=sys.stderr)  # fmt: skip
    elif bucket:
        print(f"{name}: its bucket copy is not verified (pending-upload, {receipt.name}): "
              f"{bucket}. {teardown}", file=sys.stderr)  # fmt: skip
    else:
        print(json.dumps({"case": name, "manifest": str(path), "receipt": str(receipt),
                          "exit": 0}))  # fmt: skip
    return code


def bound_export(folder: Path) -> tuple[Path, dict, Path] | None:
    """The export a release relies on: the manifest of the case's latest ``kept-check`` receipt
    of exit 0 that names it with the sha256 it still has, verified; with that receipt. None
    without one."""
    manifests = {path.name: path for _, path in numbered(folder, KEPT_MANIFEST)}
    for _, receipt in reversed(numbered(folder, KEPT_RECEIPT)):
        record = _read(receipt) or {}
        named = record.get("manifest") if isinstance(record.get("manifest"), dict) else {}
        path = manifests.get(Path(str(named.get("path"))).name)
        if record.get("exit") != 0 or path is None or file_sha256(path) != named.get("sha256"):
            continue
        manifest = _read(path)
        if manifest is not None and manifest.get("verified") is True:
            return path, manifest, receipt
    return None


def release_kept(p: Planned, root: Path, folder: Path) -> tuple[list[str], dict, list[str]]:
    """``finish --release`` of a technical case (module docstring, "A technical case's kept
    checkpoints"): the files ``case_files`` gives go only through the export a ``kept-check`` of
    exit 0 bound (``bound_export``), the inventory that receipt names still of its sha256, every
    file listed in the manifest with its size and sha256 (``source_problems``), and each checked
    again right before its deletion; then the harness's folders left empty. What it removed, the
    export it relied on (the manifest and the receipt, by path and sha256), and why it stopped
    (nothing when it did not)."""
    files = case_files(p, root)
    bound = bound_export(folder)
    relied: dict[str, Any] = {"manifests": [], "kept_check": None}
    if bound is None:
        if files:
            why = ("no export of them is bound by a kept-check of exit 0 (finish --kept DIR, then "
                   f"kept-check): {list(map(str, files))}")  # fmt: skip
            return [], relied, [why]
        entries: dict[str, Any] = {}
    else:
        path, manifest, receipt = bound
        relied = {"manifests": [{"path": str(path), "sha256": file_sha256(path)}],
                  "kept_check": {"path": str(receipt), "sha256": file_sha256(receipt)}}  # fmt: skip
        inventory = (_read(receipt) or {}).get("inventory")
        inventory = inventory if isinstance(inventory, dict) else {}
        if files and (not isinstance(inventory.get("path"), str) or inventory.get("sha256") is None
                      or file_sha256(Path(inventory["path"])) != inventory["sha256"]):  # fmt: skip
            why = (f"the inventory {inventory.get('path')} that {receipt.name} verified is gone or "
                   "changed (its sha256 is not the receipt's): the upload is checked again "
                   "(archive.py, then kept-check)")  # fmt: skip
            return [], relied, [why]
        problems = source_problems(manifest, files)
        if problems:
            return [], relied, [*problems, "another export is needed (finish --kept DIR, then "
                                           "kept-check)"]  # fmt: skip
        entries = {entry["source"]: entry for entry in manifest["files"]}
    removed: list[str] = []
    for source in files:
        if not matches_export(source, entries[str(source)]):  # right before its deletion
            return removed, relied, [f"{source}: not its exported bytes right before its "
                                     "deletion (changed since the check)"]  # fmt: skip
        source.unlink()
        removed.append(str(source))
    if p.harness.is_dir():  # the folders left empty
        for sub in sorted((d for d in p.harness.rglob("*") if d.is_dir()),
                          key=lambda d: len(d.parts), reverse=True):  # fmt: skip
            with suppress(OSError):
                sub.rmdir()
        try:
            p.harness.rmdir()
            removed.append(str(p.harness))
        except OSError:
            pass
    left = [f"{path}: not in the export (new since the check)" for path in case_files(p, root)]
    return removed, relied, left


# ------------------------------------------------------------------------------- commands


def require_clean() -> None:
    """A part or a verdict writes records only from a clean checkout of a commit."""
    state = git_state()
    if state.get("commit") is None or state.get("dirty") is not False:
        sys.exit("The checkout is not clean (tracking.git_state): records made from it never "
                 "count, and none is written from it; commit first")  # fmt: skip


def plan_text(name: str, planned: list[Planned], problems: list[str], settings, steps) -> str:
    need = disk_need_gb(settings, steps)
    lines = [f"campaign {name}: {len(planned)} cases, in this order"]
    for p in planned:
        lines.append(f"  {p.case.name}: launcher run {p.run.label}"
                     + ("; with its rate's same-path control" if p.control else ""))  # fmt: skip
    if planned:
        lines.append(f"checkpoints: {planned[0].endpoint} (the harness's under harness/)")
    lines.append(f"≈ {minutes(planned)} GPU min (a launcher run ≈ {MINUTES['launcher']}, a "
                 f"baseline ≈ {MINUTES['baseline']}, a rest ≈ {MINUTES['rest']}, a same-path "
                 f"control ≈ {MINUTES['control']} min)")  # fmt: skip
    lines.append(f"free disk per case: {need:g} GB ({2 + len(steps) + 1} checkpoints of "
                 f"{float(settings.checkpoint_gb):g} GB and {float(settings.disk_margin_gb):g} "
                 "GB of margin)")  # fmt: skip
    lines.append(f"free host memory: {float(settings.host_memory_gb):g} GB")
    lines += [f"PROBLEM: {p}" for p in problems]
    return "\n".join(lines)


def part(args, planned: Planned, out: Path) -> None:
    case, name = planned.case, args.part
    resolution = resolution_of(out, case)
    if resolution is not None:  # the engineering cycle resolved its launcher run: nothing runs
        sys.exit(f"{case.name} {name}: the engineering cycle kept the case on its launcher run "
                 f"{resolution['run_id']} ({resolution['outcome']}: "
                 f"{'; '.join(resolution['reasons'])}): no part runs, and no rerun is "
                 "opened")  # fmt: skip
    state = part_state(out, case, name)
    reason = args.infrastructure
    # The engineering cycle named its latest attempt: the rerun after it carries the mark
    cycle = None if state.state == "due" else cycle_of(case_folder(out, case), name, state.attempt)
    if args.resume:
        if state.state == "recorded":
            print(f"{case.name} {name}: attempt {state.attempt} is recorded "
                  f"({state.record.get('status')}); nothing runs")  # fmt: skip
            return
        if state.state in ("interrupted", "voided"):
            mark = "--engineering-cycle" if cycle is not None else "--infrastructure REASON"
            why = ("was started and left no record" if state.state == "interrupted"
                   else "is named by the case's restart or supersession record")  # fmt: skip
            print(f"{case.name} {name}: attempt {state.attempt} {why}: its next attempt is run "
                  f"by hand, --attempt {state.attempt + 1} {mark}", file=sys.stderr)  # fmt: skip
            sys.exit(INTERRUPTED)
        if state.state == "invalid":
            sys.exit(f"{case.name} {name}: attempt {state.attempt}'s record is unreadable or "
                     "not this part's: nothing runs")  # fmt: skip
        attempt = 1
    else:
        attempt = args.attempt
        if state.state in ("recorded", "invalid"):
            sys.exit(f"{case.name} {name}: attempt {state.attempt} is recorded (or its "
                     "record unreadable): a recorded attempt, whatever its kind, is never "
                     "rerun")  # fmt: skip
        allowed = 1 if state.state == "due" else state.attempt + 1
        if attempt != allowed:
            sys.exit(f"{case.name} {name}: the next attempt is {allowed}, not {attempt}")
        if cycle is not None:  # a re-qualification after the engineering cycle
            if attempt > 1 and (not args.engineering_cycle or reason):
                sys.exit(f"{case.name} {name}: its attempts are superseded by the engineering "
                         f"cycle ({cycle[0].name}): its rerun carries --engineering-cycle, not "
                         "--infrastructure")  # fmt: skip
        elif args.engineering_cycle:
            sys.exit(f"{case.name} {name}: no engineering cycle supersedes this part")
        elif (attempt > 1) != bool(reason):
            sys.exit("an attempt above 1 reruns an interrupted one, or one a case restart "
                     "names, and needs --infrastructure REASON; attempt 1 takes none")  # fmt: skip
        folder = case_folder(out, case)
        if (name == "rest" and state.state == "interrupted"
                and ("rest", state.attempt) not in named_attempts(folder)):  # fmt: skip
            results = interrupted_rest(folder, state.attempt)
            kind = (
                None
                if results is None
                else failure_kind(
                    "rest", results, recorded_baselines(folder), scope_of(), case.candidate
                )
            )
            if kind is not None:
                sys.exit(f"{case.name} rest: attempt {state.attempt}'s trajectory results hold a "
                         f"{kind} failure, which keeps counting: it is never rerun; record the "
                         "attempt with interrupted --technical REASON")  # fmt: skip
    if name == "rest" and part_state(out, case, "baseline").state == "voided":
        sys.exit(f"{case.name} rest: the baseline's attempts are named by the case's restart or "
                 "supersession record: its next baseline attempt runs first")  # fmt: skip
    if args.run_dir is None or args.checkpoints is None:
        sys.exit("part needs --run-dir and --checkpoints")
    require_clean()
    settings = load_settings()
    cfg = load_run_config(planned.run.config, planned.run.overrides)
    runtime.configure_numerics(cfg, args.device)  # first: before any CUDA work
    env = runtime.environment(cfg, args.device)
    path = out / case.attempt_path(name, attempt)
    q.publish(Path(f"{path}.started"), {
        "part": name, "attempt": attempt, "case": case.record(), "run_dir": str(args.run_dir),
        "infrastructure": reason, "engineering_cycle": None if cycle is None else str(cycle[0]),
        "host": socket.gethostname(), "pod_id": q.pod_id(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds")})  # fmt: skip
    infrastructure = engineering_cycle = None
    previous = [{"attempt": n, "state": s} for n, s in state.attempts.items() if n < attempt]
    if reason:  # the record carries its reason from its one publish, before any part reads it
        restarts = [{"path": str(f), "sha256": file_sha256(f)}
                    for f, _ in restart_records(case_folder(out, case))]  # fmt: skip
        infrastructure = {"reason": reason, "previous": previous, "restarts": restarts}
    if cycle is not None and attempt > 1:  # the supersession it re-qualifies after
        engineering_cycle = {"path": str(cycle[0]), "sha256": file_sha256(cycle[0]),
                             "previous": previous}  # fmt: skip
    record = q.run_part(settings, env, case, args.run_dir, out, part=name, attempt=attempt,
                        checkpoints=args.checkpoints,
                        diagnostic_steps=parse_steps(args.diagnostic_steps),
                        infrastructure=infrastructure,
                        engineering_cycle=engineering_cycle)  # fmt: skip
    print(json.dumps({key: record.get(key) for key in ("part", "attempt", "status", "kind",
                                                       "reasons")}))  # fmt: skip


def finish(
    planned: Planned, out: Path, root: Path, scope: q.QualificationScope, *, release: bool,
    kept: Path | None = None,
) -> None:  # fmt: skip
    """A complete case's checkpoints deleted (module docstring, step 7); a technical case's
    exported to ``kept`` instead (``export_kept``: exit 9, or 13), and deleted with ``release``
    only through that export, verified (``release_kept``)."""
    complete, _, why = case_state(out, planned, scope)
    if not complete:
        sys.exit(f"{planned.case.name}: {why}; its checkpoints are kept")
    technical = technical_reasons(out, planned, scope)
    if technical and not release:
        print(f"{planned.case.name}: a technical verdict; its checkpoints are kept for the "
              f"diagnosis, and exported: {technical}", file=sys.stderr)  # fmt: skip
        sys.exit(export_kept(planned, out, root, technical, kept))
    folder = case_folder(out, planned.case)
    relied: dict[str, Any] = {}
    stopped: list[str] = []
    if technical:  # released only through its verified export
        removed, relied, stopped = release_kept(planned, root, folder)
    else:
        removed = remove_checkpoints(planned, journal_run_ids(planned.run, root))
        if planned.harness.exists():
            shutil.rmtree(planned.harness)
            removed.append(str(planned.harness))
    if removed:  # removed on purpose: never evidence of an interruption (``restart-case``)
        q.publish(folder / f"{RELEASED}-{len(list(folder.glob(f'{RELEASED}-*.json'))) + 1}.json",
                  {"case": planned.case.record(), "removed": removed, "release": release,
                   "technical": technical, **relied,
                   "at": datetime.now(UTC).isoformat(timespec="seconds")})  # fmt: skip
    if stopped:
        print(f"{planned.case.name}: the release stops, and what is left is kept: "
              f"{'; '.join(stopped)}", file=sys.stderr)  # fmt: skip
        sys.exit(NOT_EXPORTED)
    print(json.dumps({"case": planned.case.name, "state": why, "removed": removed}))


def interrupted(args, planned: Planned, out: Path, scope: q.QualificationScope) -> None:
    """An interrupted attempt recorded as a technical part record with ``--technical REASON``
    ("Failure routes": an unexplained incomplete record is technical): the case, its launcher
    run as the ``.started`` file names it, the policy, and for a rest the baseline record it was
    built on (the latest baseline then); never over a record. A rest attempt that published
    trajectory results carries them, bound as they were (their launcher run, baseline record,
    bridge checks, code and settings): what they show keeps counting, a scientific failure
    included."""
    case, name, attempt = planned.case, args.part, args.attempt
    state = attempt_files((out / case.attempt_path(name, 1)).parent, name)
    if state.get(attempt) != "interrupted" or attempt != max(state):
        sys.exit(f"{case.name} {name}: attempt {attempt} is not an interrupted attempt (the "
                 f"latest, started and without a record): {state}")  # fmt: skip
    require_clean()
    path = out / case.attempt_path(name, attempt)
    started = _read(Path(f"{path}.started")) or {}
    run_dir = started.get("run_dir")
    reason = f"interrupted, recorded technical: {args.technical}"
    record: dict[str, Any] = {
        "part": name, "attempt": attempt, "case": case.record(),
        "K_star": scope.k_star(case.candidate), "bound": scope.bound(case.candidate),
        "launcher_run": run_dir, "status": "failed", "kind": q.TECHNICAL, "reasons": [reason],
        "launcher": {"kind": None, "reasons": [],
                     **(q.launcher_record(Path(run_dir)) if run_dir
                        else {"identity": None, "pod_id": None})},
        "interrupted": {"reason": args.technical, "started": started},
    }  # fmt: skip
    # The Pod is the one the attempt started on (its ``.started`` file), not this process's
    record["provenance"] = q.qualification_provenance(load_settings(),
                                                      pod_id=started.get("pod_id"))  # fmt: skip
    results = interrupted_rest(case_folder(out, case), attempt) if name == "rest" else None
    if results is not None:  # what the attempt published keeps counting, bound as it was
        record.update({key: results.get(key) for key in (
            "launcher", "baseline_record", "baseline_checkpoint_sha256", "bridge",
            "v0_unchanged", "trajectories", "comparisons", "from_trajectories",
            "provenance")})  # fmt: skip
        record["reasons"] += results.get("reasons") or []  # results bound to different inputs
        record["provenance"] = {**(results.get("provenance") or {}),
                                "pod_id": started.get("pod_id")}  # fmt: skip
    elif name == "rest":
        base = (out / case.attempt_path("baseline", 1)).parent
        found = max((n for n, s in attempt_files(base, "baseline").items() if s == "recorded"),
                    default=None)  # fmt: skip
        if found is not None:
            source = out / case.attempt_path("baseline", found)
            baseline = _read(source) or {}
            record["baseline_record"] = {"path": str(source),
                                         "sha256": q.record_sha256(baseline)}  # fmt: skip
            record["baseline_checkpoint_sha256"] = (baseline.get("checkpoint") or {}).get("sha256")
    q.publish(path, record)
    print(json.dumps({key: record[key] for key in ("part", "attempt", "status", "kind",
                                                   "reasons")}))  # fmt: skip


def scope_of() -> q.QualificationScope:
    return q.QualificationScope.of(load_settings())


def recorded_baselines(folder: Path) -> dict[str | None, dict | None]:
    """The case's recorded baseline attempts, by their record's sha256."""
    out: dict[str | None, dict | None] = {}
    for n, state in attempt_files(folder, "baseline").items():
        if state == "recorded":
            path = folder / f"baseline-attempt-{n}.json"
            out[file_sha256(path)] = _read(path)
    return out


def failure_kind(
    part: str, record: dict, baselines: dict, scope: q.QualificationScope, candidate: int
) -> str | None:
    """Whether a part's record (or an interrupted rest's trajectory results) is a completed
    failure, and of which kind: a baseline its verdict alone fails (scientific: a declared
    instability rule, the launcher run's instability stop; technical: any reason but the rest
    still to come); a rest failing with the baseline record it names, or technical when it names
    none recorded; a control not accepted. None for a record that passes so far."""
    if part == "baseline":
        verdict = q.case_verdict(record, None, scope)
        if verdict["kind"] == q.SCIENTIFIC:
            return q.SCIENTIFIC
        return q.TECHNICAL if [r for r in verdict["reasons"] if r != REST_MISSING] else None
    if part == "rest":
        bound = baselines.get((record.get("baseline_record") or {}).get("sha256"))
        if bound is None:
            return q.SCIENTIFIC if record.get("kind") == q.SCIENTIFIC else q.TECHNICAL
        return q.case_verdict(bound, record, scope)["kind"]
    verdict = q.control_verdict(record, scope, candidate)
    return None if verdict["state"] == q.ACCEPTED else verdict["kind"]


def attempt_failures(folder: Path, p: Planned, scope: q.QualificationScope) -> list[list[str]]:
    """The case's attempts no restart may name (module docstring, "A case restart"), as
    [label, kind]: every recorded attempt that is a completed failure, scientific or technical
    (``failure_kind``), or that the verdicts cannot read, and every interrupted rest attempt
    whose trajectory results are one; attempts a restart or supersession names already are
    void and left out."""
    named, listed = named_records(folder), named_attempts(folder)
    baselines = recorded_baselines(folder)
    found: list[list[str]] = []
    for name in ["baseline", "rest", *(["control"] if p.control else [])]:
        for n, state in attempt_files(folder, name).items():
            path = folder / f"{name}-attempt-{n}.json"
            label = f"{name} attempt {n}"
            if state == "recorded":
                if file_sha256(path) in named:
                    continue
                record = _read(path)
            elif name == "rest" and (name, n) not in listed:
                record = interrupted_rest(folder, n)
                if record is None:
                    continue
                label += " (its trajectory results)"
            else:
                continue
            if record is None:
                found.append([label, "unreadable"])
                continue
            try:
                kind = failure_kind(name, record, baselines, scope, p.case.candidate)
            except Exception as exc:  # a record the verdicts cannot read is never voided
                found.append([label, f"unreadable ({type(exc).__name__})"])
                continue
            if kind is not None:
                found.append([label, kind])
    return found


def case_attempts(folder: Path, p: Planned) -> list[dict[str, Any]]:
    """Every attempt of the case's parts as a voiding record names it."""
    attempts = []
    for name in ["baseline", "rest", *(["control"] if p.control else [])]:
        for n, state in attempt_files(folder, name).items():
            path = folder / f"{name}-attempt-{n}.json"
            record = _read(path) if state == "recorded" else None
            attempts.append({"part": name, "attempt": n, "state": state, "path": str(path),
                             "sha256": file_sha256(path) if state == "recorded" else None,
                             "status": (record or {}).get("status"),
                             "kind": (record or {}).get("kind")})  # fmt: skip
    return attempts


def launcher_to_supersede(planned: Planned, root: Path) -> tuple[Path | None, dict | None]:
    """The case's settled launcher run (``located``) and how a voiding record names it."""
    try:
        run_dir = located(planned.run, root)
        if run_dir is None:
            return None, None
        return run_dir, {"run_id": run_dir.name, "identity": q.launcher_identity(run_dir)}
    except Exception as exc:  # the record names what it can
        return None, {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def log_superseded(
    planned: Planned, root: Path, campaign: str, run_dir: Path | None, *, reason: str, record: Path
) -> None:
    """The launcher run, once the voiding record that names it is published, logged superseded in
    the launch journal (``launch.supersede``): kept, it no longer counts for its label, and the
    launcher's recovery round starts a new attempt. Done once."""
    if run_dir is None:
        return
    if run_dir.name not in label_superseded(planned.run, root):
        supersede(root, campaign, planned.run, run_dir.name, reason=reason, record=str(record))


RELEASED = "checkpoints-removed"  # ``finish``'s record of the checkpoints it removed


def released(folder: Path) -> set[str]:
    """Every path ``finish`` (``--release`` included) removed for the case, on purpose."""
    paths: set[str] = set()
    for file in sorted(folder.glob(f"{RELEASED}-*.json")) if folder.is_dir() else []:
        paths |= {str(x) for x in (_read(file) or {}).get("removed") or []}
    return paths


def checkpoint_missing(run_dir: Path | None, folder: Path | None = None) -> bool:
    """Whether the launcher run's recorded step-25 checkpoint is gone from disk, and not
    because ``finish`` removed it on purpose (the case's ``checkpoints-removed`` records)."""
    if run_dir is None:
        return False
    record = (_read(run_dir / "summary.json") or {}).get("endpoint_checkpoint")
    if not isinstance(record, dict) or Path(str(record.get("path"))).is_file():
        return False
    return folder is None or str(record.get("path")) not in released(folder)


def checkpoint_conflict(run_dir: Path | None) -> Path | None:
    """The launcher run's recorded endpoint checkpoint when it is on disk with another sha256
    than its recorded one, or None: a provenance conflict, never a loss (module docstring, "A
    case restart")."""
    if run_dir is None:
        return None
    record = (_read(run_dir / "summary.json") or {}).get("endpoint_checkpoint")
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return None
    path = Path(record["path"])
    if not path.is_file():
        return None
    return path if checkpoint_sha256(path) != record.get("sha256") else None


def own_conflict_end(run: Run, root: Path, run_dir: Path | None) -> dict | None:
    """The launcher run's end in the launch journal that no lost container disk explains, or None:
    an end ``outcome: conflict`` that is not settled (it carries a return code: the launcher saw
    the attempt's process end, and a lost disk takes that process with it), or one, settled or
    not, whose problem is not the missing endpoint checkpoint (``launch.is_missing_checkpoint``).
    A settled end (``settled: true``, no return code) with that problem alone is what a lost
    container disk leaves, as is an end ``done``: a clean run that lost only its checkpoint
    settles with it, the launcher having checked the rest first."""
    if run_dir is None:
        return None
    for record in launch_records(root):
        if (record.get("event") == "end" and record.get("run_id") == run_dir.name
                and record.get("run") == run.label and record.get("outcome") == "conflict"
                and (not record.get("settled")
                     or not is_missing_checkpoint(str(record.get("problem"))))):  # fmt: skip
            return record
    return None


def next_number(folder: Path, kind: str) -> int:
    return max((int(VOIDING_FILE.fullmatch(f.name)[2]) for f, _ in voiding_records(folder, kind)),
               default=0) + 1  # fmt: skip


def restart_case(
    args, planned: Planned, out: Path, root: Path, scope: q.QualificationScope, campaign: str
) -> None:
    """A case restart (module docstring, "A case restart"): a record, published atomically, that
    names every attempt of the case's parts (the baseline, the rest and its control if it holds
    one), each recorded one with its record's sha256, and the infrastructure reason; the
    attempts it names are void, and the case's launcher run is then logged superseded. Refused
    without a reason, for a complete case, with no attempt it does not name already, without
    evidence of an interruption (an interrupted attempt, or the launcher run's checkpoint gone
    from disk), when the launcher run itself is a completed failure (``launcher_problems``:
    its declared instability stop, scientific; another stop or a departing run, technical),
    when any attempt is a completed failure, scientific or technical, or unreadable
    (``attempt_failures``), when the launcher run has an end ``conflict`` in the launch journal
    that a lost container disk does not explain (``own_conflict_end``: an unsettled one, the
    launcher's own judgement of its attempt, or one whose problem is not the missing checkpoint),
    and when the launcher run's checkpoint is on disk with another sha256
    than its recorded one (``checkpoint_conflict``: a provenance conflict, not a loss, whatever
    an interrupted attempt would allow): only an interrupted case is restarted. A launcher run whose
    checkpoint is lost is evidence enough, and restarts the case even before any attempt (the
    Pod stopped between the launch and the baseline). Run again after a stop between the record
    and the journal's line, it only writes that line."""
    case, reason = planned.case, args.infrastructure
    if not reason:
        sys.exit("restart-case needs --infrastructure REASON: the documented interruption")
    resolution = resolution_of(out, case)
    if resolution is not None:  # kept by the engineering cycle: never a permission to rerun
        sys.exit(f"{case.name}: no restart: the engineering cycle kept the case on its launcher "
                 f"run {resolution['run_id']} ({resolution['outcome']}), and no rerun is "
                 "opened")  # fmt: skip
    folder = case_folder(out, case)
    attempts, listed = case_attempts(folder, planned), named_attempts(folder)
    fresh = [a for a in attempts if (a["part"], a["attempt"]) not in listed]
    pending = restart_lines_to_repair(planned, out, root)
    if pending:  # stopped before the journal's line
        repair_restart_lines(planned, root, campaign, pending)
    run_dir, launcher = launcher_to_supersede(planned, root)
    missing = checkpoint_missing(run_dir, folder)  # a checkpoint finish removed is no evidence
    if not fresh and not missing:  # a lost checkpoint alone is evidence, before any attempt
        sys.exit(f"{case.name}: no attempt to restart (none recorded or started, or every one "
                 "named by an earlier restart or supersession record), and the launcher run's "
                 "checkpoint is not lost")  # fmt: skip
    if case_state(out, planned, scope)[0]:
        sys.exit(f"{case.name}: the case is complete; a restart voids only an interrupted case")
    failures = attempt_failures(folder, planned, scope)
    if failures:
        sys.exit(f"{case.name}: no restart: a completed failure keeps counting: {failures}. A "
                 "scientific one is never rerun (no reruns to pass); a technical one is lifted "
                 "only by an engineering cycle's supersession record (supersede)")  # fmt: skip
    if run_dir is not None:
        kind, why = q.launcher_problems(run_dir, case, checkpoint=False, environment=False)
        if kind is not None:
            sys.exit(f"{case.name}: no restart: the launcher run {run_dir.name} is itself a "
                     f"completed {kind} failure, which keeps counting: {why}")  # fmt: skip
    own = own_conflict_end(planned.run, root, run_dir)
    if own is not None:
        by = (
            "the launcher's own judgement of its attempt, which cannot come from a lost "
            "container disk (its process ends with the disk and writes no end)"
            if not own.get("settled")
            else "a conflict other than a missing checkpoint"
        )
        sys.exit(f"{case.name}: no restart: the launcher run {run_dir.name} has an end with "
                 f"return code {own.get('returncode')} and outcome conflict "
                 f"({own.get('problem')}): {by}. A provenance conflict goes to diagnosis and "
                 "the technical route, never a restart")  # fmt: skip
    if not any(a["state"] == "interrupted" for a in fresh) and not missing:
        sys.exit(f"{case.name}: no restart: no evidence of an interruption (no interrupted "
                 "attempt, and the launcher run's checkpoint is on disk)")  # fmt: skip
    conflict = checkpoint_conflict(run_dir)
    if conflict is not None:  # whatever an interrupted attempt would allow
        sys.exit(f"{case.name}: no restart: the launcher run {run_dir.name}'s endpoint checkpoint "
                 f"{conflict} is on disk with another sha256 than its recorded one: a "
                 "provenance conflict, not a loss; it goes to the technical route (diagnosis, "
                 "the engineering cycle)")  # fmt: skip
    require_clean()
    number = next_number(folder, "restart")
    path = folder / f"restart-{number}.json"
    record = {"case": case.record(), "restart": number, "reason": reason, "attempts": attempts,
              "launcher": launcher, "checkpoint_missing": missing,
              "host": socket.gethostname(),
              "restarted_at": datetime.now(UTC).isoformat(timespec="seconds"),
              "provenance": q.qualification_provenance(load_settings())}  # fmt: skip
    q.publish(path, record)  # first the record, then the journal's line that points to it
    log_superseded(planned, root, campaign, run_dir, reason=reason, record=path)
    print(json.dumps({"case": case.name, "restart": str(path), "reason": reason,
                      "attempts": [(a["part"], a["attempt"]) for a in attempts],
                      "launcher": launcher}))  # fmt: skip


def restart_lines_to_repair(
    planned: Planned, out: Path, root: Path
) -> list[tuple[Path, dict | None, str]]:
    """The case's published restart records, each with the launcher run it names, whose
    ``superseded`` line the launch journal lacks (a stop between a record and its line, or an
    append that wrote no byte): ``restart-case`` run again then only writes those lines
    (``repair_restart_lines``), one of the two journal repairs allowed over a journal that is
    not whole (module docstring)."""
    have = label_superseded(planned.run, root)
    pending = []
    for path, record in restart_records(case_folder(out, planned.case)):
        launcher = (record or {}).get("launcher")
        named_run = launcher.get("run_id") if isinstance(launcher, dict) else None
        if isinstance(named_run, str) and named_run not in have:
            pending.append((path, record, named_run))
    return pending


def repair_restart_lines(
    planned: Planned, root: Path, campaign: str, pending: list[tuple[Path, dict | None, str]]
) -> None:
    """The ``superseded`` line of each published restart record ``restart_lines_to_repair``
    found, with the record's own reason; exits 1 with what it wrote."""
    for path, record, named_run in pending:
        log_superseded(planned, root, campaign, root / planned.run.experiment / named_run,
                       reason=str((record or {}).get("reason")), record=path)  # fmt: skip
    names = ", ".join(path.name for path, _, _ in pending)
    if len(pending) == 1:
        sys.exit(f"{planned.case.name}: the restart is recorded ({names}); the launch journal "
                 "now names its launcher run")  # fmt: skip
    sys.exit(f"{planned.case.name}: the restarts are recorded ({names}); the launch journal now "
             "names their launcher runs")  # fmt: skip


def cycle_to_repair(args, chosen: Planned, out: Path) -> tuple[Path, dict] | None:
    """The candidate's published supersession record when this ``supersede`` call names that
    cycle's own record again (its path and sha256): the call then only writes the journal's
    missing lines (``repair_cycle_journal``), one of the two journal repairs allowed over an
    unresolved torn line (module docstring)."""
    if args.record is None or not args.diagnosis or not args.fix:
        return None
    cycles = sorted(candidate_folder(case_folder(out, chosen.case)).glob("supersede-*.json"))
    path = args.record.resolve()
    if not cycles or not path.is_file():
        return None
    published = _read(cycles[0]) or {}
    old_record = published.get("old_record") or {}
    if old_record.get("path") == str(path) and old_record.get("sha256") == file_sha256(path):
        return cycles[0], published
    return None


def repair_cycle(root: Path, cycle: tuple[Path, dict]) -> None:
    """The journal's missing lines of a published supersession (``cycle_to_repair``)."""
    repaired = repair_cycle_journal(cycle[0], cycle[1], root)
    print(json.dumps({"supersession": str(cycle[0]), "journal_lines_written": repaired}))


def journal_repair(args, planned: list[Planned], out: Path, root: Path, campaign: str):
    """The journal repair this call is, decided before anything acts, or None: ``supersede``
    run again with its published record (``cycle_to_repair``), or ``restart-case`` with a
    published restart record whose ``superseded`` line is missing (``restart_lines_to_repair``,
    with the reason and outside a case the engineering cycle kept, as ``restart_case`` asks).
    Returns the action, which writes only the missing lines."""
    found = [p for p in planned if p.case.name == args.case]
    if len(found) != 1:
        return None
    chosen = found[0]
    if args.command == "supersede":
        cycle = cycle_to_repair(args, chosen, out)
        return None if cycle is None else lambda: repair_cycle(root, cycle)
    if args.command == "restart-case" and args.infrastructure:
        if resolution_of(out, chosen.case) is not None:
            return None
        pending = restart_lines_to_repair(chosen, out, root)
        if pending:
            return lambda: repair_restart_lines(chosen, root, campaign, pending)
    return None


def scientific_attempts(
    folder: Path, p: Planned | None, scope: q.QualificationScope, *, candidate: int
) -> set[tuple[str, int]]:
    """The case's attempts an engineering cycle keeps: every scientific failure (a record, or an
    interrupted rest's trajectory results; ``attempt_failures``), and the baseline record a kept
    rest is bound to, without which it could not be read. A case of no campaign run (another
    stage's) is read the same way, from the case its records name (``record_case``)."""
    keep: set[tuple[str, int]] = set()
    p = p if p is not None else record_case(folder)
    for label, kind in attempt_failures(folder, p, scope) if p is not None else []:
        if kind == q.SCIENTIFIC:
            part_name, _, n = label.split(" (")[0].split(" ")
            keep.add((part_name, int(n)))
    for part_name, n in list(keep):
        if part_name != "rest":
            continue
        record = _read(folder / f"rest-attempt-{n}.json")
        if record is None:  # an interrupted attempt: the baseline its results name
            record = interrupted_rest(folder, n) or {}
        named = (record.get("baseline_record") or {}).get("sha256")
        for m, state in attempt_files(folder, "baseline").items():
            if state == "recorded" and file_sha256(folder / f"baseline-attempt-{m}.json") == named:
                keep.add(("baseline", m))
    return keep


def record_case(folder: Path) -> SimpleNamespace | None:
    """The case the records in ``folder`` name, as ``attempt_failures`` reads a campaign case
    (its ``case``, and ``control`` when the folder holds a control attempt); None without a
    readable record of it."""
    fields = {f.name for f in dataclasses.fields(q.QualificationCase)}
    for part_name in q.PARTS:
        for n, state in attempt_files(folder, part_name).items():
            record = (
                _read(folder / f"{part_name}-attempt-{n}.json") if state == "recorded" else None
            )
            named = (record or {}).get("case")
            if isinstance(named, dict):
                try:
                    case = q.QualificationCase(**{k: v for k, v in named.items() if k in fields})
                except TypeError:
                    continue
                return SimpleNamespace(case=case, control=bool(attempt_files(folder, "control")))
    return None


def kept_records(folder: Path) -> set[str]:
    """The sha256 of the case's records the candidate's engineering cycle kept: every record of
    a case that holds a scientific failure, which it keeps whole."""
    case, stage = case_name_of(folder), stage_folder(folder).name
    return {e["sha256"] for _, r in voiding_records(folder, "supersede")
            for e in (r or {}).get("kept") or []
            if isinstance(e, dict) and e.get("case") == case and e.get("stage", stage) == stage
            and e.get("sha256")}  # fmt: skip


COMPLETED, UNRESOLVED = "completed", "unresolved"  # a launcher run's resolved outcome


@contextmanager
def configs_at(commit: str) -> Iterator[Path]:
    """The tracked ``configs/`` of ``commit`` as committed, never the working tree's: ``git
    archive`` in this clone, unpacked into a temporary directory that is removed on exit; yields
    its ``configs`` folder. Raises when the commit is not in this clone or is no commit id."""
    if not re.fullmatch(r"[0-9a-f]{4,64}", commit):
        raise ValueError(f"{commit!r} is not a commit id")
    resolved = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"],
                              cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()  # fmt: skip
    data = subprocess.run(["git", "archive", "--format=tar", resolved, "configs"], cwd=REPO_ROOT,
                          capture_output=True, check=True).stdout  # fmt: skip
    with tempfile.TemporaryDirectory(prefix="zfr-configs-") as tmp:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(tmp, filter="data")
        yield Path(tmp) / "configs"


def issued_config(root: Path, command: Any) -> Any:
    """The config a launcher run was issued: its journal start's command (``launch.Run.command``
    without the interpreter: the script, the config's name under ``configs/``, its overrides)
    composed from ``root`` alone (the configs of the run's own commit, ``configs_at``). Raises
    when the command is not one, or names a config outside ``root``."""
    if not (isinstance(command, list) and len(command) >= 2
            and all(isinstance(c, str) for c in command)):  # fmt: skip
        raise ValueError(f"its journal start records no command ({command!r})")
    name, overrides = command[1], command[2:]
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"its issued config {name!r} is not a name under configs/")
    path = root / (name if name.endswith((".yaml", ".yml")) else f"{name}.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"its issued config {name!r} is not among its commit's configs")
    return load_config(path, overrides, root=root)


def resolve_launcher_run(
    run_dir: Path, case: q.QualificationCase, start: Mapping[str, Any]
) -> tuple[str, list[str]]:
    """A launcher run no part record classified, against what it was issued and its own
    provenance, never this checkout's configs, code or environment (module docstring,
    "Supersession"). What it was issued: its journal start (``start``: label, config hash,
    header, command) and its own records (``config.yaml``; ``meta.json``: config hash, the
    campaign's record of its launch with the header, commit, ``inputs.numerics_setup``), with
    the configs of its own commit (``configs_at``): the command composed from them
    (``issued_config``) must give the start's config hash, and the case's confirmation config
    is composed from them too. The checks: its records agree with its start (config hash,
    label, header); its identity as ``launcher_problems`` checks it (seed, arm, rate, batch,
    steps, ``eval.k_only``), and the endpoint checkpoint's directory the issued one; config
    agreement with that confirmation config; its recorded numerics setup against
    ``launch.expected_numerics`` of the issued config (the device-independent fields); its
    commit recorded, clean, its GPU-code identity computable. Returns the outcome and why:
    ``scientific`` (it conforms, and its declared instability rule stopped it), ``completed``
    (it conforms and took its steps), ``technical`` (a check fails on readable records, or
    another stop, or an attempt the launcher never settled) or ``unresolved`` (records
    unreadable, a needed field missing, no commit or one not in this clone, configs that do not
    compose or do not give the recorded hash)."""
    departs, missing = [], []
    try:
        cfg = fcd.run_config(run_dir)
        meta = json.loads((run_dir / "meta.json").read_text())
        if not isinstance(cfg, DictConfig) or not isinstance(meta, dict):
            raise ValueError("config.yaml or meta.json is not a mapping")
    except Exception as exc:  # a saved config or meta.json that does not read: nothing to judge
        return UNRESOLVED, [f"its records are unreadable: {type(exc).__name__}: "
                            f"{str(exc)[:300]}"]  # fmt: skip
    terminal = fcd._read_object(run_dir / "terminal.json")
    if meta.get("status") not in ("completed", "diverged") and terminal is None:
        return q.TECHNICAL, [f"an attempt the launcher never settled (status "
                             f"{meta.get('status')!r})"]  # fmt: skip
    if meta.get("config_hash") != start.get("config_hash"):
        departs.append(f"its meta.json has config hash {meta.get('config_hash')!r}, its journal "
                       f"start {start.get('config_hash')!r}")  # fmt: skip
    launched = meta.get("campaign")
    if not isinstance(launched, dict) or "header" not in launched:
        missing.append("no campaign header in its meta.json (campaign.header)")
    else:
        if launched.get("label") != start.get("run"):
            departs.append(f"its meta.json names the label {launched.get('label')!r}, its "
                           f"journal start {start.get('run')!r}")  # fmt: skip
        if launched["header"] != start.get("header"):
            departs.append(f"its meta.json's header {launched['header']!r} is not its journal "
                           f"start's {start.get('header')!r}")  # fmt: skip
    expected = {"seed": case.seed, "optimizer.name": case.optimizer,
                "train.batch_size": case.batch, "train.steps": case.steps,
                "eval.k_only": True}  # fmt: skip
    for key, want in expected.items():
        have = OmegaConf.select(cfg, key, default=None)
        if have != want:
            departs.append(f"{key} is {have!r}, not {want!r}")
    lr = OmegaConf.select(cfg, "optimizer.lr", default=None)
    if not (isinstance(lr, int | float) and math.isclose(float(lr), case.lr, rel_tol=1e-9)):
        departs.append(f"optimizer.lr is {lr!r}, not {case.lr:g}")
    inputs = meta.get("inputs") if isinstance(meta.get("inputs"), dict) else {}
    recorded = inputs.get("numerics_setup")
    if not isinstance(recorded, dict):
        missing.append("no numerics setup (inputs.numerics_setup)")
    if not isinstance(inputs.get("theta0_trainable_sha256"), str):
        missing.append("no hash of θ₀'s trainable parameters")
    if q.reads_v0(case.optimizer) and not isinstance(inputs.get("v0_sha256"), str):
        missing.append("no hash of v₀")
    git = meta.get("git") if isinstance(meta.get("git"), dict) else {}
    if not git.get("commit") or git.get("dirty") not in (True, False):
        missing.append(f"no recorded commit (git: {git})")
    elif git["dirty"]:
        departs.append(f"made from a dirty checkout of {git['commit']}")
    else:
        if run_gpu_code(run_dir) is None:
            missing.append(f"the GPU code of its commit {git['commit']} cannot be told")
        try:  # what it was issued, from the configs of its own commit
            with configs_at(str(git["commit"])) as root:
                issued = issued_config(root, start.get("command"))
                reference = q.confirmation_config(case, root=root)
            issued_hash = config_hash(issued)
        except Exception as exc:
            missing.append(f"what it was issued cannot be composed from the configs of its "
                           f"commit {git['commit']}: {type(exc).__name__}: "
                           f"{str(exc)[:300]}")  # fmt: skip
        else:
            if issued_hash != start.get("config_hash"):
                missing.append(f"its issued config, composed from the configs of its commit, "
                               f"has config hash {issued_hash!r}, not its journal start's "
                               f"{start.get('config_hash')!r}")  # fmt: skip
            key = "tracking.endpoint_checkpoint"
            have, want = (OmegaConf.select(c, key, default=None) for c in (cfg, issued))
            if have != want:
                departs.append(f"{key} is {have!r}, its issued config's {want!r}")
            departs += q.config_departures(cfg, reference)
            for key, want in expected_numerics(issued).items():
                if not isinstance(recorded, dict):
                    break
                if key not in recorded:
                    missing.append(f"no numerics_setup.{key}")
                elif recorded[key] != want:
                    departs.append(f"numerics_setup.{key} is {recorded[key]!r}, its issued "
                                   f"config asks {want!r}")  # fmt: skip
    if departs:
        return q.TECHNICAL, departs
    stop = (terminal or {}).get("stop")
    if not (isinstance(stop, dict) and isinstance(stop.get("reason"), str)
            and isinstance((terminal or {}).get("steps"), int)):  # fmt: skip
        missing.append("no whole terminal record")
    if missing:
        return UNRESOLVED, missing
    reason, steps = stop["reason"], terminal["steps"]
    if reason in INSTABILITY:
        return q.SCIENTIFIC, [f"the launcher run {run_dir.name}, conforming to what it was "
                              f"issued, stopped by its declared rule {reason!r} after {steps} "
                              "steps"]  # fmt: skip
    if reason == "steps" and steps == case.steps:
        return COMPLETED, [f"the launcher run {run_dir.name} took its {steps} steps"]
    return q.TECHNICAL, [f"the launcher run {run_dir.name} stopped by {reason!r} after "
                         f"{steps} steps"]  # fmt: skip


def resolution_of(out: Path, case: q.QualificationCase) -> dict | None:
    """How the candidate's engineering cycle resolved the case's launcher run, when it kept the
    case for it (``scientific`` or ``unresolved``), or None."""
    folder = case_folder(out, case)
    stage = stage_folder(folder).name
    for _, record in voiding_records(folder, "supersede"):
        for kept in (record or {}).get("kept_cases") or []:
            if (kept.get("case"), kept.get("stage")) == (case.name, stage) and kept.get(
                    "resolution"):  # fmt: skip
                return kept["resolution"]
    return None


def resolved_verdict(resolution: Mapping[str, Any]) -> dict[str, Any]:
    """A case verdict of a launcher run's resolved outcome (``resolve_launcher_run``), for a
    case no part record of which classified that run (records that raised before their check
    keep their own verdicts beside it): ``failing`` and scientific for a conforming declared
    stop, an ``incomplete`` case of no kind for an unresolved one. Its GPU code is the run's,
    never the current one, so ``rate_verdict`` does not count it as a qualified case: a
    scientific one keeps the rate failing, an unresolved one keeps it from ``qualified``."""
    scientific = resolution.get("outcome") == q.SCIENTIFIC
    return {"case": resolution.get("case_record"),
            "state": q.FAILING if scientific else q.INCOMPLETE,
            "kind": q.SCIENTIFIC if scientific else None,
            "reasons": [("" if scientific else "unresolved: ") + r
                        for r in resolution.get("reasons") or []],
            "complete": False, "comparisons": {}, "bound": None, "tau": None, "warning": None,
            "reference_mode": None, "fallback": False,
            "launcher": resolution.get("identity"), "gpu_code": resolution.get("gpu_code"),
            "settings": None, "records": {}, "resolution": dict(resolution),
            "record_sha256": hashlib.sha256(json.dumps(resolution, sort_keys=True).encode())
            .hexdigest()}  # fmt: skip


def journal_starts(root: Path) -> dict[str, tuple[str, dict]]:
    """Every launcher attempt in the launch journals, by run id: (campaign, its start record)."""
    starts: dict[str, tuple[str, dict]] = {}
    for log in sorted((root / "launch").glob("*.jsonl")):
        for record in journal_records(log):  # the launcher's reading: a torn line is skipped
            if record.get("event") == "start" and isinstance(record.get("run_id"), str):
                starts.setdefault(record["run_id"], (log.stem, record))
    return starts


def in_attempt_order(run_ids: set[str], starts: Mapping[str, tuple[str, dict]]) -> list[str]:
    """Run ids ordered as their journal starts' attempts (``launch.attempt_key``: the attempt
    number, then the start's time; the run id breaks a tie), whatever journal file holds them;
    an id without a start comes first."""
    return sorted(run_ids, key=lambda r: (*attempt_key(starts.get(r, ("", {}))[1]), r))


def run_gpu_code(run_dir: Path) -> str | None:
    """The GPU-code identity (sha256) of the commit a launcher run recorded, from a clean
    checkout; None when it cannot be told."""
    git = (_read(run_dir / "meta.json") or {}).get("git") or {}
    if not git.get("commit") or git.get("dirty") is not False:
        return None
    try:
        return code_identity.gpu_code_identity(git["commit"]).get("sha256")
    except Exception:  # an unknown commit: its code cannot be told
        return None


def supersede_case(
    args, chosen: Planned, planned: list[Planned], out: Path, root: Path,
    scope: q.QualificationScope, campaign: str,
) -> None:  # fmt: skip
    """The candidate's engineering cycle, recorded (module docstring, "Supersession"; "Failure
    routes", "Voiding and supersession"): ``c<N>/supersede-1.json`` names the technical record that
    started it (its path and sha256), the diagnosis and the fix, the GPU code of the fix (this clean
    checkout's) and the old one, every record of the candidate made under the old GPU code, in all
    its stage folders, passed ones included, with the cases re-qualified, and every launcher run of
    the candidate in the launch journal that is not of the fix's GPU code (another or an unknown
    code included: the runs its records name and those of the campaign's cases, a case without any
    part attempt included, found by the case's label whatever their config hash:
    ``label_run_ids``), which it then logs superseded; a case whose run is superseded is
    re-qualified. A case that holds a scientific failure, as its bound, valid part records read with
    the verdicts show it (``scientific_attempts``; a launcher run's declared stop counts only when
    such a record classified it scientific), is kept whole: none of its records, nor its launcher
    run, is voided, and it is listed as kept, with why. A launcher run no part record classified
    (``launcher.classified``: a record that names it without it raised before its check) is
    resolved against what it was issued first (``resolve_launcher_run``, ``configs_at``: the
    configs of its own commit; a case of another stage as its records name it): a conforming
    declared stop
    keeps its case whole (scientific); an unresolvable one keeps it as it is, listed unresolved, the
    run held (no rerun opened); a departing or completed run is superseded. Refused: a second time
    for the candidate (one cycle per candidate: an unresolved technical failure then routes to the
    technical feasibility report); for a record that is not a completed technical failure of the
    case, or that belongs to a case the cycle keeps whole; when the fix's GPU code is the old one
    (the same path is not simply rerun); from a checkout that is not clean. Run again with the same
    record, it only writes the journal's lines that are missing."""
    case = chosen.case
    if args.record is None or not args.diagnosis or not args.fix:
        sys.exit("supersede needs --record PATH, --diagnosis TEXT and --fix TEXT")
    folder = case_folder(out, case)
    candidate_dir = candidate_folder(folder)
    path = args.record.resolve()
    sha = file_sha256(path)
    cycles = sorted(candidate_dir.glob("supersede-*.json"))
    again = cycle_to_repair(args, chosen, out)
    if again is not None:
        repair_cycle(root, again)
        return
    if cycles:
        sys.exit(f"candidate {case.candidate}: its engineering cycle is recorded already "
                 f"({cycles[0]}): one cycle per candidate, and a technical failure left "
                 "unresolved after it routes to the technical feasibility report")  # fmt: skip
    match = re.fullmatch(r"(baseline|rest|control)-attempt-(\d+)\.json", path.name)
    if (not match or path.parent != folder.resolve()
            or attempt_files(folder, match[1]).get(int(match[2])) != "recorded"):  # fmt: skip
        sys.exit(f"{case.name}: {args.record} is not a recorded attempt of this case")
    if sha in named_records(folder):
        sys.exit(f"{case.name}: {path.name} is void already")
    failures = attempt_failures(folder, chosen, scope)
    label = f"{match[1]} attempt {match[2]}"
    if dict(map(tuple, failures)).get(label) != q.TECHNICAL:
        sys.exit(f"{case.name}: {path.name} is not a completed technical failure: an "
                 "engineering cycle supersedes only one")  # fmt: skip
    science = scientific_attempts(folder, chosen, scope, candidate=case.candidate)
    if science:
        sys.exit(f"{case.name}: the case holds a scientific failure ({sorted(science)}), which "
                 "keeps counting: an engineering cycle keeps the case whole, so it cannot start "
                 "from one of its records")  # fmt: skip
    old = ((_read(path) or {}).get("provenance") or {}).get("gpu_code") or {}
    require_clean()
    fix = code_identity.gpu_code_identity()
    if not old.get("sha256") or fix.get("sha256") == old.get("sha256"):
        sys.exit(f"{case.name}: the fix's GPU code is the superseded record's "
                 f"({old.get('sha256')}): the same path is not simply rerun")  # fmt: skip
    by_name = {(p.case.name, ROLES[p.case.engineering]): p for p in planned}
    attempts, requalified, kept, kept_cases, run_ids, kept_runs = [], [], [], [], set(), set()
    entries_of: dict[tuple[str, str], list[dict]] = {}  # every attempt of a case, as named
    folder_of: dict[tuple[str, str], Path] = {}  # every case with an attempt: its folder
    # The launcher runs a part record classified: it names the run and says so explicitly
    # (``launcher.classified``, written once ``launcher_problems`` returned); a record whose setup
    # or check raised first names the run without classifying it
    classified: set[str] = set()
    for stage in sorted(d for d in candidate_dir.iterdir() if d.is_dir()):
        for case_dir in sorted({f.parent for f in stage.glob("s*/*/*-attempt-*")}):
            name = case_name_of(case_dir)
            p = by_name.get((name, stage.name))
            entries, mine = [], []
            named = named_records(case_dir)
            for part_name in q.PARTS:
                for n, state in attempt_files(case_dir, part_name).items():
                    file = case_dir / f"{part_name}-attempt-{n}.json"
                    record = _read(file) if state == "recorded" else None
                    launcher = (record or {}).get("launcher") or {}
                    identity = launcher.get("identity") or {}
                    entry = {"case": name, "stage": stage.name, "part": part_name,
                             "attempt": n, "state": state, "path": str(file),
                             "sha256": file_sha256(file) if state == "recorded" else None,
                             "status": (record or {}).get("status"),
                             "kind": (record or {}).get("kind"),
                             "launcher_run_id": identity.get("run_id")}  # fmt: skip
                    entries.append(entry)
                    if (state == "recorded" and entry["launcher_run_id"]
                            and launcher.get("classified") is True):  # fmt: skip
                        classified.add(entry["launcher_run_id"])
                    if state == "recorded" and (entry["sha256"] in named or (
                            ((record or {}).get("provenance") or {}).get("gpu_code") or {}
                    ).get("sha256") != old["sha256"]):  # fmt: skip
                        continue  # void already, or not of the old code
                    mine.append(entry)
            entries_of[(name, stage.name)] = entries
            folder_of[(name, stage.name)] = case_dir
            keep = scientific_attempts(case_dir, p, scope, candidate=case.candidate)
            if keep:  # a scientific failure keeps the case whole, its launcher run included
                kept += entries
                why = sorted(f"{part_name} attempt {n}" for part_name, n in keep)
                kept_cases.append(
                    {
                        "case": name,
                        "stage": stage.name,
                        "why": f"a scientific failure keeps counting: {why}",
                    }
                )
                kept_runs |= {e["launcher_run_id"] for e in entries if e["launcher_run_id"]}
                if p is not None:
                    kept_runs |= set(label_run_ids(p.run, root))
                continue
            if any(e["state"] == "recorded" for e in mine):
                attempts += mine
                requalified.append({"case": name, "stage": stage.name})
                run_ids |= {e["launcher_run_id"] for e in mine if e["launcher_run_id"]}
    kept_names = {(k["case"], k["stage"]) for k in kept_cases}
    # Every case not kept, of the campaign or with an attempt in a stage folder: its launcher
    # runs (its campaign run's attempts in the journals, found by its label whatever their config
    # hash, and the runs its records name), each judged by its own issued config. A run a
    # part record classified (``launcher.classified``) is judged by that record (scientific: the
    # case is kept above). One no part record classified (none names it, or each that does
    # raised before its check returned) is resolved against what it was issued
    # (``resolve_launcher_run``): a conforming declared stop keeps the case whole (scientific);
    # an outcome nobody can resolve keeps it as it is and holds the run (no rerun opened); a
    # nonconforming or completed run is superseded below, as any run not of the fix's code
    starts = journal_starts(root)
    subjects = {key: p.case for key, p in by_name.items()}
    for key, case_dir in folder_of.items():  # another stage's case: as its records name it
        named_case = record_case(case_dir) if key not in subjects else None
        if named_case is not None:
            subjects[key] = named_case.case
    by_case: dict[tuple[str, str], set[str]] = {}
    held: list[dict] = []
    for (name, stage_name), subject in subjects.items():
        if (name, stage_name) in kept_names:
            continue
        p = by_name.get((name, stage_name))
        ids = set(label_run_ids(p.run, root)) if p is not None else set()  # any config hash
        mentioned = {e["launcher_run_id"] for e in entries_of.get((name, stage_name), [])
                     if e["state"] == "recorded" and e["launcher_run_id"]}  # fmt: skip
        outcomes = {}  # in attempt order, never by journal file or run id
        for run_id in in_attempt_order((ids | mentioned) - classified, starts):
            if run_id not in starts:
                continue
            run_dir = root / str(starts[run_id][1].get("experiment")) / run_id
            if run_dir.is_dir():
                outcomes[run_id] = resolve_launcher_run(run_dir, subject, starts[run_id][1])
        found = [k for k in (q.SCIENTIFIC, UNRESOLVED) if any(o == k for o, _ in outcomes.values())]
        if found:
            outcome = found[0]
            run_id = next(r for r, (o, _) in outcomes.items() if o == outcome)  # the earliest
            campaign_name, start = starts[run_id]
            run_dir = root / str(start.get("experiment")) / run_id
            resolution = {"run_id": run_id, "outcome": outcome, "reasons": outcomes[run_id][1],
                          "case_record": subject.record(), "gpu_code": run_gpu_code(run_dir),
                          "identity": q.launcher_identity(run_dir)}  # fmt: skip
            why = (
                "a scientific failure keeps counting: "
                if outcome == q.SCIENTIFIC
                else "its launcher run's outcome is unresolved, and no rerun is opened: "
            )
            kept_cases.append({"case": name, "stage": stage_name,
                               "why": why + "; ".join(outcomes[run_id][1]),
                               "resolution": resolution})  # fmt: skip
            kept += entries_of.get((name, stage_name), [])
            kept_runs |= ids | mentioned
            if outcome == UNRESOLVED:
                held.append({"run_id": run_id, "campaign": campaign_name,
                             "run": start.get("run"), "experiment": start.get("experiment"),
                             "config_hash": start.get("config_hash"),
                             "reason": "; ".join(outcomes[run_id][1])})  # fmt: skip
            continue
        if p is not None:
            by_case[(name, stage_name)] = ids
            run_ids |= ids
    if (case.name, stage_folder(folder).name) in {(k["case"], k["stage"]) for k in kept_cases}:
        sys.exit(f"{case.name}: the cycle would keep this case whole (its launcher run's "
                 "declared stop, or an outcome nobody can resolve): a cycle cannot start from "
                 "one of its records")  # fmt: skip
    kept_names = {(k["case"], k["stage"]) for k in kept_cases}
    requalified = [c for c in requalified if (c["case"], c["stage"]) not in kept_names]
    attempts = [e for e in attempts if (e["case"], e["stage"]) not in kept_names]
    launchers = []
    for run_id in sorted(run_ids - kept_runs):
        if run_id not in starts:
            continue
        campaign_name, start = starts[run_id]
        run_dir = root / str(start.get("experiment")) / run_id
        # Every launcher run not of the fix's code (another code, or none to be told): a
        # re-qualification under the fix cannot use it
        if run_dir.is_dir() and run_gpu_code(run_dir) != fix.get("sha256"):
            launchers.append({"run_id": run_id, "campaign": campaign_name,
                              "run": start.get("run"), "experiment": start.get("experiment"),
                              "config_hash": start.get("config_hash"),
                              "identity": q.launcher_identity(run_dir)})  # fmt: skip
            for (name, stage_name), ids in by_case.items():  # its case re-qualifies, records
                entry = {"case": name, "stage": stage_name}  # or none
                if run_id in ids and entry not in requalified:
                    requalified.append(entry)
    record_path = candidate_dir / "supersede-1.json"
    record = {"candidate": case.candidate, "supersession": 1,
              "old_record": {"case": case.name, "stage": stage_folder(folder).name,
                             "path": str(path), "sha256": sha, "part": match[1],
                             "attempt": int(match[2])},
              "diagnosis": args.diagnosis, "fix": args.fix, "old_gpu_code": old,
              "fix_gpu_code": fix, "cases_requalified": requalified, "attempts": attempts,
              "kept_cases": kept_cases, "kept": kept, "launchers": launchers, "held": held,
              "host": socket.gethostname(),
              "superseded_at": datetime.now(UTC).isoformat(timespec="seconds"),
              "provenance": q.qualification_provenance(load_settings())}  # fmt: skip
    q.publish(record_path, record)  # first the record, then the journal's lines
    repair_cycle_journal(record_path, record, root)
    print(json.dumps({"candidate": case.candidate, "supersession": str(record_path),
                      "old_record": record["old_record"], "cases_requalified": requalified,
                      "kept_cases": kept_cases, "held": [r["run_id"] for r in held],
                      "launchers": [r["run_id"] for r in launchers]}))  # fmt: skip


def repair_cycle_journal(path: Path, record: dict, root: Path) -> list[str]:
    """The journal's ``superseded`` line of every launcher run the supersession names, where it
    is missing; returns the run ids whose line ``launch.hold`` or ``launch.supersede`` wrote
    (neither writes a line identical to one in the journal)."""
    reason = (
        f"superseded by an engineering cycle: {record.get('diagnosis')}; fix: {record.get('fix')}"
    )
    written = []
    for entry in record.get("held") or []:  # an unresolved outcome: the run is never rerun
        run = SimpleNamespace(label=entry.get("run"), experiment=entry.get("experiment"),
                              config_hash=entry.get("config_hash"))  # fmt: skip
        if entry["run_id"] not in held_runs(run, launch_records(root)) and hold(
                root, entry["campaign"], run, entry["run_id"], reason=entry["reason"],
                record=str(path)):  # fmt: skip
            written.append(entry["run_id"])
    for entry in record.get("launchers") or []:
        run = SimpleNamespace(label=entry.get("run"), experiment=entry.get("experiment"),
                              config_hash=entry.get("config_hash"))  # fmt: skip
        if entry["run_id"] not in label_superseded(run, root) and supersede(
                root, entry["campaign"], run, entry["run_id"], reason=reason,
                record=str(path)):  # fmt: skip
            written.append(entry["run_id"])
    return written


def superseded_run_ids(run: Run, root: Path) -> list[str]:
    """The attempts of the run that the launch journal shows superseded: named so by a case
    restart or supersession (``launch.supersede``), or started, and ended technical or not ended
    at all, with no settled directory (completed, whether it stands or is in conflict, or
    diverged). A completed run that cannot stand keeps its checkpoint for the diagnosis."""
    settled_outcomes = {r["run_id"] for r in launch_records(root)
                        if r.get("event") == "end" and r.get("run") == run.label
                        and isinstance(r.get("run_id"), str)
                        and r.get("outcome") in ("done", "diverged", "conflict")}  # fmt: skip
    superseded = []
    named = superseded_runs(run, launch_records(root))  # a restart or supersession names them
    for run_id in journal_run_ids(run, root):
        if run_id in named:
            superseded.append(run_id)
            continue
        run_dir = root / run.experiment / run_id
        meta = _read(run_dir / "meta.json") or {}
        if (run_id in settled_outcomes or meta.get("status") in ("completed", "diverged")
                or (run_dir / "terminal.json").is_file()):  # fmt: skip
            continue
        superseded.append(run_id)
    return superseded


def clear_stale(planned: Planned, root: Path) -> None:
    stale = superseded_run_ids(planned.run, root)
    print(json.dumps({"case": planned.case.name, "removed": remove_checkpoints(planned, stale)}))


def latest(folder: Path, part: str, named: dict[str, str] | None = None) -> dict | None:
    """The record of a part's latest attempt in a case's folder, or None (none, interrupted,
    unreadable, or named by a restart record: ``named``)."""
    attempts = attempt_files(folder, part)
    if not attempts or attempts[max(attempts)] != "recorded":
        return None
    path = folder / f"{part}-attempt-{max(attempts)}.json"
    if file_sha256(path) in (named or {}):
        return None
    return _read(path)


def attempt_listing(
    folder: Path, named: dict[str, str]
) -> tuple[dict[str, list], list[tuple[str, int, dict | None, str | None]]]:
    """Every attempt of every part in a case's folder (number, state, path, the record's sha256,
    status, kind and, when a restart record names it, the restart's reason), and the superseded
    records with that reason or None: a record a restart names (void), and one under a later
    attempt that no restart names, which the commands here never write and which is not hidden
    by it."""
    listing, superseded = {}, []
    cycles = named_records(folder, "supersede")
    for part in q.PARTS:
        attempts = attempt_files(folder, part)
        rows = []
        for n, state in attempts.items():
            path = folder / f"{part}-attempt-{n}.json"
            record = _read(path) if state == "recorded" else None
            sha = file_sha256(path) if state == "recorded" else None
            void = named.get(sha) if sha is not None else None
            cycle = void is not None and sha in cycles
            restart, supersession = (None, void) if cycle else (void, None)
            rows.append({"attempt": n, "state": state, "path": str(path), "sha256": sha,
                         "status": (record or {}).get("status"),
                         "kind": (record or {}).get("kind"), "restart": restart,
                         "supersession": supersession})  # fmt: skip
            if state == "recorded" and (void is not None or n < max(attempts)):
                superseded.append((part, n, record, restart, supersession))
        if rows:
            listing[part] = rows
    return listing, superseded


def verdict(args, settings, scope, planned: list[Planned], out: Path, root: Path) -> Path:
    candidate, stage = args.candidate, ROLES[args.engineering]
    base = out / f"c{candidate}" / stage
    folders = sorted({f.parent for f in base.glob("s*/*/*-attempt-*")})
    cases, records, attempts, superseded = [], [], {}, []
    old_controls: list[dict[str, Any]] = []  # the superseded controls' verdicts
    voided: dict[str, str] = {}  # a void record's verdict (record_sha256) → the restart's reason
    warnings: list[str] = []
    for folder in folders:
        name = str(folder.relative_to(base))
        named = named_records(folder)
        attempts[name], stale = attempt_listing(folder, named)
        baseline, rest = latest(folder, "baseline", named), latest(folder, "rest", named)
        rests = attempt_files(folder, "rest")
        if (rests and rests[max(rests)] == "interrupted"
                and ("rest", max(rests)) not in named_attempts(folder)):  # fmt: skip
            rest = interrupted_rest(folder, max(rests))  # its trajectory results count
        if baseline is not None or rest is not None:
            records += [r for r in (baseline, rest) if r is not None]
            cases.append(q.case_verdict(baseline, rest, scope, e_eng=args.e_eng))
        for part, n, record, restart, supersession in stale:  # shown; void, or counted
            void = restart or supersession
            mark = {"part": part, "attempt": n, "restart": restart, "supersession": supersession}
            superseded.append({"case": name, "sha256": q.record_sha256(record), **mark})
            if part == "control":  # counted against its rate as a superseded case is
                checked = {**q.control_verdict(record, scope, candidate), "superseded": mark}
                old_controls.append(checked)
                if void is not None and checked["record_sha256"] is not None:
                    voided[checked["record_sha256"]] = void
                continue
            pair = (record, None) if part == "baseline" else (None, record)
            verdict_ = {**q.case_verdict(*pair, scope, e_eng=args.e_eng), "superseded": mark}
            cases.append(verdict_)
            if void is not None:
                voided[verdict_["record_sha256"]] = void
    # A case the engineering cycle kept on its launcher run's resolved outcome (no part record
    # classified that run: none names it, or each that does raised before its check): its
    # verdict is that outcome too (``resolved_verdict``), beside its records' own
    for _, cycle in [
        (f, _read(f)) for f in sorted((out / f"c{candidate}").glob("supersede-*.json"))
    ]:
        for kept_case in (cycle or {}).get("kept_cases") or []:
            resolution = kept_case.get("resolution")
            if resolution and kept_case.get("stage") == stage:
                cases.append(resolved_verdict(resolution))
                if resolution["outcome"] == UNRESOLVED:
                    warnings.append(f"{kept_case['case']}: its launcher run "
                                    f"{resolution['run_id']}'s outcome is unresolved")  # fmt: skip
    campaign = {"path": str(args.campaign.resolve()),
                "sha256": hashlib.sha256(args.campaign.read_bytes()).hexdigest()}  # fmt: skip
    body: dict[str, Any] = {"candidate": candidate, "stage": stage, "campaign": campaign,
                            "e_eng": args.e_eng, "cases": cases, "attempts": attempts,
                            "superseded": superseded}  # fmt: skip
    if args.engineering:
        body["summary"] = q.engineering_summary(records)
        target = out / f"c{candidate}" / "engineering" / "verdict.json"
    else:
        code, policy = code_identity.gpu_code_identity(), q.settings_record(settings)
        rates, states, assigned = {}, {}, set()
        for optimizer, lr in dict.fromkeys((p.case.optimizer, p.case.lr) for p in planned):
            key = f"{optimizer}-{lr:g}"
            mine = [
                v
                for v in cases
                if (v["case"] or {}).get("optimizer") == optimizer
                and math.isclose((v["case"] or {}).get("lr") or math.nan, lr, rel_tol=1e-9)
            ]
            assigned |= {id(v) for v in mine}
            control = None
            holder = next((p for p in planned if p.control and p.case.lr == lr
                           and p.case.optimizer == optimizer), None)  # fmt: skip
            if holder is not None:
                where = case_folder(out, holder.case)  # a restart-named control is not current
                record = latest(where, "control", named_records(where))
                control = None if record is None else q.control_verdict(record, scope, candidate)
            runs = {p.case.seed: located(p.run, root) for p in planned
                    if (p.case.optimizer, p.case.lr) == (optimizer, lr)}  # fmt: skip
            launchers = {seed: None if d is None else q.launcher_identity(d)
                         for seed, d in runs.items()}  # fmt: skip
            if any(identity is None for identity in launchers.values()):
                warnings.append(f"{key}: launcher runs not located "
                                f"({sorted(s for s, i in launchers.items() if i is None)}): "
                                "the rate is at most incomplete")  # fmt: skip
            olds = [c for c in old_controls if (c["case"] or {}).get("optimizer") == optimizer
                    and math.isclose((c["case"] or {}).get("lr") or math.nan, lr,
                                     rel_tol=1e-9)]  # fmt: skip
            assigned |= {id(c) for c in olds}
            rates[key] = q.rate_verdict(mine, scope=scope, candidate=candidate,
                                        optimizer=optimizer, lr=lr, code=code, policy=policy,
                                        control=control, superseded_controls=olds,
                                        launchers=launchers, voided=voided)  # fmt: skip
            states[key] = rates[key]["state"]
            warnings += [f"{key}: τ exceeded by {name}" for name in rates[key]["warnings"]]
        body.update(
            states=states,
            rates=rates,
            unassigned=[
                (v["case"] or {}).get("name")
                for v in [*cases, *old_controls]
                if id(v) not in assigned
            ],
        )
        target = out / f"c{candidate}" / "verdict.json"
    body.update(warnings=warnings, provenance=q.qualification_provenance(settings))
    q.publish(target, body)
    for key, state in (body.get("states") or {}).items():
        print(f"{key}: {state}")
    return target


def driver_script(
    commit: str, written_at: str, campaign: Path, args, planned: list[Planned], settings,
    steps: list[int],
) -> str:  # fmt: skip
    """The bash driver of the qualification on a Pod (module docstring)."""
    stamp = "$(date -u +%FT%TZ)"
    sha = hashlib.sha256(campaign.read_bytes()).hexdigest()
    need = disk_need_gb(settings, steps)
    host = float(settings.host_memory_gb)
    stage = ROLES[args.engineering]
    common = f"--campaign \"$CAMPAIGN\" --candidate {args.candidate}" + (
        " --engineering" if args.engineering else "")  # fmt: skip
    diagnostic = f" --diagnostic-steps {','.join(map(str, steps))}" if steps else ""
    endpoint = planned[0].endpoint
    remote = shlex.quote(args.kept_remote.strip().rstrip("/"))
    lines = [
        "#!/usr/bin/env bash",
        *comment(f"Written by scripts/qualify.py driver at {written_at}; do not edit."),
        *comment(f"Study 3's numerical qualification ({stage}) of candidate {args.candidate} "
                 f"from commit {commit}: {len(planned)} cases of the campaign {campaign}, one at "
                 "a time: the launcher run alone (one attempt), the baseline part, then, unless "
                 "the baseline is already a scientific failure, the rest, and the FA rate's "
                 "same-path control, each in a fresh process (a case whose control alone is "
                 "due, not technical, runs that control alone, control-only: not the launcher, "
                 "whose checkpoint the control does not read); then the case's checkpoints are "
                 "deleted, unless its verdict is technical: then they are kept for the "
                 "diagnosis, exported to $KEPT on the Pod's volume and verified there (finish), "
                 "uploaded to $ARTIFACT_REMOTE (scripts/archive.py) and verified there "
                 f"(kept-check), and the driver stops (exit {TECHNICAL_CASE}; {NOT_EXPORTED} "
                 "while the export is not verified: the planned teardown waits for kept-check, "
                 "and the hard deadline stays). K-only and blind to outcomes."),
        *comment(f"A case needs {need:g} GB free under {endpoint} (the container disk). A "
                 "complete case is skipped, so the driver can be run again after a stop. It "
                 "never reruns a part or a launcher run: an interrupted part (exit "
                 f"{INTERRUPTED}) is rerun by hand as its next attempt with --infrastructure "
                 "REASON, a same-path control's after a lost container disk too (it reads no "
                 "checkpoint; a decided case is never restarted), a failed launcher run with "
                 "scripts/launch.py --only LABEL --max-attempts 1 --recovery REASON "
                 "--incident-time UTC --cause TEXT "
                 "--evidence REF, one attempt per documented incident, each round's reason one "
                 "no earlier round of the label gave. The verdict is written on every exit, "
                 "but over a launch journal that is not whole (an unresolved torn line, or a "
                 "line a published record requires missing), which verdict refuses too (exit "
                 f"{TORN_JOURNAL}): no verdict.json is written then, and an older one stays on "
                 "disk."),
        *comment(f"Exits: 0 every case complete; 1 a precondition (the host memory, the "
                 "bucket's settings and the place of $KEPT included), the disk, a case finish "
                 "refuses (not complete) "
                 "or a case control-only cannot bind to its launcher run without the "
                 f"launcher; {INTERRUPTED} an interrupted attempt (or a part a case restart "
                 f"names); {TECHNICAL_CASE} a technical case, its checkpoints kept, on the volume "
                 f"and in the bucket, verified; {STALE} a stale case (records of other GPU code "
                 f"or settings); {HELD} a case whose label has a held launcher run; "
                 f"{TORN_JOURNAL} a torn line in a launch journal, or a line a published record "
                 "requires missing from it (complete's refusal, or the launcher's); "
                 f"{NOT_EXPORTED} a technical case whose kept checkpoints are not verified: from "
                 "finish, its export refused or failed (no verified volume copy: the checkpoints "
                 "still only on the container disk, which a stop loses), or after kept-check "
                 "(run whatever archive.py's exit), the volume copy verified and the upload "
                 "pending (its receipt's pending-upload true) or the volume copy not verified "
                 "either (false, the worse state); either way the planned teardown waits for "
                 "kept-check exit 0, and the hard deadline stays; the launcher's own when it "
                 "stops; another a failed command."),
        "set -euo pipefail",
        *POD_ENVIRONMENT,
        f"SRC={SOURCES}/{commit}/source",
        'cd "$SRC"',
        f'test "$(git rev-parse HEAD)" = {commit}',
        f'project-run "$SRC" python -c "{CLEAN_CHECKOUT}"',
        f"CAMPAIGN={shlex.quote(str(campaign))}",
        f'test "$(sha256sum "$CAMPAIGN" | cut -d " " -f 1)" = {sha}',
        f"CKPT={shlex.quote(str(endpoint))}",
        f"KEPT={shlex.quote(os.path.normpath(args.kept))}",
        'OUT="${RUNS_DIR:?}/qualification"',
        f'mkdir -p "$OUT/c{args.candidate}" "$CKPT"',
        'mkdir -p "$KEPT"',
        f'exec > >(tee -a "$OUT/c{args.candidate}/driver-{stage}-$(date -u +%Y%m%dT%H%M%SZ).log")'
        " 2>&1",
        f'step() {{ echo "== {stamp} start: $*"; "$@"; echo "== {stamp} done: $*"; }}',
        f'stop() {{ echo "== {stamp} $1" >&2; exit "$2"; }}',
        "# A step whose exit the driver routes: its start and done markers, and its exit",
        f'routed() {{ echo "== {stamp} start: $*"; local rc=0; "$@" || rc=$?; '
        f'echo "== {stamp} done (exit $rc): $*"; return "$rc"; }}',
        "# The verdict on every exit, whatever stopped the driver (it refuses a journal not whole)",
        f"trap 'rc=$?; {RUN} verdict {common} || true; exit $rc' EXIT",
        "# No GPU job beside the qualification: no compute process, and (as a container may not",
        "# see other processes) no GPU memory in use",
        'if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then',
        '  stop "a process holds the GPU; the qualification does not start" 1',
        "fi",
        "used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)",
        f"if echo \"$used\" | awk '$1 + 0 >= {GPU_USED_MIB} {{busy = 1}} END {{exit !busy}}'; then",
        '  stop "GPU memory in use ($used MiB); the qualification does not start" 1',
        "fi",
        *comment("The host holds θ₀, v₀, the second model and the p1 direction, and a "
                 "checkpoint save's transient (host_memory_gb; the free memory: MemAvailable, "
                 "or the cgroup's when smaller)"),
        f'if ! project-run "$SRC" python -c "{HOST_FREE}" {host:g}; then',
        f'  stop "{host:g} GB of free host memory are needed; the qualification does not '
        'start" 1',
        "fi",
        *comment("A technical case's checkpoints go to the bucket before the driver stops "
                 "(scripts/archive.py under the key ARTIFACT_GCS_KEY): without ARTIFACT_REMOTE, "
                 "a gs:// path, a readable key file or gcloud, no case starts"),
        'case "${ARTIFACT_REMOTE:-}" in',
        "  gs://?*) ;;",
        "  *) stop \"a technical case's kept checkpoints could not reach the bucket: "
        'ARTIFACT_REMOTE is not a gs:// path; the qualification does not start" 1 ;;',
        "esac",
        'if [ ! -f "${ARTIFACT_GCS_KEY:-}" ] || [ ! -r "${ARTIFACT_GCS_KEY:-}" ] '
        "|| ! command -v gcloud > /dev/null; then",
        "  stop \"a technical case's kept checkpoints could not reach the bucket: "
        "ARTIFACT_GCS_KEY is not a readable file, or gcloud is not on PATH; the qualification "
        'does not start" 1',
        "fi",
        *comment("The kept checkpoints' folder is on the Pod's volume: never on the checkpoint "
                 "directory's device (the container disk), never under $RUNS_DIR or the "
                 "checkpoint directory"),
        'if [ "$(stat -L -c %d "$KEPT")" = "$(stat -L -c %d "$CKPT")" ]; then',
        '  stop "KEPT $KEPT is on the device of the checkpoint directory $CKPT (the container '
        'disk); the qualification does not start" 1',
        "fi",
        'case "$(realpath "$KEPT")/" in',
        '  "$(realpath "$RUNS_DIR")"/*|"$(realpath "$CKPT")"/*)',
        '    stop "KEPT $KEPT lies under RUNS_DIR or under the checkpoint directory $CKPT; the '
        'qualification does not start" 1 ;;',
        "esac",
    ]  # fmt: skip
    for index, p in enumerate(planned, 1):
        case = f"--case {shlex.quote(p.case.name)}"
        parts = f'{common} {case} --run-dir "$RUN_DIR" --checkpoints {shlex.quote(str(p.harness))}'
        # The case's folder of its exports under $KEPT (each export kept-<n>/ in it)
        kept_path = shlex.quote(f"qualification/{case_folder(Path(), p.case).as_posix()}")
        control = f"step {RUN} part {parts} --part control --resume"
        usual = [  # complete's no: the launcher run, then the parts
            f"{RUN} clear-stale {common} {case}",
            "free_kb=$(df -Pk \"$CKPT\" | awk 'NR == 2 {print $4}')",
            "case \"$free_kb\" in",
            "  ''|*[!0-9]*) stop \"df gave no free space for $CKPT: '$free_kb'\" 1 ;;",
            "esac",
            f'if [ "$free_kb" -lt {need_kb(need)} ]; then',
            f'  stop "{need:g} GB free are needed under $CKPT, $free_kb KB are" 1',
            "fi",
            f'step {LAUNCH} "$CAMPAIGN" --only {shlex.quote(p.run.label)} --max-attempts 1',
            f"RUN_DIR=$({RUN} locate {common} {case})",
            f"step {RUN} part {parts} --part baseline --resume{diagnostic}",
            "rc=0",
            f"{RUN} decided {common} {case} || rc=$?",
            'case "$rc" in',
            f'  0) echo "== {stamp} {p.case.name}: the baseline is a scientific failure; '
            'no rest" ;;',
            f"  {NO}) step {RUN} part {parts} --part rest --resume{diagnostic} ;;",
            '  *) stop "decided failed (exit $rc)" "$rc" ;;',
            "esac",
            *([control] if p.control else []),
        ]  # fmt: skip
        if p.control:  # its control alone outstanding, the case not technical: that alone
            usual = [
                "rc=0",
                f"RUN_DIR=$({RUN} control-only {common} {case}) || rc=$?",
                'case "$rc" in',
                f'  0) echo "== {stamp} {p.case.name}: only its same-path control is due (its '
                "baseline decided it, or its rest is recorded): the launcher, the baseline and "
                'the rest are not called"',
                f"     {control} ;;",
                f"  {NO})",
                *[f"    {line}" for line in usual],
                "    ;;",
                f'  *) stop "{p.case.name}: control-only failed (exit $rc)" "$rc" ;;',
                "esac",
            ]  # fmt: skip
        lines += [
            f"# ---- case {index}/{len(planned)}: {p.case.name}, launcher run {p.run.label}",
            "rc=0",
            f"{RUN} complete {common} {case} || rc=$?",
            'case "$rc" in',
            f'  0) echo "== {stamp} complete: {p.case.name} is skipped" ;;',
            f'  {STALE}) stop "{p.case.name}: records of other GPU code or settings (stale); '
            f'nothing is rerun or skipped" {STALE} ;;',
            f'  {HELD}) stop "{p.case.name}: its label has a held launcher run (an outcome '
            f'nobody could resolve); the launcher is not called" {HELD} ;;',
            f'  {TORN_JOURNAL}) stop "{p.case.name}: the launch journal is not whole (an '
            "unresolved torn line, or a line a published record requires); restore it by its "
            "record's repair, or scripts/launch.py --resolve-torn (docs/operations.md)"
            f'" {TORN_JOURNAL} ;;',
            f"  {NO})",
            *[f"    {line}" for line in usual],
            "    ;;",
            '  *) stop "complete failed (exit $rc)" "$rc" ;;',
            "esac",
            *comment("A technical case stops the driver here, its checkpoints kept: finish exports "
                     f"them to $KEPT, verified (exit {TECHNICAL_CASE}; {NOT_EXPORTED} when "
                     "refused or failed), then the bucket copy is uploaded and, whatever the "
                     "upload's exit, checked (kept-check: its receipt records a pending upload); "
                     f"{TECHNICAL_CASE} only when both exit 0"),
            "rc=0",
            f'routed {RUN} finish {common} {case} --kept "$KEPT" || rc=$?',
            'case "$rc" in',
            "  0) ;;",
            f'  {TECHNICAL_CASE}) INV="$RUNS_DIR/archive/inventory-kept-$(date -u +%Y%m%dT%H%M%SZ)'
            '.tsv"',
            f'     DEST="$ARTIFACT_REMOTE"/{remote}',
            "     arc=0",
            '     routed project-run "$SRC" python scripts/archive.py "$INV" "$DEST" "$KEPT" '
            f"{kept_path} || arc=$?",
            "     kc=0",
            f'     routed {RUN} kept-check {common} {case} --inventory "$INV" --remote "$DEST" '
            "|| kc=$?",
            '     if [ "$arc" -ne 0 ] || [ "$kc" -ne 0 ]; then',
            f'       stop "{p.case.name}: its kept checkpoints are not verified in the bucket '
            "(archive.py exit $arc); if kept-check published a receipt for this export, it says "
            "whether the volume copy is verified and the upload pending (pending-upload true) or "
            "the volume copy is not verified either (pending-upload false, the worse state); "
            "kept-check exit $kc; do not stop the Pod by plan; the hard deadline stays"
            f'" {NOT_EXPORTED}',
            "     fi",
            f'     stop "{p.case.name}: a technical case: its checkpoints are kept, on the volume '
            f'and in the bucket, verified" {TECHNICAL_CASE} ;;',
            f'  {NOT_EXPORTED}) stop "{p.case.name}: finish refused or failed its export (exit '
            f"{NOT_EXPORTED}): no verified volume copy; its checkpoints are still only on the "
            "container disk, which a stop loses; fix the cause finish names, then run the driver "
            "again; do not stop the Pod by plan; the hard deadline stays"
            f'" {NOT_EXPORTED} ;;',
            f'  *) stop "{p.case.name}: finish failed (exit $rc)" "$rc" ;;',
            "esac",
        ]  # fmt: skip
    lines.append(f'echo "== {stamp} every case is complete"')
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=[
            "plan",
            "driver",
            "part",
            "complete",
            "decided",
            "control-only",
            "clear-stale",
            "locate",
            "finish",
            "kept-check",
            "verdict",
            "interrupted",
            "restart-case",
            "supersede",
        ],
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--engineering", action="store_true", help="the engineering stage")
    parser.add_argument("--diagnostic-steps", metavar="T1,T2,...",
                        help="engineering: the steps of the intermediate D")  # fmt: skip
    parser.add_argument("--case", metavar="NAME", help="s<seed>/<optimizer>-<lr>")
    parser.add_argument("--part", choices=list(q.PARTS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="part: the driver's call")
    mode.add_argument("--attempt", type=int, help="part: the next attempt")
    parser.add_argument(
        "--infrastructure",
        metavar="REASON",
        help="part: the documented interruption an attempt above 1 reruns; restart-case: why "
        "the case starts again",
    )
    parser.add_argument("--technical", metavar="REASON",
                        help="interrupted: why the attempt is recorded technical")  # fmt: skip
    parser.add_argument(
        "--record",
        type=Path,
        metavar="PATH",
        help="supersede: the technical record the engineering cycle supersedes",
    )
    parser.add_argument("--diagnosis", metavar="TEXT", help="supersede: the cycle's diagnosis")
    parser.add_argument(
        "--engineering-cycle",
        action="store_true",
        help="part: a rerun after the candidate's engineering cycle",
    )
    parser.add_argument("--fix", metavar="TEXT", help="supersede: the cycle's fix")
    parser.add_argument(
        "--release",
        action="store_true",
        help="finish: delete a technical case's checkpoints once diagnosed",
    )
    parser.add_argument(
        "--kept",
        type=Path,
        metavar="DIR",
        help="finish: the folder on the Pod's volume a technical case's checkpoints are exported "
        "to (absolute); driver: the same, for each case's finish",
    )
    parser.add_argument("--kept-remote", default=KEPT_REMOTE, metavar="PREFIX",
                        help="driver: the export's place under ARTIFACT_REMOTE")  # fmt: skip
    parser.add_argument("--inventory", type=Path, metavar="FILE",
                        help="kept-check: archive.py's inventory of the upload")  # fmt: skip
    parser.add_argument("--remote", metavar="PREFIX",
                        help="kept-check: the bucket prefix the upload went to")  # fmt: skip
    parser.add_argument("--run-dir", type=Path, help="part: the launcher run")
    parser.add_argument("--checkpoints", type=Path, help="part: the harness's checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--e-eng", type=float, help="verdict: e_eng, for the warning τ")
    parser.add_argument("--runs-root", type=Path, help="RUNS_DIR")
    parser.add_argument("--out", type=Path, help="the part records (driver: the driver file)")
    parser.add_argument("--commit", help="driver: the source checkout's commit")
    parser.add_argument("--dry-run", action="store_true", help="driver: print it")
    args = parser.parse_args(argv)
    load_env()
    settings = load_settings()
    scope = q.QualificationScope.of(settings)
    steps = parse_steps(args.diagnostic_steps)
    if steps and not args.engineering:
        sys.exit("--diagnostic-steps belong to the engineering stage (--engineering)")
    name, planned, problems = campaign_cases(args.campaign, scope, args.candidate,
                                             args.engineering)  # fmt: skip
    root = args.runs_root or runs_root()
    if args.command == "plan":
        print(plan_text(name, planned, problems, settings, steps))
        if problems or not planned:
            sys.exit(1)
        return
    if args.command == "driver":
        if problems or not planned:
            sys.exit("\n".join(["The campaign is not a qualification campaign:", *problems]))
        kept = None if args.kept is None else Path(os.path.normpath(args.kept))
        if kept is None or not kept.is_absolute():
            sys.exit("driver needs --kept DIR, an absolute folder on the Pod's volume "
                     "(/workspace/kept-checkpoints): a technical case's checkpoints are exported "
                     "there")  # fmt: skip
        if kept.is_relative_to(os.path.normpath(planned[0].endpoint)):
            sys.exit(f"--kept {kept} lies under the checkpoint directory {planned[0].endpoint}, "
                     "which goes with the container disk")  # fmt: skip
        remote = PurePosixPath(args.kept_remote.strip().rstrip("/"))
        if remote.is_absolute() or ".." in remote.parts or str(remote) == ".":
            sys.exit(f"--kept-remote {args.kept_remote!r} is not a relative path under "
                     "ARTIFACT_REMOTE")  # fmt: skip
        script = driver_script(args.commit or "<SHA>",
                               datetime.now(UTC).isoformat(timespec="seconds"),
                               args.campaign.resolve(), args, planned, settings,
                               steps)  # fmt: skip
        if args.dry_run:
            print(script, end="")
            return
        if not args.commit or not args.out or args.out.exists():
            sys.exit("driver needs --commit SHA and an --out FILE that does not exist")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(script)
        print(args.out)
        return
    out = args.out or root / "qualification"
    if args.command in ("part", "verdict", "interrupted", "restart-case", "supersede") and problems:
        sys.exit("\n".join(["The campaign is not a qualification campaign:", *problems]))
    unresolved = [line for line in torn_lines(root) if not line.resolved]
    missing = missing_lines(root)
    if unresolved or missing:  # the launcher's refusal; only a repair of lost lines goes on
        repair = journal_repair(args, planned, out, root, name)
        if repair is None:
            print(TornJournalError(unresolved, missing), file=sys.stderr)
            sys.exit(TORN_JOURNAL)
        try:
            repair()
        finally:  # what it leaves: the resolution, or another record's repair, is next
            left = [line for line in torn_lines(root) if not line.resolved]
            gone = missing_lines(root)
            if left or gone:
                print(TornJournalError(left, gone), file=sys.stderr)
        return
    if args.command == "verdict":
        require_clean()
        print(verdict(args, settings, scope, planned, out, root))
        return
    found = [p for p in planned if p.case.name == args.case]
    if len(found) != 1:
        sys.exit(f"The campaign has no case {args.case!r}: {[p.case.name for p in planned]}")
    chosen = found[0]
    if args.command == "part":
        if args.part is None or not (args.resume or args.attempt is not None):
            sys.exit("part needs --part and --resume or --attempt N")
        if args.part == "control" and not chosen.control:
            sys.exit(f"{chosen.case.name} holds no same-path control")
        part(args, chosen, out)
    elif args.command in ("complete", "decided", "control-only"):
        stale = stale_records(out, chosen, code_identity.gpu_code_identity(),
                              q.settings_record(settings))  # fmt: skip
        if stale:
            print(f"{chosen.case.name}: stale, records of other GPU code or settings than this "
                  f"checkout's: {stale}", file=sys.stderr)  # fmt: skip
            sys.exit(STALE)
        held = held_runs(chosen.run, launch_records(root))  # any config hash of the label
        if held:
            print(f"{chosen.case.name}: its label {chosen.run.label} has a held launcher run, an "
                  f"outcome nobody could resolve, and nothing of it runs: {held}",
                  file=sys.stderr)  # fmt: skip
            sys.exit(HELD)
        if args.command == "control-only":  # the launcher run's directory, alone on stdout
            print(control_only(chosen, out, root, scope))
            return
        complete, decided, why = case_state(out, chosen, scope)
        print(f"{chosen.case.name}: {why}")
        if not (complete if args.command == "complete" else decided):
            sys.exit(NO)
    elif args.command == "finish":
        finish(chosen, out, root, scope, release=args.release, kept=args.kept)
    elif args.command == "kept-check":
        if args.inventory is None or not args.remote:
            sys.exit("kept-check needs --inventory FILE (scripts/archive.py's) and --remote PREFIX "
                     "(archive.py's REMOTE_PREFIX: $ARTIFACT_REMOTE/<--kept-remote>)")  # fmt: skip
        code = kept_check(chosen, out, args.inventory, args.remote.rstrip("/"))
        if code:
            sys.exit(code)
    elif args.command == "interrupted":
        if args.part is None or args.attempt is None or not args.technical:
            sys.exit("interrupted needs --part, --attempt N and --technical REASON")
        interrupted(args, chosen, out, scope)
    elif args.command == "restart-case":
        restart_case(args, chosen, out, root, scope, name)
    elif args.command == "supersede":
        supersede_case(args, chosen, planned, out, root, scope, name)
    elif args.command == "clear-stale":
        clear_stale(chosen, root)
    else:  # locate
        run_dir = settled(chosen.run, root)
        if run_dir is None:
            sys.exit(f"{chosen.run.label}: no settled launcher run (done or diverged)")
        print(run_dir)


if __name__ == "__main__":
    main()
