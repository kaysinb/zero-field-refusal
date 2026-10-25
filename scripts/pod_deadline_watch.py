"""Hard-stop deadline for a Pod, run in its start command.

    bash -c "python3 - <<'PY' &\\n<this file>\\nPY\\nexec vllm serve ..."

The deadline is a file on the volume, ZFR_DEADLINE_FILE (default /workspace/zfr-deadline), holding
one integer, Unix seconds (ASCII digits only). The loop re-reads it every 20 s, so the deadline
changes without a container restart; ZFR_DEADLINE_EPOCH (Unix seconds, in the Pod's environment)
is only the upper limit: a file value above it is ignored (logged "zfr-deadline file above cap")
and the cap applies. A file value already past when the watcher started (a deadline left from an
earlier start) is ignored too (logged "zfr-deadline file stale"), and the cap applies; each of
these states is logged when it begins. Before any valid read, an absent, empty or unparsable file
means the cap; after one, it keeps the last valid deadline. Each change prints
"zfr-deadline <ISO time> source <file|last|env>". When the deadline fires, the file is renamed to
<file>.fired-<epoch>, so that the next start cannot read it. Update the file atomically (temp file
in the same directory, then os.replace), from an SSH session:
`python3 scripts/pod_deadline_watch.py --set EPOCH [--file PATH]`.

At start it checks the Pod's own key with a read of its Pod (GraphQL, Bearer RUNPOD_API_KEY) and
logs one line, "zfr-deadline <cap> pod <id> auth True"; at the deadline it stops the Pod (podStop,
as the in-Pod watcher of the training Pods), retrying up to five times. It never prints the key.
Uses the standard library only. The procedure: docs/operations.md, "Deadlines between campaigns".
"""

import json
import os
import sys
import tempfile
import time
import urllib.request

API = "https://api.runpod.io/graphql"
READ = "query p($i: String!) { pod(input: {podId: $i}) { id desiredStatus } }"
STOP = "mutation s($i: String!) { podStop(input: {podId: $i}) { id desiredStatus } }"


def call(query: str, pod: str, key: str) -> dict:
    request = urllib.request.Request(
        API,
        data=json.dumps({"query": query, "variables": {"i": pod}}).encode(),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "User-Agent": "zfr-deadline/1.0",
        },  # fmt: skip
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read())


DEFAULT_FILE = "/workspace/zfr-deadline"
POLL = 20


def iso(epoch: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def parse_epoch(text: str | None) -> int | None:
    """The integer of ASCII digits the text holds (surrounding whitespace aside), or None."""
    stripped = (text or "").strip()
    return int(stripped) if stripped.isascii() and stripped.isdigit() else None


def effective_deadline(
    cap: int, text: str | None, last: int | None, started: int | None = None
) -> tuple[int, str, str | None]:
    """The deadline, its source and why the file was ignored: the file's integer when it is at
    most `cap` and after `started` ("file"); `cap` ("env") when it is above the cap ("above
    cap"), at or before `started` ("stale"), or when there was no valid read yet; the last valid
    deadline ("last") when the file is absent, empty or unparsable after a valid read."""
    value = parse_epoch(text)
    if value is None:
        return ((cap, "env") if last is None else (last, "last")) + (None,)
    if value > cap:
        return cap, "env", "above cap"
    if started is not None and value <= started:
        return cap, "env", "stale"
    return value, "file", None


def read_file(path: str) -> str | None:
    try:
        with open(path) as handle:
            return handle.read()
    except OSError:
        return None


def write_deadline(path: str, epoch: int) -> None:
    """Replace the deadline file atomically: a temp file in its directory, then os.replace."""
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".zfr-deadline-")
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(f"{int(epoch)}\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def retire(path: str, epoch: int) -> None:
    """Rename the deadline file once it fired (<file>.fired-<epoch>), so no later start reads it."""
    try:
        os.replace(path, f"{path}.fired-{epoch}")
        print("zfr-deadline file retired", f"{path}.fired-{epoch}", flush=True)
    except FileNotFoundError:
        pass
    except OSError as error:
        print("zfr-deadline file not retired", type(error).__name__, flush=True)


def main(*, clock=time.time, sleep=time.sleep, read=read_file, call=call) -> None:
    cap = int(os.environ["ZFR_DEADLINE_EPOCH"])
    started = int(clock())
    path = os.environ.get("ZFR_DEADLINE_FILE", DEFAULT_FILE)
    pod, key = os.environ.get("RUNPOD_POD_ID", ""), os.environ.get("RUNPOD_API_KEY", "")
    try:
        auth: object = call(READ, pod, key)["data"]["pod"]["id"] == pod
    except Exception as error:  # noqa: BLE001 - the log line names the failure
        auth = type(error).__name__
    print("zfr-deadline", iso(cap), "pod", pod, "auth", auth, flush=True)
    last: int | None = None
    shown: tuple[int, str] | None = None
    ignored: str | None = None
    while True:
        deadline, source, why = effective_deadline(cap, read(path), last, started)
        if why != ignored:  # each state once, when it begins
            ignored = why
            if why is not None:
                print("zfr-deadline file", why, flush=True)
        if source == "file":
            last = deadline
        if (deadline, source) != shown:
            shown = (deadline, source)
            print("zfr-deadline", iso(deadline), "source", source, flush=True)
        if clock() >= deadline:
            break
        sleep(POLL)
    retire(path, int(clock()))
    for _ in range(5):
        try:
            print("zfr-deadline stop", call(STOP, pod, key).get("data"), flush=True)
            return
        except Exception as error:  # noqa: BLE001
            print("zfr-deadline stop failed", type(error).__name__, flush=True)
            sleep(30)


USAGE = "usage: pod_deadline_watch.py [--set EPOCH [--file PATH]]"


def cli(args: list[str]) -> None:
    """The watcher without arguments; `--set EPOCH [--file PATH]` writes the deadline file."""
    if not args:
        main()
        return
    epoch = parse_epoch(args[1]) if len(args) >= 2 else None
    if args[0] != "--set" or epoch is None or len(args) not in (2, 4):
        sys.exit(USAGE)
    if len(args) == 4 and args[2] != "--file":
        sys.exit(USAGE)
    default = os.environ.get("ZFR_DEADLINE_FILE", DEFAULT_FILE)
    write_deadline(args[3] if len(args) == 4 else default, epoch)


if __name__ == "__main__":
    cli(sys.argv[1:])
