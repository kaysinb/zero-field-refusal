"""Hard-stop deadline for a Pod, run in its start command.

    bash -c "python3 - <<'PY' &\\n<this file>\\nPY\\nexec vllm serve ..."

The deadline is a file on the volume, ZFR_DEADLINE_FILE (default /workspace/zfr-deadline), holding
one integer, Unix seconds. The loop re-reads it every 20 s, so the deadline changes without a
container restart; ZFR_DEADLINE_EPOCH (Unix seconds, in the Pod's environment) is only the upper
limit: a file value above it is ignored (logged "zfr-deadline file above cap") and the cap applies.
Before any valid read, an absent, empty or unparsable file means the cap; after one, it keeps the
last valid deadline. Each change prints "zfr-deadline <ISO time> source <file|last|env>".
Update the file atomically (temp file in the same directory, then os.replace), from an SSH session:
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


def effective_deadline(cap: int, text: str | None, last: int | None) -> tuple[int, str]:
    """The deadline and its source: the file's integer when it is at most `cap` ("file"); `cap`
    ("env") when it is above it (logged) or when there was no valid read yet; the last valid
    deadline ("last") when the file is absent, empty or unparsable after a valid read."""
    try:
        value = int((text or "").strip())
    except ValueError:
        return (cap, "env") if last is None else (last, "last")
    if value > cap:
        print("zfr-deadline file above cap", flush=True)
        return cap, "env"
    return value, "file"


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


def main(*, clock=time.time, sleep=time.sleep, read=read_file, call=call) -> None:
    cap = int(os.environ["ZFR_DEADLINE_EPOCH"])
    path = os.environ.get("ZFR_DEADLINE_FILE", DEFAULT_FILE)
    pod, key = os.environ.get("RUNPOD_POD_ID", ""), os.environ.get("RUNPOD_API_KEY", "")
    try:
        auth: object = call(READ, pod, key)["data"]["pod"]["id"] == pod
    except Exception as error:  # noqa: BLE001 - the log line names the failure
        auth = type(error).__name__
    print("zfr-deadline", iso(cap), "pod", pod, "auth", auth, flush=True)
    last: int | None = None
    shown: tuple[int, str] | None = None
    while True:
        deadline, source = effective_deadline(cap, read(path), last)
        if source == "file":
            last = deadline
        if (deadline, source) != shown:
            shown = (deadline, source)
            print("zfr-deadline", iso(deadline), "source", source, flush=True)
        if clock() >= deadline:
            break
        sleep(POLL)
    for _ in range(5):
        try:
            print("zfr-deadline stop", call(STOP, pod, key).get("data"), flush=True)
            return
        except Exception as error:  # noqa: BLE001
            print("zfr-deadline stop failed", type(error).__name__, flush=True)
            sleep(30)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--set"] and len(args) in (2, 4):
        target = (
            args[3]
            if args[2:3] == ["--file"]
            else os.environ.get("ZFR_DEADLINE_FILE", DEFAULT_FILE)
        )
        write_deadline(target, int(args[1]))
    elif args:
        sys.exit("usage: pod_deadline_watch.py [--set EPOCH [--file PATH]]")
    else:
        main()
