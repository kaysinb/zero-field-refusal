"""Hard-stop deadline for a Pod without SSH (the judge server), run in its start command.

    bash -c "python3 - <<'PY' &\\n<this file>\\nPY\\nexec vllm serve ..."

Reads ZFR_DEADLINE_EPOCH (Unix seconds) from the Pod's environment. At start it checks the Pod's
own key with a read of its Pod (GraphQL, Bearer RUNPOD_API_KEY) and logs one line,
"zfr-deadline <deadline> pod <id> auth True"; at the deadline it stops the Pod (podStop, as the
in-Pod watcher of the training Pods), retrying up to five times. It never prints the key. To
extend the deadline, change ZFR_DEADLINE_EPOCH with update-pod (the container restarts). Uses the
standard library only. The procedure: docs/operations.md, "Judge server".
"""

import json
import os
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


def main() -> None:
    deadline = int(os.environ["ZFR_DEADLINE_EPOCH"])
    pod, key = os.environ.get("RUNPOD_POD_ID", ""), os.environ.get("RUNPOD_API_KEY", "")
    try:
        auth: object = call(READ, pod, key)["data"]["pod"]["id"] == pod
    except Exception as error:  # noqa: BLE001 - the log line names the failure
        auth = type(error).__name__
    at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline))
    print("zfr-deadline", at, "pod", pod, "auth", auth, flush=True)
    while time.time() < deadline:
        time.sleep(20)
    for _ in range(5):
        try:
            print("zfr-deadline stop", call(STOP, pod, key).get("data"), flush=True)
            return
        except Exception as error:  # noqa: BLE001
            print("zfr-deadline stop failed", type(error).__name__, flush=True)
            time.sleep(30)


if __name__ == "__main__":
    main()
