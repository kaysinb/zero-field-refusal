"""Study 3's pins: the spread's static import closure and its pin, and the control list's closed
schema.

docs/study3-preregistration.md: "Sample size from validation" (the decision script is hashed
before labeling), "Judge and audit" (every labeling batch includes 40 fixed control items drawn
once by a recorded seed from study 1's judge calibration, 20 that WildGuard labeled harmful there
and 20 not, each pinned by its id, the sha256 of its prompt and of its response, and its expected
label) and "Seeds and order" (order steps 6 and 7: the choice of candidate, then validation's
spread and N fixed). Label-free: nothing here opens a response, a harm label or a judge output;
the control list holds ids, hashes and labels of study 1's calibration items, never a text.

- ``import_closure``: the files a script loads from the project, found statically from its
  imports.
- ``spread_pin``: the pin of the spread, ``scripts/study3_spread.py`` and every file of its static
  import closure under ``src/zero_field_refusal/``, each ``{"path", "sha256"}`` in the order of
  its paths, read from a commit or from the working tree.
- ``pin_problems``: a pin's form and, with ``now``, the files as they are now against it.
- ``control_list_problems`` and ``read_control_list``: the control list
  ``docs/study3-judge-controls.json`` and its problems.

**The spread pin.** The validation fixation pins the spread (``study3_versions``): its reader
requires the pin to be the import closure at the fixation's commit. The pin is checked against
the current files only when the n record is written, before any label is read and again at the
write, and the n record keeps it (``spread.code``, ``study3_decisions``); every later reader
compares the record with the fixation, record to record, and never hashes the current files.

**The control list** (``CONTROLS``) is a closed record:
- ``kind`` (``CONTROLS_KIND``), ``what`` (the fixed text ``CONTROLS_WHAT``), ``written_at`` (ISO
  8601 in UTC), ``code`` (``commit``, 40 hexadecimal digits) and ``seed`` (the draw's recorded
  seed, an integer in [0, 2³²));
- ``source``: ``sample`` and ``labels``, each ``{"path", "sha256"}`` with its path relative to
  RUNS_DIR, study 1's ``sample.jsonl`` and ``judge-wildguard.jsonl`` of one judge-calibration
  folder, and ``archive``, the bucket URI it is archived at (``gs://``, no space);
- ``judge``: the identity of the judge that labeled study 1's items (``judge.IDENTITY_FIELDS``);
- ``items``: exactly 40, in the order of their ids, each once; each ``{"id", "prompt_sha256",
  "response_sha256", "expected", "label"}``: an id ``c`` and 12 hexadecimal digits, the two
  sha256s, the label WildGuard gave in study 1 (``judge.LABELS``) and the expected label,
  ``harmful`` exactly when that label is ``harmful_compliance`` and ``not_harmful`` otherwise; 20
  of each.

**Rulings** where the registration leaves a detail open:
- *What is pinned.* "The decision script is hashed" covers what it runs: the script and every
  file of the project it imports, directly or through another, so that a change of a module the
  spread reads changes the pin.
- *The static closure contains the dynamic one.* Every ``import`` and ``from … import`` anywhere
  in a file is followed (in a function, in ``try``, under ``if TYPE_CHECKING``); every package's
  ``__init__.py`` on the way, which Python runs, is added and followed; ``from M import n`` names
  the module ``M.n`` when it has a file, else the module ``M``, which must then bind ``n`` at its
  top level. A package (``<m>/__init__.py``) is taken before a module (``<m>.py``), as Python's
  path finder takes it. What cannot be shown fails closed (``ValueError``, naming it): a project
  module with no file, a name a package does not bind, a relative import outside the package or
  beyond its top, a file that does not parse. Imports outside the project are not followed, nor
  imports by a name given at run time (the spread makes none).
- *Checked now* means the files and their closure: a pinned file whose sha256 differs, one that
  is missing, and a closure that gained or lost a file are each named by path, never by content.
- *The control list* is a file of the checkout, committed before the validation fixation, which
  pins it by its sha256 at the fixation's commit; its schema is closed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn

from ..judge import IDENTITY_FIELDS
from ..judge import LABELS as JUDGE_LABELS
from .study3_records import SEALED_WHY, sealed_why, under_sealed

SPREAD_SCRIPT = "scripts/study3_spread.py"
CONTROLS = "docs/study3-judge-controls.json"
PACKAGE_DIR = "src/zero_field_refusal"
PROJECT = "zero_field_refusal"  # the project's import name: src/zero_field_refusal/
CONTROL_ITEMS = 40
PER_CLASS = 20
CONTROLS_KIND = "study3-judge-controls"
EXPECTED = ("harmful", "not_harmful")
HARMFUL = "harmful_compliance"  # the WildGuard label that makes a control item harmful
CONTROLS_WHAT = ("The 40 control items of study 3's judge: immutable archived prompt-response "
                 "pairs from study 1's judge calibration, drawn once by the recorded seed, 20 that "
                 "WildGuard labeled harmful there and 20 not, each pinned by its id, the sha256 of "
                 "its prompt and of its response, and its expected label.")  # fmt: skip
CONTROLS_FIELDS = ("kind", "what", "written_at", "code", "seed", "source", "judge", "items")
ITEM_FIELDS = ("id", "prompt_sha256", "response_sha256", "expected", "label")
SOURCE_FILES = {"sample": "sample.jsonl", "labels": "judge-wildguard.jsonl"}  # study 1's files
SEED_LIMIT = 2**32
_ID = re.compile(r"c[0-9a-f]{12}")
_SHA256, _COMMIT = re.compile(r"[0-9a-f]{64}"), re.compile(r"[0-9a-f]{40}")
# A repository id or revision (the judge's model, its revision, its prompt version)
_PART = r"[A-Za-z0-9][A-Za-z0-9._@+-]{0,127}"
_NAME = re.compile(rf"{_PART}(?:/{_PART}){{0,3}}")
_ARCHIVE = re.compile(r"gs://[^\s/]+(?:/[^\s/]+)*/?")

Read = Callable[[str], bytes | None]  # a repository-relative path -> its bytes, None: no file


# ------------------------------------------------------------------ the import closure


def import_closure(entry: str, read: Read) -> list[str]:
    """The files ``entry`` (a repository-relative path) loads from the project, found from its
    imports (module docstring, "Rulings"): sorted, each once, repository-relative, ``entry``
    included. ``read`` gives a file's bytes, or None when there is none. ``ValueError``, naming
    it, for what cannot be shown: ``entry`` or a project module with no file, a name a package
    does not bind, a relative import outside the package or beyond its top, a file that does not
    parse."""
    cache: dict[str, bytes | None] = {}

    def get(path: str) -> bytes | None:
        if path not in cache:
            cache[path] = read(path)
        return cache[path]

    found: set[str] = set()
    queue = [entry]
    while queue:
        path = queue.pop()
        if path in found:
            continue
        data = get(path)
        if data is None:
            raise ValueError(f"{path} cannot be read: the import closure of {entry} cannot be "
                             "shown")  # fmt: skip
        found.add(path)
        tree = _parse(path, data)
        package = _package_of(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _in_project(alias.name):
                        queue += _module_files(alias.name, get, path)
            elif isinstance(node, ast.ImportFrom):
                queue += _from_files(node, package, get, path)
    return sorted(found)


def _parse(path: str, data: bytes) -> ast.Module:
    try:
        return ast.parse(data, filename=path)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{path} does not parse ({type(exc).__name__}): its imports cannot be "
                         "shown") from exc  # fmt: skip


def _in_project(name: str) -> bool:
    return name == PROJECT or name.startswith(f"{PROJECT}.")


def _package_of(path: str) -> str | None:
    """The package a file of the project resolves its relative imports against; None for a file
    outside ``PACKAGE_DIR``."""
    if not (path.startswith(f"{PACKAGE_DIR}/") and path.endswith(".py")):
        return None
    dotted = path[len("src/") : -len(".py")].replace("/", ".")
    if dotted.endswith(".__init__"):
        return dotted.removesuffix(".__init__")
    return dotted.rpartition(".")[0]


def _module_file(name: str, get: Read) -> str | None:
    """The file of a project module: its package's ``__init__.py``, else its ``.py``."""
    base = f"src/{name.replace('.', '/')}"
    for path in (f"{base}/__init__.py", f"{base}.py"):
        if get(path) is not None:
            return path
    return None


def _module_files(name: str, get: Read, where: str) -> list[str]:
    """A project module's file and every existing ``__init__.py`` of the packages above it;
    ``ValueError`` for a module with no file."""
    own = _module_file(name, get)
    if own is None:
        raise ValueError(f"{where} imports {name}, a project module with no file under "
                         f"{PACKAGE_DIR}/: the import closure cannot be shown")  # fmt: skip
    parts = name.split(".")
    parents = [f"src/{'/'.join(parts[:i])}/__init__.py" for i in range(1, len(parts))]
    return [own, *(p for p in parents if get(p) is not None)]


def _from_files(node: ast.ImportFrom, package: str | None, get: Read, where: str) -> list[str]:
    """The files of ``from M import n, …``: ``M.n`` where it has a file, else ``M``, which must
    bind ``n``."""
    if node.level:
        if package is None:
            raise ValueError(f"{where}: a relative import (level {node.level}) outside "
                             f"{PACKAGE_DIR}/ cannot be resolved")  # fmt: skip
        parts = package.split(".")
        if node.level - 1 >= len(parts):
            raise ValueError(f"{where}: a relative import of level {node.level} reaches beyond "
                             f"the package {PROJECT}")  # fmt: skip
        module = ".".join([*parts[: len(parts) - node.level + 1], *filter(None, [node.module])])
    else:
        module = node.module or ""
        if not _in_project(module):
            return []
    files: list[str] = []
    for alias in node.names:
        sub = f"{module}.{alias.name}"
        if alias.name != "*" and _module_file(sub, get) is not None:
            files += _module_files(sub, get, where)
            continue
        own = _module_file(module, get)
        if own is None or (alias.name != "*" and own.endswith("/__init__.py")
                           and alias.name not in _bound(_parse(own, get(own)))):  # fmt: skip
            raise ValueError(f"{where} imports {sub}, which is no project module with a file "
                             f"under {PACKAGE_DIR}/ and no name {module} binds: the import "
                             "closure cannot be shown")  # fmt: skip
        files += _module_files(module, get, where)
    return files


def _bound(tree: ast.Module) -> set[str]:
    """The names a module binds at its top level (in its ``if``, ``try`` and ``with`` blocks too):
    definitions, assignments and imports."""
    names: set[str] = set()

    def visit(body: Sequence[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.Import):
                names.update(a.asname or a.name.partition(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.update(a.asname or a.name for a in node.names if a.name != "*")
            elif isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
            else:
                for field in ("body", "orelse", "finalbody"):
                    block = getattr(node, field, None)
                    if isinstance(block, list):
                        visit(block)
                for handler in getattr(node, "handlers", None) or []:
                    visit(handler.body)

    visit(tree.body)
    return names


# ------------------------------------------------------------------ the spread's pin


def spread_pin(repo: Path, commit: str | None = None) -> list[dict[str, str]]:
    """The pin of the spread in the checkout ``repo``: ``SPREAD_SCRIPT`` and every file of its
    import closure, each ``{"path", "sha256"}``, in the order of their paths; read at ``commit``
    (``git cat-file``) when one is given, else from the working tree. ``ValueError`` when the
    closure cannot be shown."""
    repo = Path(repo)
    source = _in_tree(repo) if commit is None else _at_commit(repo, commit)
    cache: dict[str, bytes | None] = {}

    def read(path: str) -> bytes | None:
        if path not in cache:
            cache[path] = source(path)
        return cache[path]

    return [{"path": path, "sha256": hashlib.sha256(read(path)).hexdigest()}
            for path in import_closure(SPREAD_SCRIPT, read)]  # fmt: skip


def _in_tree(repo: Path) -> Read:
    """A file of the working tree; None when there is none (a module looked for under another
    name: ``judge/__init__.py`` beside ``judge.py``). A file that resolves under a sealed folder
    (a link into a run's, say) is refused unopened."""

    def read(path: str) -> bytes | None:
        file = repo / path
        if not file.is_file():
            return None
        if under_sealed(file, repo):
            raise ValueError(f"{path} {sealed_why(file, repo, f'resolves {SEALED_WHY}')}: never "
                             "opened")  # fmt: skip
        return file.read_bytes()

    return read


def _at_commit(repo: Path, commit: str) -> Read:
    """A file at ``commit``; None when the commit does not hold it as a file."""

    def read(path: str) -> bytes | None:
        if commit.startswith("-"):
            return None
        try:
            return subprocess.run(["git", "cat-file", "blob", f"{commit}:{path}"], cwd=repo,
                                  capture_output=True, check=True).stdout  # fmt: skip
        except (OSError, subprocess.CalledProcessError):
            return None

    return read


def pin_problems(pin: Any, repo: Path, *, now: bool = True) -> list[str]:
    """The problems of a spread pin: its form (a non-empty list of ``{"path", "sha256"}`` in the
    order of its paths, each once, naming ``SPREAD_SCRIPT``, every other path a ``.py`` file under
    ``PACKAGE_DIR``, each sha256 64 hexadecimal digits); with ``now``, the files of the checkout
    ``repo`` as they are now against it, and their import closure against its paths. Problems
    name paths, never content."""
    problems = _pin_form(pin)
    if problems or not now:
        return problems
    read = _in_tree(Path(repo))
    for entry in pin:
        path, pinned = entry["path"], entry["sha256"]
        try:
            data = read(path)
        except (OSError, ValueError) as exc:
            problems.append(f"{path} cannot be read now ({exc})")
            continue
        if data is None:
            problems.append(f"{path} is missing now, pinned with sha256 {pinned}")
        elif (digest := hashlib.sha256(data).hexdigest()) != pinned:
            problems.append(f"{path} has sha256 {digest} now, not the pinned {pinned}")
    try:
        closure = set(import_closure(SPREAD_SCRIPT, read))
    except (OSError, ValueError) as exc:
        problems.append(f"the import closure of {SPREAD_SCRIPT} cannot be shown now ({exc})")
        return problems
    named = {entry["path"] for entry in pin}
    if closure - named:
        problems.append(f"the import closure of {SPREAD_SCRIPT} now holds "
                        f"{sorted(closure - named)}, which the pin does not name")  # fmt: skip
    if named - closure:
        problems.append(f"the import closure of {SPREAD_SCRIPT} no longer holds "
                        f"{sorted(named - closure)}, which the pin names")  # fmt: skip
    return problems


def _pin_form(pin: Any) -> list[str]:
    if not isinstance(pin, list) or not pin:
        return [f"is {type(pin).__name__ if pin else pin!r}, not a non-empty list of "
                "{path, sha256}"]  # fmt: skip
    problems, paths = [], []
    for i, entry in enumerate(pin):
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            keys = sorted(map(str, entry)) if isinstance(entry, dict) else type(entry).__name__
            problems.append(f"[{i}]: is {keys}, not an object of exactly path and sha256")
            continue
        path, digest = entry["path"], entry["sha256"]
        if not _pinned_path(path):
            problems.append(f"[{i}].path: is {path!r}, not {SPREAD_SCRIPT} or a .py file under "
                            f"{PACKAGE_DIR}/")  # fmt: skip
        if not (isinstance(digest, str) and _SHA256.fullmatch(digest)):
            problems.append(f"[{i}].sha256: is {digest!r}, not a sha256 (64 hexadecimal digits)")
        paths.append(path)
    if all(isinstance(p, str) for p in paths) and paths != sorted(set(paths)):
        problems.append("is not in the order of its paths, each once")
    if SPREAD_SCRIPT not in paths:
        problems.append(f"does not name {SPREAD_SCRIPT}")
    return problems


def _pinned_path(path: Any) -> bool:
    """``SPREAD_SCRIPT``, or a plain repository-relative ``.py`` path under ``PACKAGE_DIR``."""
    if path == SPREAD_SCRIPT:
        return True
    return (isinstance(path, str) and path.startswith(f"{PACKAGE_DIR}/") and path.endswith(".py")
            and "\\" not in path
            and all(part not in ("", ".", "..") for part in path.split("/")))  # fmt: skip


# ------------------------------------------------------------------ the control list


class _Form:
    """The problems of a closed record, each ``where: what``."""

    def __init__(self) -> None:
        self.problems: list[str] = []

    def add(self, where: str, text: str) -> None:
        self.problems.append(f"{where}: {text}" if where else text)

    def object(self, where: str, value: Any, fields: Sequence[str]) -> dict[str, Any] | None:
        """``value`` as an object of exactly ``fields``: None when it is not one or misses one."""
        if not isinstance(value, dict):
            self.add(where, f"is not an object ({type(value).__name__})")
            return None
        missing = [key for key in fields if key not in value]
        for key in missing:
            self.add(where, f"has no field `{key}`")
        for key in value:
            if key not in fields:
                self.add(where, f"`{key}` is not a field of this record")
        return None if missing else value

    def pattern(self, where: str, value: Any, pattern: re.Pattern, what: str) -> bool:
        ok = isinstance(value, str) and pattern.fullmatch(value) is not None
        if not ok:
            self.add(where, f"is {value!r}, not {what}")
        return ok


def control_list_problems(record: Any) -> list[str]:
    """Every problem of a control list (module docstring, "The control list"); none: it is one."""
    form = _Form()
    top = form.object("", record, CONTROLS_FIELDS)
    if top is None:
        return form.problems
    if top["kind"] != CONTROLS_KIND:
        form.add("kind", f"is {top['kind']!r}, not {CONTROLS_KIND!r}")
    if top["what"] != CONTROLS_WHAT:
        form.add("what", "is not the control list's fixed text")
    _written_at(form, top["written_at"])
    code = form.object("code", top["code"], ("commit",))
    if code is not None:
        form.pattern("code.commit", code["commit"], _COMMIT, "a commit (40 hexadecimal digits)")
    if not (type(top["seed"]) is int and 0 <= top["seed"] < SEED_LIMIT):
        form.add("seed", f"is {top['seed']!r}, not a seed (an integer in [0, 2³²))")
    _source(form, top["source"])
    judge = form.object("judge", top["judge"], IDENTITY_FIELDS)
    for key in IDENTITY_FIELDS if judge is not None else ():
        if judge[key] is not None:
            form.pattern(f"judge.{key}", judge[key], _NAME, "a repository id or revision")
    _items(form, top["items"])
    return form.problems


def _written_at(form: _Form, value: Any) -> None:
    try:
        when = datetime.fromisoformat(value)
        ok = when.tzinfo is not None and when.utcoffset() == timedelta(0)
    except (TypeError, ValueError):
        ok = False
    if not ok:
        form.add("written_at", f"is {value!r}, not an ISO 8601 time in UTC (Z or +00:00)")


def _source(form: _Form, value: Any) -> None:
    """Study 1's sample and its WildGuard labels, in one folder under RUNS_DIR, and the bucket URI
    they are archived at."""
    source = form.object("source", value, ("sample", "labels", "archive"))
    if source is None:
        return
    folders = set()
    for key, name in SOURCE_FILES.items():
        entry = form.object(f"source.{key}", source[key], ("path", "sha256"))
        if entry is None:
            continue
        path = entry["path"]
        parts = path.split("/") if isinstance(path, str) else []
        if not (parts and parts[-1] == name and "\\" not in path
                and all(part not in ("", ".", "..") for part in parts)):  # fmt: skip
            form.add(f"source.{key}.path", f"is {path!r}, not study 1's {name} by a plain path "
                     "relative to RUNS_DIR")  # fmt: skip
        else:
            folders.add("/".join(parts[:-1]))
        form.pattern(f"source.{key}.sha256", entry["sha256"], _SHA256, "a sha256")
    if len(folders) > 1:
        form.add("source", f"names the files of two folders {sorted(folders)}, not one "
                 "judge-calibration folder")  # fmt: skip
    form.pattern("source.archive", source["archive"], _ARCHIVE, "a bucket URI (gs://, no space)")


def _items(form: _Form, items: Any) -> None:
    """The 40 control items, in the order of their ids, each once; 20 expected harmful and 20
    not, each expected label the one its study-1 label gives."""
    if not isinstance(items, list):
        form.add("items", f"is not a list of the {CONTROL_ITEMS} control items")
        return
    if len(items) != CONTROL_ITEMS:
        form.add("items", f"has {len(items)} items, not {CONTROL_ITEMS}")
    ids, counts = [], Counter()
    for i, value in enumerate(items):
        where = f"items[{i}]"
        item = form.object(where, value, ITEM_FIELDS)
        if item is None:
            continue
        form.pattern(f"{where}.id", item["id"], _ID, "an id (c and 12 hexadecimal digits)")
        for key in ("prompt_sha256", "response_sha256"):
            form.pattern(f"{where}.{key}", item[key], _SHA256, "a sha256")
        label, expected = item["label"], item["expected"]
        if label not in JUDGE_LABELS:
            form.add(f"{where}.label", f"is {label!r}, not one of {list(JUDGE_LABELS)}")
        if expected not in EXPECTED:
            form.add(f"{where}.expected", f"is {expected!r}, not one of {list(EXPECTED)}")
        else:
            counts[expected] += 1
            given = EXPECTED[0] if label == HARMFUL else EXPECTED[1]
            if label in JUDGE_LABELS and expected != given:
                form.add(f"{where}.expected", f"is {expected!r}, but its study-1 label {label!r} "
                         f"makes it {given!r}")  # fmt: skip
        ids.append(item["id"])
    if all(isinstance(i, str) for i in ids) and ids != sorted(set(ids)):
        form.add("items", "is not in the order of its ids, each once")
    if [counts[e] for e in EXPECTED] != [PER_CLASS, PER_CLASS]:
        form.add("items", f"has {counts['harmful']} expected harmful and {counts['not_harmful']} "
                 f"expected not_harmful, not {PER_CLASS} of each")  # fmt: skip


def read_control_list(repo: Path, data: bytes | None = None) -> tuple[dict | None, list[str]]:
    """The control list (``CONTROLS`` in the checkout ``repo``, or ``data``, its bytes at a
    commit) and its problems: ``(record, [])``, or ``(None, problems)``. A missing file, one that
    resolves under a sealed folder (never opened), bytes that are no strict JSON object (a
    repeated key, NaN) and every schema problem are problems."""
    if data is None:
        path = Path(repo) / CONTROLS
        if not (path.exists() or path.is_symlink()):
            return None, [f"{CONTROLS} is missing"]
        if under_sealed(path, Path(repo)):
            why = sealed_why(path, Path(repo), f"resolves {SEALED_WHY}")
            return None, [f"{CONTROLS} {why}: never opened"]
        try:
            data = path.read_bytes()
        except OSError as exc:
            return None, [f"{CONTROLS} is unreadable ({type(exc).__name__})"]
    try:
        record = json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                            parse_constant=_no_constant)  # fmt: skip
    except ValueError as exc:  # UnicodeDecodeError included
        return None, [f"{CONTROLS} is no strict JSON record ({type(exc).__name__})"]
    problems = control_list_problems(record)
    return (None, problems) if problems else (record, [])


def _unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("a repeated key")
    return dict(pairs)


def _no_constant(name: str) -> NoReturn:
    raise ValueError(f"{name} is no JSON number")
