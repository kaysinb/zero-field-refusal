"""Offline documentation checks and advisory, file-level change impact.

Only --write-index writes: the marked region of docs/README.md. Markdown is
parsed, not executed. Configuration examples are loaded, never trained.
"""

from __future__ import annotations

# Config imports must not create bytecode during a read-only check.
# ruff: noqa: E402
import sys

sys.dont_write_bytecode = True

import argparse
import fnmatch
import importlib
import os
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
from omegaconf import OmegaConf

BEGIN = "<!-- BEGIN GENERATED DOC MAP -->"
END = "<!-- END GENERATED DOC MAP -->"
PARSER = MarkdownIt("commonmark")


class HtmlRefs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("name"):
            self.ids.add(attrs["name"])
        key = "href" if tag == "a" else "src" if tag == "img" else None
        if key and attrs.get(key):
            self.links.append(attrs[key])


def markdown_info(text):
    """CommonMark links and GitHub-style heading IDs; fenced examples are ignored."""
    tokens = PARSER.parse(text)
    anchors, links, seen = set(), [], {}
    for i, token in enumerate(tokens):
        line = (token.map or [0])[0] + 1
        if token.type == "heading_open":
            children = tokens[i + 1].children or []
            title = "".join(t.content for t in children if t.type in {"text", "code_inline"})
            slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            anchors.add(f"{slug}-{count}" if count else slug)
        for item in [token, *(token.children or [])]:
            if item.type == "link_open":
                links.append((item.attrGet("href"), line))
            elif item.type == "image":
                links.append((item.attrGet("src"), line))
            elif item.type in {"html_block", "html_inline"}:
                html = HtmlRefs()
                html.feed(item.content)
                anchors.update(html.ids)
                links.extend((ref, line) for ref in html.links)
    return anchors, links


def repo_path(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"Expected a repository-relative path: {name!r}")
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Path must stay within the repository: {name!r}")
    target = (root / name).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Path escapes the repository: {name!r}")
    return target


def load_map(root):
    data = OmegaConf.to_container(OmegaConf.load(root / "docs/map.yaml"), resolve=False)
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("docs/map.yaml must be a mapping with version: 1")
    extra = set(data) - {
        "version",
        "package",
        "documents",
        "topics",
        "config_examples",
        "historical_docs",
    }
    if extra:
        raise ValueError(f"Unknown map fields: {sorted(extra)}")
    for key in ["documents", "topics", "config_examples", "historical_docs"]:
        if not isinstance(data.get(key), list):
            raise ValueError(f"docs/map.yaml: {key} must be a list")
    if not re.fullmatch(r"[A-Za-z_]\w*", str(data.get("package", ""))):
        raise ValueError("docs/map.yaml: package must name the local Python package")
    paths, ids = set(), set()
    for doc in data["documents"]:
        if not isinstance(doc, dict) or not all(
            isinstance(doc.get(k), str) and doc[k] for k in ["path", "purpose"]
        ):
            raise ValueError("Every document needs path and purpose")
        if set(doc) - {"path", "purpose"}:
            raise ValueError(f"Unknown document fields: {doc['path']}")
        repo_path(root, doc["path"])
        if doc["path"] in paths:
            raise ValueError(f"Duplicate document: {doc['path']}")
        paths.add(doc["path"])
    for topic in data["topics"]:
        if not isinstance(topic, dict) or not all(
            isinstance(topic.get(k), str) and topic[k] for k in ["id", "title", "source"]
        ):
            raise ValueError("Every topic needs id, title and source")
        if topic["id"] in ids:
            raise ValueError(f"Duplicate topic id: {topic['id']}")
        if set(topic) - {"id", "title", "source", "review", "watch"}:
            raise ValueError(f"Unknown topic fields: {topic['id']}")
        if "review" not in topic:
            raise ValueError(
                f"Topic {topic['id']}: review must be explicit (an empty list is allowed)"
            )
        ids.add(topic["id"])
        for key in ["review", "watch"]:
            if not isinstance(topic.get(key, []), list) or not all(
                isinstance(v, str) and v for v in topic.get(key, [])
            ):
                raise ValueError(f"Topic {topic['id']}: {key} must be a list of strings")
        for ref in [topic["source"], *topic.get("review", [])]:
            parts = urlsplit(ref)
            if parts.scheme or parts.netloc or parts.query:
                raise ValueError(f"Map references must be local, without queries: {ref}")
            repo_path(root, unquote(parts.path))
        for pattern in topic.get("watch", []):
            repo_path(root, pattern)
    for pattern in data["historical_docs"]:
        repo_path(root, pattern)
    for example in data["config_examples"]:
        if not isinstance(example, dict) or not isinstance(example.get("config"), str):
            raise ValueError("Every config example needs a config name")
        if set(example) - {"config", "overrides", "allow_missing"}:
            raise ValueError("Unknown config example fields")
        repo_path(root / "configs", example["config"])
        if not isinstance(example.get("overrides", []), list) or not all(
            isinstance(v, str) for v in example.get("overrides", [])
        ):
            raise ValueError("Config overrides must be a list of strings")
        if not isinstance(example.get("allow_missing", False), bool):
            raise ValueError("allow_missing must be true or false")
    return data


def index_content(data):
    def link(ref):
        path, sep, fragment = ref.partition("#")
        relative = os.path.relpath(path, "docs").replace(os.sep, "/")
        target = relative + (sep + fragment if sep else "")
        label = ref.replace("|", "\\|")
        return f"[{label}]({target})"

    def cell(text):
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [BEGIN, "", "| Document | Responsibility |", "|---|---|"]
    for doc in data["documents"]:
        lines.append(f"| {link(doc['path'])} | {cell(doc['purpose'])} |")
    lines += ["", "| Topic | Authoritative source | Review after changes |", "|---|---|---|"]
    for topic in data["topics"]:
        review = "; ".join(link(r) for r in topic.get("review", [])) or "—"
        lines.append(f"| {cell(topic['title'])} | {link(topic['source'])} | {review} |")
    lines += ["", END]
    return "\n".join(lines)


class Check:
    def __init__(self, root):
        self.root = root.resolve()
        self.errors = []
        self.cache = {}
        self.ignore_cache = {}
        self.links = 0
        self.configs = 0

    def ignored(self, path):
        relative = path.relative_to(self.root).as_posix()
        if relative not in self.ignore_cache:
            try:
                result = subprocess.run(
                    [
                        "git",
                        "--no-optional-locks",
                        "-C",
                        str(self.root),
                        "check-ignore",
                        "--quiet",
                        "--",
                        relative,
                    ],
                    capture_output=True,
                )
                self.ignore_cache[relative] = result.returncode == 0
            except FileNotFoundError:
                self.ignore_cache[relative] = False
        return self.ignore_cache[relative]

    def document(self, path):
        if path not in self.cache:
            self.cache[path] = markdown_info(path.read_text(encoding="utf-8"))
        return self.cache[path]

    def reference(self, origin, ref, where):
        parts = urlsplit(ref)
        if parts.scheme or parts.netloc:
            return  # Offline checker: never fetch external links.
        self.links += 1
        raw_path = unquote(parts.path)
        target = (
            (self.root / raw_path.lstrip("/"))
            if raw_path.startswith("/")
            else (origin.parent / raw_path)
            if raw_path
            else origin
        ).resolve()
        if not target.is_relative_to(self.root):
            self.errors.append(f"{where}: link escapes repository: {ref}")
        elif self.ignored(target):
            self.errors.append(f"{where}: Git-ignored target is not a public dependency: {ref}")
        elif not target.exists():
            self.errors.append(f"{where}: missing target: {ref}")
        elif (
            parts.fragment
            and target.is_file()
            and target.suffix.lower() == ".md"
            and unquote(parts.fragment) not in self.document(target)[0]
        ):
            self.errors.append(f"{where}: missing anchor: {ref}")

    def validate(self, data, write_index=False):
        registered = {d["path"] for d in data["documents"]}
        origin = self.root / "map-origin.md"  # Map paths start at repository root.
        for doc in data["documents"]:
            self.reference(origin, doc["path"], "docs/map.yaml")
        for topic in data["topics"]:
            for ref in [topic["source"], *topic.get("review", [])]:
                self.reference(origin, ref, f"docs/map.yaml [{topic['id']}]")

        index = repo_path(self.root, "docs/README.md")
        if not index.is_file():
            self.errors.append("Missing docs/README.md")
        else:
            current = index.read_text(encoding="utf-8")
            if (
                current.count(BEGIN) != 1
                or current.count(END) != 1
                or current.index(BEGIN) > current.index(END)
            ):
                self.errors.append(
                    "docs/README.md must contain one ordered generated-map marker pair"
                )
            else:
                start, end = current.index(BEGIN), current.index(END) + len(END)
                expected = index_content(data)
                if current[start:end] != expected:
                    if write_index and not self.errors:
                        index.write_text(
                            current[:start] + expected + current[end:], encoding="utf-8"
                        )
                        self.cache.pop(index, None)
                    else:
                        self.errors.append(
                            "Stale docs/README.md map: run scripts/check_docs.py --write-index"
                        )

        files = sorted(
            {
                self.root / "README.md",
                *(self.root / "docs").rglob("*.md"),
                *(self.root / p for p in registered if p.endswith(".md")),
            }
        )
        files = [path for path in files if not self.ignored(path)]
        for path in files:
            relative = path.relative_to(self.root).as_posix()
            if relative not in registered and not any(
                fnmatch.fnmatchcase(relative, p) for p in data["historical_docs"]
            ):
                self.errors.append(f"Unregistered document: {relative}; add it to docs/map.yaml")
            if path.is_file():
                for ref, line in self.document(path)[1]:
                    self.reference(path, ref, f"{relative}:{line}")

        if data["config_examples"]:
            sys.path.insert(0, str(self.root / "src"))
            module = importlib.import_module(data["package"] + ".config")
            for example in data["config_examples"]:
                label = " ".join([example["config"], *example.get("overrides", [])])
                try:
                    cfg = module.load_config(
                        example["config"], example.get("overrides", []), root=self.root / "configs"
                    )
                    if not example.get("allow_missing", False):
                        module.config_hash(cfg)
                    self.configs += 1
                except Exception as exc:
                    self.errors.append(f"Config example [{label}]: {type(exc).__name__}: {exc}")
        return len(files)


def changes_from_git(root, base=None):
    def git(*args):
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), *args], capture_output=True, check=True
        ).stdout

    try:
        top = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
    except (OSError, subprocess.CalledProcessError):
        if base:
            raise ValueError("--base requires a Git checkout") from None
        return [], "No Git baseline; use --changed to request an impact report."
    if top != root:
        raise ValueError("The project root must be the Git checkout root")
    if base and base.startswith("-"):
        raise ValueError("Invalid --base revision")
    try:
        git("rev-parse", "--verify", base or "HEAD")
    except subprocess.CalledProcessError:
        if base:
            raise ValueError(f"Unknown Git revision: {base}") from None
        names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
        note = "No initial commit: all non-ignored files are treated as new."
    else:
        names = git("diff", "--name-only", "--no-renames", "-z", base or "HEAD", "--")
        names += git("ls-files", "--others", "--exclude-standard", "-z")
        note = "Impact is advisory and file-level; it does not prove semantic agreement."
    return sorted({os.fsdecode(n) for n in names.split(b"\0") if n}), note


def impacted_topics(data, changed):
    changed = sorted({PurePosixPath(path).as_posix() for path in changed})
    out = []
    for topic in data["topics"]:
        refs = [topic["source"], *topic.get("review", [])]
        paths = {PurePosixPath(unquote(urlsplit(ref).path)).as_posix() for ref in refs}
        patterns = [PurePosixPath(pattern).as_posix() for pattern in topic.get("watch", [])]
        matched = [
            p
            for p in changed
            if p in paths
            or any(fnmatch.fnmatchcase(p, pattern) for pattern in patterns)
            or p == "docs/map.yaml"
        ]
        if matched:
            out.append((topic, matched))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write-index", action="store_true", help="refresh only the generated map")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--base", help="Git revision to compare with, including working-tree changes"
    )
    group.add_argument("--changed", nargs="*", help="explicit repository-relative changed paths")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        data = load_map(root)
        checker = Check(root)
        count = checker.validate(data, args.write_index)
        if args.changed is None:
            changed, note = changes_from_git(root, args.base)
        else:
            for path in args.changed:
                repo_path(root, path)
            changed = sorted(set(args.changed))
            note = "Explicit paths; impact is advisory and file-level."
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if checker.errors:
        for message in checker.errors:
            print(f"ERROR: {message}", file=sys.stderr)
    else:
        print(
            f"PASS: {count} Markdown files, {checker.links} local references, "
            f"{checker.configs} configuration examples"
        )
    print(note)
    for topic, paths in impacted_topics(data, changed):
        print(f"REVIEW [{topic['id']}] because: {', '.join(paths)}")
        print(f"  source: {topic['source']}")
        for ref in dict.fromkeys(topic.get("review", [])):
            print(f"  review: {ref}")
    if changed:
        print(
            "Review each affected topic; update dependent text only when needed. "
            "Do not rewrite historical reports to match a new protocol."
        )
    return 1 if checker.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
