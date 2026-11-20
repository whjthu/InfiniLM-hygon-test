"""Verify the common training entry and Git tree against the original deployment."""

import hashlib
import re


def git_object(kind, contents):
    header = f"{kind} {len(contents)}\0".encode("ascii")
    return hashlib.sha1(header + contents).hexdigest()


def manifest_tree(files):
    tree = {}
    for path, record in files.items():
        parts = path.encode("utf-8").split(b"/")
        if any(part in {b"", b".", b".."} or b"\0" in part or b"\\" in part for part in parts):
            raise ValueError(f"invalid framework manifest path: {path}")
        directory = tree
        for part in parts[:-1]:
            child = directory.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"framework manifest directory conflict: {path}")
            directory = child
        if parts[-1] in directory:
            raise ValueError(f"framework manifest file conflict: {path}")
        mode, blob = record["mode"], record["git_blob"]
        if mode not in (0o644, 0o755) or not re.fullmatch(r"[0-9a-f]{40}", blob):
            raise ValueError(f"invalid framework Git file identity: {path}")
        directory[parts[-1]] = (f"{0o100000 | mode:o}".encode("ascii"), bytes.fromhex(blob))

    def tree_oid(directory):
        def sort_key(item):
            name, entry = item
            return name + b"/" if isinstance(entry, dict) else name

        body = bytearray()
        for name, entry in sorted(directory.items(), key=sort_key):
            if isinstance(entry, dict):
                mode, object_id = b"40000", bytes.fromhex(tree_oid(entry))
            else:
                mode, object_id = entry
            body.extend(mode + b" " + name + b"\0" + object_id)
        return git_object("tree", body)

    if not tree:
        raise ValueError("empty framework manifest")
    return tree_oid(tree)


def verify_registry_source(candidate, entry, registry, observed, manifest, entry_bytes):
    source = observed["source"]
    files = manifest["files"]
    tree = manifest_tree(files)
    if (candidate != source["framework_commit"] or candidate != manifest["framework_commit"]
            or source["manifest_SHA256"] != registry["source_manifest"]["SHA256"]
            or tree != source["framework_tree"] or tree != manifest["framework_tree"]
            or len(files) != source["framework_files"]):
        raise ValueError("compute registry framework identity differs from the acceptance candidate")
    record = files[entry]
    entry_hash = hashlib.sha256(entry_bytes).hexdigest()
    if (len(entry_bytes) != record["bytes"] or git_object("blob", entry_bytes) != record["git_blob"]
            or entry_hash != record["SHA256"] or entry_hash != observed["entry_SHA256"]):
        raise ValueError("compute registry entry differs from the published source bytes")
    return {"framework_commit": candidate, "framework_tree": tree, "framework_files": len(files),
            "entry_SHA256": entry_hash}
