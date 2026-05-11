from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py


# ----------------------------- Public API types -----------------------------

@dataclass
class ValidationMessage:
    level: str  # "ERROR" | "WARN"
    path: str
    message: str


@dataclass
class ValidationReport:
    file_path: Path
    errors: List[ValidationMessage]
    warnings: List[ValidationMessage]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def format_lines(self) -> List[str]:
        lines: List[str] = []
        for m in self.errors:
            lines.append(f"[ERROR] {m.path}: {m.message}")
        for m in self.warnings:
            lines.append(f"[WARN]  {m.path}: {m.message}")
        return lines


# ----------------------------- Internal helpers -----------------------------

def _as_str(v: Any) -> str:
    if isinstance(v, (bytes, bytearray)):
        return v.decode(errors="replace")
    return str(v)


def _nx_class(obj: Union[h5py.Group, h5py.Dataset]) -> Optional[str]:
    v = obj.attrs.get("NX_class")
    return None if v is None else _as_str(v)


def _collect_all_paths(h5: h5py.File) -> List[str]:
    """Collect all absolute paths in the HDF5 file."""
    paths: List[str] = []

    def walk(g: h5py.Group, prefix: str) -> None:
        for k, obj in g.items():
            p = f"{prefix}/{k}" if prefix else f"/{k}"
            paths.append(p)
            if isinstance(obj, h5py.Group):
                walk(obj, p)

    walk(h5["/"], "")
    return paths


def _expand_node_matches(node: Dict[str, Any], all_paths: List[str]) -> List[str]:
    """Return all matching paths for a node spec (explicit path or regex pattern)."""
    if "path" in node:
        return [node["path"]]
    if "path_pattern" in node:
        rx = re.compile(node["path_pattern"])
        return [p for p in all_paths if rx.match(p)]
    return []


def _discover_entry_path(
    h5: h5py.File,
    *,
    schema_entry_path: str,
    requested_entry_path: Optional[str],
) -> str:
    if requested_entry_path and requested_entry_path in h5:
        return requested_entry_path
    if schema_entry_path in h5:
        return schema_entry_path
    for cand in ("/raw_data", "/entry", "/entry0", "/entry1"):
        if cand in h5:
            return cand
    for key in h5.keys():
        path = f"/{key}"
        obj = h5[path]
        if isinstance(obj, h5py.Group) and _nx_class(obj) == "NXentry":
            return path
    return schema_entry_path


def _remap_schema_path(path: str, *, schema_entry_path: str, actual_entry_path: str) -> str:
    if schema_entry_path == actual_entry_path:
        return path
    if path == schema_entry_path:
        return actual_entry_path
    if path.startswith(f"{schema_entry_path}/"):
        return f"{actual_entry_path}{path[len(schema_entry_path):]}"
    return path


def _remap_schema_pattern(pattern: str, *, schema_entry_path: str, actual_entry_path: str) -> str:
    if schema_entry_path == actual_entry_path:
        return pattern
    if pattern.startswith(f"^{schema_entry_path}"):
        return f"^{actual_entry_path}{pattern[len(schema_entry_path) + 1:]}"
    if pattern.startswith(schema_entry_path):
        return f"{actual_entry_path}{pattern[len(schema_entry_path):]}"
    return pattern


def _get_link_target(group: h5py.Group, name: str) -> Optional[str]:
    """
    If `group/name` is a SoftLink or ExternalLink, return its target path (or filename:path).
    If it's a hard link or an inline dataset, return None.
    """
    link = group.get(name, getlink=True)
    if isinstance(link, h5py.SoftLink):
        return link.path
    if isinstance(link, h5py.ExternalLink):
        return f"{link.filename}:{link.path}"
    return None


# ----------------------------- Main validator -----------------------------

def validate_nexus_file(
    file_path: Path,
    schema: Dict[str, Any],
    *,
    entry_path: Optional[str] = None,
    strict: bool = False,
    strict_links: bool = False,
) -> ValidationReport:
    """
    Validate a NeXus/HDF5 file using a YAML-derived schema dictionary.

    Parameters
    ----------
    file_path:
        Path to the NeXus/HDF5 file.
    schema:
        Parsed YAML schema (dict), containing a top-level key `nodes`.
    entry_path:
        Optional override for the entry group path (future-proof).
        Note: SCARLET schemas currently use absolute paths, so this does not remap paths.
    strict:
        If True, warnings are promoted to errors (useful for CI).
    strict_links:
        If True, enforce that link_targets fields are SoftLinks matching the target regex.
    """
    schema_entry_path = schema.get("schema", {}).get("entry_path", "/entry")

    nodes = schema.get("nodes", [])
    errors: List[ValidationMessage] = []
    warnings: List[ValidationMessage] = []

    def err(path: str, msg: str) -> None:
        errors.append(ValidationMessage("ERROR", path, msg))

    def warn(path: str, msg: str) -> None:
        warnings.append(ValidationMessage("WARN", path, msg))

    if not isinstance(nodes, list):
        err("<schema>", "Invalid schema: 'nodes' must be a list")
        return ValidationReport(file_path=file_path, errors=errors, warnings=warnings)

    try:
        with h5py.File(file_path, "r") as h5:
            actual_entry_path = _discover_entry_path(
                h5,
                schema_entry_path=schema_entry_path,
                requested_entry_path=entry_path,
            )
            all_paths = _collect_all_paths(h5)

            for node in nodes:
                node = dict(node)
                if "path" in node:
                    node["path"] = _remap_schema_path(
                        node["path"],
                        schema_entry_path=schema_entry_path,
                        actual_entry_path=actual_entry_path,
                    )
                if "path_pattern" in node:
                    node["path_pattern"] = _remap_schema_pattern(
                        node["path_pattern"],
                        schema_entry_path=schema_entry_path,
                        actual_entry_path=actual_entry_path,
                    )
                required = bool(node.get("required", False))
                kind = node.get("kind")  # "group" | "dataset"
                nxcls = node.get("nx_class")
                const = node.get("const")

                matches = _expand_node_matches(node, all_paths)

                # Required semantics:
                # - explicit path: must exist
                # - pattern: at least one match must exist
                if "path" in node:
                    p = node["path"]
                    if required and p not in h5:
                        err(p, "Missing required node")
                        continue
                    if p not in h5:
                        continue
                    match_paths = [p]
                else:
                    if required and len(matches) == 0:
                        err(node.get("path_pattern", "<pattern>"), "Missing required node(s) for pattern")
                        continue
                    match_paths = matches

                for p in match_paths:
                    if p not in h5:
                        continue
                    obj = h5[p]

                    # kind check
                    if kind == "group" and not isinstance(obj, h5py.Group):
                        err(p, "Expected a group")
                        continue
                    if kind == "dataset" and not isinstance(obj, h5py.Dataset):
                        err(p, "Expected a dataset")
                        continue

                    # nx_class check
                    if nxcls and isinstance(obj, h5py.Group):
                        found = _nx_class(obj)
                        if found != nxcls:
                            err(p, f"NX_class must be {nxcls!r} (found {found!r})")

                    # const check (dataset value)
                    if const is not None:
                        if not isinstance(obj, h5py.Dataset):
                            err(p, "const check requires a dataset")
                        else:
                            val = _as_str(obj[()]).strip()
                            if val != str(const):
                                err(p, f"Must equal {const!r} (found {val!r})")

                    # recommended datasets
                    for rec in node.get("recommended_datasets", []) or []:
                        rec_name = rec if isinstance(rec, str) else rec.get("name")
                        if isinstance(obj, h5py.Group) and rec_name and rec_name not in obj:
                            warn(f"{p}/{rec_name}", "Recommended field missing")

                    # children (relative to current node)
                    for child in node.get("children", []) or []:
                        rel = child["path"]
                        cp = f"{p}/{rel}"
                        if child.get("required", False) and cp not in h5:
                            err(cp, "Missing required child node")
                            continue
                        if cp not in h5:
                            continue
                        cobj = h5[cp]
                        ckind = child.get("kind")
                        cnx = child.get("nx_class")
                        if ckind == "group" and not isinstance(cobj, h5py.Group):
                            err(cp, "Expected a group")
                        if ckind == "dataset" and not isinstance(cobj, h5py.Dataset):
                            err(cp, "Expected a dataset")
                        if cnx and isinstance(cobj, h5py.Group):
                            found = _nx_class(cobj)
                            if found != cnx:
                                err(cp, f"NX_class must be {cnx!r} (found {found!r})")

                    # rules (group-level)
                    rules = node.get("rules", {}) or {}
                    if isinstance(obj, h5py.Group) and rules:
                        found_nx = _nx_class(obj)

                        def apply_group_rules(r: Dict[str, Any]) -> None:
                            # allowed_nx_classes (alternative to node['nx_class'])
                            allowed_nx = r.get("allowed_nx_classes")
                            if allowed_nx:
                                if found_nx is None:
                                    err(p, f"Missing NX_class (expected one of {allowed_nx!r})")
                                elif found_nx not in allowed_nx:
                                    err(p, f"NX_class must be one of {allowed_nx!r} (found {found_nx!r})")

                            # required_datasets
                            for name in r.get("required_datasets", []) or []:
                                if name not in obj:
                                    err(f"{p}/{name}", "Missing required dataset")
                                elif not isinstance(obj[name], h5py.Dataset):
                                    err(f"{p}/{name}", "Must be a dataset")

                            # optional_datasets (type-check only if present)
                            for name in r.get("optional_datasets", []) or []:
                                if name in obj and not isinstance(obj[name], h5py.Dataset):
                                    err(f"{p}/{name}", "Must be a dataset")

                            # required_groups
                            for name in r.get("required_groups", []) or []:
                                if name not in obj or not isinstance(obj[name], h5py.Group):
                                    err(f"{p}/{name}", "Missing required group")

                            # required_in_group
                            for gname, req_list in (r.get("required_in_group", {}) or {}).items():
                                if gname not in obj or not isinstance(obj[gname], h5py.Group):
                                    err(f"{p}/{gname}", "Missing required group for required_in_group")
                                    continue
                                gg = obj[gname]
                                for field in req_list or []:
                                    if field not in gg:
                                        err(f"{p}/{gname}/{field}", "Missing required field in group")

                            # one_of
                            if "one_of" in r:
                                alts = r["one_of"] or []
                                ok = any(all(field in obj for field in alt) for alt in alts)
                                if not ok:
                                    err(p, f"Must contain one of: {alts}")

                            # enums
                            for field, allowed in (r.get("enums", {}) or {}).items():
                                if field in obj and isinstance(obj[field], h5py.Dataset):
                                    v = _as_str(obj[field][()]).strip().lower()
                                    allowed_l = [str(a).strip().lower() for a in allowed]
                                    if v not in allowed_l:
                                        err(f"{p}/{field}", f"Invalid value {v!r}, allowed={allowed}")

                            # booleans
                            for field in r.get("booleans", []) or []:
                                if field in obj and isinstance(obj[field], h5py.Dataset):
                                    raw = obj[field][()]
                                    s = _as_str(raw).strip().lower()
                                    if isinstance(raw, (bool, int)):
                                        continue
                                    if s not in ("0", "1", "true", "false", "t", "f", "yes", "no"):
                                        warn(f"{p}/{field}", f"Field present but not clearly boolean: {s!r}")

                            # linked_axis_if_present
                            for arr_field, axis_field in (r.get("linked_axis_if_present", {}) or {}).items():
                                if arr_field in obj and isinstance(obj[arr_field], h5py.Dataset):
                                    arr = obj[arr_field][()]
                                    if getattr(arr, "shape", ()) not in ((), (1,)):
                                        if axis_field not in obj:
                                            warn(f"{p}/{axis_field}", f"{arr_field} is non-scalar; expected {axis_field} axis")
                                        else:
                                            ax = obj[axis_field][()]
                                            if hasattr(ax, "shape") and ax.shape != arr.shape:
                                                warn(
                                                    f"{p}/{arr_field}",
                                                    f"{arr_field} shape {arr.shape} differs from {axis_field} shape {getattr(ax, 'shape', None)}",
                                                )

                            # recommended_attrs
                            for attr in r.get("recommended_attrs", []) or []:
                                if attr not in obj.attrs:
                                    warn(p, f"Recommended attribute {attr!r} missing")

                            # link_targets
                            for field, rx_s in (r.get("link_targets", {}) or {}).items():
                                if field not in obj:
                                    continue
                                rx_s = _remap_schema_pattern(
                                    rx_s,
                                    schema_entry_path=schema_entry_path,
                                    actual_entry_path=actual_entry_path,
                                )
                                rx = re.compile(rx_s)
                                target = _get_link_target(obj, field)

                                if strict_links:
                                    if target is None:
                                        err(f"{p}/{field}", "Expected SoftLink (NXlink) in strict_links mode")
                                    else:
                                        if ":" in target:
                                            err(f"{p}/{field}", f"External link not allowed: {target}")
                                        elif not rx.match(target):
                                            err(f"{p}/{field}", f"Link target {target!r} does not match {rx_s!r}")
                                        elif target not in h5:
                                            err(f"{p}/{field}", f"Broken link: target {target!r} does not exist")
                                else:
                                    if target is not None:
                                        if ":" in target:
                                            warn(f"{p}/{field}", f"External link detected: {target}")
                                        elif not rx.match(target):
                                            warn(f"{p}/{field}", f"Link target {target!r} does not match {rx_s!r}")

                        apply_group_rules(rules)

                        per = rules.get("per_nx_class")
                        if isinstance(per, dict) and found_nx in per and isinstance(per[found_nx], dict):
                            apply_group_rules(per[found_nx])

                        conditional = rules.get("conditional_rules")
                        if isinstance(conditional, list):
                            for cond in conditional:
                                if not isinstance(cond, dict):
                                    continue
                                nx = cond.get("apply_if_nx_class")
                                if isinstance(nx, str) and found_nx == nx:
                                    apply_group_rules(cond)

    except OSError as e:
        err("<file>", f"Could not open/read file: {e}")

    # strict: promote warnings to errors
    if strict and warnings:
        for w in warnings:
            errors.append(ValidationMessage("ERROR", w.path, f"(from warning) {w.message}"))
        warnings = []

    return ValidationReport(file_path=file_path, errors=errors, warnings=warnings)
