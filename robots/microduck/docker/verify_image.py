# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Build-time provenance and offline image checks, without stepping physics."""

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path("/opt/microduck")
SOURCE = ROOT / "rl"
INFO = ROOT / "build-info.json"
PINS = Path("/opt/rpent/robots/microduck/dependencies.json")
PROTECTED_PACKAGES = (
    "mujoco",
    "numpy",
    "scipy",
    "onnxruntime",
    "mjlab",
    "warp-lang",
    "better-actuator-models",
)
# pyvers requires packaging>=25,<26; the frozen inventory contains 25.0.
PRESERVED_PACKAGES = (*PROTECTED_PACKAGES, "packaging")
REQUIRED_POLICIES = (
    "velstand.onnx",
    "alpha_sitstand.onnx",
    "alpha_ground_pick.onnx",
    "ball_kick_left.onnx",
    "ball_kick_right.onnx",
    "roulade.onnx",
)
EXPECTED_VERSIONS = {"mujoco": "3.10.0", "numpy": "2.4.1", "onnxruntime": "1.24.4"}


def package_inventory() -> dict[str, str]:
    """Return only distribution names and versions, never index URLs."""
    return {
        dist.metadata["Name"].lower().replace("_", "-"): dist.version
        for dist in importlib.metadata.distributions()
    }


def check_inventory(before: dict[str, str], after: dict[str, str]) -> None:
    """Preserve critical simulator versions and the packaging compatibility pin."""
    for name in PRESERVED_PACKAGES:
        if name not in before or before[name] != after.get(name):
            raise ValueError(f"Original package changed or missing: {name}")


def discover_interfaces(source: Path) -> dict[str, dict[str, str]]:
    """Locate unique original definitions; fail closed on ambiguous modules."""
    names = ("World", "Body", "pose_table", "Server", "Handler")
    matches: dict[str, list[Path]] = {name: [] for name in names}
    for directory, children, files in os.walk(source):
        children[:] = [name for name in children if not name.startswith(".")]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = Path(directory) / name
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    if node.name in matches:
                        matches[node.name].append(path)
    result = {}
    for name, paths in matches.items():
        if len(paths) != 1:
            raise ValueError(
                f"Expected unique original {name} definition; found {len(paths)}"
            )
        path = paths[0]
        parent = path.parent
        parts = [] if path.name == "__init__.py" else [path.stem]
        while (parent / "__init__.py").is_file():
            parts.insert(0, parent.name)
            parent = parent.parent
        result[name] = {"module": ".".join(parts), "import_root": str(parent)}
    return result


def policy_inventory(current: Path) -> dict[str, str]:
    """Require the release manifest and every protocol policy, and hash assets."""
    manifest = current / "manifest.json"
    if not manifest.is_file():
        raise ValueError("Missing policy manifest.json")
    json.loads(manifest.read_text())
    for name in REQUIRED_POLICIES:
        if not (current / name).is_file() or (current / name).stat().st_size == 0:
            raise ValueError(f"Missing or empty policy: {name}")
    return {
        str(path.relative_to(current)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(current.rglob("*"))
        if path.is_file()
    }


def fetch_policy_notices() -> None:
    """Retain pinned model cards/licenses during build, without guessed paths."""
    pins = json.loads(PINS.read_text())["policies"]
    repository = pins["repository"]
    revision = pins["revision"]
    listing = json.loads(
        subprocess.check_output(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                f"https://huggingface.co/api/models/{repository}/tree/{revision}?recursive=true&limit=1000",
            ],
            text=True,
        )
    )
    if not isinstance(listing, list) or len(listing) >= 1000:
        raise ValueError("Policy notice listing incomplete; source evidence needed")
    paths = []
    for item in listing:
        path = Path(item["path"])
        if item["type"] != "file":
            continue
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe policy repository path")
        if path.name.lower().startswith(("license", "copying", "notice", "readme")):
            paths.append(path)
    if not paths:
        raise ValueError("Missing pinned policy license/model-card evidence")
    for path in paths:
        destination = ROOT / "policies/current/notices" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                f"https://huggingface.co/{repository}/resolve/{revision}/{quote(path.as_posix())}",
                "--output",
                str(destination),
            ],
            check=True,
        )


def source_revisions(pins: dict) -> None:
    """Check actual fetched revisions against committed dependency metadata."""
    controller = (ROOT / "runtime/controller-revision").read_text().strip()
    simulator = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if controller != pins["controller"]["revision"]:
        raise ValueError("Controller revision mismatch")
    if simulator != pins["simulator"]["revision"]:
        raise ValueError("Simulator revision mismatch")


def prepare() -> None:
    """Write image provenance after dependency installation and pinned seeding."""
    before = json.loads((ROOT / "packages-before.json").read_text())
    after = package_inventory()
    check_inventory(before, after)
    for name, version in EXPECTED_VERSIONS.items():
        if after.get(name) != version:
            raise ValueError(f"Unexpected locked version: {name}")
    pins = json.loads(PINS.read_text())
    source_revisions(pins)
    libraries = {
        path.resolve()
        for path in (SOURCE / ".venv").glob(
            "lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*"
        )
        if path.is_file()
    }
    if len(libraries) != 1:
        raise ValueError("Expected one unambiguous ONNX Runtime shared library")
    scenes = list(SOURCE.rglob("scene_backlash.xml"))
    if len(scenes) != 1:
        raise ValueError("Expected one original scene_backlash.xml")
    if not any(path.is_file() for path in (ROOT / "runtime/notices").rglob("*")):
        raise ValueError("Missing controller license notices")
    if not any(
        path.is_file()
        for path in SOURCE.iterdir()
        if path.name.lower().startswith(("license", "copying", "notice"))
    ):
        raise ValueError("Missing simulator license notices")
    if not any(
        path.is_file() for path in (ROOT / "policies/current/notices").rglob("*")
    ):
        raise ValueError("Missing policy license/model-card evidence")
    info = {
        "protocol": 1,
        "dependencies": pins,
        "bases": {
            "builder": "rust:1.94.1-bookworm",
            "application": "python:3.12-bookworm",
        },
        "uv": "0.9.0",
        "python": sys.version.split()[0],
        "os_release": Path("/etc/os-release").read_text(),
        "system_packages": subprocess.check_output(
            ["dpkg-query", "-W", "-f=${Package}\t${Version}\n"], text=True
        ).splitlines(),
        "packages_before": before,
        "packages_after": after,
        "ort_dylib_path": str(next(iter(libraries))),
        "scene": str(scenes[0]),
        "interfaces": discover_interfaces(SOURCE),
        "policy_sha256": policy_inventory(ROOT / "policies/current"),
    }
    INFO.write_text(json.dumps(info, indent=2) + "\n")


def verify() -> None:
    """Verify baked provenance, original imports, assets and controller help."""
    info = json.loads(INFO.read_text())
    pins = json.loads(PINS.read_text())
    if info["protocol"] != 1 or info["dependencies"] != pins:
        raise ValueError("Image provenance mismatch")
    source_revisions(pins)
    packages = package_inventory()
    check_inventory(info["packages_before"], packages)
    if packages != info["packages_after"]:
        raise ValueError("Image package inventory changed")
    for name, version in EXPECTED_VERSIONS.items():
        if packages.get(name) != version:
            raise ValueError(f"Unexpected locked version: {name}")
    if policy_inventory(ROOT / "policies/current") != info["policy_sha256"]:
        raise ValueError("Policy SHA256 inventory mismatch")
    if not Path(info["ort_dylib_path"]).is_file():
        raise ValueError("Missing recorded ONNX Runtime shared library")
    interfaces = discover_interfaces(SOURCE)
    if interfaces != info["interfaces"]:
        raise ValueError("Original source interface discovery changed")
    for name, location in interfaces.items():
        if location["import_root"] not in sys.path:
            sys.path.insert(0, location["import_root"])
        module = importlib.import_module(location["module"])
        if not callable(getattr(module, name)):
            raise ValueError(f"Original interface is not callable: {name}")

    # Optional simulator imports belong exclusively to this in-image check.
    import mujoco

    model = mujoco.MjModel.from_xml_path(info["scene"])
    for kind, name in (
        (mujoco.mjtObj.mjOBJ_KEY, "SIT"),
        (mujoco.mjtObj.mjOBJ_BODY, "trunk_base"),
    ):
        if mujoco.mj_name2id(model, kind, name) < 0:
            raise ValueError(f"Missing scene name: {name}")
    for module_name, names in (
        ("robots.microduck.body", ("MicroduckBody",)),
        ("robots.microduck.recording", ("TrajectoryWriter", "render_trajectory")),
        ("robots.microduck.env_server", ()),
    ):
        module = importlib.import_module(module_name)
        for name in names:
            if not callable(getattr(module, name)):
                raise ValueError(f"Missing adapter: {name}")
    executable = ROOT / "runtime/bin/robotd"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("Missing executable robotd")
    subprocess.run(
        [str(executable), "--help"],
        env={**os.environ, "ORT_DYLIB_PATH": info["ort_dylib_path"]},
        check=True,
        timeout=30,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    from rpent.utils.logging import get_logger

    get_logger("microduck_image").info(
        "Offline image verification passed (no gait executed)"
    )


def main() -> None:
    """Select the build inventory, provenance preparation or offline check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("inventory", "policy-notices", "prepare", "verify"),
        nargs="?",
        default="verify",
    )
    mode = parser.parse_args().mode
    if mode == "inventory":
        before = package_inventory()
        (ROOT / "packages-before.json").write_text(json.dumps(before, indent=2) + "\n")
        # Preserve only critical versions and pyvers' packaging compatibility.
        # Other distributions may upgrade; final uv pip check validates them.
        (ROOT / "packages-before.constraints").write_text(
            "".join(f"{name}=={before[name]}\n" for name in sorted(PRESERVED_PACKAGES))
        )
    elif mode == "policy-notices":
        fetch_policy_notices()
    elif mode == "prepare":
        prepare()
    else:
        verify()


if __name__ == "__main__":
    main()
