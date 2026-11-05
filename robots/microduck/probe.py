# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Container-only diagnostic locomotion gate; never a planner observation API."""

import argparse
import json
import math
import signal
import time
from pathlib import Path
from typing import Any

from robots.microduck.session import MicroduckSession, _healthy, _number, _standing
from rpent.utils.logging import get_logger

logger = get_logger(__name__)


def _projection(before: dict[str, Any], after: dict[str, Any]) -> float:
    w, x, y, z = before["quat_wxyz"]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    a, b = before["trunk_xyz"], after["trunk_xyz"]
    return (b[0] - a[0]) * math.cos(yaw) + (b[1] - a[1]) * math.sin(yaw)


def _distance(before: dict[str, Any], after: dict[str, Any]) -> float:
    return math.dist(before["trunk_xyz"][:2], after["trunk_xyz"][:2])


def _evaluate(evidence: dict[str, Any]) -> dict[str, Any]:
    verdicts: dict[str, bool] = {}
    verdicts["sit_height"] = (
        abs(evidence["sit"]["trunk_xyz"][2] - 0.07) <= 0.005 + 1e-12
    )
    stand = evidence["stand"]
    verdicts["standing"] = _standing(stand["sample"]) and stand["elapsed_s"] <= 10
    motions = evidence["motions"]
    verdicts["sequence"] = [m["name"] for m in motions] == ["forward", "backward"]
    projections, timings = {}, {}
    for motion in motions:
        name, before, after = motion["name"], motion["before"], motion["after"]
        projection = _projection(before, after)
        projections[name] = projection
        sim = after["sim_time_s"] - before["sim_time_s"]
        wall = after["monotonic_time_s"] - before["monotonic_time_s"]
        pace = sim / wall if wall > 0 else 0
        notification_s = motion["notification_end_s"] - motion["notification_start_s"]
        timings[name] = {
            "sim_s": sim,
            "wall_s": wall,
            "ratio": pace,
            "notification_s": notification_s,
        }
        duration = 5 if name == "forward" else 3
        verdicts[f"{name}_displacement"] = (
            projection >= 0.10 if name == "forward" else projection <= -0.05
        )
        verdicts[f"{name}_pace"] = 0.95 <= pace <= 1.05
        verdicts[f"{name}_duration"] = abs(notification_s - duration) <= 0.1
        verdicts[f"{name}_standing"] = _standing(before) and _standing(after)
        verdicts[f"{name}_health"] = _healthy(motion["health_before"]) and _healthy(
            motion["health_after"]
        )
        requested = motion["requested_after_stop"]
        verdicts[f"{name}_stop"] = (
            isinstance(requested, list)
            and len(requested) == 3
            and all(_number(value) and value == 0 for value in requested)
        )
    holds = evidence["holds"]
    verdicts["holds"] = len(holds) == 3 and all(
        len(hold) >= 3
        and hold[-1]["monotonic_time_s"] - hold[0]["monotonic_time_s"] >= 1 - 1e-6
        and abs(hold[-1]["monotonic_time_s"] - hold[-2]["monotonic_time_s"] - 0.5)
        <= 0.05
        and _distance(hold[-2], hold[-1]) <= 0.05
        for hold in holds
    )
    states = evidence["health_states"]
    begin, end = evidence["health_window"]
    states = [state for state in states if begin <= state["at"] <= end]
    times = [begin, *(state["at"] for state in states), end]
    verdicts["diagnostic_frequency"] = bool(states) and all(
        state["hz"] >= 45 for state in states
    )
    verdicts["diagnostic_gaps"] = bool(states) and all(
        0 <= b - a <= 1 for a, b in zip(times, times[1:])
    )
    video, trajectory = evidence["video"], evidence["trajectory"]
    verdicts["recording"] = (
        trajectory["completion"] is True
        and trajectory["failure"] is None
        and trajectory["count"] > 0
    )
    verdicts["video"] = (
        video["decoded"] is True
        and video["width"] == 640
        and video["height"] == 480
        and abs(video["fps"] - 25) <= 1e-6
        and video["frames"] == trajectory["count"]
        and video["codec"] == "h264"
        and video["pixel_format"] == "yuv420p"
        and abs(video["duration_s"] - trajectory["count"] / 25) <= 1e-3
    )
    verdicts["cleanup"] = evidence["cleanup"] is True
    verdicts["source_inventory"] = bool(evidence["source_inventory"])
    return {
        "numeric_pass": all(verdicts.values()),
        "visual_inspection": "pending",
        "verdicts": verdicts,
        "projections_m": projections,
        "timing": timings,
    }


def _hold(session: MicroduckSession) -> list[dict[str, Any]]:
    samples = [session._body.sample()]
    deadline = time.monotonic() + 1
    for offset in (0.5, 0.0):
        target = deadline - offset
        while time.monotonic() < target:
            session._check()
            time.sleep(min(0.02, target - time.monotonic()))
        samples.append(session._body.sample())
    return samples


def _decode_video(
    video: Path,
    trajectory: dict[str, Any],
    boundaries: list[float],
    contact_sheet: Path,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np

    selected = []
    frame_times = trajectory["frame_times"]
    indices = {
        min(range(len(frame_times)), key=lambda i: abs(frame_times[i][1] - at))
        for at in boundaries
    }
    with imageio.get_reader(str(video)) as reader:
        meta = reader.get_meta_data()
        count, shape = 0, None
        for index, frame in enumerate(reader):
            if shape is not None and frame.shape != shape:
                raise ValueError("Decoded video changes dimensions")
            shape = frame.shape
            count += 1
            if index in indices:
                selected.append(frame)
        if shape is None or len(selected) != 4:
            raise ValueError("Video lacks four distinct phase boundary frames")
    imageio.imwrite(str(contact_sheet), np.concatenate(selected, axis=1))
    return {
        "decoded": True,
        "frames": count,
        "width": shape[1],
        "height": shape[0],
        "fps": meta["fps"],
        "duration_s": meta["duration"],
        "codec": meta["codec"],
        "pixel_format": meta["pix_fmt"].split("(", 1)[0],
    }


def _run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gate = output_dir / "gate.json"
    if gate.exists():
        raise FileExistsError(gate)
    session = MicroduckSession(output_dir / ".microduck")
    previous = {}
    # Handlers only set a flag. The operation owner performs all cleanup.
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda *_: session._cancel.set())
    evidence: dict[str, Any] = {"motions": [], "holds": []}
    result: dict[str, Any] = {
        "numeric_pass": False,
        "visual_inspection": "pending",
        "verdicts": {"execution": False},
    }
    try:
        session.start()
        evidence.update(
            sit=session._sit_sample,
            stand=session._stand,
            source_inventory=session._info,
        )
        evidence["holds"].append(_hold(session))
        begin = session._stand["sample"]["monotonic_time_s"] + 1
        for name, vx, duration in (("forward", 0.4, 5), ("backward", -0.4, 3)):
            session.move(vx=vx, duration_s=duration)
            motion = {"name": name, **session._last_motion}
            evidence["motions"].append(motion)
            evidence["holds"].append(_hold(session))
        video = session.finish()
        trajectory = json.loads((session.output_dir / "trajectory.json").read_text())
        boundaries = [
            evidence["sit"]["monotonic_time_s"],
            session._stand["sample"]["monotonic_time_s"],
            *(motion["after"]["monotonic_time_s"] for motion in evidence["motions"]),
        ]
        evidence.update(
            trajectory=trajectory,
            health_states=session._health_states,
            health_window=[begin, trajectory["monotonic_time_end_s"]],
            cleanup=session._state == "CLOSED",
        )
        evidence["video"] = _decode_video(
            video, trajectory, boundaries, output_dir / "contact-sheet.png"
        )
        result = _evaluate(evidence)
    except Exception as exc:
        logger.exception("MicroDuck probe failed")
        result["error"] = str(exc)
    finally:
        session.close()
        evidence["source_inventory"] = session._info
        evidence["health_states"] = session._health_states
        if session._background_failure is not None:
            evidence["diagnostic_failure"] = str(session._background_failure)
            result["verdicts"]["diagnostics"] = False
            result["numeric_pass"] = False
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        result["evidence"] = evidence
        gate.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    """Run the fixed SIT/stand/forward/backward sequence inside a built image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = _run(parser.parse_args().output_dir)
    logger.info(
        "Probe numeric_pass=%s; inspect contact-sheet.png and full video",
        result["numeric_pass"],
    )
    if not result["numeric_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
