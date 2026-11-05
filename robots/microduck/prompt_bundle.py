# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Open-loop motion instructions, without sensed or verified success claims."""

from collections.abc import Mapping


def system_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Describe the fixed motion API and its observation limits."""
    return """Execute the user's Microduck task using robot_move, one sequential
action at a time. Defaults: vx=0.4 m/s, vy=0, vyaw=0, duration_s=3 seconds.
Backward motion requires negative vx. Smaller commands are valid but walking
is not guaranteed: the simulator has an empirically observed low-speed deadzone.
Velocity intent is not measured travel speed. Use finite nonboolean numbers with
|vx| <= 0.4 m/s, |vy| <= 0.10 m/s, |vyaw| <= 0.5 rad/s and
0 < duration_s <= 30 seconds. Do not retry a partially failed movement.
Successful motion returns empty observations: there is no sensed feedback,
image, position, or verified task success. Never claim verified success.
After the requested commands, call finish with a summary of execution only.
The video is finalized and registered automatically when the run closes.
No recipe or audit is required."""


def user_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Insert the original task verbatim through the shared template renderer."""
    return "Task (verbatim):\n{{task}}\nEnd of task."
