"""Tickover: a weekly listening routine for machines with no sensors.

Registers the tickover_* tools, two hooks that route chat voice notes, and the tickover skill. numpy and onnxruntime load
when a tool runs, never at registration.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def register(ctx) -> None:
    skills_dir = _ROOT / "skills"
    if skills_dir.is_dir():
        for child in sorted(p for p in skills_dir.iterdir() if p.is_dir()):
            if (child / "SKILL.md").is_file():
                ctx.register_skill(child.name, child / "SKILL.md")
    try:
        from tickover_core.tools import register_tools
    except ImportError:
        sys.path.insert(0, str(_ROOT))
        from tickover_core.tools import register_tools
    register_tools(ctx)
