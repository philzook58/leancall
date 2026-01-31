from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def _python_blocks(text: str) -> list[str]:
    pattern = re.compile(r"```(?:python|py)\s+([\s\S]*?)```")
    return [match.group(1).strip() for match in pattern.finditer(text)]


def test_readme_python_blocks_run():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    blocks = _python_blocks(readme.read_text(encoding="utf-8"))
    assert blocks, "No python code blocks found in README.md"
    for idx, block in enumerate(blocks, start=1):
        result = subprocess.run(
            [sys.executable, "-"],
            input=block,
            text=True,
            capture_output=True,
            cwd=readme.parent,
            check=False,
        )
        assert result.returncode == 0, (
            f"README python block {idx} failed with exit code {result.returncode}\\n"
            f"stdout:\\n{result.stdout}\\n"
            f"stderr:\\n{result.stderr}"
        )
