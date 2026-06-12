from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


class PyProjectDependencyTests(unittest.TestCase):
    def test_torch_xla_dependency_is_platform_gated(self) -> None:
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text())

        dependencies = data["project"]["dependencies"]
        torch_xla_dependencies = [dep for dep in dependencies if dep.startswith("torch-xla")]

        self.assertEqual(len(torch_xla_dependencies), 1)
        dependency = torch_xla_dependencies[0]
        self.assertIn("sys_platform == 'linux'", dependency)
        self.assertIn("platform_machine == 'x86_64'", dependency)


if __name__ == "__main__":
    unittest.main()
