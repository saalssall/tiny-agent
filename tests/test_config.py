"""Tests for settings parsing and API-key discovery."""

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from tiny_agent.config import ConfigError, load_api_key, parse_settings


class ApiKeyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        patcher = unittest.mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("ANTHROPIC_API_KEY", None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_environment_variable_wins(self):
        (self.dir / "API.KEY").write_text("from-file\n")
        os.environ["ANTHROPIC_API_KEY"] = "from-env"
        self.assertEqual(load_api_key(self.dir), "from-env")

    def test_key_file_is_used_when_env_is_unset_or_blank(self):
        (self.dir / "API.KEY").write_text("  sk-file  \n")
        self.assertEqual(load_api_key(self.dir), "sk-file")
        os.environ["ANTHROPIC_API_KEY"] = "   "
        self.assertEqual(load_api_key(self.dir), "sk-file")

    def test_missing_or_empty_key_file(self):
        self.assertIsNone(load_api_key(self.dir))
        (self.dir / "API.KEY").write_text("\n")
        self.assertIsNone(load_api_key(self.dir))


class ParseSettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        patcher = unittest.mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for var in ("AGENT_MODEL", "AGENT_EFFORT"):
            os.environ.pop(var, None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_defaults(self):
        settings = parse_settings([str(self.dir)], self.dir)
        self.assertEqual(settings.workspace, self.dir.resolve())
        self.assertEqual(settings.api_key, "sk-test")
        self.assertEqual(settings.model, "claude-opus-5")
        self.assertEqual(settings.effort, "high")
        self.assertFalse(settings.auto_approve)
        self.assertFalse(settings.always_ask)
        self.assertFalse(settings.verbose)

    def test_flags(self):
        argv = [
            str(self.dir),
            "--yolo",
            "--always-ask",
            "--verbose",
            "--model",
            "claude-sonnet-5",
            "--effort",
            "low",
        ]
        settings = parse_settings(argv, self.dir)
        self.assertTrue(settings.auto_approve and settings.always_ask and settings.verbose)
        self.assertEqual((settings.model, settings.effort), ("claude-sonnet-5", "low"))

    def test_environment_defaults_for_model_and_effort(self):
        os.environ.update({"AGENT_MODEL": "claude-haiku-4-5", "AGENT_EFFORT": "max"})
        settings = parse_settings([str(self.dir)], self.dir)
        self.assertEqual((settings.model, settings.effort), ("claude-haiku-4-5", "max"))

    def test_invalid_effort_is_rejected_by_argparse(self):
        with self.assertRaises(SystemExit):
            parse_settings([str(self.dir), "--effort", "ultra"], self.dir)

    def test_bad_workspace_is_a_config_error(self):
        with self.assertRaises(ConfigError) as ctx:
            parse_settings([str(self.dir / "nope")], self.dir)
        self.assertIn("not a directory", str(ctx.exception))

    def test_missing_key_is_a_config_error_with_guidance(self):
        os.environ.pop("ANTHROPIC_API_KEY")
        with self.assertRaises(ConfigError) as ctx:
            parse_settings([str(self.dir)], self.dir)
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))
        self.assertIn("API.KEY", str(ctx.exception))

    def test_version_flag(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_settings(["--version"], self.dir, version="9.9.9")
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
