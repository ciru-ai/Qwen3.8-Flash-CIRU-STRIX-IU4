#!/usr/bin/env python3
"""Check launcher configuration without loading a model or requiring a GPU."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/ciru/run-server.sh"


class CiruLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.server = root / "server"
        self.server.write_text(
            "#!/usr/bin/env python3\nimport json,os,sys\n"
            "print(json.dumps({'argv':sys.argv[1:],"
            "'shortlist':os.environ.get('CIRU_MTP_SHORTLIST')}))\n"
        )
        self.server.chmod(0o755)
        self.model = root / "model"
        for name in ["Qwen3.8-Flash-CIRU-STRIX-IU4.gguf", "ple/ple.payload.bin",
                     "ple/ple.manifest.json", "ple/ple.scale.bf16",
                     "mtp/Qwen3.8-Flash-CIRU-STRIX-IU4-MTP-Q8_0.gguf"]:
            p = self.model / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("CIRU_", "GGML_", "LLAMA_ARG_"))}
        self.env.update(SERVER_BIN=str(self.server), MODEL_DIR=str(self.model),
                        SLOT_DIR=str(root / "slots"), ENABLE_MTP="1", PARALLEL_SLOTS="1")

    def launch(self, *args, **env):
        return subprocess.run(["bash", str(LAUNCHER), *args],
                              env={**self.env, **env}, capture_output=True, text=True)

    def test_default_mtp_profile(self):
        r = self.launch()
        self.assertEqual(r.returncode, 0, r.stderr)
        result = json.loads(r.stdout)
        self.assertEqual(result["shortlist"], "32768")
        self.assertIn("draft-mtp", result["argv"])
        self.assertEqual(result["argv"][result["argv"].index("--parallel") + 1], "1")

    def test_parallel_mtp_rejected_before_model_validation(self):
        for args, env in [((), {"PARALLEL_SLOTS": "2"}),
                          (("--parallel", "2"), {}), (("-np", "2"), {}),
                          (("--parallel=2",), {}), (("-np=2",), {}),
                          (("--parallel", "1", "-np", "2"), {}),
                          (("-np", "-1"), {})]:
            with self.subTest(args=args, env=env):
                r = self.launch(*args, MODEL_DIR="/missing-model", **env)
                self.assertEqual(r.returncode, 2)
                self.assertIn("requires exactly one slot", r.stderr)
                self.assertIn("ENABLE_MTP=0", r.stderr)
                self.assertEqual(r.stdout, "")

    def test_last_cli_slot_override_wins(self):
        r = self.launch("-np", "2", "--parallel", "1", PARALLEL_SLOTS="2")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_target_only_parallel_does_not_require_draft(self):
        (self.model / "mtp/Qwen3.8-Flash-CIRU-STRIX-IU4-MTP-Q8_0.gguf").unlink()
        r = self.launch("--no-kv-unified", ENABLE_MTP="0", PARALLEL_SLOTS="2")
        self.assertEqual(r.returncode, 0, r.stderr)
        args = json.loads(r.stdout)["argv"]
        self.assertFalse(any(a.startswith("--spec-") for a in args))
        self.assertEqual(args[args.index("--parallel") + 1], "2")
        self.assertIn("--no-kv-unified", args)


if __name__ == "__main__":
    unittest.main()
