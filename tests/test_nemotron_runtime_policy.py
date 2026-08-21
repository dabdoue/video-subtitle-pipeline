import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "providers"))
from nemotron_runtime_policy import GIB, choose_runtime  # noqa: E402


class RuntimePolicyTests(unittest.TestCase):
    def choose(self, requested: str, **overrides: object):
        values = {
            "duration_seconds": 60.0,
            "free_gpu_bytes": 20 * GIB,
            "memory_limit_gb": None,
            "max_offline_seconds": None,
            "auto_max_offline_seconds": 900.0,
            "reserve_gb": 1.0,
            "offline_fixed_mib": 256.0,
            "offline_mib_per_second": 4.0,
        }
        values.update(overrides)
        return choose_runtime(requested, **values)

    def test_auto_selects_offline_when_input_fits(self) -> None:
        decision = self.choose("auto")
        self.assertEqual(decision.selected, "offline")
        self.assertEqual(decision.reason, "offline_fits_budget")

    def test_explicit_streaming_never_selects_offline(self) -> None:
        decision = self.choose("streaming")
        self.assertEqual(decision.selected, "streaming")
        self.assertEqual(decision.reason, "explicit_streaming")

    def test_auto_duration_limit_selects_streaming(self) -> None:
        decision = self.choose("auto", duration_seconds=901.0)
        self.assertEqual(decision.selected, "streaming")
        self.assertEqual(decision.reason, "offline_duration_limit")

    def test_throughput_ignores_auto_duration_limit(self) -> None:
        decision = self.choose(
            "throughput",
            duration_seconds=901.0,
            offline_mib_per_second=1.0,
        )
        self.assertEqual(decision.selected, "offline")

    def test_explicit_duration_limit_applies_to_throughput(self) -> None:
        decision = self.choose(
            "throughput", duration_seconds=601.0, max_offline_seconds=600.0
        )
        self.assertEqual(decision.selected, "streaming")
        self.assertEqual(decision.reason, "offline_duration_limit")

    def test_memory_limit_selects_streaming(self) -> None:
        decision = self.choose("auto", memory_limit_gb=0.25)
        self.assertEqual(decision.selected, "streaming")
        self.assertEqual(decision.reason, "offline_memory_budget")

    def test_invalid_runtime_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown runtime"):
            self.choose("turbo")


if __name__ == "__main__":
    unittest.main()
