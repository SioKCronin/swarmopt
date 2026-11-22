"""Checkpoint/resume and run logs on the Rust store.

The key property: a run that loses power and resumes from its last
checkpoint finishes *bit-for-bit* the same as one that never stopped.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from swarmopt import Swarm
from swarmopt import persist
from swarmopt.functions import rastrigin

needs_store = unittest.skipUnless(persist.available(), "swarmopt-persist extension not installed")


class PowerCut(Exception):
    pass


_BUDGET = {"calls": None, "made": 0}


def flaky_rastrigin(x):
    """rastrigin that 'loses power' after a set number of evaluations."""
    _BUDGET["made"] += 1
    if _BUDGET["calls"] is not None:
        if _BUDGET["calls"] <= 0:
            raise PowerCut()
        _BUDGET["calls"] -= 1
    return rastrigin(x)


BASE = dict(n_particles=8, dims=3, c1=2.0, c2=2.0, w=0.9, epochs=40,
            obj_func=flaky_rastrigin, seed=7)

VARIANTS = {
    "global": {},
    "local": dict(algo="local", k=3),
    "unified": dict(algo="unified"),
    "sa": dict(algo="sa"),
    "random_inertia": dict(inertia_func="random"),
    "chaotic_inertia": dict(inertia_func="chaotic", velocity_clamp_func="chaotic"),
    "adaptive": dict(inertia_func="adaptive", velocity_clamp_func="adaptive"),
    "variation": dict(variation_strategy="gaussian"),
    "respect_boundary": dict(target_position=[1.0, 1.0, 1.0], respect_boundary=0.5),
}


def particle_state(s):
    return [(p.pos.copy(), p.best_pos.copy(), p.velocity.copy(), float(p.best_cost)) for p in s.swarm]


class TestEncoding(unittest.TestCase):
    def test_roundtrip_is_exact(self):
        obj = {
            "a": np.array([[0.1, -0.0], [np.inf, 1e-300]]),
            "s": np.float64(np.nextafter(1.0, 2.0)),
            "i": np.int64(-3),
            "f": [float("inf"), float("-inf"), 1.5],
            "t": (1, "x", None, True),
            "big": 2**70,
        }
        back = persist.decode(persist.encode(obj))
        np.testing.assert_array_equal(back["a"], obj["a"])
        self.assertEqual(back["s"], obj["s"])
        self.assertIsInstance(back["s"], np.float64)
        self.assertTrue(np.isnan(persist.decode(persist.encode(float("nan")))))
        self.assertEqual(back["t"], obj["t"])
        self.assertEqual(back["big"], 2**70)

    def test_output_is_strict_json(self):
        raw = persist.encode({"x": float("inf"), "y": np.array([np.nan])})
        json.loads(raw, parse_constant=lambda c: self.fail(f"emitted {c}"))

    def test_refuses_code_bearing_values(self):
        for bad in (object(), np.array([1, "a"], dtype=object), {1: 2}, lambda: 0):
            with self.assertRaises(persist.StateEncodingError):
                persist.encode(bad)

    def test_rejects_hostile_payloads(self):
        hostile = [
            b'{"v":1,"data":{"__nd__":["object",[1],"AA=="]}}',
            b'{"v":1,"data":{"__nd__":["float64",[1000000],"AA=="]}}',
            b'{"v":1,"data":{"__nd__":["float64",[-1],""]}}',
            b'{"v":1,"data":NaN}',
            b'{"v":2,"data":1}',
            b'not json',
        ]
        for raw in hostile:
            with self.assertRaises((ValueError, UnicodeDecodeError)):
                persist.decode(raw)


@needs_store
class TestResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        _BUDGET["calls"] = None

    def tearDown(self):
        _BUDGET["calls"] = None
        self.tmp.cleanup()

    def _reference(self, cfg):
        s = Swarm(**cfg)
        s.optimize()
        return s

    def test_resume_after_power_cut_is_bit_exact(self):
        for name, extra in VARIANTS.items():
            for cut_after in (37, 150, 301):
                with self.subTest(variant=name, cut_after=cut_after):
                    cfg = {**BASE, **extra}
                    ref = self._reference(cfg)
                    ck_dir = self.dir / f"{name}-{cut_after}"

                    _BUDGET["calls"] = cut_after
                    with persist.SwarmCheckpoint(ck_dir) as ck:
                        with self.assertRaises(PowerCut):
                            Swarm(**cfg).optimize(checkpoint=ck, checkpoint_every=3)
                    _BUDGET["calls"] = None

                    # The "rebooted" drone builds the same swarm and carries on.
                    with persist.SwarmCheckpoint(ck_dir) as ck:
                        resumed = Swarm(**cfg)
                        _BUDGET["made"] = 0
                        resumed.optimize(checkpoint=ck, checkpoint_every=3)
                        resumed_calls = _BUDGET["made"]
                        self.assertIsNotNone(ck.last_generation)

                    # It really continued from the checkpoint rather than
                    # quietly starting over (which a fixed seed would also
                    # make bit-exact).
                    _BUDGET["made"] = 0
                    Swarm(**cfg).optimize()
                    full_calls = _BUDGET["made"]
                    self.assertLess(resumed_calls, full_calls)

                    self.assertEqual(resumed.best_cost, ref.best_cost)
                    np.testing.assert_array_equal(resumed.best_pos, ref.best_pos)
                    for a, b in zip(particle_state(resumed), particle_state(ref)):
                        for x, y in zip(a, b):
                            np.testing.assert_array_equal(x, y)
                    self.assertTrue(np.isclose(rastrigin(resumed.best_pos), resumed.best_cost))

    def test_finished_run_reloads_without_recomputing(self):
        with persist.SwarmCheckpoint(self.dir / "done") as ck:
            first = Swarm(**BASE)
            first.optimize(checkpoint=ck)
        with persist.SwarmCheckpoint(self.dir / "done") as ck:
            again = Swarm(**BASE)  # evaluates the initial particles
            _BUDGET["calls"] = 0   # any evaluation during optimize "crashes"
            again.optimize(checkpoint=ck)
        self.assertEqual(again.best_cost, first.best_cost)
        np.testing.assert_array_equal(again.best_pos, first.best_pos)

    def test_mismatched_config_is_refused(self):
        with persist.SwarmCheckpoint(self.dir / "m") as ck:
            Swarm(**BASE).optimize(checkpoint=ck, checkpoint_every=5)
        for change in (dict(c1=1.5), dict(seed=8), dict(epochs=41), dict(obj_func=rastrigin)):
            with self.subTest(change=change):
                with persist.SwarmCheckpoint(self.dir / "m") as ck:
                    with self.assertRaises(persist.CheckpointMismatch):
                        Swarm(**{**BASE, **change}).optimize(checkpoint=ck)

    def test_unsupported_algorithms_say_so(self):
        with persist.SwarmCheckpoint(self.dir / "u") as ck:
            for extra in (dict(algo="cpso"), dict(algo="hhoa"), dict(ppso_enabled=True)):
                with self.subTest(extra=extra), self.assertRaises(NotImplementedError):
                    Swarm(**{**BASE, **extra}).optimize(checkpoint=ck)
        with self.assertRaises(ValueError):
            Swarm(**BASE).optimize(checkpoint_every=5)

    def test_corrupted_latest_checkpoint_falls_back(self):
        ck_dir = self.dir / "fb"
        _BUDGET["calls"] = 200
        with persist.SwarmCheckpoint(ck_dir) as ck:
            with self.assertRaises(PowerCut):
                Swarm(**BASE).optimize(checkpoint=ck, checkpoint_every=2)
            newest = ck.last_generation
        _BUDGET["calls"] = None
        slot = ck_dir / ("ckpt-a.swc" if newest % 2 else "ckpt-b.swc")
        raw = bytearray(slot.read_bytes())
        raw[len(raw) // 2] ^= 0x20  # a single radiation-induced bit flip
        slot.write_bytes(bytes(raw))
        with persist.SwarmCheckpoint(ck_dir) as ck:
            resumed = Swarm(**BASE)
            resumed.optimize(checkpoint=ck, checkpoint_every=2)
            self.assertTrue(ck.fell_back)
        self.assertEqual(resumed.best_cost, self._reference(BASE).best_cost)


@needs_store
class TestRunLog(unittest.TestCase):
    def test_events_roundtrip_and_checkpoint_events(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with persist.RunLog(d / "log", stream_id=0x4A55, replicas=2) as log:
                log.append({"event": "start", "best_cost": float("inf"), "pos": np.zeros(3)})
                with persist.SwarmCheckpoint(d / "ck", log=log) as ck:
                    Swarm(**{**BASE, "obj_func": rastrigin}).optimize(checkpoint=ck, checkpoint_every=10)
            events, stats = persist.read_run_log(d / "log", stream_id=0x4A55)
            self.assertEqual([seq for seq, _ in events], list(range(1, 6)))
            self.assertEqual(events[0][1]["best_cost"], float("inf"))
            self.assertEqual([e["epoch"] for _, e in events[1:]], [10, 20, 30, 40])
            self.assertEqual(stats["undecodable"], 0)
            self.assertEqual(stats["repaired_from_replica"], 0)

    def test_storage_full_is_an_exception(self):
        import swarmopt_persist
        with tempfile.TemporaryDirectory() as d:
            with persist.RunLog(Path(d) / "log", replicas=1, max_total_bytes=4096,
                                segment_bytes=4096, max_record_bytes=1024) as log:
                with self.assertRaises(swarmopt_persist.StorageFullError):
                    for i in range(1000):
                        log.append({"i": i, "pad": "x" * 200})

    def test_second_writer_locked(self):
        import swarmopt_persist
        with tempfile.TemporaryDirectory() as d:
            with persist.RunLog(Path(d) / "log"):
                with self.assertRaises(swarmopt_persist.LockedError):
                    persist.RunLog(Path(d) / "log")


if __name__ == "__main__":
    unittest.main()
