"""Communication topologies, the message channel, and lying peers."""

import unittest

import numpy as np

from context import Swarm, functions
from swarmopt.comms import LIE_COST


def _swarm(**kwargs):
    params = dict(
        n_particles=12,
        dims=2,
        c1=2.0,
        c2=2.0,
        w=0.9,
        epochs=4,
        obj_func=functions.sphere,
        algo="local",
        seed=7,
        velocity_clamp=(-5.12, 5.12),
        k=2,
    )
    params.update(kwargs)
    return Swarm(**params)


class TestKnnOracleUnchanged(unittest.TestCase):
    def test_default_local_run_matches_the_recorded_oracle(self):
        swarm = _swarm(n_particles=20, epochs=12, k=5)
        swarm.optimize()
        # Recorded after positions were clipped to velocity_clamp; before that
        # this run let particles leave the search box.
        self.assertAlmostEqual(swarm.best_cost, 0.0023932803075623686)
        np.testing.assert_allclose(
            swarm.best_pos, [0.0488729032434006, -0.0021724723527750123]
        )
        self.assertEqual(swarm.messages_sent, 0)

    def test_explicit_oracle_matches_the_default(self):
        default = _swarm(epochs=8)
        explicit = _swarm(epochs=8, topology="knn-oracle")
        default.optimize()
        explicit.optimize()
        self.assertEqual(default.best_cost, explicit.best_cost)
        np.testing.assert_array_equal(default.best_pos, explicit.best_pos)

    def test_oracle_refuses_a_message_channel(self):
        with self.assertRaises(ValueError):
            _swarm(topology="knn-oracle", liar_fraction=0.2)


class TestMessageChannel(unittest.TestCase):
    def test_ring_sends_one_claim_per_neighbour_per_epoch(self):
        swarm = _swarm(topology="ring", epochs=5, k=2)
        swarm.optimize()
        self.assertEqual(swarm.messages_sent, 12 * 2 * 5)
        self.assertEqual(swarm.messages_lost, 0)
        self.assertEqual(swarm.messages_delivered, 12 * 2 * 4)

    def test_loss_drops_claims_and_still_counts_the_send(self):
        swarm = _swarm(topology="ring", epochs=3, k=2, message_loss=1.0)
        swarm.optimize()
        self.assertEqual(swarm.messages_sent, 12 * 2 * 3)
        self.assertEqual(swarm.messages_lost, swarm.messages_sent)
        self.assertEqual(swarm.messages_delivered, 0)

    def test_a_lie_reaches_only_neighbours_until_it_is_checked(self):
        trusted = _swarm(
            n_particles=6, epochs=2, k=2, topology="ring",
            liar_fraction=1 / 6, defense="trust", seed=1,
        )
        trusted.optimize()
        liar = min(trusted.channel.liars)
        for neighbour in trusted.channel.neighbors[liar]:
            self.assertEqual(trusted.swarm[neighbour].local_best_cost, LIE_COST)
        outsiders = [
            i for i in range(6)
            if i != liar and i not in trusted.channel.neighbors[liar]
        ]
        self.assertTrue(outsiders)
        for outsider in outsiders:
            self.assertGreater(trusted.swarm[outsider].local_best_cost, LIE_COST)

        checked = _swarm(
            n_particles=6, epochs=2, k=2, topology="ring",
            liar_fraction=1 / 6, defense="spot-check", seed=1,
        )
        checked.optimize()
        self.assertGreater(checked.channel.spot_checks, 0)
        for neighbour in checked.channel.neighbors[min(checked.channel.liars)]:
            self.assertGreater(checked.swarm[neighbour].local_best_cost, LIE_COST)

    def test_spot_check_beats_trust_when_half_the_swarm_lies(self):
        common = dict(
            n_particles=24, epochs=20, k=4, topology="ring",
            liar_fraction=0.5, seed=3, obj_func=functions.sphere,
        )
        trust = _swarm(**common, defense="trust")
        checked = _swarm(**common, defense="spot-check")
        trust.optimize()
        checked.optimize()
        self.assertLess(checked.best_cost, trust.best_cost)

    def test_other_topologies_build_a_bounded_neighbour_list(self):
        for topology in ("random-k", "geometric", "dynamic"):
            with self.subTest(topology=topology):
                swarm = _swarm(topology=topology, epochs=2, k=3)
                swarm.optimize()
                for index, neighbors in enumerate(swarm.channel.neighbors):
                    self.assertLessEqual(len(neighbors), 3)
                    self.assertNotIn(index, neighbors)
                if topology != "geometric":
                    self.assertGreater(swarm.messages_sent, 0)

    def test_checkpoint_refuses_a_communication_topology(self):
        swarm = _swarm(topology="ring")
        with self.assertRaises(NotImplementedError):
            swarm._check_checkpointable()


if __name__ == "__main__":
    unittest.main()
