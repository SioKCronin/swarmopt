"""Edge cases of the message channel: total loss, delay, bandwidth, collusion."""

import numpy as np
import pytest

from swarmopt import Swarm
from swarmopt.comms import LIE_COST
from swarmopt.functions import sphere

N, EPOCHS, K = 10, 8, 2


def _swarm(**kwargs):
    params = dict(n_particles=N, dims=2, c1=2.0, c2=2.0, w=0.9, epochs=EPOCHS,
                  obj_func=sphere, algo="local", velocity_clamp=(-5.12, 5.12),
                  k=K, topology="ring", seed=3)
    params.update(kwargs)
    return Swarm(**params)


def test_total_loss_isolates_every_particle():
    s = _swarm(message_loss=1.0)
    s.optimize()
    assert s.messages_sent == N * K * EPOCHS
    assert s.messages_lost == s.messages_sent
    assert s.messages_delivered == 0
    # Isolated, a particle never learns of a point better than its own best
    # (its local best lags its own best by one epoch).
    assert all(p.local_best_cost >= p.best_cost for p in s.swarm)


def test_without_loss_some_particle_learns_from_a_neighbour():
    s = _swarm()
    s.optimize()
    assert any(p.local_best_cost < p.best_cost for p in s.swarm)


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_delay_holds_back_exactly_the_last_epochs(delay):
    s = _swarm(message_delay=delay)
    s.optimize()
    per_epoch = N * K
    assert s.messages_sent == per_epoch * EPOCHS
    # A claim sent at the end of epoch e is first read at epoch e + 1 + delay.
    assert s.messages_delivered == per_epoch * max(EPOCHS - 1 - delay, 0)


def test_delay_longer_than_the_run_delivers_nothing():
    s = _swarm(message_delay=EPOCHS)
    s.optimize()
    assert s.messages_delivered == 0


@pytest.mark.parametrize("bandwidth", [1, 2])
def test_bandwidth_caps_claims_per_particle(bandwidth):
    s = _swarm(k=4, message_bandwidth=bandwidth)
    s.optimize()
    assert s.messages_sent == N * min(bandwidth, 4) * EPOCHS


def test_colluding_liars_advertise_one_point():
    s = _swarm(liar_fraction=0.3, liar_mode="colluding")
    liars = [p for p in s.swarm if p.index in s.channel.liars]
    assert len(liars) == 3
    claims = [s.channel._claim(p) for p in liars]
    for pos, cost in claims:
        np.testing.assert_array_equal(pos, claims[0][0])
        assert cost == LIE_COST


def test_independent_liars_advertise_different_points():
    s = _swarm(liar_fraction=0.3, liar_mode="independent")
    liars = [p for p in s.swarm if p.index in s.channel.liars]
    points = [tuple(s.channel._claim(p)[0]) for p in liars]
    assert len(set(points)) == len(points)


def test_spot_check_never_accepts_the_lie_cost():
    s = _swarm(liar_fraction=0.5, liar_mode="colluding", defense="spot-check")
    s.optimize()
    assert s.channel.spot_checks == s.messages_delivered
    assert all(p.local_best_cost > LIE_COST for p in s.swarm)
    assert s.best_cost > LIE_COST
