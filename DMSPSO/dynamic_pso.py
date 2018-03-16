# -*- coding: utf-8 -*-

# Import from __future__
from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function

# Import modules
import logging
import numpy as np
import pyswarms as ps
from scipy.spatial import cKDTree
from past.builtins import xrange

# Import from package
from base import SwarmBase
from ps.utils.console_utils import cli_print, end_report

class DynamicLPSO(SwarmBase):

    def assertions(self):
        """Assertion method to check various inputs.

        Raises
        ------
        KeyError
            When one of the required dictionary keys is missing.
        ValueError
            When the number of neighbors is not within the range
                :code:`[0, n_particles]`.
            When the p-value is not in the list of values :code:`[1,2]`.
        """
        super(LocalBestPSO, self).assertions()

        if not all(key in self.options for key in ('k', 'p')):
            raise KeyError('Missing either k or p in options')
        if not 0 <= self.k <= self.n_particles:
            raise ValueError('No. of neighbors must be between 0 and no. '
                             'of particles.')
        if self.p not in [1, 2]:
            raise ValueError('p-value should either be 1 (for L1/Minkowski) '
                             'or 2 (for L2/Euclidean).')

    def __init__(self, n_particles, n_swarms, R, max_gen, dimensions, options, bounds=None,
                 velocity_clamp=None):
        """Initializes the swarm.

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        n_swarms : int
            number swarms.
        R : int
            regrouping interval
        max_gen: int
            max iterations
        dimensions : int
            number of dimensions in the space.
        bounds : tuple of np.ndarray, optional (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound
            while the second entry is the maximum bound. Each array must
            be of shape :code:`(dimensions,)`.
        velocity_clamp : tuple (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        options : dict with keys :code:`{'c1', 'c2', 'w', 'R'}`
            a dictionary containing the parameters for the specific
            optimization technique
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
                * k : int
                    number of neighbors to be considered. Must be a
                    positive integer less than :code:`n_particles`
                * p: int {1,2}
                    the Minkowski p-norm to use. 1 is the
                    sum-of-absolute values (or L1 distance) while 2 is
                    the Euclidean (or L2) distance.
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        # Assign k-neighbors and p-value as attributes
        self.k, self.p = options['k'], options['p']
        # Initialize parent class
        super(LocalBestPSO, self).__init__(n_particles, dimensions, options,
                                           bounds, velocity_clamp)
        # Invoke assertions
        self.assertions()
        # Initialize the resettable attributes
        self.reset()

    def optimize(self, objective_func, iters, print_step=1, verbose=1):
        """Optimizes the swarm for a number of iterations.

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : function
            objective function to be evaluated
        iters : int
            number of iterations
        print_step : int (default is 1)
            amount of steps for printing into console.
        verbose : int  (default is 1)
            verbosity setting.

        Returns
        -------
        tuple
            the local best cost and the local best position among the
            swarm.
        """

        # Start with LPSO
        for i in xrange(self.max_gen * 0.9):
            if mod(i,self.R) == 0:
                self.reset_subswarms()

            # Compute cost for current position and personal best
            current_cost = objective_func(self.pos)
            pbest_cost = objective_func(self.personal_best_pos)

            # Update personal bests if the current position is better
            # Create a 1-D mask then update pbest_cost
            m = (current_cost < pbest_cost)
            pbest_cost = np.where(~m, pbest_cost, current_cost)
            # Create a 2-D mask to update positions
            _m = np.repeat(m[:, np.newaxis], self.dimensions, axis=1)
            self.personal_best_pos = np.where(~_m, self.personal_best_pos,
                                              self.pos)

            # Obtain the indices of the best position for each
            # neighbour-space, and get the local best cost and
            # local best positions from it.
            nmin_idx = self._get_neighbors(pbest_cost)
            self.best_cost = pbest_cost[nmin_idx]
            self.best_pos = self.personal_best_pos[nmin_idx]

            # Print to console
            if i % print_step == 0:
                cli_print('Iteration %s/%s, cost: %s' %
                          (i+1, iters, np.min(self.best_cost)), verbose, 2,
                          logger=self.logger)

            # Save to history
            hist = self.ToHistory(
                best_cost=np.min(self.best_cost),
                mean_pbest_cost=np.mean(pbest_cost),
                mean_neighbor_cost=np.mean(self.best_cost),
                position=self.pos,
                velocity=self.velocity
            )
            self._populate_history(hist)

            # Perform position velocity update
            self._update_velocity()
            self._update_position()

        # End with GPSO
        for i in xrange(self.max_gen * 0.1):

            # Compute cost for current position and personal best
            current_cost = objective_func(self.pos)
            pbest_cost = objective_func(self.personal_best_pos)

            # Update personal bests if the current position is better
            # Create a 1-D mask then update pbest_cost
            m = (current_cost < pbest_cost)
            pbest_cost = np.where(~m, pbest_cost, current_cost)
            # Create a 2-D mask to update positions
            _m = np.repeat(m[:, np.newaxis], self.dimensions, axis=1)
            self.personal_best_pos = np.where(~_m, self.personal_best_pos,
                                              self.pos)

            # Get the minima of the pbest and check if it's less than
            # the saved gbest
            if np.min(pbest_cost) < self.best_cost:
                 self.best_cost = np.min(pbest_cost)
                 self.best_pos = self.personal_best_pos[np.argmin(pbest_cost)]

            # Print to console
            if i % print_step == 0:
                cli_print('Iteration %s/%s, cost: %s' %
                          (i+1, iters, np.min(self.best_cost)), verbose, 2,
                          logger=self.logger)

            # Save to history
            hist = self.ToHistory(
                best_cost=np.min(self.best_cost),
                mean_pbest_cost=np.mean(pbest_cost),
                mean_neighbor_cost=np.mean(self.best_cost),
                position=self.pos,
                velocity=self.velocity
            )
            self._populate_history(hist)

            # Perform position velocity update
            self._update_velocity()
            self._update_position()


        # Obtain the final best_cost and the final best_position
        final_best_cost_arg = np.argmin(self.best_cost)
        final_best_cost = np.min(self.best_cost)
        final_best_pos = self.best_pos[final_best_cost_arg]

        end_report(final_best_cost, final_best_pos, verbose,
                   logger=self.logger)
        return final_best_cost, final_best_pos

    def _get_neighbors(self, pbest_cost):
        """Helper function to obtain the best position found in the
        neighborhood. This uses the cKDTree method from :code:`scipy`
        to obtain the nearest neighbours

        Parameters
        ----------
        pbest_cost : numpy.ndarray of size :code:`(n_particles, )`
            the cost incurred at the historically best position. Will be used
            for mapping the obtained indices to its actual cost.

        Returns
        -------
        array of size (n_particles, ) dtype=int64
            indices containing the best particles for each particle's
            neighbour-space that have the lowest cost
        """
        # Use cKDTree to get the indices of the nearest neighbors
        tree = cKDTree(self.pos)
        _, idx = tree.query(self.pos, p=self.p, k=self.k)

        # Map the computed costs to the neighbour indices and take the
        # argmin. If k-neighbors is equal to 1, then the swarm acts
        # independently of each other.
        if self.k == 1:
            # The minimum index is itself, no mapping needed.
            best_neighbor = pbest_cost[idx][:, np.newaxis].argmin(axis=1)
        else:
            idx_min = pbest_cost[idx].argmin(axis=1)
            best_neighbor = idx[np.arange(len(idx)), idx_min]

        return best_neighbor

    def _update_velocity(self):
        """Updates the velocity matrix of the swarm.

        This method updates the attribute :code:`self.velocity` of
        the instantiated object. It is called by the
        :code:`self.optimize()` method.
        """
        # Define the hyperparameters from options dictionary
        c1, c2, w = self.options['c1'], self.options['c2'], self.options['w']

        # Compute for cognitive and social terms
        cognitive = (c1 * np.random.uniform(0, 1, self.swarm_size)
                     * (self.personal_best_pos - self.pos))
        social = (c2 * np.random.uniform(0, 1, self.swarm_size)
                  * (self.best_pos - self.pos))
        temp_velocity = (w * self.velocity) + cognitive + social

        # Create a mask to clamp the velocities
        if self.velocity_clamp is not None:
            # Create a mask depending on the set boundaries
            min_velocity, max_velocity = self.velocity_clamp[0], \
                                         self.velocity_clamp[1]
            _b = np.logical_and(temp_velocity >= min_velocity,
                                temp_velocity <= max_velocity)
            # Use the mask to finally clamp the velocities
            self.velocity = np.where(~_b, self.velocity, temp_velocity)
        else:
            self.velocity = temp_velocity

    def _update_position(self):
        """Updates the position matrix of the swarm.

        This method updates the attribute :code:`self.pos` of
        the instantiated object. It is called by the
        :code:`self.optimize()` method.
        """
        # Update position and store it in a temporary variable
        temp = self.pos.copy()
        temp += self.velocity

        if self.bounds is not None:
            # Create a mask depending on the set boundaries
            b = (np.all(self.min_bounds <= temp, axis=1)
                 * np.all(temp <= self.max_bounds, axis=1))
            # Broadcast the mask
            b = np.repeat(b[:, np.newaxis], self.dimensions, axis=1)
            # Use the mask to finally guide position update
            temp = np.where(~b, self.pos, temp)
        self.pos = temp
