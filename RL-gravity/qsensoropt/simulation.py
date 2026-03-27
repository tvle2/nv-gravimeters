"""Module containg the :py:obj:`~.Simulation` class.
"""

from typing import List, Callable

from tensorflow import zeros, ones, reduce_mean, \
    stop_gradient, expand_dims, constant, while_loop, \
    cast, where, concat, reshape, broadcast_to, cond, \
    tensor_scatter_nd_add, sort, Tensor
import tensorflow as tf
from tensorflow.math import less, count_nonzero, \
    logical_not, reciprocal_no_nan, add, log, \
    reduce_sum, less_equal, logical_or, \
    greater_equal, reduce_max
from tensorflow.random import Generator

from .physical_model import PhysicalModel
from .particle_filter import ParticleFilter
from .simulation_parameters import SimulationParameters


class Simulation:
    """This is a blueprint class for all the
    Bayesian estimations.

    **Achtung!** The user should not derive this class,
    but instead :py:obj:`~.StatefulSimulation` or
    :py:obj:`~.StatelessSimulation`, according to the
    physical model of the probe.

    Attributes
    ----------
    bs: int
        Batchsize of the simulation,
        i.e. number of Bayesian estimations
        performed simultaneously. It is
        taken from the `phys_model` attribute.
    phys_model: :py:obj:`~.PhysicalModel`
        Parameter `phys_model` passed to the
        class constructor.
    control_strategy: Callable
        Parameter `control_strategy` passed to the
        class constructor.
    pf: :py:obj:`~.ParticleFilter`
        Parameter `particle_filter` passed to the
        class constructor.
    input_size: int
        Parameter `input_size` passed to the
        class constructor.
    input_name: List[str]
        Parameter `input_name` passed to the
        class constructor.
    simpars: :py:obj:`~.SimulationParameters`
        Parameter `simpars` passed to the
        class constructor.
    ns: int
        Maximum number of steps of the
        measurement loop in the :py:meth:`execute` method.
        It is the :py:attr:`~.SimulationParameters.num_steps`
        attribute of `simpars`.
    """

    def __init__(
            self, particle_filter: ParticleFilter,
            phys_model: PhysicalModel,
            control_strategy: Callable,
            input_size: int,
            input_name: List[str],
            simpars: SimulationParameters,
    ):
        """Parameters passed to the :py:obj:`~.Simulation` class
        constructor.

        Parameters
        ----------
        particle_filter: :py:obj:`~.ParticleFilter`
            Particle filter responsible for the update
            of the Bayesian posterior on the parameters
            and on the state of the probe. It
            contains the methods for applying the Bayes
            rule and computing Bayesian estimators
            from the posterior.
        phys_model: :py:obj:`~.PhysicalModel`
            Abstract description of the physical model
            of the quantum probe.
            It contains the method
            :py:meth:`~.StatefulPhysicalModel.perform_measurement`
            that simulates the measurement
            on the probe.
        control_strategy: Callable
            Callable object (normally a
            function or a lambda function) that
            computes the values of the controls
            for the next measurement from
            the `Tensor` `input_strategy`, which is
            produced by the method
            :py:meth:`~.StatefulSimulation.generate_input`
            defined by the user. The :py:obj:`~.Simulation`
            class expects a callable with the following
            header

            ``controls =
            control_strategy(input_strategy)``

            However, if at least one of the controls is
            discrete, then the expected header
            for `control_strategy` is

            ``controls, log_prob_control =
            control_strategy(input_strategy, rangen)``

            That means that some stochastic operations
            can be performed inside the function,
            like the extraction of the discrete controls
            from a probability distribution. The parameter
            `rangen` is the random number generator while
            `log_prob_control` is the log-likelihood
            of sampling the selected discrete controls.
        input_size: int
            Size of the last dimension of the `input_strategy`
            object passed to the callable attribute
            `control_strategy`
            and returned by the user-defined method
            :py:meth:`~.StatefulSimulation.generate_input`.
            It is the number of scalars that the function
            generating the controls (a neural network
            typically) takes as input.
        input_name: List[str]
            List of names for each of the scalar inputs
            returned by the user-defined method
            :py:meth:`~.StatefulSimulation.generate_input`.
            The length of `input_name` should be
            `input_size`. This list of names is used in
            the function
            :py:func:`~.utils.store_input_control`, that
            saves a history of every input to
            the neural network, in order for the user
            to be able to
            interpret the actions
            of the optimized control strategy.
        simpars: :py:obj:`~.SimulationParameters`
            Contains the flags and parameters
            that regulate the stopping
            condition of the measurement loop
            and modify the loss function used in the
            training.
        """
        self.bs = phys_model.bs
        self.phys_model = phys_model
        self.control_strategy = control_strategy
        self.pf = particle_filter
        self.input_size = input_size
        self.input_name = input_name
        self.simpars = simpars
        self.ns = simpars.num_steps

        if not simpars.prec in ("float64", "float32"):
            raise ValueError("The allowed values of \
                             prec are float32 and float64.")

        if len(input_name) != input_size:
            raise ValueError("The length of the list \
                             input_name should be input_size.")

        # Is there at least a discrete control?
        self.discrete_controls = False
        for contr in self.phys_model.controls:
            self.discrete_controls = contr.is_discrete or \
                self.discrete_controls

        self.state_size = phys_model.state_specifics['size']
        self.state_type = phys_model.state_specifics['type']

        # If we resample the particle filter ensemble
        # we must also resample the states ensemble
        self.recompute_state = self.pf.phys_model.recompute_state and \
            self.pf.resampling and self.pf.state_size > 0

    def _generate_input(
        self, weights: Tensor, particles: Tensor,
        state_ensemble: Tensor, meas_step: Tensor,
        used_resources: Tensor,
        rangen: Generator,
    ):
        raise NotImplementedError("You should override this method!")

    def _loss_function(
        self, weights: Tensor, particles: Tensor,
        true_state: Tensor, state_ensemble: Tensor,
        true_values: Tensor, used_resources: Tensor,
        meas_step: Tensor,
    ) -> Tensor:
        raise NotImplementedError("You should override this method!")

    def _compute_scalar_loss(
        self, weights: Tensor, particles: Tensor,
        true_state: Tensor, state_ensemble: Tensor,
        sum_log_prob: Tensor, true_values: Tensor,
        used_resources: Tensor, meas_step: Tensor,
        continue_flag: Tensor,
    ):
        """Compute the scalar loss to to be
        differentiated according to the flags specified
        in simpars."""
        pars = self.simpars

        loss_values = self._loss_function(
            weights, particles, true_state, state_ensemble,
            true_values, used_resources, meas_step,
        )
        loss_mean = reduce_mean(loss_values)
        # Precision of the estimation mixed with the logarithm
        # of the probabilities, this will be differentiated.
        if pars.loss_logl_outcomes:
            added = loss_mean if pars.baseline else 0
            loss_diff_partial_complete = add(
                loss_values,
                (stop_gradient(loss_values) - added)*sum_log_prob,
                name="loss_diff_partial",
            )
        else:
            loss_diff_partial_complete = loss_values

        if pars.end_batch:
            if pars.log_loss:
                loss_diff_partial = reduce_mean(
                    loss_diff_partial_complete)*reciprocal_no_nan(
                    stop_gradient(loss_mean),
                    name="loss_diff_partial",
                )
                loss_partial = log(loss_mean)
            else:
                loss_diff_partial = reduce_mean(
                    loss_diff_partial_complete,
                    name="loss_diff_partial",
                )
                loss_partial = loss_mean
        else:
            loss_diff_partial = reduce_sum(where(
                continue_flag,
                loss_diff_partial_complete, zeros(
                    (self.bs, 1), dtype=pars.prec),
            )
            )
            loss_partial = reduce_sum(where(
                continue_flag,
                loss_values,
                zeros((self.bs, 1), dtype=pars.prec)),
            )
        return loss_diff_partial, loss_partial

    def _compute_finished(
            self, continue_flag: Tensor,
            meas_step: Tensor,
    ):
        """Compute the number of completed estimations,
        either because they have run out of resources or because
        they have reached the maximum number of measurements."""
        pars = self.simpars

        finished = logical_or(
            logical_not(continue_flag),
            greater_equal(meas_step, pars.num_steps *
                          ones((self.bs, 1), dtype="int32")),
        )
        num_finished_estimations = cast(count_nonzero(
            finished,
            name="num_estimation_completed",
        ), dtype=pars.prec)

        return finished, num_finished_estimations

    def _loop_cond(
            self, pars: SimulationParameters,
            continue_flag: Tensor, *args,
    ):
        """On the basis of the fraction of completed estimations
        in the batch decides whether the measurement loop should
        be continued or stopped. The relevant attribute is
        `simpars.resources_fraction`
`
        Returns
        -------
        continue_resources: Tensor
            Bool value that regulates whether the loop
            in _loop_body should be stopped or not.
        """
        num_estimation_completed = cast(count_nonzero(
            logical_not(continue_flag),
            name="num_estimation_completed",
        ), dtype=pars.prec)

        continue_resources = less(
            num_estimation_completed,
            constant(pars.resources_fraction*self.bs,
                     dtype=pars.prec),
            name="continue_resources"
        )
        return continue_resources

    def _loop_body(
            self, pars, deploy, rangen,
            continue_flag, index, meas_step, used_resources,
            outcomes,
            sum_log_prob, true_values, weights, particles,
            true_state, state_ensemble, loss_diff, loss,
            history_input, history_control, history_resources,
            history_precision, hist_outcomes_rec, hist_controls_rec,
            hist_continue_rec, hist_step_rec
    ):
        """Measurement loop.
        """
        # Input of the control strategy
        input_strategy = self._generate_input(
            weights, particles, state_ensemble,
            cast(meas_step, dtype=pars.prec),
            used_resources,
            rangen,
        )

        log_prob_control = zeros((self.bs, 1), dtype=pars.prec)

        # Evaluation of the control strategy.
        # The stop gradient here greatly
        # reduces the memory consumption of the simulation
        cond_input = stop_gradient(input_strategy) if \
            pars.stop_gradient_input else input_strategy
        if self.discrete_controls:
            controls, log_prob_control = self.control_strategy(
                cond_input, rangen,
            )
        else:
            controls = self.control_strategy(cond_input)

        # tf.debugging.assert_all_finite(input_strategy, "input_strategy has NaN/Inf")
        # tf.debugging.assert_all_finite(controls, "controls has NaN/Inf")

        # Update the used resources
        new_used_resources = self.phys_model.wrapper_count_resources(
            used_resources, outcomes, controls, true_values,
            true_state, meas_step,
        )
        # tf.debugging.assert_all_finite(new_used_resources, "new_used_resources has NaN/Inf")

        # tf.print(
        #     "index=", index,
        #     " alive=", tf.math.count_nonzero(continue_flag),
        #     " ctrl0=", controls[0],
        #     " res0=", new_used_resources[0],
        #     summarize=-1
        # )

        # Which estimations in th batch have not
        # yet run out of resources?
        continue_flag = less_equal(
            new_used_resources,
            pars.max_resources*ones((self.bs, 1),
                                    dtype=pars.prec),
            name="continue_flag",
        )

        # Updates the number of already consumed resources
        used_resources = where(
            continue_flag, new_used_resources, used_resources,
            name="used_resources",
        )

        # Measurement on the probe
        outcomes, log_prob, post_true_state = \
            self.phys_model.wrapper_perform_measurement(
                expand_dims(controls, axis=1), true_values,
                true_state,
                expand_dims(meas_step, axis=1), rangen,
            )

        # We eliminate the extra dimension generated
        # by the method perform_measurement
        outcomes = outcomes[:, 0, :]

        # Update the true state
        continue_flag_state = reshape(
            continue_flag, (self.bs, 1, 1),
        )

        # Updates the true state of the probe
        true_state = where(
            broadcast_to(
                continue_flag_state,
                (self.bs, 1, self.state_size),
            ), post_true_state, true_state,
            name="true_state_update",
        )

        # Accumulation of the trajectory likelihood
        # of the measurement outcomes:
        if pars.loss_logl_outcomes:
            sum_log_prob = where(
                continue_flag,
                sum_log_prob+log_prob,
                sum_log_prob,
                name="sum_log_prob_outcomes"
            )

        # Accumulation of the trajectory likelihood
        # of the measurement controls (if they are
        # stochastically extracted):
        if pars.loss_logl_controls:
            sum_log_prob = where(
                continue_flag,
                sum_log_prob+log_prob_control,
                sum_log_prob,
                name="sum_log_prob_controls"
            )

        # Update of the particle filter ensemble
        if pars.stop_gradient_pf:
            post_weights, post_state_ensemble = \
                self.pf.apply_measurement(
                    stop_gradient(weights),
                    stop_gradient(particles),
                    stop_gradient(state_ensemble),
                    outcomes, controls,
                    meas_step,
                )
        else:
            post_weights, post_state_ensemble = \
                self.pf.apply_measurement(
                    weights,
                    particles,
                    state_ensemble,
                    outcomes, controls,
                    meas_step,
                )

        continue_flag_state = reshape(
            continue_flag, (self.bs, 1, 1),
        )
        
        # Updates the states ensemble
        state_ensemble = where(
            broadcast_to(
                continue_flag_state,
                (self.bs, self.pf.np, self.pf.state_size),
            ), post_state_ensemble, state_ensemble,
            name="state_ensemble_update",
        )

        # Weights update for the particle filter
        weights = where(
            continue_flag, post_weights, weights,
            name="weights",
        )

        hist_outcomes_rec = tensor_scatter_nd_add(
            hist_outcomes_rec, reshape(index, (1, 1)),
            expand_dims(outcomes, axis=0)
        )
        hist_controls_rec = tensor_scatter_nd_add(
            hist_controls_rec, reshape(index, (1, 1)),
            expand_dims(controls, axis=0)
        )
        hist_continue_rec = tensor_scatter_nd_add(
            hist_continue_rec, reshape(index, (1, 1)),
            cast(expand_dims(continue_flag, axis=0),
                 dtype="int32"),
        )
        hist_step_rec = tensor_scatter_nd_add(
            hist_step_rec, reshape(index, (1, 1)),
            expand_dims(meas_step, axis=0)
        )

        # Resample the particle filter ensemble
        if self.pf.resampling:
            if self.pf.res_frac is not None:
                    weights, particles, resampled = \
                        self.pf.full_resampling(
                            weights, particles, continue_flag, rangen,
                        )
                    if pars.permutation_invariant:
                        particles = sort(particles, axis=2)

                    # Recompute the states ensemble if the particle
                    # filter has been resampled (the particles
                    # have changed).
                    if self.recompute_state:
                        state_ensemble = cond(
                            resampled,
                            lambda: self.pf.recompute_state(
                                index, particles, hist_controls_rec,
                                hist_outcomes_rec, hist_continue_rec,
                                hist_step_rec,
                                pars.num_steps,
                            ),
                            lambda: state_ensemble,
                            name="state_resampling",
                        )
            else:
                weights, particles = \
                    self.pf.partial_resampling(
                        weights, particles, continue_flag, rangen,
                    )
                if pars.permutation_invariant:
                    particles = sort(particles, axis=2)

        # Measurement step counter update for
        # each estimation in the batch.
        meas_step = where(
            continue_flag, meas_step+1, meas_step,
            name="meas_step",
        )

        # Updates the iteration number
        index += 1

        # Computes the loss after each step of the
        # measurement loop if it needs to be accumulated.
        if pars.cumulative_loss and not deploy:
            loss_diff_partial, loss_partial = \
                self._compute_scalar_loss(
                    weights, particles, true_state,
                    state_ensemble, sum_log_prob,
                    true_values, used_resources, meas_step,
                    continue_flag,
                )
            loss_diff += loss_diff_partial
            loss += loss_partial

        # In the deploy mode we save the history of the inputs
        # and the precision after each measurement.
        if deploy:
            loss_value = self._loss_function(
                weights, particles, true_state, state_ensemble,
                true_values, used_resources, meas_step,
            )
            # Preprocessing of the history tensors to eliminate those
            # corresponding to already terminated estimations.
            processed_input_tensor = where(
                broadcast_to(continue_flag,
                             (self.bs, self.input_size)),
                input_strategy,
                zeros((self.bs, self.input_size), dtype=pars.prec)
            )
            processed_control = where(
                broadcast_to(continue_flag,
                             (self.bs, self.phys_model.controls_size)),
                controls,
                zeros(
                    (self.bs, self.phys_model.controls_size),
                    dtype=pars.prec),
            )
            processed_resources = where(
                broadcast_to(continue_flag, (self.bs, 1)),
                used_resources,
                zeros((self.bs, 1), dtype=pars.prec),
            )
            processed_precision = where(
                broadcast_to(continue_flag, (self.bs, 1)),
                loss_value,
                zeros((self.bs, 1), dtype=pars.prec),
            )
            history_input = concat(
                [expand_dims(processed_input_tensor, axis=0),
                 history_input], 0,
                name="history_input", )[:pars.num_steps, :, :]
            history_control = concat(
                [expand_dims(processed_control, axis=0),
                 history_control], 0,
                name="history_control", )[:pars.num_steps, :, :]
            history_resources = concat(
                [expand_dims(processed_resources, axis=0),
                 history_resources], 0,
                name="history_recources", )[:pars.num_steps, :, :]
            history_precision = concat(
                [expand_dims(processed_precision, axis=0),
                 history_precision], 0,
                name="history_precision", )[:pars.num_steps, :, :]

        return continue_flag, index, meas_step, used_resources, \
            outcomes, \
            sum_log_prob, true_values, weights, particles, \
            true_state, state_ensemble, loss_diff, loss, \
            history_input, history_control, \
            history_resources, history_precision, \
            hist_outcomes_rec, hist_controls_rec, \
            hist_continue_rec, hist_step_rec

    def execute(
            self, rangen: Generator, deploy: bool = False,
    ):
        """Measurement loop and loss evaluation.

        This method codifies the interaction between
        the sensor, the particle filter and the neural network.
        It contains the measurement loop schematically
        represented in the figure.

        .. image:: ../docs/_static/measurement_loop.png
            :width: 600
            :alt: measurement_loop

        When this method is executed in the context of a
        `GradientTape`, the return value `loss_diff`
        can be differentiated with respect to the
        `Variable` objects of the control strategy
        for training purposes. It can also be used
        to evaluate the performances of a control strategy
        without training it. For this use, setting
        `deploy=True` is particularly useful, since it
        lets the method return the history of the
        input to the neural network, of the controls,
        of the consumed resources, and of the precision
        (the used-defined loss).
        This methods is used in the
        functions :py:func:`~.utils.train` and
        :py:func:`~.utils.performance_evaluation`.

        **Achtung!** Ideally the user never needs to use
        directly this method; the functions of the
        :py:mod:`~.utils` module should suffice.

        The simulation of the metrological task
        is performed in a loop (called
        the measurement loop) which stops when the maximum
        number of iterations :py:attr:`~.SimulationParameters.num_steps`
        (called measurement steps) is reached or when the
        strict upper bound
        :py:attr:`~.SimulationParameters.max_resources`
        on the amount of resources has been saturated,
        as computed by the method
        :py:meth:`~.PhysicalModel.count_resources`

        Parameters
        ----------
        rangen: Generator
            Random number generator from the module
            :py:mod:`tensorflow.random`.
        deploy: bool
           If this flag is `True`, the function
           doesn't return the `loss_diff` parameter,
           but in its place the four histories
           `history_input`, `history_control`, `history_resources`,
           and `history_precision`.

        Returns
        -------
        loss_diff: Tensor
            `Tensor` of shape (,) and type `prec` containing
            the mean value of the loss on the batch of estimations
            performed in the measurement loop. This is the
            quantity to be differentiated if the
            :py:meth:`execute` method is called inside
            a `GradientTape`. This produces
            an unbiased estimator of the gradient
            of the precision for the strategy
            training. This version
            of the loss has been mixed with the
            log-likelihood terms for the measurement outcomes
            and/or the controls, according to the flags of the
            `simpars` attribute of the :py:obj:`~.Simulation`
            class, which is an instance of the
            :py:obj:`~.SimulationParameters` class. It is the
            mean loss as written in :eq:`log_mixing` (or in
            :eq:`baseline_correction` if there is a baseline)

            **Achtung!** This `Tensor` is returned only if the
            `deploy` flag passed to :py:meth:`execute` is `False`.
        loss: Tensor
            `Tensor` of shape (,) and type `prec` containing
            the mean value of the loss on the batch of estimations
            without the mixing with the
            log-likelihood terms. It is
            the "pure" version of the mean loss computed from the
            user define method
            :py:meth:`~.StatefulSimulation.loss_function` and
            takes into account the modifying flags
            :py:attr:`~.SimulationParameters.cumulative_loss`
            and :py:attr:`~.SimulationParameters.log_loss`.
            This scalar is not
            suitable to be the differentiated loss
            and it is only meant to be a reference value
            of the true loss during the gradient descent
            updates that uses instead `loss_diff`.
            This return value is used in the function
            :py:func:`~.utils.train` that produces,
            along with the trained strategy, the history of the
            mean losses during the training. This value
            is returned independently on the parameter
            `deploy`.
        history_input: Tensor
            `Tensor` of shape (`ns`, `bs`, `input_size`) of
            type `prec`, where `ns`, `bs`, and `input_size`
            are attributes of the :py:obj:`~.Simulation` class.
            It contains all the objects `input_strategy`
            generated by the user-defined method
            :py:meth:`~.StatefulSimulation.generate_input`
            and passed to the `control_strategy` attribute
            of the :py:obj:`~.Simulation` class, for a single
            call of the :py:meth:`execute` method, for
            all the estimations in the batch separately and all the
            steps of the measurement loop. For the measurement
            steps that are not performed because the estimation
            has already run out of resources the `history_input`
            is filled with zeros.

            **Achtung!** This `Tensor` is returned only if the
            `deploy` flag passed to :py:meth:`execute` is `True`.
        history_controls: Tensor
            `Tensor` of shape
            (`ns`, `bs`, `phys_model.controls_size`) of
            type `prec`, where `ns`, `bs`, and `phys_model`
            are attributes of the :py:obj:`~.Simulation` class.
            It contains all the controls returned by the
            callable attribute `control_strategy`
            of the :py:obj:`~.Simulation` class, for a single
            call of the :py:meth:`execute` method, for
            all the estimations in the batch separately and all the
            steps of the measurement loop. For the measurement
            steps that are not performed because the estimation
            has already run out of resources `history_controls`
            is filled with zeros.

            **Achtung!** This `Tensor` is returned only if the
            `deploy` flag passed to :py:meth:`execute` is `True`.
        history_resources: Tensor
            `Tensor` of shape (`ns`, `bs`, 1) of
            type `prec`, where `ns`, `bs` are attributes of
            the :py:obj:`~.Simulation` class.
            It contains the used resources accumulated during
            the estimation, as computed by the user-defined method
            :py:meth:`~.PhysicalModel.count_resources`, for a single
            call of the :py:meth:`execute` method, for
            all the estimations in the batch separately and all the
            steps of the measurement loop. For the measurement
            steps that are not performed because the estimation
            has already run out of resources `history_resources`
            is filled with zeros.

            **Achtung!** This `Tensor` is returned only if the
            `deploy` flag passed to :py:meth:`execute` is `True`.
        history_precision: Tensor
            `Tensor` of shape (`ns`, `bs`, 1) of
            type `prec`, where `ns`, `bs` are attributes of
            the :py:obj:`~.Simulation` class.
            It contains the loss as computed by the user-defined
            method :py:meth:`~.StatefulSimulation.loss_function`,
            for a single call of the :py:meth:`execute` method
            for all the estimations in the batch separately and all the
            steps of the measurement loop. For the measurement
            steps that are not performed because the estimation
            has already run out of resources `history_precision`
            is filled with zeros.

            **Achtung!** This `Tensor` is returned only if the
            `deploy` flag passed to :py:meth:`execute` is `True`.
        """
        pars = self.simpars

        # Index of the iteration in the measurement loop
        index = constant(0, dtype="int32")

        # Logarithmic probability of the string of
        # observed measurements and controls.
        # It must be accumulated during the procedure.
        sum_log_prob = zeros(
            (self.bs, 1), dtype=pars.prec,
            name="sum_log_prob",
        )

        true_values = self.phys_model.true_values(rangen)

        if pars.permutation_invariant:
            true_values = sort(true_values, axis=2)

        # Initialization of the particle ensemble
        weights, particles = self.pf.reset(rangen)

        if pars.permutation_invariant:
            particles = sort(particles, axis=2)

        # True state of the quantum probe
        true_state = self.phys_model.wrapper_initialize_state(
            true_values, 1,
        )

        # State particle filter
        state_ensemble = \
            self.pf.phys_model.wrapper_initialize_state(
                particles, self.pf.np,
            )

        # Cumulated consumed resources in the estimation
        used_resources = zeros(
            (self.bs, 1), dtype=pars.prec,
            name="used_resources",
        )

        # Are the resources terminated?
        continue_flag = ones(
            (self.bs, 1), dtype="bool",
            name="continue_flag",
        )

        outcomes = zeros(
            (self.bs, self.phys_model.outcomes_size), dtype=self.phys_model.prec,
            name="outcomes",
        )

        # Loss tensor to the differentiated
        loss_diff = zeros(
            (), dtype=pars.prec, name="loss_diff",
        )

        # Loss tensor without mixing with the
        # log-likelihood
        loss = zeros(
            (), dtype=pars.prec, name="loss",
        )

        # History of the input to the control strategy,
        # the precision and the consumed
        # resources for each measurement step.
        history_input = zeros(
            (pars.num_steps, self.bs, self.input_size),
            dtype=pars.prec,
            name="history_input",
        )
        history_controls = zeros(
            (pars.num_steps, self.bs,
             self.phys_model.controls_size),
            dtype=pars.prec,
            name="history_input",
        )
        history_resources = zeros(
            (pars.num_steps, self.bs, 1), dtype=pars.prec,
            name="history_resources",
        )
        history_precision = zeros(
            (pars.num_steps, self.bs, 1), dtype=pars.prec,
            name="history_precision"
        )

        # Measurement counter for each estimation in the batch
        meas_step = zeros((self.bs, 1), dtype="int32", name="step")

        # Histories used to recomputing the state ensemble
        # after a resampling of the particles.
        hist_outcomes_rec = zeros(
            (pars.num_steps, self.bs, self.phys_model.outcomes_size),
            dtype=pars.prec,
        )
        hist_control_rec = zeros(
            (pars.num_steps, self.bs, self.phys_model.controls_size),
            dtype=pars.prec,
        )
        hist_continue_rec = zeros(
            (pars.num_steps, self.bs, 1), dtype="int32",
        )
        hist_step_rec = zeros(
            (pars.num_steps, self.bs, 1), dtype="int32",
        )

        # For using the tensorflow loop in the simulation
        # we have to list all
        # the tensors used and modified in the loop.
        loop_variables = (
            continue_flag, index, meas_step, used_resources,
            outcomes,
            sum_log_prob, true_values, weights, particles,
            true_state, state_ensemble, loss_diff, loss,
            history_input, history_controls,
            history_resources, history_precision,
            hist_outcomes_rec, hist_control_rec,
            hist_continue_rec, hist_step_rec,
        )

        # Measurement loop
        continue_flag, index, meas_step, used_resources, outcomes, sum_log_prob, \
            true_values, weights, particles, true_state, state_ensemble, \
            loss_diff, loss, history_input, history_controls, \
            history_resources, history_precision, \
            hist_outcomes_rec, hist_control_rec, \
            hist_continue_rec, hist_step_rec = \
            while_loop(
                lambda *args: self._loop_cond(pars, *args),
                lambda *args: self._loop_body(pars, deploy,
                                              rangen, *args),
                loop_variables,
                maximum_iterations=pars.num_steps,
                name="main_loop"
            )

        # Normalization of the losses
        if not deploy:
            # Training mode
            if pars.cumulative_loss:
                total_step_number = cast(
                    reduce_max(meas_step) if pars.end_batch
                    else reduce_sum(meas_step),
                    dtype=pars.prec,
                )
                loss_diff *= reciprocal_no_nan(total_step_number)
                loss *= reciprocal_no_nan(total_step_number)
            else:
                finished, num_finished_estimations = \
                    self._compute_finished(
                        continue_flag, meas_step,
                    )
                loss_diff, loss = self._compute_scalar_loss(
                    weights, particles, true_state,
                    state_ensemble, sum_log_prob,
                    true_values, used_resources,
                    meas_step, finished,
                )

                loss_diff *= reciprocal_no_nan(
                    num_finished_estimations,
                )
                loss *= reciprocal_no_nan(
                    num_finished_estimations,
                )
            return loss_diff, loss
        # Deploy mode
        history_input = reshape(
            history_input, (self.bs*pars.num_steps,
                            self.input_size),
            name="history_input",
        )
        history_controls = reshape(
            history_controls, (self.bs*pars.num_steps,
                               self.phys_model.controls_size),
            name="history_control",
        )
        history_resources = reshape(
            history_resources, (self.bs*pars.num_steps, 1),
            name="history_resources",
        )
        history_precision = reshape(
            history_precision, (self.bs*pars.num_steps, 1),
            name="history_precision",
        )
        return true_values, history_input, history_controls, \
            history_resources, history_precision

    def __str__(self):
        return f"{self.simpars.sim_name}_batchsize_" \
            f"{self.bs}_num_steps_{self.simpars.num_steps}_" \
            f"max_resources_{self.simpars.max_resources:.2f}_" \
            f"ll_{self.simpars.log_loss}_cl_" \
            f"{self.simpars.cumulative_loss}"
