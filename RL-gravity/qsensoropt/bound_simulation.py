"""Module containing the :py:obj:`~.BoundSimulation` class,
for optimizing the Cramér-Rao bound for
an arbitrary estimation task.
"""

from typing import List, Callable, Optional

from tensorflow import zeros, ones, \
    stop_gradient, expand_dims, constant, while_loop, \
    cast, where, concat, reshape, broadcast_to, \
    tensor_scatter_nd_add, eye, transpose, \
    einsum, print, Tensor, GradientTape
from tensorflow.math import less, count_nonzero, \
    logical_not, log, exp, reduce_sum, less_equal, cumsum
from tensorflow.random import Generator
from tensorflow.linalg import inv, matmul, trace

from .physical_model import PhysicalModel
from .simulation_parameters import SimulationParameters
from .utils import normalize, random_uniform, get_seed


class BoundSimulation:
    r"""This simulation optimizes the Cramér-Rao bound
    (based on the Fisher information matrix)
    for the local estimation of the parameters
    specified in the :py:obj:`~.PhysicalModel`
    object. This class is used for frequentist inference,
    instead of :py:obj:`~.Simulation` which applies
    to Bayesian inference.

    The Cramér-Rao bound (CR bound) is a lower
    bound on the Mean Square Error (MSE) matrix of
    the frequentist estimator :math:`\hat{\vec{\theta}}`
    at the position :math:`\vec{\theta}`. In formula

    .. math::
        \Sigma_{ij} = \mathbb{E} \left[ (\hat{\vec{\theta}}
        - \vec{\theta})_i
        (\hat{\vec{\theta}} - \vec{\theta})_j
        \right] \ge (F^{-1})_{ij} \; ,
        :label: cr_bound

    in other words it codifies the maximum
    achievable precision
    of an estimator around :math:`\vec{\theta}`, that
    is, the ability to distinguish reliably two close
    values :math:`\vec{\theta}` and
    :math:`\vec{\theta}+\delta \vec{\theta}`.
    In the above formula we didn't write
    explicitly the dependence of the estimator
    :math:`\hat{\vec{\theta}} (\vec{x}, \vec{y})` on the
    outcomes :math:`\vec{y}` and controls :math:`\vec{x}`
    trajectories of the experiment in order
    not to make the notation too heavy.
    The expectation
    value is taken on many realizations
    of the experiment, i.e. on the
    probability distribution of the trajectories.
    On the right side of :eq:`cr_bound`
    the Fisher information matrix (FI matrix)
    appears, which is defined as following

    .. math::
        F_{ij} = \mathbb{E} \left[
        \frac{\partial}{\partial
        \theta_i}
        \log p(\vec{y}|\vec{\theta}, \vec{x}) \cdot
        \frac{\partial}
        {\partial \theta_j}
        \log p(\vec{y}|\vec{\theta}, \vec{x}) \right] \; ,

    being :math:`p(\vec{y}|\vec{\theta}, \vec{x})` the
    probability of the observed trajectory of outcomes
    at the point :math:`\vec{\theta}`.
    Also in this case the expectation value
    is taken on the experiment realizations.
    A scalar value for the precision
    is typically build from the matrix
    :math:`\Sigma_{ij}`
    by contracting it with a positive
    semidefinite weight matrix :math:`G`,
    that is the attribute `cov_weight_matrix_tensor`
    of :py:obj:`~.BoundSimulation`, this
    gives the scalar CR bound:

    .. math::
        \text{tr} (G \cdot \Sigma_{ij}) \ge
        \text{tr} ( G \cdot F^{-1}) :=
        \mathcal{L}(\vec{\lambda})\; .

    This scalar value is the loss to be minimized
    in the training. Let us indicate
    with :math:`\vec{\lambda}` the parameters
    of the control strategy (the weights and biases
    of a neural network), then the derivative of the
    loss with respect to them can be written as

    .. math::
        \frac{\partial \mathcal{L}}
        {\partial \vec{\lambda}} = \text{tr} \left(
        F^{-1} G F^{-1} \cdot \frac{\partial F}{\partial
        \vec{\lambda}} \right) \; .

    The expectation value in the definition
    of :math:`F` is approximated in the simulation
    by averaging the product of the
    derivatives log-likelihoods in the batch,
    i.e.

    .. math::
        F \simeq \hat{F} = \frac{1}{B}\sum_{k=1}^{B}
        \frac{\partial}{\partial
        \theta_i}
        \log p(\vec{y}_k|\vec{\theta}, \vec{x}_k)
        \cdot \frac{\partial}{\partial \theta_j}
        \log p(\vec{y}_k|\vec{\theta}, \vec{x}_k) =
        \frac{1}{B} \sum_{k=1}^B f_k \; .
        :label: def_f_k

    where :math:`(\vec{x}_k, \vec{y}_k)`
    is the trajectory of a particular
    realization of the experiment
    in the batch of simulations and
    :math:`f_k` is the observed
    Fisher information. The unbiased gradient of
    the FI matrix, that takes into account
    also the gradient of the probability
    distribution in the expectation value,
    can be computed as following

    .. math::
        \frac{\partial \mathcal{L}}
        {\partial \vec{\lambda}} \simeq
        \frac{1}{B} \frac{\partial} {\partial \vec{\lambda}}
        \text{tr} \Big \lbrace \text{sg} \left( \hat{F}^{-1}
        G \hat{F}^{-1} \right) \sum_{k=1}^B \left[
        f_k + \text{sg} (f_k) \log p(\vec{y},
        \vec{x}|\vec{\theta}) \right] \Big \rbrace \; .
        :label: grad_cr

    The :math:`\text{sg} (\cdot)` is the `stop_gradient`
    operator, and the probability :math:`p(\vec{y},
    \vec{x}|\vec{\theta})` is the likelihood
    of the particular trajectory, that contains
    both the probability of the stochastic outcome
    and the probability of the control
    (in case it is stochastically generated).

    By activating the flag
    :py:attr:`~.SimulationParameters.log_loss`
    the loss function becomes

    .. math::
        \mathcal{L}_{\text{log}} (\vec{\lambda})
        = \log \text{tr} ( G \cdot F^{-1}) \; ,
        :label: log_loss_cr

    this is particularly useful to stabilize
    the training when the
    CR bounds spans multiple orders of
    magnitude.

    For minimizing the CR bound
    around the point :math:`\vec{\theta}`
    the :py:obj:`~.Parameter` objects
    in `phys_model` should be initialized with
    the parameter `values` equal to the
    tuple containing a single element, that
    is the value of the parameter at the
    estimation point. For example the estimation
    around the point :math:`\theta=0` can be
    set with the following code

    ``Parameter(values=(0.0, ), name='Theta')``

    This would correspond to a delta-like
    prior for the parameter `Theta`.
    If a different prior :math:`\pi(\vec{\theta})`
    is used, for example a uniform prior
    in a range, realized through the parameter
    `bounds` of the :py:obj:`~.Parameter`
    constructor, then the FI matrix
    approximated by the simulation is

    .. math::
        \bar{F} = \int F (\vec{\theta})
        \pi (\vec{\theta}) d \vec{\theta} \; ,

    and the minimized loss is

    .. math::
        \text{tr} \left[ G \cdot \bar{F}^{-1} \right] \le
        \int \text{tr} \left( G
        \cdot F^{-1} (\vec{\theta}) \right)
        d \vec{\theta} \; ,

    which is a lower bound on the expectation
    value of the CR bound, because of the
    Jensen inequality applied to the matrix inverse.
    In the case of a single parameter the
    training maximizes the expected value of
    the Fisher information on the
    prior :math:`\pi(\theta)`.

    With
    the parameter `importance_sampling` it
    is possible to use a custom distribution
    for extracting the trajectory. See
    the documentation of the class
    constructor for more details.

    Notes
    -----
    We can use this class also for the optimization
    of unitary quantum controls on a systems, where the
    only measurement is performed at the end.

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
    simpars: :py:obj:`~.SimulationParameters`
        Parameter `simpars` passed to the
        class constructor.
    ns: int
        Maximum number of steps of the
        measurement loop in the :py:meth:`execute` method.
        It is the :py:attr:`~.SimulationParameters.num_steps`
        attribute of `simpars`.
    input_size: int
        Size of the last dimension of the `input_strategy`
        object passed to the callable attribute
        `control_strategy`
        and generated by the method
        :py:meth:`~.StatefulSimulation.generate_input`.
        It is the number of scalars on which
        `control_strategy` bases the prediction
        for the next control. By default
        it is computed
        in the constructor of the class as

        ``input_size = d+state_size+outcomes_size+2``

        where `d` is the number of parameters
        to be estimated. See also the documentation
        of the
        :py:meth:`~.BoundSimulation.generate_input`
        method.
    input_name: List[str]
        List of names for each of the scalar inputs
        returned by the method
        :py:meth:`~.BoundSimulation.generate_input`.

        The list `input_name` contains the following
        elements:

        * The name of the `d` parameters
          in `phys_model.parameters`. These
          are `d` elements.
        * The strings "State_#, one for each
          scalar component of the state of the
          physical model. These are
          `state_size` entries.
        * The strings "Outcome_#", one for each
          scalar outcome. These are
          `phys_model.outcomes_size`
          elements.
        * The strings "Step" and "Res", referring
          respectively to the index of the
          measurement and the consumed resources.

        This list of names is used in
        the function
        :py:func:`~.utils.store_input_control`, that
        saves a history of every input to
        the neural network, in order for the user
        to be able to
        interpret the actions
        of the optimized control strategy.
        See also the documentation
        of the
        :py:meth:`~.BoundSimulation.generate_input`
        method.
    cov_weight_matrix_tensor: Tensor
        Tensorial version of the parameter
        `cov_weight_matrix` passed to the
        class constructor. It is a `Tensor` of shape
        (`bs`, `d`, `d`) and of type `prec` that
        contains `bs` repetitions of
        `cov_weight_matrix`.
        In case no `cov_weight_matrix`
        is passed to the constructor,
        `cov_weight_matrix_tensor`
        contains `bs` copies of the
        `d`-dimensional identity matrix.
    discrete_controls: bool
        This flag is `True` if at least
        one of the controls is stochastically
        generated in the `control_strategy`
        function.
    random: bool = False
        Flag `random` passed to the
        class constructor.
    importance_sampling: bool = False
        Flag `importance_sampling`
        passed to the class constructor.
    """

    def __init__(
            self,
            phys_model: PhysicalModel,
            control_strategy: Callable,
            simpars: SimulationParameters,
            cov_weight_matrix: Optional[List] = None,
            importance_sampling: bool = False,
            random: bool = False,
    ):
        r"""Parameters passed to the
        :py:obj:`~.BoundSimulation` class
        constructor.

        Parameters
        ----------
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
            :py:meth:`~.BoundSimulation.generate_input`.
            The :py:obj:`~.BoundSimulation`
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
        simpars: :py:obj:`~.SimulationParameters`
            Contains the flags and parameters
            that regulate the stopping
            condition of the measurement loop
            and modify the loss function used in the
            training.
        cov_weight_matrix: List, optional
            Weight matrix for the multiparametric
            estimation problem. It should be a
            `(d, d)` list, with `d` being the number
            of estimation parameters.
            If this parameter is not passed the
            default value of the weight matrix is the
            identity.
        random: bool = False
            Append a random number to the
            `input_strategy` `Tensor` produced by
            :py:meth:`~.BoundSimulation.generate_input`
            for each element in `phys_model.controls`.
        importance_sampling: bool = False
            It is possible to implement the method
            :py:meth:`~.StatefulPhysicalModel.perform_measurement`
            so that is extracts an outcome not from
            the true probability determined
            by the Born rule on the state of the
            probe, called
            :math:`p(y|\vec{\theta}, \vec{x})`,
            but from an arbitrary modified
            distribution
            :math:`\widetilde{p}(y|\vec{\theta}, \vec{x})`.
            As long as the probability implemented in
            the method
            :py:meth:`~.StatefulPhysicalModel.model`
            is the correct one, the class
            :py:obj:`~.BoundSimulation` will
            be able to perform the CR bound
            optimization. The trajectory of the
            system is sampled according to
            :math:`\widetilde{p}`, and the FI matrix
            is computed as

            .. math::
                F = \mathbb{E}_{\widetilde{p}} \left[
                \frac{\partial}{\partial \theta_i}
                \log p(\vec{y}|\vec{\theta}, \vec{x})
                \cdot
                \frac{\partial}{\partial \theta_j}
                \log p(\vec{y}|\vec{\theta}, \vec{x})
                \frac{p(\vec{y}|\vec{\theta}, \vec{x})}
                {\widetilde{p}(\vec{y}|\vec{\theta}, \vec{x})}
                \right]

            which can be approximated on a batch as

            .. math::
                F \simeq \frac{1}{B} \sum_{k=1}^B f_k
                \frac{p(\vec{y}_k|\vec{\theta}, \vec{x}_k)}
                {\widetilde{p}(\vec{y}_k|\vec{\theta}, \vec{x}_k)}

            with :math:`f_k` defined in
            :eq:`def_f_k`. Also the gradient
            of :math:`F` is changed accordingly.

            Typically the distribution :math:`\widetilde{p}`
            is some perturbation of :math:`p`, for example
            it can be obtained by mixing :math:`p`
            with a uniform distribution on the outcomes.

            The importance sampling is useful when some
            trajectories have vanishingly small
            probability of occurring according to
            the model :math:`p`
            but contribute significantly to the
            Fisher information. If these trajectories
            have some probability of occurring
            sampling with :math:`\widetilde{p}`,
            then the estimator of the FI might
            be more accurate.

            The drawback is that the
            perturbed probability of the
            complete trajectory
            :math:`\widetilde{p}(\vec{y}|\vec{\theta}, \vec{x})`
            might be too different from the
            real probability (because of the
            accumulation of the perturbation at each step),
            so that the simulation might entirely miss the
            region in the space of trajectories
            in which the system state moves, thus
            delivering a bad estimate of the
            Fisher information and a bad control strategy,
            upon completion of the training.
            Whether or not the importance sampling can
            be beneficial to the
            optimization should be checked case by case.
        """
        self.bs = phys_model.bs
        self.phys_model = phys_model
        self.control_strategy = control_strategy
        self.simpars = simpars
        self.ns = simpars.num_steps
        self.random = random
        self.importance_sampling = importance_sampling

        if not simpars.prec in ("float64", "float32"):
            raise ValueError("The allowed values of \
                             prec are float32 and float64.")

        # Weight matrix or the loss
        if cov_weight_matrix is None:
            self.cov_weight_matrix_tensor = \
                eye(self.phys_model.d, dtype=simpars.prec)
        else:
            self.cov_weight_matrix_tensor = constant(
                cov_weight_matrix, dtype=simpars.prec
            )

        # Is there at least a discrete control?
        self.discrete_controls = False
        for contr in self.phys_model.controls:
            self.discrete_controls = contr.is_discrete or \
                self.discrete_controls

        self.state_size = phys_model.state_specifics['size']
        self.state_type = phys_model.state_specifics['type']

        self.input_size = \
            self.phys_model.d+self.state_size +\
            self.phys_model.outcomes_size+2
        self.input_name = []
        for param in self.phys_model.params:
            self.input_name.append(param.name+"_norm")
        for i in range(self.state_size):
            self.input_name.append(f"State_{i}")
        for i in range(self.phys_model.outcomes_size):
            self.input_name.append(f"Outcome_{i}")
        self.input_name.append("Step")
        self.input_name.append("Res")
        if self.random:
            self.input_size += self.phys_model.controls_size
            for control in self.phys_model.controls:
                self.input_name.append(f"{control.name}_Rnd")

    def generate_input(
        self, true_values: Tensor, true_state: Tensor,
        meas_step: Tensor, used_resources: Tensor,
        outcomes: Tensor,
        rangen: Generator,
    ):
        """Constructs the input of `control_strategy`
        on the basis of which the next controls
        are predicted (possibly adaptively).

        **Achtung!** If the user desires a different
        `input_strategy` `Tensor` he can redefine
        this method in a child class of
        :py:obj:`~.BoundSimulation`.

        Parameters
        ----------
        true_values: Tensor
            Contains the true values of the unknown
            parameters at which the Fisher information
            and the Cramér-Rao bound should be computed.
            It is a `Tensor` of shape
            (`bs`, 1, `phys_model.d`) and type `prec`, where
            `bs`, `pf`, and `prec` are
            attributes of the :py:obj:`~.BoundSimulation` class.
        true_states: Tensor
            Contains the state of the quantum
            probe at the current time for
            all the estimations in the batch.
            The stochastic evolution induced
            by
            :py:meth:`~.StatefulPhysicalModel.perform_measurement`
            causes the trajectories of the
            state of the system to diverge.
            This is a `Tensor` of shape
            (`bs`, 1, `state_size`),
            where `bs` and `state_size` are attributes of
            the :py:obj:`~.BoundSimulation` class.
            Its type is `state_type`.
        meas_step: Tensor
            The index of the current measurement
            on the probe system.
            The counting starts from zero.
            It is a `Tensor` of shape (`bs`, 1) and of type `prec`.
            `bs` is the attribute of
            the :py:obj:`~.BoundSimulation` class.
        used_resources: Tensor
            A `Tensor` of shape (`bs`, 1) containing
            the total amount of consumed resources for
            each estimation in the batch up to the
            point this method is called.
            `bs` is the attribute of the
            :py:obj:`~.BoundSimulation` class.
            The resources
            are counted according to the user defined method
            :py:meth:`~.PhysicalModel.count_resources`
            from the attribute `phys_model` of
            the :py:obj:`~.BoundSimulation` class.
        outcomes: Tensor
            Outcomes of the last measurement
            on the probe, generated by the method
            :py:meth:`~.StatefulPhysicalModel.perform_measurement`.
            It is a `Tensor` of shape
            (`bs`, 1, `phys_model.outcomes_size`) of type `prec`.
        rangen: Generator
            A random number generator from the module
            :py:mod:`tensorflow.random`.

        Return
        ------
        Tensor:
            `Tensor` of shape (`bs`, `input_size`)
            and type `prec`, with its columns being
            in order

            * `true_values` normalized in `[-1, +1]`,
              that is, the lowest possible value of each
              parameter is mapped to `-1` and the greates
              to `+1`. If only a single value is admissible
              for the parameter, then its corresponding
              column contains a series of ones.
            * `true_state`, i.e. the unnormalized
              state of `phys_model` at the moment
              this function is called.
            * `outcomes`, the unnormalized outcomes
              of the last measurement.
            * `meas_step` normalized in `[-1, +1]`
            * `used_resources` normalized in `[-1, +1]`
            * `d` columns of random number uniformly
              extracted in `[0, 1]`, where `d` is the
              number of parameters to estimate. This
              column is present only if `random=True`.
        """
        pars = self.simpars
        input_tensor = zeros((self.bs, 0), dtype=pars.prec)
        for i, param in enumerate(self.phys_model.params):
            if (not param.continuous) and len(param.values) == 1:
                input_tensor = concat(
                    [input_tensor,
                     true_values[:, 0, i:(i+1)]/param.values[0]], 1,
                )
            else:
                input_tensor = concat(
                    [input_tensor, normalize(
                        true_values[:, 0, i:(i+1)],
                        self.phys_model.params[i].bounds
                    )], 1
                )
        input_tensor = concat(
            [input_tensor,
             cast(true_state[:, 0, :], dtype=pars.prec),
             outcomes,
             normalize(meas_step, (0, pars.num_steps)),
             normalize(used_resources, (0, pars.max_resources))], 1,
        )
        if self.random:
            for _ in range(self.phys_model.controls_size):
                seed = get_seed(rangen)
                random_num = expand_dims(random_uniform(
                    self.bs, pars.prec, -1, +1, seed,
                ), axis=1)
            input_tensor = concat([input_tensor, random_num], 1)
        return input_tensor

    def _loop_cond(
            self, pars: SimulationParameters,
            continue_flag: Tensor, *args,
    ):
        """On the basis of the fraction of completed estimations
        in the batch this function decides whether
        the measurement loop should
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
            continue_flag, index, meas_step, outcomes,
            used_resources, table_log_prob_outcomes,
            table_target_log_prob_outcomes,
            table_log_prob_controls, true_values,
            true_state, history_input, history_control,
            history_resources,
    ):
        """Measurement loop.
        """
        # Input of the control strategy
        input_strategy = self.generate_input(
            true_values, true_state, cast(meas_step, dtype=pars.prec),
            used_resources, outcomes, rangen,
        )

        # Evaluation of the control strategy.
        # The stop gradient here greatly
        # reduces the memory consumption of the simulation
        cond_input = stop_gradient(input_strategy) if \
            pars.stop_gradient_input else input_strategy
        if self.discrete_controls:
            controls, log_prob_controls = self.control_strategy(
                cond_input, rangen,
            )
        else:
            controls = self.control_strategy(cond_input)

        # Update the used resources
        new_used_resources = self.phys_model.count_resources(
            used_resources, controls, true_values, meas_step,
        )

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
        outcomes, log_prob_outcomes, post_true_state = \
            self.phys_model.wrapper_perform_measurement(
                expand_dims(controls, axis=1), true_values,
                true_state,
                expand_dims(meas_step, axis=1), rangen,
            )

        # If the importance sampling is used
        # the true state of the system
        # and the probability of the
        # observed outcomes must be recomputed.
        if self.importance_sampling:
            target_prob_outcomes, post_true_state = \
                self.phys_model.wrapper_model(
                    outcomes, expand_dims(controls, axis=1), true_values,
                    true_state, expand_dims(meas_step, axis=1),
                )
            log_target_prob_outcomes = log(target_prob_outcomes)
        else:
            log_target_prob_outcomes = log_prob_outcomes

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

        table_log_prob_outcomes = tensor_scatter_nd_add(
            table_log_prob_outcomes, reshape(index, (1, 1)),
            transpose(log_prob_outcomes, [1, 0]),
            name="table_log_prob_outcomes"
        )

        table_target_log_prob_outcomes = tensor_scatter_nd_add(
            table_target_log_prob_outcomes, reshape(index, (1, 1)),
            transpose(log_target_prob_outcomes, [1, 0]),
            name="table_target_log_prob_outcomes"
        )

        # Accumulation of the trajectory likelihood
        # of the measurement controls (if they are
        # stochastically extracted):
        if pars.loss_logl_controls:
            table_log_prob_controls = tensor_scatter_nd_add(
                table_log_prob_controls, reshape(index, (1, 1)),
                transpose(log_prob_controls, [1, 0]),
                name="table_log_prob_controls"
            )

        continue_flag_state = reshape(
            continue_flag, (self.bs, 1, 1),
        )

        # Measurement step counter update for
        # each estimation in the batch.
        meas_step = where(
            continue_flag, meas_step+1, meas_step,
            name="meas_step",
        )

        # Updates the iteration number
        index += 1

        # In the deploy mode we save the history of the inputs
        # and the precision after each measurement.
        if deploy:
            # Preprocessing of the history tensors
            # to eliminate those
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

        return continue_flag, index, meas_step, outcomes, \
            used_resources, table_log_prob_outcomes, \
            table_target_log_prob_outcomes, \
            table_log_prob_controls, true_values, true_state, \
            history_input, history_control, \
            history_resources

    def execute(
            self, rangen: Generator, deploy: bool = False,
    ):
        r"""Measurement loop and Fisher information
        computation.

        This method codifies the interaction between
        the sensor and the neural network (NN).
        It contains the measurement loop inside
        a `GradientTape` object used to compute the
        Fisher information matrix.
        This is done by taking the derivatives
        of the trajectory log-likelihood with respect to the
        parameters to estimate.

        .. image:: ../docs/_static/measurement_loop_bound.png
            :width: 650
            :alt: measurement_loop_fisher

        When this method is executed in the context of
        an external
        `GradientTape`, the return value `loss_diff`
        can be differentiated with respect to the
        `Variable` objects of the control strategy
        for training purposes. It can also be used
        to evaluate the performances of a control strategy
        without training it. For this use, setting
        `deploy=True` is particularly useful, since it
        lets the method return the history of the
        input to the neural network, of the controls,
        of the consumed resources, and of the Cramér-Rao
        bound for each step of the measurement loop.
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
           `history_input`, `history_control`,
           `history_resources`,
           and `history_precision`.

        Returns
        -------
        loss_diff: Tensor
            `Tensor` of shape (,) and type `prec` containing
            the differentiable loss
            estimated on the batch
            of simulations. It is the
            quantity to be differentiated if the
            :py:meth:`execute` method is called inside
            a `GradientTape`. This produces
            an unbiased estimator of the gradient
            of the CR bound to be used in the
            training. To compute `loss_diff`
            the CR bound has been mixed with the
            log-likelihood terms for the measurement outcomes
            and/or the controls, according to the flags of the
            `simpars` attribute of the :py:obj:`~.BoundSimulation`
            class, which is an instance of the
            :py:obj:`~.SimulationParameters` class.
            In formula it is the argument of the gradient
            in :eq:`grad_cr`.

            **Achtung!** This `Tensor` is returned only if the
            `deploy` flag passed to :py:meth:`execute` is `False`.
        loss: Tensor
            `Tensor` of shape (,) and type `prec` containing
            the estimator of the CR computed from
            :math:`\hat{F}`, defined in :eq:`def_f_k`.
            If
            :py:attr:`~.SimulationParameters.log_loss`
            is true, the loss is
            :eq:`log_loss_cr`. This scalar is not
            suitable to be differentiated
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
            are attributes of the :py:obj:`~.BoundSimulation` class.
            It contains all the objects `input_strategy`
            generated by the method
            :py:meth:`~.StatefulSimulation.generate_input`
            and passed to the `control_strategy` attribute
            of the :py:obj:`~.BoundSimulation` class, for a single
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
            are attributes of the :py:obj:`~.BoundSimulation` class.
            It contains all the controls returned by the
            callable attribute `control_strategy`
            of the :py:obj:`~.BoundSimulation` class, for a single
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
            the :py:obj:`~.BoundSimulation` class.
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
            the :py:obj:`~.BoundSimulation` class.
            It contains the CR bound for a single call
            of the :py:meth:`execute` method
            for all the steps of the measurement loop.
            In the second dimension the values of this
            object are repeated. For the measurement
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
        # observed measurements.
        # It must be accumulated during the procedure.
        table_log_prob_outcomes = zeros(
            (pars.num_steps, self.bs), dtype=pars.prec,
            name="table_log_prob_outcomes",
        )

        table_target_log_prob_outcomes = zeros(
            (pars.num_steps, self.bs), dtype=pars.prec,
            name="table_target_log_prob_outcomes",
        )

        # Logarithmic probability of the string of
        # observed controls.
        # It must be accumulated during the procedure.
        table_log_prob_controls = zeros(
            (pars.num_steps, self.bs), dtype=pars.prec,
            name="table_log_prob_controls",
        )

        true_values = self.phys_model.true_values(rangen)

        # True state of the quantum probe
        true_state = self.phys_model.wrapper_initialize_state(
            true_values, 1,
        )

        # Cumulated consumed resources in the estimation
        used_resources = zeros(
            (self.bs, 1), dtype=pars.prec,
            name="used_resources",
        )

        # Outcomes
        outcomes = zeros(
            (self.bs, self.phys_model.outcomes_size),
            dtype=pars.prec,
            name="outcomes",
        )

        # Are the resources terminated?
        continue_flag = ones(
            (self.bs, 1), dtype="bool",
            name="continue_flag",
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

        # Measurement counter for each estimation in the batch
        meas_step = zeros((self.bs, 1), dtype="int32", name="step")

        # For using the tensorflow loop in the simulation
        # we have to list all
        # the tensors used and modified in the loop.
        loop_variables = (
            continue_flag, index, meas_step,
            outcomes, used_resources,
            table_log_prob_outcomes, table_target_log_prob_outcomes,
            table_log_prob_controls, true_values, true_state,
            history_input, history_controls,
            history_resources
        )

        # GradientTape to compute the Fisher information
        with GradientTape(persistent=False) as tape:
            tape.watch(true_values)
            # Measurement loop
            continue_flag, index, meas_step, outcomes, used_resources, \
                table_log_prob_outcomes, table_target_log_prob_outcomes, \
                table_log_prob_controls, \
                _, true_state, history_input, history_controls, \
                history_resources = \
                while_loop(
                    lambda *args: self._loop_cond(pars, *args),
                    lambda *args: self._loop_body(pars, deploy,
                                                  rangen, *args),
                    loop_variables,
                    maximum_iterations=pars.num_steps,
                    name="main_loop"
                )
            table_log_prob_outcomes = transpose(
                table_log_prob_outcomes, [1, 0],
            )
            table_target_log_prob_outcomes = transpose(
                table_target_log_prob_outcomes, [1, 0],
            )
            table_log_prob_controls = transpose(
                table_log_prob_controls, [1, 0],
            )
        # First derivatives of the outcome probabilities
        der1_likelihood = tape.batch_jacobian(
            table_target_log_prob_outcomes, true_values,
        )
        der1_likelihood = der1_likelihood[:, :, 0, :]
        if not deploy:
            # Derivative of the likelihood of the trajectories
            # in the batch
            sum_der1_likelihood = reduce_sum(
                der1_likelihood, axis=1,
            )
            # Square of the derivatives
            hist_der1der1_likelihood = einsum(
                'ai,aj->aij', sum_der1_likelihood,
                sum_der1_likelihood,
            )
            # The likelihood of the trajectory
            # depends also on the controls
            if pars.loss_logl_outcomes:
                sum_log_prob = reduce_sum(
                    table_log_prob_outcomes+table_log_prob_controls,
                    axis=1,
                )
            else:
                sum_log_prob = reduce_sum(
                    table_log_prob_controls, axis=1,
                )
            importance_sampling_factor = broadcast_to(reshape(exp(
                reduce_sum(
                    table_target_log_prob_outcomes-table_log_prob_outcomes,
                    axis=1,)
            ), (self.bs, 1, 1)),
                (self.bs, self.phys_model.d, self.phys_model.d))
            hist_der1der1_likelihood = importance_sampling_factor *\
                hist_der1der1_likelihood
            # Expectation value on the batch of the
            # squared first derivatives
            fisher = reduce_sum(
                hist_der1der1_likelihood, axis=0)/self.bs
            # Differentiable fisher
            der_fisher = -reduce_sum(
                hist_der1der1_likelihood +
                stop_gradient(hist_der1der1_likelihood) *
                broadcast_to(reshape(sum_log_prob,
                                     (self.bs, 1, 1)),
                             (self.bs, self.phys_model.d,
                             self.phys_model.d)), axis=0)/self.bs
            inv_fisher = inv(fisher)
            # Transformed weights to account for the
            # inverse of the Fisher matrix
            trans_weights = stop_gradient(matmul(
                matmul(inv_fisher, self.cov_weight_matrix_tensor),
                inv_fisher,
            ))
            # Differentiable loss
            loss_diff = trace(matmul(trans_weights, der_fisher))
            # Loss
            loss = trace(matmul(
                self.cov_weight_matrix_tensor, inv_fisher),
            )
            # Logarithmic loss, it is useful if the
            # Fisher information spans several orders of
            # magnitude during the training
            print("Loss:")
            print(loss)
            if pars.log_loss:
                return loss_diff/stop_gradient(loss), log(loss)
            else:
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
        # The precision history if the
        # Cramer-Rao bound as a function
        # of measurement step
        # Likelihood of the partial trajectories
        cum_der1_likelihood = cumsum(
            der1_likelihood, axis=1, reverse=True,
        )
        # Square of the likelihoods of the
        # partial trajectories
        hist_der1der1_likelihood = einsum(
            'abi,abj->abij', cum_der1_likelihood,
            cum_der1_likelihood,
        )
        hist_der1der1_likelihood *= broadcast_to(reshape(exp(
            cumsum(
                table_target_log_prob_outcomes-table_log_prob_outcomes,
                axis=1,)
        ), (self.bs, pars.num_steps, 1, 1)),
            (self.bs, pars.num_steps, self.phys_model.d, self.phys_model.d)
        )
        # Fisher if expectation value
        history_fisher = reduce_sum(
            hist_der1der1_likelihood, axis=0)/self.bs
        # Weight matrix
        broad_weight_matrix = broadcast_to(
            reshape(self.cov_weight_matrix_tensor,
                    (1, self.phys_model.d, self.phys_model.d),
                    ),
            (pars.num_steps, self.phys_model.d, self.phys_model.d)
        )
        # Fisher info inverse
        inv_history_fisher = inv(history_fisher)
        history_precision = broadcast_to(reshape(trace(matmul(
            broad_weight_matrix, inv_history_fisher
        )), (pars.num_steps, 1, 1)), (pars.num_steps, self.bs, 1))

        return true_values, history_input, history_controls, \
            history_resources, history_precision

    def __str__(self):
        return f"{self.simpars.sim_name}_batchsize_" \
            f"{self.bs}_num_steps_{self.simpars.num_steps}_" \
            f"max_resources_{self.simpars.max_resources:.2f}_" \
            f"ll_{self.simpars.log_loss}_cl_" \
            f"{self.simpars.cumulative_loss}"
