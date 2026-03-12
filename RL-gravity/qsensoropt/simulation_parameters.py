"""Module containing the :py:obj:`SimulationParameters`
dataclass.
"""

from dataclasses import dataclass


@dataclass
class SimulationParameters:
    r"""Flags and parameters to tune the
    :py:obj:`~.Simulation` class.

    This dataclass contains the attributes
    :py:attr:`num_steps`, :py:attr:`max_resources`, and
    :py:attr:`resources_fraction` that determine the
    stopping condition of the
    measurement loop in the :py:meth:`~.Simulation.execute`
    method, some flags that regulate the gradient propagation
    and the choice of the loss in the training, and
    the name of the simulation in the :py:attr:`sim_name`
    attribute, along with the numerical precision
    of the simulation in :py:attr:`prec`.

    .. image:: ../docs/_static/gradient_flow.png
        :width: 700
        :alt: gradient_flow

    This picture represents schematically the default
    setting of the gradient propagation in the training.
    The solid arrows are the flows of
    information along which the gradient is propagated,
    while the dashed lines are the ones along which
    the gradient is not computed. By default the gradient
    is propagated along the particle filter,
    whose evolving state is represented by
    the thick blank line across the picture, along the
    evolving state of the probe, and through the neural
    network. The backpropagation of the gradient goes in the
    opposite direction of the information flow.

    For a certain estimation in the batch we will refer
    to the loss produced by the method
    :py:meth:`~.StatefulSimulation.loss_function` with
    the symbol :math:`\ell(\omega, \vec{\lambda})`, with
    :math:`\omega = (\vec{y}, \vec{\theta})`, containing
    the list of measurement outcomes
    :math:`\vec{y} = (y_0, y_2, \dots, y_{t})` up to step
    :math:`t`, and the true values :math:`\vec{\theta}`
    of the unknown parameters to be
    estimated. The vector :math:`\vec{\lambda}` contains
    all the degrees of freedom of the control
    strategy. In case of a neural network these are
    the weights and the biases. The loss to be
    optimized is then the average of
    :math:`\ell(\omega, \vec{\lambda})` on a batch
    of estimations, i.e.

    .. math::
        \mathcal{L} (\vec{\lambda}) = \frac{1}{B}
        \sum_{k=1}^B \ell (\omega_k, \vec{\lambda}) \;,
        :label: loss

    where :math:`B` is the batchsize.
    If the flag :py:attr:`cumulative_loss` is deactivated
    the loss is computed at the end of the simulation,
    which is triggered when at least a fraction
    :py:attr:`resources_fraction` of estimations in the
    batch have terminated or when the maximum
    number of steps :py:attr:`num_steps` in the
    measurement loop has been achieved.
    If an estimation in the batch has terminated
    because of the resources requirements,
    the last estimator before the termination is
    used to compute the mean loss. """
    sim_name: str
    """Name of the simulation. It is the first
    part of the name of the files saved by the functions
    :py:func:`~.utils.train`,
    :py:func:`~.utils.train_nn_graph`,
    :py:func:`~.utils.train_nn_profiler`,
    :py:func:`~.utils.store_input_control`,
    and :py:func:`~.utils.performance_evaluation`
    """
    num_steps: int
    """Maximum number of steps in the measurement cycle
    of the :py:meth:`~.Simulation.execute` method in the
    :py:obj:`~.Simulation` class."""
    max_resources: float
    """Strict upper bound on the number of resources
    usable in an estimation, as counted by the
    :py:meth:`~.PhysicalModel.count_resources` method.
    This parameter enters in the stopping condition
    of the measurement loop."""
    resources_fraction: float = 1.0
    """Fraction of the estimations in a batch that must have
    exhausted the available resources for the measurement
    loop to stop and the batch of estimations to be declared
    complete. For example if `resources_fraction=0.9`
    then only 90% of the estimations in the batch need to be
    terminated to stop the measurement loop of the
    :py:meth:`~.Simulation.execute` method. This parameter
    should be used when some of the possible simulation
    trajectories take a large
    number of measurements to saturate
    :py:attr:`max_resources`. Thanks to this attribute,
    the simulation will be stopped prematurely even if
    not all the estimations have terminated according
    to the resource availability.

    **Achtung!** The mean loss is computed also with
    those estimations that have been prematurely
    stopped."""
    prec: str = "float64"
    """Floating point precision of used in the
    simulation, can be either `float32` or `float64`.

    **Achtung!** The type of the probe state defined
    in the attribute `state_specifics` of the class
    :py:obj:`~.PhysicalModel` can be different from
    `prec`, but the
    precisions of the :py:obj:`~.ParticleFilter`,
    :py:obj:`~.PhysicalModel`,
    :py:obj:`~.SimulationParameters` objects, and
    of the neural network should all agree."""
    stop_gradient_input: bool = True
    """Stops the propagation of the gradient through
    the inputs of the neural network (or in general
    of the control strategy).
    This contribution to the
    gradient is usually redundant, and can be
    neglected to spare time and memory during the
    training. In the picture of the
    :py:obj:`SimulationParameters` class description
    the stopping of the gradient is represented
    by the dashed line going from the
    particle filter the input of the neural network"""
    loss_logl_outcomes: bool = True
    r"""If `True` the logarithm of the likelihood of
    the observed measurement outcomes is mixed
    with the loss :math:`\ell(\omega, \vec{\lambda})`
    in order to compute the correct gradient.
    The modified loss is

    .. math::
        \widetilde{\ell} (\omega, \vec{\lambda}) :=
        \ell (\omega, \vec{\lambda})+
        \text{sg}[{\ell (\omega, \vec{\lambda})}]
        \log P(\vec{y}|\vec{\theta}, \vec{\lambda}) \; .
        :label: log_mixing

    The additional term is needed in order to
    propagate the gradient through the physical
    model of the probe, described by the differentiable
    methods
    :py:meth:`~.StatefulPhysicalModel.perform_measurement`
    and :py:meth:`~.StatefulPhysicalModel.model`,
    which return the logarithm of the probability of
    the outcomes. Without this extra term the gradient
    used in the training is biased.
    This flag must be deactivated only when the
    measurement outcomes are generated through a
    differentiable reparametrization. That is,
    instead of extracting :math:`y` directly from
    :math:`P(y|\vec{\theta}, \vec{\lambda})`
    it is obtained as
    :math:`y=f_{\vec{\theta}, \vec{\lambda}}(z)`,
    with :math:`z` being a stochastic variable extracted
    from a distribution independent on
    :math:`\vec{\theta}` and :math:`\vec{\lambda}`."""
    loss_logl_controls: bool = False
    """If one ore more controls are discrete, that is,
    their flag `continuous` of the
    :py:obj:`~.Control` class is `False`, then
    they are probably extracted stochastically from
    a probability distribution produced by the
    control strategy. If this is a neural network
    for example, its output could be the
    categorical distribution
    from which the controls are extracted, instead
    of the controls themselves. In such cases the
    callable attribute `control_strategy`
    of the :py:obj:`~.Simulation` class takes as
    input the random number generator `rangen`
    alongside `input_tensor` and returns
    `controls` and `log_prob_control`, respectively
    the values of the controls and the logarithm
    of the probability of having extracted the
    said values.

    If this flag is `True`, then `log_prob_control`
    is mixed with the loss to compute the correct gradient.
    This means that an extra terms analogous to the one
    in :eq:`log_mixing` is added to the loss, which
    refers to the controls instead of the outcomes.

    This option should be activated only if the controls
    are generated stochastically."""
    log_loss: bool = False
    r"""If this flag is `True`, then the used loss
    is the logarithm of :math:`\mathcal{L} (\vec{\lambda})`
    in :eq:`loss`. In this case, if the flags
    :py:attr:`loss_logl_controls`
    and :py:attr:`loss_logl_outcomes` are also
    activated, the simulation modifies accordingly the
    log-likelihood terms to account for the
    extra logarithm in the definition of the loss.
    This flag can be used in conjunction with
    :py:attr:`cumulative_loss` to weight in the
    precisions at all steps in the training,
    while avoiding the use of a reference
    precision :math:`\eta`.
    The cumulative logarithmic loss would be

    .. math::
        \mathcal{L}_{\text{log}} (\vec{\lambda}) = \frac{1}{T}
        \sum_{t=0}^{T-1} \log \left[ \frac{1}{B}
        \sum_{k=1}^B \ell (\omega_k^t, \vec{\lambda}) \right] \; .
    """
    cumulative_loss: bool = False
    r"""With this flag on the loss of :eq:`loss` is computed
    after each measurement and accumulated, so that
    the quantity that is differentiated for the training
    at the end of the loop in :py:meth:`~.Simulation.execute` is

    .. math::
        \mathcal{L}_{\text{cum}} (\vec{\lambda}) = \frac{1}{TB}
        \sum_{t=1}^T \sum_{k=1}^B \ell (\omega_k^t,
        \vec{\lambda}) \;.

    The maximum number of measurement performed
    is :math:`T`, but some estimations in the batch might
    terminate with less measurements. In any case the last
    estimator :math:`\hat{\vec{\theta}}` from each simulation
    in the batch is used to compute the mean loss in all the
    subsequent steps.

    .. image:: ../docs/_static/table_simulation.png
        :width: 400
        :alt: table_simulation

    In the picture a cells of the table contains
    an estimator for the unknown parameters. The
    rows are the different simulations in the batch
    and the columns the steps of the measurement cycle.
    The current measurements step is indicated with
    the red box.
    The cells highlighted in grey are those without an
    estimator, either because it is not yet been
    computed or because the estimation is already
    finished for that particular batch element.
    In green are highlighted the :math:`B=6`
    estimators that are used in the computation of
    the mean loss at the current step. For the third
    estimation in the batch, that is already terminated,
    the most recent version of :math:`\hat{\vec{\theta}}_3`
    is used for computing the mean loss in all the
    steps after its termination.

    It follows that the last estimator of those
    simulations that have already exhausted the resources
    appear multiple times in the evaluation of the loss.

    Mixing precisions that refer to different number of
    measurements might not be a fair way to compute the
    loss in the metrological task. In this case the user
    should define the loss function so that
    it contains a normalization factor
    :math:`\eta(\vec{\theta}_k, t)`,
    that should be of the order of magnitude of the
    expected loss, i.e.

    .. math::
        \frac{\ell (\omega_k^t, \vec{\lambda})}{\eta(
            \vec{\theta}_k, t)}
        \sim \mathcal{O} (1) \; .

    The cumulative loss is then

    .. math::
        \mathcal{L}_{\text{cum}} (\vec{\lambda}) = \frac{1}{TB}
        \sum_{t=1}^T \sum_{k=1}^B \frac{\ell (\omega_k^t,
        \vec{\lambda})}{\eta(\vec{\theta}_k, t)} \;.

    **Achtung!** This flag has no effect
    on a :py:obj:`~.BoundSimulation`
    object. For which the Fisher information
    on the probe trajectory is already a sort of
    cumulative loss.
    """
    stop_gradient_pf: bool = False
    """Stops the propagation of the gradient propagation
    through the Bayesian updates
    of the particle filter performed
    by the :meth:`~.ParticleFilter.apply_measurement`
    method and represented by the thick blank line
    in the picture in the description of the
    :obj:`SimulationParameters` class.
    It can be used in conjunction with :py:attr:`log_loss`
    and :py:attr:`cumulative_loss` to obtain a
    fully greedy optimization analogous to that
    of the library optbayesexpt [6]_.

    .. [6] https://github.com/usnistgov/optbayesexpt
    """
    baseline: bool = False
    r"""If both this flag and the attribute
    :py:attr:`loss_logl_outcomes` are `True`
    then the loss of :eq:`log_mixing` is changed to

    .. math::
        \widetilde{\ell} (\omega, \vec{\lambda}) :=
        \ell (\omega, \vec{\lambda})+
        \text{sg}[{\ell (\omega, \vec{\lambda})} -
        \mathcal{B}]
        \log P(\vec{y}|\vec{\theta}, \vec{\lambda}) \; ,
        :label: baseline_correction


    with

    .. math::
    	\mathcal{B} := \frac{1}{B} \sum_{k=1}^B
          \ell (\omega_k, \vec{\lambda}) \; ,

    The additional term should reduce the variance of
    the gradient computed from the batch. The baseline
    :math:`\mathcal{B}` is recomputed for each step of
    the measurement loop."""
    permutation_invariant: bool = False
    r"""If the model is invariant under permutations
    in the unknown parameters (for all the values
    of the controls) this flag should be activated.
    Given the unknown parameters :math:`\theta_i`, the
    ambiguity is lifted by fixing
    :math:`\theta_1<\theta_2<\cdots<\theta_d`."""
    @property
    def end_batch(self):
        """If this flag is `True` the the cumulative
        loss counts multiple times the estimators
        of the simulations that have terminated the
        available resources."""
        return self.cumulative_loss or self.log_loss
