"""Submodule containing the :py:obj:`PhysicalModel` class
alongside with :py:obj:`StateSpecifics` and :py:obj:`Control`.
"""

from dataclasses import dataclass
from typing import List, Tuple, TypedDict

from tensorflow import Tensor, concat
from tensorflow.random import Generator
from tensorflow import dtypes

from .parameter import Parameter


class StateSpecifics(TypedDict):
    """Size and type StateSpeof the probe state
    collected in a single object.

    Examples
    --------
    For a single-qubit probe, the state can be
    represented by a 2x2 complex matrix with four
    entries, or by a 3-dimensional vector of real
    values using the Bloch representation.
    For a cavity with a single bosonic mode,
    which is always in a Gaussian state, we need
    a 2x2 real covariance matrix and a 2-dimensional
    real vector to represent the state.
    This requires a total of 6 real values, which can
    be reduced to 5 noticing that the covariance
    matrix is symmetric.
    """
    size: int
    """Size of a 1D `Tensor` necessary to describe
    unambiguously the state of the probe."""
    type: str
    """Type of the `Tensor` representing the probe
    state. Can be whatever type admissible in
    Tensorflow.
    """


@dataclass
class Control:
    """Control parameter for the physical sensor.

    A control is a tunable parameter that can be
    adjusted during the experiment. This could be
    for example the measurement duration, the detuning
    of the laser frequency driving a cavity, a
    tunable phase in an interferometer, or other
    similar parameters.

    In a pictorial sense, control parameters are
    all the buttons and knobs on the electronics
    of the experiment.

    .. image:: ../docs/_static/instr_measure.png
        :width: 500
        :alt: Alternative text

    A control can be continuous if it takes
    values in an interval, or discrete if it
    takes only a finite set of values (like on
    and off).
    """
    name: str
    """Name of the control."""
    is_discrete: bool = False
    """If the parameter is discrete
    this flag is true.

    If in the simulation at least one of the
    controls is discrete, then the `control_strategy`
    attribute of the :py:obj:`Simulation`
    class should be a callable with
    header

    ``controls, log_prob_control =
    control_strategy(input_strategy, rangen)``

    that is, it can contain stochastic operations
    (like the extraction of the discrete control
    from a probability distribution)
    and it must return the logarithm of the
    extraction probability of the chosen controls.
    """


class PhysicalModel:
    """Abstract representation of the physical
    system used as quantum probe.

    This class and its children,
    :py:obj:`~.StatefulPhysicalModel` and
    :py:obj:`~.StatelessPhysicalModel`,
    contain a description of the physics
    of the quantum probe, of the unknown
    parameters to estimate,
    of the dynamics of their encoding on the probe,
    and of the measurements.

    **Achtung!** In implementing a class for some particular
    device, the user should not directly derive
    :py:obj:`PhysicalModel`, but instead the classes
    :py:obj:`~.StatefulPhysicalModel` and
    :py:obj:`~.StatelessPhysicalModel` should
    be the base objects.

    When programming the physical model,
    the first and most important
    decision to make is whether the model is
    stateful or stateless. In the
    former case, :py:obj:`~.StatefulPhysicalModel` should
    be used, while in
    the latter case :py:obj:`~.StatelessPhysicalModel` should
    be chosen.
    The probe is stateless if it is reinitialized
    after each measurement
    and no classical information needs to
    be passed from one measurement
    to the next, other than that contained in the
    particle filter ensemble.
    If this is not the case, then the probe is stateful.

    **Achtung!** If some classical information
    on the measurement outcomes
    needs to be passed down from measurement to measurement,
    other than the information
    contained in the particle filter ensemble, then
    the probe is **stateful**.
    For example, if we perform multiple weak
    measurements on a signal and
    want to keep track of the total number
    of photons observed, then this
    information is part of the signal state.
    See the :py:mod:`dolinar` module for an example.

    Attributes
    ----------
    bs: int
        The batch size of the physical model,
        i.e., the number of probes
        on which the estimation is performed
        simultaneously in the simulation.
    controls: List[:py:obj:`~.physical_model.Control`]
        A list of controls on the probe,
        i.e., the buttons and knobs
        of the experiment.
    params: List[:py:obj:`~.Parameter`]
        A list of unknown parameters to be estimated,
        along with the corresponding
        sets of admissible values.
    controls_size: int
        The number of controls on the probe, i.e.,
        the length of the `controls` attribute.
    d: int
        The number of unknown parameters to estimate,
        i.e., the length of the `params` attribute.
    state_specifics: :py:obj:`~.physical_model.StateSpecifics`
        The size of the last dimension
        and type of the `Tensor`
        used to represent the state of
        the probe internally in the simulation.
    recompute_state: bool
        A flag passed to the constructor of
        the :py:obj:`PhysicalModel` class;
        it controls whether the state ensemble should be
        recomputed after a resampling of the particles
        or not.
    outcomes_size: int
        A parameter passed to the constructor
        of the :py:obj:`PhysicalModel` class;
        it is the number of scalar outcomes
        of a measurement on the probe.
    prec: str
        The floating point precision of the controls,
        outcomes, and parameters.

    Examples
    --------
    For a metrological task performed on a cavity
    that is driven by a laser,
    and where we can control the laser frequency,
    we have `control_size=1`.
    If a heterodyne measurement is performed on
    the cavity, then `outcomes_size=2`,
    because a measurement produces an estimate
    for both the quadratures.
    Assuming the cavity state is Gaussian,
    it can be represented by 6 real
    numbers (two for the mean and four for the covariance).
    The number of scalar quantities can be reduced to
    5 by observing that the covariance matrix
    is symmetric.

    .. image:: ../docs/_static/cavity.png
        :width: 600
        :alt: Cavity state
    """

    def __init__(
            self, batchsize: int,
            controls: List[Control],
            params: List[Parameter],
            state_specifics: StateSpecifics, *,
            recompute_state: bool = True,
            outcomes_size: int = 1,
            prec: str = "float64",
    ):
        r"""Parameters passed to the constructor of
        the :py:obj:`~.PhysicalModel` class.

        Parameters
        ----------
        batchsize: int
            Batch size of the physical model, i.e.,
            the number of simultaneous estimations
            in the simulation.
        controls: List[:py:obj:`~.Control`]
            A list of controls for the probe
            (the buttons and knobs
            of the experiment).
        params: List[:py:obj:`~.Parameter`]
            A list of :py:obj:`~.Parameter` objects
            that represent the unknowns of the estimation.
        state_specifics: :py:obj:`~.StateSpecifics`
            The size of the last dimension and type of the `Tensor`
            used to represent internally in the simulation the
            state of the probe.
        recompute_state: bool = True
            Controls whether the state ensemble should be recomputed
            with the method
            :py:meth:`~.ParticleFilter.recompute_state`
            after a resampling of the particles or not.

            **Achtung!** This flag should be deactivated
            only if
            the states in the ensemble don't depend on the
            particles or if the resampling has been deactivated.

            Recomputing the state is very expensive,
            especially when there have already been many
            measurement steps. One should consider whether to
            resample the particles at all in a simulation
            involving a relatively large quantum state.
        outcomes_size: int = 1
            Number of scalars collected in a measurement on
            the probe.
        prec: str = "float64"
            Floating point precision of the controls,
            outcomes, and parameters.
        """
        self.bs = batchsize
        self.controls = controls
        self.params = params
        self.controls_size = len(controls)
        self.state_specifics = state_specifics
        self.d = len(params)
        self.prec = prec
        self.recompute_state = recompute_state
        self.outcomes_size = outcomes_size

        # Loads the batchsize and the precision in
        # the parameters objects
        for par in self.params:
            par.bs = self.bs
            par.prec = prec

        if not prec in ("float64", "float32"):
            raise ValueError("The allowed values of \
                             prec are float32 and float64.")

    def true_values(
            self, rangen: Generator,
    ) -> Tuple[Tensor]:
        """Provides a batch of fresh "true
        values" for the parameters of the system,
        from which the measurement is simulated.

        Parameters
        ----------
        rangen: Generator
            A random number generator from the module
            :py:mod:`tensorflow.random`.

        Returns
        -------
        Tensor
            `Tensor` of shape (`bs`, 1, `d`),
            where `bs` and `d` are attributes of the
            :py:obj:`PhysicalModel` class. For each estimation
            in the batch, this method produces a single instance
            of each parameters in the attribute `params`,
            extracted uniformly from the
            allowed values. For doing so the method
            :py:meth:`~.Parameter.reset`
            is called with `num_particles=1`, for
            ech parameter.
        """
        list_seed = [rangen.uniform(
            [2, ], minval=0, maxval=dtypes.int32.max,
            dtype="int32",
        ) for _ in range(self.d)]

        compact_list_fragment = [1]
        for param in self.params:
            if (not param.randomize) and (not param.continuous):
                compact_list_fragment.append(
                    compact_list_fragment[-1]*len(param.values),
                )
        compact_list_fragment = compact_list_fragment[:-1]
        derandomized_counter = 0
        list_fragment = []
        for param in self.params:
            if (not param.randomize) and (not param.continuous):
                list_fragment.append(
                    compact_list_fragment[derandomized_counter],
                )
                derandomized_counter += 1
            else:
                list_fragment.append(1)

        return concat(list(map(
            lambda param, seed, frag: param.reset(seed, 1, frag),
            self.params, list_seed, list_fragment,
        )), 2, name="uniform_particles")

    def wrapper_initialize_state(
            self, parameters: Tensor,
            num_systems: int,
    ) -> Tensor:
        """wrapper_initialize_state"""
        raise NotImplementedError("You should override this method!")

    def wrapper_perform_measurement(
        self, controls: Tensor, parameters: Tensor,
        true_state: Tensor, meas_step: float,
        rangen: Generator,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """wrapper_perform_measurement"""
        raise NotImplementedError("You should override this method!")

    def wrapper_model(
        self, outcomes: Tensor, controls: Tensor,
        parameters: Tensor, state: Tensor, meas_step: Tensor,
        num_systems: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        """wrapper_model"""
        raise NotImplementedError("You should override this method!")
    
    def wrapper_count_resources(
        self, resources: Tensor, outcomes: Tensor, controls: Tensor,
        true_values: Tensor, state: Tensor, meas_step: Tensor,
    ) -> Tuple[Tensor]:
        """wrapper_model"""
        raise NotImplementedError("You should override this method!")
