"""Submodule containing the class :py:obj:`ParticleFilter`
necessary for Bayesian estimation.
"""

from typing import Tuple, Optional

from tensorflow import constant, ones, broadcast_to, \
    concat, expand_dims, where, gather_nd, stop_gradient, \
    expand_dims, gather_nd, while_loop, transpose, \
    cond, zeros, cast, squeeze, reshape, concat, \
    boolean_mask, zeros_like, tensor_scatter_nd_update, \
        Tensor
from tensorflow.math import multiply, reciprocal_no_nan, \
    less, log, reduce_sum, count_nonzero, \
    argmax, greater_equal, reduce_any, equal, \
    logical_and, floor, add, divide, \
    less_equal, logical_not
from tensorflow.random import stateless_normal, \
    stateless_categorical, Generator
from tensorflow.linalg import matvec, matmul
from math import floor

from .physical_model import PhysicalModel
from .utils import sqrt_hmatrix, get_seed
from .parameter import trim_single_param

class ParticleFilter:
    r"""
    Particle filter object, with methods to reset the
    particle ensemble and perform Bayesian filtering
    for quantum metrology.

    Given the tuple of unknown parameters
    :math:`\vec \theta=(\theta_1, \theta_2, \cdots, \theta_d)`,
    in the Bayesian framework we start from a prior
    :math:`p(\theta)` and update it after each observation
    through the Bayes rule. In simulating such procedure
    (called Bayesian filtering) we need a computer
    representation of a generic probability distribution
    on the parameters. We can approximate a distribution
    :math:`p(\theta)` with a weighted sum of delta functions
    as follows:

    .. math::
        p(\vec \theta) = \sum_{j=1}^{N} w_j
        \delta(\vec \theta - \vec \theta_j),

    where the vectors :math:`\vec \theta_j` are called particles,
    while the real numbers :math:`w_i` are the weights.
    The integer :math:`N` is the attribute `np` of the
    :py:obj:`ParticleFilter` object. The core idea is that
    the posterior distribution is represented by an ensemble of
    particles and weights
    :math:`\lbrace (\vec \theta_j, w_j) \rbrace_{j=1}^N`.
    The Bayesian filtering techniques that use this kind of
    representations for the posterior are called particle
    filter methods.

    In the figure, we represent the ensemble of a
    two-dimensional particle filter, where the intensity
    of the color is proportional to the weight
    associated with each particle.

    .. image:: ../docs/_static/particle_filter_representation.png
        :width: 400
        :alt: particle_filter_representation

    The update of the posterior in the particle filter
    becomes the update of the weights :math:`w_i`.

    Attributes
    ----------
    bs: int
        Batchsize of the particle filter,
        i.e. number of Bayesian estimations
        performed simultaneously.
    np: int
        Parameter `num_particles` passed to the
        constructor of the class.
        It is the number of particles in the ensemble.
    phys_model: :py:obj:`~.PhysicalModel`
        Abstract description of the physical model
        of the quantum probe passed to the
        :py:obj:`ParticleFilter` constructor.
        It also contains the description of the parameters
        to estimate, on the basis of which the particles of
        the ensemble are initialized.
    resampling: bool
        Flag `resampling_allowed` passed
        to the constructor of
        :py:obj:`ParticleFilter`.

        **Achtung!** This attribute has no effect inside the methods
        of the :py:obj:`ParticleFilter`. Calling the method
        :py:meth:`full_resampling` will resample the
        particles if needed also with `resampling=False`.

    state_size: int
        Size of the state of the probe,
        it is `phys_model.state_specifics["size"]`,
        where `phys_model` is the attribute of
        :py:obj:`ParticleFilter`.
    state_type: str
        Type of the state vector of the probe,
        it is `phys_model.state_specifics["type"]`,
        where `phys_model` is the attribute
        of :py:obj:`ParticleFilter`.
    d: int
        Number of unknown parameters of the simulation,
        it is the length of the attribute `params`.
    prec: str
        Floating point precision of the parameter values.
        Can be either `float32` or `float64`.
        It is the parameter `prec` passed to the constructor.

    Notes
    -----
    This class can deal with a whole batch of particle
    filter ensembles that are updated simultaneously.
    The batchsize attribute `bs` is taken from
    `phys_model` in the initialization.

    This implementation of the particle filter is
    fully differentiable and can, therefore, be
    used in the training loop of a neural network agent
    that controls the experiment. It is trivial to see
    that the Bayes update is differentiable, while the
    differentiability of the resampling [3]_ is achieved
    through a combination of soft resampling [4]_
    and the method of Ścibior and Wood [5]_.
    """

    def __init__(self, num_particles: int,
                 phys_model: PhysicalModel,
                 resampling_allowed: bool = True,
                 resample_threshold: float = 0.5,
                 resample_fraction: float = 0.98,
                 alpha: float = 0.5,
                 beta: float = 0.98,
                 gamma: float = 1.00,
                 scibior_trick: bool = True,
                 trim: bool = True,
                 prec: str = "float64"):
        r"""Constructor of the :py:obj:`ParticleFilter` class.

        Parameters
        ----------
        num_particles: int
            Number of particles in the ensemble of the
            particle filter.
        phys_model: :py:obj:`~.PhysicalModel`
            Contains the physics of the quantum probe and the
            methods to operate on its state and compute the
            probabilities of observing
            a certain outcome in a measurement.
            This object is used in the
            :py:meth:`apply_measurement` method,
            which performs the Bayesian update on the
            particle weights.
        resampling_allowed: bool = True
            Controls whether the resampling is allowed
            or not for a particular instance of
            :py:obj:`ParticleFilter`. The typical case in
            which we set `resampling_flag=False` is that
            of hypothesis testing, where we have a single
            discrete parameter and `num_particles` equals
            the number of hypotheses.

            This flag isn't used directly in this class,
            but in the
            :py:obj:`~.StatelessSimulation`
            and
            :py:obj:`~.StatefulSimulation`
            classes, which use
            the methods of :py:obj:`ParticleFilter`.
        resample_threshold: float = 0.5
            The method :py:meth:`full_resampling` verifies the need
            for resampling for a certain particle filter
            ensemble according to the *effective particle number*,
            defined as

            .. math::
                N_{\text{eff}} := \frac{1}{\sum_{j=1}^N
                w_j^2} \; .
                :label: effN

            Indicating with :math:`r_t` the `resample_threshold`,
            the particle filter identifies an ensemble as
            in need of resampling when
            :math:`N_{\text{eff}}<r_t N`.
        resample_fraction: float = 0.98
            The method :py:meth:`full_resampling`
            triggers the resampling
            for the whole batch of particle filter ensembles
            if at least
            a fraction `resample_fraction` of simulations
            are in need of resampling. Also those simulations in
            the batch that don't need resampling will
            be resampled.
        alpha: float = 0.5
            In the method :py:meth:`resample`
            an importance sampling of the particles is performed,
            where the extraction is done from a distribution,
            being the one defined by the ensemble
            mixed with a uniform distribution on
            the particles. This procedure, called *soft resampling*,
            aims to make the update step differentiable.
            The mixing coefficient is the parameter `alpha`,
            and it is indicated with
            :math:`\alpha` in the equations.

            This parameter quantitatively controls the flow of
            derivatives through the resampling routine and the
            effectiveness of the resampling. With `alpha=1`,
            regular resampling without mixing occurs, but no
            derivatives can flow through this operation.
            With `alpha=0`, differentiability is achieved,
            but no resampling is done. There is a tradeoff
            between the efficiency of the soft resampling
            and its differentiability.
        beta: float = 0.98
            In the method :py:meth:`resample`, a Gaussian
            perturbation of the particles obtained from the
            soft resampling is performed. The intensity of
            this perturbation is controlled by the parameter
            `1.0-beta`, which is indicated by :math:`\beta`
            in the equations.
        gamma: float = 1.0
            This parameter controls the fraction of `num_particles`
            that are extracted in the soft resampling.
            The remaining fraction of particles, `1.0-gamma`,
            are newly generated from a Gaussian
            distribution having the same mean and variance as
            the posterior distribution before the resampling.
            This parameter, like `alpha` and `beta`, regulates
            the behavior of the :py:meth:`resample` method and
            is indicated by :math:`\gamma` in the equations.

            **Achtung!**  Proposing new particles during a
            resampling won't work when
            one ore more discrete parameters are present.
            The advice in these cases is to keep `gamma=1.0`.
        scibior_trick: bool = True
            This flag controls the use of the method proposed
            by Ścibior and Wood to make the resampling differentiable.
            It can be used alone or together with soft resampling
            to improve the flow of the gradient
            through the particle ensemble resampling steps.
        trim: bool = True
            After the Gaussian perturbation of the particles
            or their extraction from scratch from a Gaussian,
            they may fall outside the admissible bounds for the
            parameters. If `trim=True`, they are cast again inside
            the bounds using the function
            :py:func:`~.parameter.trim_single_param`.
        prec: str = "float64"
            This parameter indicates the floating-point precision
            of the particles and weights, it can be either
            `float64` or `float32`.

        .. [3] C. E. Granade et al, New J. Phys. 14 103013 (2012).

        .. [4] X. Ma, P. Karkus, D. Hsu, and W. S. Lee, arXiv:1905.12885 (2019).

        .. [5] A. Ścibior, F. Wood, arXiv:2106.10314 (2021).
        """

        # Public attributes
        self.bs = phys_model.bs  # Batchsize
        self.np = num_particles
        self.phys_model = phys_model
        self.resampling = resampling_allowed
        # State related private attributes
        self.state_size = self.phys_model.state_specifics['size']
        self.state_type = self.phys_model.state_specifics['type']
        # Number of unknowns parameters
        self.d = len(self.phys_model.params)
        self.prec = prec

        # Hyperparameters of the PF that regulate the
        # behavior of the resampling
        self.res_frac = resample_fraction
        self.res_thres = resample_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.nrp = floor(self.gamma*self.np)
        self.nnp = self.np-self.nrp
        self.trim = trim
        # The `Scibior trick` relates to the differentiability
        # of the resampling operations.
        self.scibior_trick = scibior_trick

        if not prec in ("float64", "float32"):
            raise ValueError("The allowed values of \
                             prec are float32 and float64.")

    def reset(self, rangen: Generator) -> Tuple[Tensor, Tensor]:
        r"""Initializes the particles and the weights of
        the particle filter ensemble, both uniformly.

        Parameters
        ----------
        rangen: Generator
            A random number generator from the module
            :py:mod:`tensorflow.random`.

        Returns
        -------
        weights: Tensor
            Contains the weights of the particles initialized to
            `1/np`, where `np` is an attribute of :py:obj:`ParticleFilter`.
            It is a `Tensor` of shape (`bs`, `np`)
            and type `prec`, where the first dimension is
            the number of estimations performed in the batch
            simultaneously (the batchsize).

        particles:
            Contains the particles extracted uniformly within
            the admissible bounds of each parameter.
            It is a `Tensor` of shape (`bs`, `np`, `d`),
            of type `prec`, where `d` is an attribute of
            :py:obj:`ParticleFilter`.

        Notes
        -----
        This method calls the
        :py:meth:`~.Parameter.reset`
        methods of each
        :py:obj:`~.Parameter`
        object in the attribute `params`.
        """
        # Generates seed from the Generator
        list_seed = [get_seed(rangen) for _ in range(self.d)]

        # Generate fragmentation list
        compact_list_fragment = [1]
        for param in self.phys_model.params:
            if (not param.randomize) and (not param.continuous):
                compact_list_fragment.append(
                    compact_list_fragment[-1]*len(param.values),
                )
        compact_list_fragment = compact_list_fragment[:-1]
        derandomized_counter = 0
        list_fragment = []
        for param in self.phys_model.params:
            if (not param.randomize) and (not param.continuous):
                list_fragment.append(
                    compact_list_fragment[derandomized_counter],
                )
                derandomized_counter += 1
            else:
                list_fragment.append(1)

        # Calls the reset function of each parameter
        particles_init = [par.reset(seed, self.np, frag)
                          for par, seed, frag in
                          zip(self.phys_model.params, list_seed, list_fragment)]
        # Concatenates the dimentions, each is an unknown parameter
        particles = concat(particles_init, 2, name='particles')
        # Uniformly initialized weights
        weights = 1.0/(self.np) * ones(
            (self.bs, self.np), dtype=self.prec, name='weights',
        )
        return weights, particles

    def apply_measurement(
        self, weights: Tensor, particles: Tensor,
        state_ensemble: Tensor, outcomes: Tensor,
        controls: Tensor, meas_step: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""Bayesian update of the weights of the particle filter.

        Consider :math:`x` the outcome of the last observation
        on the probe and :math:`y` the control.
        The weights are updated as

        .. math::
            w_j \leftarrow \frac{p(x|\vec{\theta}_j, y, \rho_j)
            w_j}{\sum_{j=1}^N p(x|\vec{\theta}_j, y, \rho_j)
            w_j} \; ,

        where :math:`p(x|\vec{\theta}_j, y, \rho_j)` is the probability
        of observing the outcome :math:`x` under the control :math:`y`,
        and :math:`\vec{\theta}_j` the `j`-th particles with its associated
        state :math:`\rho_j` before the encoding and the measurement.

        Parameters
        ----------
        weights: Tensor
            A `Tensor` of weights with shape (`bs`, `np`)
            and type `prec`. `bs`, `np`, and `prec` are attributes
            of the :py:obj:`~.ParticleFilter` class.
        particles: Tensor
            A `Tensor` of particles with shape (`bs`, `np`, `d`)
            and type `prec`. `bs`, `np`, `d`, and `prec` are attributes
            of the :py:obj:`~.ParticleFilter` class.
        state_ensemble: Tensor
            The state of the probe system before the last measurement
            for each particle. It is a `Tensor` of shape
            (`bs`, `np`, `state_size`) and of type `state_type`.
            Here, `state_size` and `state_type` are attributes
            of :py:obj:`ParticleFilter`. In the Bayes rule formula,
            the `state_ensemble` is indicated with :math:`\rho_j`.
        outcomes: Tensor
            The outcomes of the last measurement on the probe.
            It is a `Tensor` of shape (`bs`, `phys_model.outcomes_size`)
            and of type `prec`. In the Bayes rule formula,
            it is indicated with :math:`x`.
        controls: Tensor
            The controls of the last measurement on the probe.
            It is a `Tensor` of shape (`bs`, `phys_model.controls_size`)
            and of type `prec`. In the Bayes rule formula,
            it is indicated with :math:`y`.
        meas_step: Tensor
            The index of the current measurement on the probe system.
            The counting starts from zero.
            It is a `Tensor` of shape (`bs`, 1) and of type `int32`.

        Returns
        -------
        weights: Tensor
            Updated weights after the application of the Bayes rule.
            It is a `Tensor` of shape (`bs`, `np`)
            and of type `prec`. `bs`, `np`, and `prec` are attributes
            of the :py:obj:`~.ParticleFilter` class.
        state_ensemble: Tensor
            Probe state associated to each particle after the
            measurement backreaction associated to `outcomes`.
            It is a `Tensor` of shape
            (`bs`, `np`, `state_size`) and of type
            `state_type`. `bs`, `np`, `state_size`, and `state_type`
            are attributes
            of the :py:obj:`~.ParticleFilter` class.

        Notes
        -----
        If an outcome is observed that is not compatible
        with any of the particles, this method ignores the measurements
        for the corresponding batch element. Such observation is
        treated as an outlier.
        """
        # We pass to the model function a broadcasted
        # version of the controls
        # the outcomes, which are ready to be used together with
        # weights and particles in the `model` method.
        controls_broad = broadcast_to(
            expand_dims(controls, axis=1),
            (self.bs, self.np, self.phys_model.controls_size),
            name='control_broad'
        )
        outcomes_broad = broadcast_to(
            expand_dims(outcomes, axis=1),
            (self.bs, self.np, self.phys_model.outcomes_size),
            name='outcomes_broad',
        )
        step_broad = broadcast_to(
            expand_dims(meas_step, axis=2), (self.bs, self.np, 1),
            name='step_broad',
        )
        # Prob. of observing the given outcome for each particle
        # and backreaction on the states.
        prob_outcomes, post_state_ensemble = \
            self.phys_model.wrapper_model(
                outcomes_broad, controls_broad,
                particles, state_ensemble, step_broad,
                num_systems=self.np,
            )
        # Bayes rule
        posterior_weights = multiply(
            weights, prob_outcomes,
            name="posterior_update",
        )
        # The observation that do no pass to any particle/hypothesis
        # are ignored.
        invalid_weights_flag = \
            self._invalid_weights(posterior_weights)
        weights = cond(
            reduce_any(invalid_weights_flag),
            lambda: where(
                invalid_weights_flag, weights, posterior_weights,
                name='find_invalid_weights',
            ),
            lambda: posterior_weights,
            name='remove_invalid_weights',
        )
        # Also the probe states are not updated if the observation
        # is ignored.
        invalid_weights_flag_broad = broadcast_to(
            reshape(invalid_weights_flag, (self.bs, 1, 1)),
            (self.bs, self.np, self.state_size),
        )
        state_ensemble = cond(
            reduce_any(invalid_weights_flag),
            lambda: where(
                invalid_weights_flag_broad,
                state_ensemble, post_state_ensemble,
                name='find_invalid_weights_state',
            ),
            lambda: post_state_ensemble,
            name='remove_invalid_weights_state',
        )
        # Computing the norm of the weights.
        norm = reduce_sum(
            weights, axis=1, keepdims=True,
            name='compute_weights_norm')
        norm_broad = broadcast_to(
            norm, (self.bs, self.np), name='norm_broad',
        )
        # Posterior normalization.
        weights = multiply(
            weights, reciprocal_no_nan(norm_broad),
            name='weights_normalization',
        )
        return weights, state_ensemble

    def check_resampling(
        self, weights: Tensor,
        count_for_resampling: Tensor,
    ) -> Tensor:
        """Checks the resampling condition on the effective
        number of particles for each simulation in the
        batch.

        A particle filter ensemble is in need of resampling when
        the effective number of particles, defined in :eq:`effN`,
        is less then a fraction `resample_threshold` of the total
        particle number.
        
        Parameters
        ----------
        weights: Tensor
            `Tensor` of weights of shape
            (`bs`, `np`)
            and of type `prec`. `bs`, `np`, and `prec` are attributes
            of the :py:obj:`~.ParticleFilter` class.
        count_for_resampling: Tensor
            `Tensor` of `bool` of shape (`bs`, 1), that says
            whether a certain ensemble in the batch should
            be counted or not in the total number of
            active simulation. Some estimations in the batch
            may have already terminated and are not to be
            counted in assessing the need of launching
            the resampling routine.
            If a fraction `resample_fraction`
            of the active simulations needs resampling then the whole
            batch is resampled.

        Returns
        -------
        trigger_resampling: Tensor
            `Tensor` of
            shape (`bs`, ) and of
            type `bool` that indicates whether the corresponding
            batch element should undergo resampling
            or not.
        """
        need_resampling = less(
            1.0/reduce_sum(weights**2, axis=1, name='effective_np'),
            self.res_thres*self.np *
            ones((self.bs, ), dtype=self.prec, name='res_thres'),
            name='need_resampling',
        )
        count_for_resampling = squeeze(
            count_for_resampling, name="count_for_resampling",
        )
        triggering_resampling = logical_and(
            count_for_resampling, need_resampling,
            name='trigger_resampling',
        )
        return triggering_resampling
        
    
    def full_resampling(
        self, weights: Tensor, particles: Tensor,
        count_for_resampling: Tensor, rangen: Generator,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Checks the resampling condition on the effective
        number of particles and applies the resampling
        if needed.

        A particle filter ensemble is in need of resampling when
        the effective number of particles, defined in :eq:`effN`,
        is less then a fraction `resample_threshold` of the total
        particle number.
        The whole batch of particle filters gets resampled if at least
        a fraction `resample_fraction` of the active simulations,
        defined by `count_for_resampling`, is in need of resampling.

        Parameters
        ----------
        weights: Tensor
            `Tensor` of weights of shape
            (`bs`, `np`)
            and of type `prec`. `bs`, `np`, and `prec` are attributes
            of the :py:obj:`~.ParticleFilter` class.
        particles: Tensor
            `Tensor` of particles of shape
            (`bs`, `np`, `d`)
            and of type `prec`. `bs`, `np`, `d`, and `prec` are attributes
            of the :py:obj:`~.ParticleFilter` class.
            `d` is the attribute of :py:obj:`ParticleFilter`.
        count_for_resampling: Tensor
            `Tensor` of `bool` of shape (`bs`, 1), that says
            whether a certain ensemble in the batch should
            be counted or not in the total number of
            active simulation. Some estimations in the batch
            may have already terminated and are not to be
            counted in assessing the need of launching
            the resampling routine.
            If a fraction `resample_fraction`
            of the active simulations needs resampling then the whole
            batch is resampled.
        rangen: Generator
            Random number generator from the module
            :py:mod:`tensorflow.random`.

        Returns
        -------
        weights: Tensor
            Resampled weights. `Tensor` of
            shape (`bs`, `np`) and of
            type `prec`. `bs`, `np`, and `prec` are attributes
            of the :py:obj:`~.ParticleFilter` class.
        particles: Tensor
            Resampled particles. `Tensor` of
            shape (`bs`, `np`, `d`)
            and of type `prec`. `bs`, `np`, `d`, and `prec` are attributes
            of the :py:obj:`~.ParticleFilter` class.
        resample_flag: Tensor
            `Tensor` of shape (, ) of type `bool` that
            informs whether the resampling has been executed or not.
        """
        triggering_resampling = self.check_resampling(
            weights, count_for_resampling,
        )
        threshold_for_resampling = self.res_frac*cast(
            count_nonzero(count_for_resampling),  dtype="float64",
            name='effective_res_threshold',
        )
        # Single boolen flag that tells if
        # the whole batch should be resampled or not
        resample_flag = cond(
            greater_equal(
                cast(count_nonzero(
                    triggering_resampling,
                    name='num_in_need_of_resampling'),
                    dtype="float64",
                ),
                threshold_for_resampling,
                name='thereshold_comparison',
            ),
            lambda: constant(True), lambda: constant(False),
            name="resample_flag"
        )
        # The batchsize if self.bs for full_resampling
        weights, particles = cond(
            resample_flag,
            lambda: self.resample(
                weights, particles, rangen, self.bs,
                ),
            lambda: (weights, particles),
            "perform_resampling"
        )
        return weights, particles, resample_flag

    def partial_resampling(
        self, weights: Tensor, particles: Tensor,
        count_for_resampling: Tensor, rangen: Generator,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Checks the resampling condition on the effective
        number of particles and applies the resampling
        if needed.

        A particle filter ensemble is in need of resampling when
        the effective number of particles, defined in :eq:`effN`,
        is less then a fraction `resample_threshold` of the total
        particle number.
        The whole batch of particle filters gets resampled if at least
        a fraction `resample_fraction` of the active simulations,
        defined by `count_for_resampling`, is in need of resampling.

        Parameters
        ----------
        weights: Tensor
            `Tensor` of weights of shape
            (`bs`, `np`)
            and of type `prec`. `bs`, `np`, and `prec` are attributes
            of the :py:obj:`~.ParticleFilter` class.
        particles: Tensor
            `Tensor` of particles of shape
            (`bs`, `np`, `d`)
            and of type `prec`. `bs`, `np`, `d`, and `prec` are attributes
            of the :py:obj:`~.ParticleFilter` class.
            `d` is the attribute of :py:obj:`ParticleFilter`.
        count_for_resampling: Tensor
            `Tensor` of `bool` of shape (`bs`, 1), that says
            whether a certain ensemble in the batch should
            be counted or not in the total number of
            active simulation. Some estimations in the batch
            may have already terminated and are not to be
            counted in assessing the need of launching
            the resampling routine.
            If a fraction `resample_fraction`
            of the active simulations needs resampling then the whole
            batch is resampled.
        rangen: Generator
            Random number generator from the module
            :py:mod:`tensorflow.random`.

        Returns
        -------
        weights: Tensor
            Resampled weights. `Tensor` of
            shape (`bs`, `np`) and of
            type `prec`. `bs`, `np`, and `prec` are attributes
            of the :py:obj:`~.ParticleFilter` class.
        particles: Tensor
            Resampled particles. `Tensor` of
            shape (`bs`, `np`, `d`)
            and of type `prec`. `bs`, `np`, `d`, and `prec` are attributes
            of the :py:obj:`~.ParticleFilter` class.
        """
        triggering_resampling = self.check_resampling(
            weights, count_for_resampling,
        )
        
        # Indices of the particles in need of
        # resampling
        indices = cast(
            where(triggering_resampling),
            dtype="int32",
            )
        # Indeces of the particles that
        # must not be resampled
        indices_false = cast(
            where(logical_not(triggering_resampling)),
            dtype="int32",
            )

        # Use boolean_mask to extract batches where
        # triggering_resampling is True
        filtered_weights = boolean_mask(
            weights, triggering_resampling,
            )
        filtered_particles = boolean_mask(
            particles, triggering_resampling,
            )
        
        bs = reduce_sum(
            cast(triggering_resampling, dtype="int32"),
        )
        
        filtered_weights, filtered_particles = self.resample(
            filtered_weights, filtered_particles,
            rangen, batchsize=bs,
            )

        # Use tf.boolean_mask to extract batches
        # where triggering_resampling is False
        original_weights = boolean_mask(
            weights, logical_not(triggering_resampling),
            )
        original_particles = boolean_mask(
            particles, logical_not(triggering_resampling),
            )

        # Create zero tensors of the original shapes
        scatter_weights, scatter_particles = \
            zeros_like(weights), zeros_like(particles)
            
        # Use tensor_scatter_nd_update to put the filtered weights
        # and particles back in place after the resampling
        scatter_weights = tensor_scatter_nd_update(
            scatter_weights, indices, filtered_weights,
            )
        scatter_particles = tensor_scatter_nd_update(
            scatter_particles, indices, filtered_particles,
            )
        
        # Use tensor_scatter_nd_update to put the
        # original weights and particles back in place
        scatter_weights = tensor_scatter_nd_update(
            scatter_weights, indices_false, original_weights,
            )
        scatter_particles = tensor_scatter_nd_update(
            scatter_particles, indices_false, original_particles,
            )

        return scatter_weights, scatter_particles
    
    def resample(
            self, weights: Tensor, particles: Tensor,
            rangen: Generator,
            batchsize: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Resample routine of the particle filter.

        The resampling procedure works in three steps.
        First, we mix the probability distribution on the
        particles represented by the weights :math:`w_j`
        with a uniform distribution and construct the
        soft-weights :math:`q_j` defined as:

        .. math::
            q_j = \alpha w_j + (1-\alpha) \frac{1}{N} \; ,

        The new particles :math:`\vec{\theta}_j'` are resampled
        from the ensemble :math:`\lbrace \vec{\theta}_j, q_j \rbrace`
        and the new weight associated with each resampled particle is:

        .. math::
            w_j' \propto \frac{w_{\phi(j)}}{\alpha w_{\phi(j)} +
            (1-\alpha)\frac{1}{N}} \; ,

        where :math:`\phi(j)` is the original index in the
        old ensemble of the `j`-th particle in the new ensemble.
        The function :math:`\phi` is in general neither injective
        nor surjective. Only a fraction `gamma` of the total particles
        `np` is resampled in this way, and their weights are normalized
        such that they sum to `gamma`, i.e.

        .. math::
            \sum_{j=1}^{\gamma N} w_j' = \gamma \; .

        The second step is to perturb the newly extracted
        particles :math:`\vec{\theta}_j'` as follows:

        .. math::
            \vec{\theta}_j'' = \beta \vec{\theta}_j' +
            (1-\beta) \vec{\mu} + \vec{\delta} \; ,

        where :math:`\vec{\mu}` is the mean value of the
        parameters calculated from the ensemble before
        the resampling with the method :py:meth:`compute_mean`,
        and :math:`\vec{\delta}` is a random variable extracted
        from a Gaussian distribution, i.e.

        .. math::
            \vec{\delta} \sim \mathcal{N} (0, (1-\beta^2) \Sigma) \; ,

        with :math:`\Sigma` being the covariance matrix
        of the ensemble, computed with the method
        :py:meth:`compute_covariance`. The third and last step
        of the resampling routine is to propose new particles.
        These are again extracted from a Gaussian distribution, i.e.

        .. math::
            \vec{\theta}_j'' \sim \mathcal{N}(\vec{\mu}, \Sigma) \; .

        A number :math:`(1-\gamma)N` of particles are sampled
        in this way with :math:`N`
        being the total number of particles `np`. Their weights
        are uniform and normalized to :math:`1-\gamma`, so that
        together with the particles produced in the first two
        steps, their weights sum to one.

        Parameters
        ----------
        weights: Tensor
            `Tensor` of weights of shape (`bs`, `np`) and of type `prec`.
        particles: Tensor
            `Tensor` of particles of shape (`bs`, `np`, `d`) and
            of type `prec`. `d` is the attribute
            of :py:obj:`ParticleFilter`.
        rangen: Generator
            Random number generator from the module
            :py:mod:`tensorflow.random`.

        Returns
        -------
        new_weights: Tensor
            Resampled weights. `Tensor` of shape
            (`bs`, `np`) and of type `prec`.
        new_particles: Tensor
            Resampled particles. `Tensor` of shape
            (`bs`, `np`, `d`) and of type `prec`.

        Notes
        -----
        It is possible that the soft resampling produces
        an invalid set of new weights (i.e., all zeros) for cases
        where there are very few particles and highly
        concentrated weights. In such limit cases, the soft
        resampling is aborted, and a normal resampling
        (without mixing with the uniform distribution) is executed
        to ensure that valid weights are produced.
        """
        bs = batchsize if batchsize is not None else self.bs
        soft_weights, soft_particles = self._resample_soft(
            weights, particles, rangen, batchsize=bs,
        )
        mean = self.compute_mean(
            weights, particles, batchsize=bs,
            )
        mean_expanded = broadcast_to(
            expand_dims(mean, axis=1), (bs, self.nrp, self.d),
            name="mean_broad",
        )
        dev = sqrt_hmatrix(
            self.compute_covariance(
                weights, particles, batchsize=bs,
                ),
        )
        dev_expanded = broadcast_to(
            expand_dims(dev, axis=1),
            (bs, self.nrp, self.d, self.d),
            name="dev_broad"
        )
        soft_particles = add(
            self.beta*soft_particles, (1-self.beta)*mean_expanded,
            name="part_perturbation_mu",
        )
        seed = get_seed(rangen)
        soft_particles = add(
            soft_particles,
            matvec((1-self.beta**2)*dev_expanded,
                   stateless_normal(
                (bs, self.nrp, self.d), seed, dtype=self.prec
            ),
                name="delta",
            ),
            name="part_perturbation_dev",
        )
        extra_weights, extra_particles = \
            self._resample_gaussian(
                mean, dev, rangen, batchsize=bs,
                )
        new_weights = concat(
            [soft_weights, extra_weights], 1,
            name="extra_weights_added",
        )
        new_particles = concat(
            [soft_particles, extra_particles], 1,
            name="extra_paticles_added"
        )
        if self.trim:
            new_particles = self._trim_particles(
                new_particles, batchsize=bs,
                )
        return new_weights, new_particles

    def _resample_soft(
            self, weights: Tensor, particles: Tensor,
            rangen: Generator,
            batchsize: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Performs differentiable resampling of particles using
        importance sampling.

        If the soft-resampling produces invalid weights,
        hard resampling is performed instead. The soft
        resampling may produce invalid weights when
        the weights are concentrated and the
        number of particles is very small.
        """
        bs = batchsize if batchsize is not None else self.bs
        unif_weights = 1.0/(self.np) * \
            ones((bs, self.np), dtype=self.prec,
                 name="unif_weights")
        resampling_weights = add(
            self.alpha*weights, (1-self.alpha)*unif_weights,
            name="resampling_weights",
        )
        # categorical can deal with particles having exactly
        # zero probability
        seed = get_seed(rangen)
        position_resample = expand_dims(
            stateless_categorical(
                log(resampling_weights, name="logit"), self.nrp,
                seed, dtype="int32",
                name="sampled_position",
            ), axis=2, name="sampled_position_expand",
        )
        new_particles = gather_nd(
            params=particles, indices=position_resample, batch_dims=1,
            name="new_particles",
        )
        selected_weights = gather_nd(
            params=weights, indices=position_resample, batch_dims=1,
            name="selected_weights",
        )
        selected_resampling_weights = gather_nd(
            params=resampling_weights, indices=position_resample,
            batch_dims=1,
            name="selected_resampling_weights",
        )
        unif_weights = (1.0/self.np) * \
            ones((bs, self.nrp), dtype=self.prec,
                 name="unif_weights")
        new_weights = divide(
            selected_weights,
            add(
                self.alpha*selected_weights, (1-self.alpha)*unif_weights,
                name="selected_resampling_weights",
            ),
            name="importance_sampling_correction",
        )
        norm = reduce_sum(
            new_weights, axis=1, keepdims=True,
            name="weight_normalization",
        )
        norm_broad = broadcast_to(
            norm, (bs, self.nrp), name="norm_broad",
        )
        new_weights = multiply(
            new_weights, reciprocal_no_nan(norm_broad),
            name="weights_normalization",
        )*self.nrp/self.np
        # Hard resampling is applied if the soft resampling fails
        new_weights, selected_resampling_weights, new_particles = cond(
            reduce_any(self._invalid_weights(new_weights, batchsize=bs),
                       name="count_inv_weights"),
            lambda: self._resample_hard(weights, particles, rangen, batchsize=bs),
            lambda: (new_weights, selected_resampling_weights, new_particles),
            name="hard_res_if_inv",
        )
        if self.scibior_trick:
            new_weights = \
                divide(new_weights*selected_resampling_weights,
                       stop_gradient(selected_resampling_weights),
                       name="scibior_trick",
                       )
        return new_weights, new_particles

    def _resample_hard(
            self, weights: Tensor, particles: Tensor,
            rangen: Generator,
            batchsize: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Soft resampling with :math:`\alpha`=1.
        """
        bs = batchsize if batchsize is not None else self.bs
        seed = get_seed(rangen)
        position_resample = expand_dims(
            stateless_categorical(
                log(weights, name="logit"),
                self.nrp, seed, dtype="int32",
                name="position_resample",
            ),
            axis=2, name="position_resample_expanded",
        )
        new_particles = gather_nd(
            params=particles, indices=position_resample,
            batch_dims=1,
            name="new_particles"
        )
        selected_weights = gather_nd(
            params=weights, indices=position_resample,
            batch_dims=1,
            name="selected_weights",
        )
        new_weights = 1.0/(self.np) * \
            ones((bs, self.nrp),
                 dtype=self.prec, name="new_weights")
        return new_weights, selected_weights, new_particles

    def _resample_gaussian(
            self, mean: Tensor, dev: Tensor,
            rangen: Generator,
            batchsize: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Extract new particles from a Gaussian with
        mean `mean` and standard deviation `dev`.
        """
        bs = batchsize if batchsize is not None else self.bs
        mean_expanded = broadcast_to(expand_dims(
            mean, axis=1), (bs, self.nnp, self.d),
            name="mean_expanded",
        )
        dev_expanded = broadcast_to(expand_dims(
            dev, axis=1), (bs, self.nnp, self.d, self.d),
            name="dev_expanded",
        )
        seed = get_seed(rangen)
        particles = add(matvec(
            dev_expanded,
            stateless_normal(
                (bs, self.nnp, self.d),
                seed, dtype=self.prec, name="gauss_values",
            ),
            name="gaussian_repar_dev",
        ),
            mean_expanded,
            name="sum_mu",
        )
        weights = 1.0/(self.np)*ones(
            (bs, self.nnp), dtype=self.prec,
            name="unif_weights",
        )
        return weights, particles

    def recompute_state(
        self, index: Tensor, particles: Tensor,
        hist_control_rec: Tensor,
        hist_outcomes_rec: Tensor,
        hist_continue_rec: Tensor,
        hist_step_rec: Tensor,
        num_steps: int,
    ) -> Tensor:
        """Routine that recomputes the state ensemble
        starting from the measurement outcomes and the controls.

        When a resampling of the particle filter ensemble is
        carried out, a new set of particles is produced.
        In this situation `state_ensemble` generated by
        :py:meth:`~.StatefulPhysicalModel.initialize_state`
        and updated in the measurement loop
        inside the method
        :py:meth:`~.Simulation.execute`
        still refers to the old particles.
        The only way to align
        the state ensemble with the new particles
        is to re-initialise and re-evolve the
        state ensemble with the applied controls,
        the observed outcomes,
        and the new particles. That is the role of this method.

        Parameters
        ----------
        index: Tensor
            `Tensor` of shape (,) and type `int32` that
            contains the number of iterations of the measurement
            loop in
            :py:meth:`~.Simulation.execute`,
            as this method is called.
        particles: Tensor
            `Tensor` of particles of shape
            (`bs`, `np`, `d`)
            and of type `prec`.
            `d` is an attribute of :py:obj:`ParticleFilter`.
        hist_outcomes_rec: Tensor
            `Tensor` of shape
            (`num_step`, `bs`, `phys_model.outcomes_size`)
            and of type `prec`. It contains the
            history of the outcomes
            of the measurements for all the
            estimations in the batch
            up to the point at which this method is called.
            It is generated
            automatically in the
            :py:meth:`~.Simulation.execute`
            method.
        hist_control_rec: Tensor
            `Tensor` of shape
            (`num_steps`, `bs`, `phys_model.control_size`)
            and of type `prec`.
            It contains the history of the controls
            of the measurements for all the
            estimations in the batch
            up to the point at which
            this method is called. It is generated
            automatically in the
            :py:meth:`~.Simulation.execute`
            method.
        hist_step_rec: Tensor
            `Tensor` of shape (`num_steps`, `bs`, 1)
            that contains, for
            each loop of the measurement cycle, the
            index of the last
            measurement performed on each estimation
            in the batch. Different estimations
            can terminate with a different number of
            total measurements
            performed.
            This `Tensor` is fundamentally a
            table that shows for each loop of the
            measurement cycle
            (row index) the number of measurements performed
            up to that point for each estimation (column index).
            Because for some loops the measurement
            might be applied only
            to a subset of estimations, some cells can show
            a number of measurements lower than the current
            number of completed loops in the measurement cycle.
            This `Tensor` is generated
            automatically in the
            :py:meth:`~.Simulation.execute`
            method.
        hist_continue_rec: Tensor
            A `Tensor` of shape (`num_steps`, `bs`, 1)
            and type `int32`,
            `hist_continue_rec` is a flag `Tensor`
            indicating whether
            a simulation in the batch is ended or not.
            This flag is necessary because different simulations
            may stop at different numbers of measurements.
            The typical column of `hist_continue_rec` consists
            of a series of contiguous ones, say in number of `M`,
            followed by zeros until the end of the row.
            This tells us that the particular simulation to which
            the row refers, consisted of `M` measurements so far.
            In recomputing the state with different values
            for the particles, `M` measurement should be performed
            for that particular simulation.
            The parameter `hist_continue_rec`
            is generated automatically in the
            `Simulation.execute` method and could be easily derived
            from `hist_step_rec`, which contains more information.
        batchsize: Tensor
            Dynamical batchsize of the recomputed states.
        num_steps: int
            The maximum number of steps in the training loop.
            It is the attribute `num_steps`
            of the data class :py:obj:`SimulationParameters`.

        Returns
        -------
        state_ensemble: Tensor
            The state of the quantum probe associated
            with each entry of `particles`, that has been
            recomputed using the applied controls and the
            observed outcomes. Each entry of `state_ensemble` is
            the state of the probe computed as if the true values
            of the parameters were the particles of the ensemble.
            When these are resampled, the states do not refer
            anymore to the correct values of the parameters, and
            the only way to realign them is to recompute them
            from scratch. `state_ensemble` is a `Tensor` of
            shape (`bs`, `np`, `state_size`) and type `state_type`,
            where `state_size` and `state_type` are attributes
            of :py:obj:`ParticleFilter`.

        Notes
        -----
        The stopping condition in the
        measurement loop,
        which determines the number of measurements
        for an estimation,
        depends on the number of computed resources,
        which depends only on the applied controls and
        observed outcomes.
        For this reason, it is possible to reuse the objects
        `hist_step_rec` and `hist_continue_rec`, which
        were generated
        before the resampling. For the same reason, these objects
        are actually superfluous in the function call,
        as they could be
        recomputed from the `hist_outcomes_rec` and
        `hist_continue_rec` parameters.
        """
        # Recasting of estimation_not_finished
        hist_continue_rec = cast(hist_continue_rec, dtype="bool")
        # Initialization of the state particle filter
        state_ensemble = self.phys_model.wrapper_initialize_state(
            particles, self.np,
        )
        j = constant(0, dtype="int32")

        def loop_body(j, state_ensemble):

            slice_outcomes = hist_outcomes_rec[j, :, :]
            slice_control = hist_control_rec[j, :, :]
            slice_enf = hist_continue_rec[j, :, :]
            slice_meas_step = hist_step_rec[j, :, :]
            slice_enf = reshape(slice_enf, (self.bs, 1, 1))

            controls_broad = broadcast_to(
                expand_dims(slice_control, axis=1),
                (self.bs, self.np, self.phys_model.controls_size),
                name='control_broad'
            )
            outcomes_broad = broadcast_to(
                expand_dims(slice_outcomes, axis=1),
                (self.bs, self.np, self.phys_model.outcomes_size),
                name='outcomes_broad',
            )
            step_broad = broadcast_to(
                expand_dims(slice_meas_step, axis=2),
                (self.bs, self.np, 1),
                name='slice_meas_step',
            )
            _, post_state_ensemble = self.phys_model.wrapper_model(
                outcomes_broad, controls_broad, particles,
                state_ensemble, step_broad, num_systems=self.np,
            )
            state_ensemble = where(
                broadcast_to(
                    slice_enf,
                    (self.bs, self.np, self.state_size),
                ), post_state_ensemble, state_ensemble,
                name="state_ensemble_update",
            )
            j += 1  # Update of the 'local' index
            return j, state_ensemble

        loop_variables = (j, state_ensemble,)

        j, state_ensemble = while_loop(
            lambda j, _: less_equal(j, index),
            loop_body,
            loop_variables,
            maximum_iterations=num_steps,
            name="resample_loop"
        )

        return state_ensemble

    def compute_state_mean(
        self, weights: Tensor, state_ensemble: Tensor,
        batchsize: Tensor,
    ) -> Tensor:
        r"""Compute the mean state of a particle
        filter ensemble.

        Given the weight :math:`w_j` and state
        :math:`\rho_j` of the `j`-th particle
        in the ensemble, the mean state is defined as:

        .. math::
            \bar{\rho} = \sum_{j=1}^N w_j \rho_j.

        Parameters
        ----------
        weights: Tensor
            Tensor of particle weights with
            shape (`bs`, `np`) and type `prec`.
        state: Tensor
            Tensor of states for the probe, with
            shape (`bs`, `np`, `state_size`) and
            type `state_type`.

        Returns
        -------
        Tensor
            Mean state for each estimation in the
            batch computed with the posterior
            distribution.
            It is a `Tensor` of shape (`bs`, `state_size`)
            and type `state_type`. `state_size` and
            `state_type` are attributes of the
            :py:obj:`ParticleFilter` class.
        """
        broad_weights = broadcast_to(
            expand_dims(weights, axis=2),
            (batchsize, self.np, self.state_size),
            name="broad_weights",
        )
        return reduce_sum(
            state_ensemble*broad_weights, axis=1,
            name="mean_state",
        )

    def _trim_particles(
            self, new_particles: Tensor,
            batchsize: Optional[Tensor] = None,
    ) -> Tensor:
        """Caps to bounds the new particles that are
        outside of the parameter bounds.
        """
        bs = batchsize if batchsize is not None else self.bs
        one_arr = ones(
            (bs, self.np, ), dtype=self.prec,
            name="ones",
        )
        newparams = [
            trim_single_param(
                new_particles[:, :, i], param, one_arr,
            )
            for i, param in enumerate(self.phys_model.params)
        ]
        return concat(newparams, 2, name="concat_param")

    def compute_mean(
            self, weights: Tensor, particles: Tensor,
            batchsize: Optional[Tensor] = None,
            dims: Optional[int] = None,
    ) -> Tensor:
        r"""Computes the empirical mean of the
        unknown parameters based on the particle
        filter posterior distribution.

        The mean :math:`\vec{\mu}` is computed as follows:

        .. math::
            \vec{\mu} = \sum_{j=1}^N w_j \vec{\theta}_j \; ,

        :math:`w_j` are the weights of the particles,
        and :math:`\vec{\theta}_j` are the particles.

        Parameters
        ----------
        weights: Tensor
            A `Tensor` of weights with shape (`bs`, `np`)
            and type `prec`.
        particles: Tensor
            A `Tensor` of particles with shape (`bs`, `np`, `d`)
            and type `prec`, where `d` is the attribute of
            the :py:obj:`ParticleFilter` object.
        dims: int, optional
            This method can handle a `particles` parameter
            with a last dimension different from the
            attribute `d` of the :py:obj:`ParticleFilter`
            class. If `dims` is passed to the method call,
            the shape of `particles` is expected to be
            (`bs`, `np`, `dims`).
            This parameter is typically used when computing the mean
            of a subset of the parameters.

        Returns
        -------
        Tensor
            The mean of the unknown parameters
            computed on the posterior
            distribution represented by
            `weights` and `particles`.
            It is a `Tensor` of shape
            (`bs`, `d`), or
            possibly (`bs`, `dims`) if `dims` is defined.
            Its type is `prec`.
        """
        # We allow also array of parameters of
        # size different from the one that we have
        # by default on the particle filter.
        d = self.d if dims is None else dims
        bs = self.bs if batchsize is None else batchsize
        broad_weights = broadcast_to(expand_dims(
            weights, axis=2), (bs, self.np, d),
            name="broad_weights",
        )
        return reduce_sum(
            particles*broad_weights, axis=1,
            name="mean",
        )

    def compute_max_weights(
            self, weights: Tensor, particles: Tensor,
    ) -> Tensor:
        """Returns the particle with the maximum weight
        in the ensemble.

        **Achtung!** This method does not return the true
        maximum of the posterior if the particles
        are not equidistant in the parameter space. In general,
        it should not be used if the resampling is active
        or if the initialization of the particles in the
        ensemble is stochastic. Its use case is for pure
        **hypothesis testing**, where it returns the most
        plausible hypothesis.

        Parameters
        ----------
        weights: Tensor
            A `Tensor` of weights with shape
            (`bs`, `np`)
            and type `prec`.
        particles: Tensor
            A `Tensor` of particles with shape
            (`bs`, `np`, `d`)
            and type `prec`.
            Here, `d` is the attribute of the
            :py:obj:`ParticleFilter` object.

        Returns
        -------
        Tensor
            A `Tensor` of shape (`bs`, `d`)
            and type `prec`
            with the particle having the highest weight for
            each ensemble in the batch.
        """
        max_positions = expand_dims(
            argmax(weights, axis=1), axis=1,
            name="expanded_max_position",
        )
        return gather_nd(
            params=particles, indices=max_positions,
            batch_dims=1,
            name="max_posterior_estimator",
        )

    def extract_particles(
        self, weights: Tensor, particles: Tensor,
        num_extractions: int, rangen: Generator,
    ) -> Tensor:
        """Extracts `num_extractions` particles from
        the posterior distribution represented by the particle
        filter ensemble.

        Parameters
        ----------
        weights: Tensor
            A `Tensor` of weights with shape
            (`bs`, `np`)
            and type `prec`.
        particles: Tensor
            A `Tensor` of particles with shape
            (`bs`, `np`, `d`)
            and type `prec`.
            Here, `d` is the attribute of the
            :py:obj:`ParticleFilter` object.
        num_extraction: int
            The number of particles to be
            stochastically extracted.
        rangen: Generator
            A random number generator from the module
            :py:mod:`tensorflow.random`.

        Returns
        -------
        Tensor
            A `Tensor` of shape
            (`bs`, `num_extractions`, `d`),
            of type `prec`, containing the `num_extraction`
            particles
            sampled from the posterior distribution.
        """
        seed = get_seed(rangen)
        pos_extraction = expand_dims(stateless_categorical(
            log(weights), num_extractions, seed, dtype="int32"),
            axis=2,
            name="pos_extracted_particles",
        )
        return gather_nd(
            params=particles, indices=pos_extraction, batch_dims=1,
            name="extracted_particles",
        )

    def compute_covariance(
            self, weights: Tensor, particles: Tensor,
            batchsize: Optional[Tensor] = None,
            dims: Optional[int] = None,
    ) -> Tensor:
        r"""Computes the covariance matrix of the Bayesian
        posterior distribution.

        The empirical covariance matrix of the particle
        filter ensemble is defined as:

        .. math::
            \Sigma = \sum_{j=1}^N w_j^t (\vec{\theta}_j -
            \vec{\mu}) (\vec{\theta}_j - \vec{\mu})^\intercal,

        where :math:`\vec{\mu}` is the mean value
        returned by the method :py:meth:`compute_mean`.

        Parameters
        ----------
        weights: Tensor
            A `Tensor` of weights of shape
            (`bs`, `np`) and of type `prec`.
            `bs`, `np`, and `prec` are attributes
            of the :py:obj:`~.ParticleFilter` class.
        particles: Tensor
            A `Tensor` of particles of shape
            (`bs`, `np`, `d`) and of type `prec`,
            `bs`, `np`, `d`, and `prec` are attributes
            of the :py:obj:`~.ParticleFilter` class.
        dims: int, optional
            This method can deal with a last dimension
            for the `particles` parameter different
            from the attribute `d` of the
            :py:obj:`ParticleFilter` object.
            If this optional parameter is passed to the method
            call, then the shape of `particles` is expected to
            be (`bs`, `np`, `dims`). Its typical use is when
            we compute the covariance of a subset only of
            the parameters.

        Returns
        -------
        Tensor
            A `Tensor` of shape (`bs`, `d`, `d`) of type
            `prec` with the covariance matrix of the Bayesian
            posterior for each ensemble in the batch.
            if `dims` was passed to the function call
            then the outcome `Tensor` has shape
            (`bs`, `dims`, `dims`).
        """
        d = self.d if dims is None else dims
        bs = self.bs if batchsize is None else batchsize
        broad_mean = broadcast_to(expand_dims(self.compute_mean(
            weights, particles, batchsize=bs, dims=d), axis=1),
            (bs, self.np, d),
            name="broad_mean"
        )
        broad_weights = broadcast_to(expand_dims(
            weights, axis=2), (bs, self.np, d),
            name="broad_weights",
        )
        return matmul(
            transpose(
                broad_weights*(particles - broad_mean),
                (0, 2, 1),
            ),
            (particles - broad_mean),
            name="cov",
        )

    def _invalid_weights(
            self, weights: Tensor,
            batchsize: Optional[Tensor] = None,
    ) -> Tensor:
        """Checks if some elements of the batch
        has null weights.
        """
        bs = batchsize if batchsize is not None else self.bs
        return equal(
            reduce_sum(weights, axis=1, keepdims=True),
            zeros((bs, 1), dtype=self.prec),
            name="invalide_weights"
        )

    def __str__(self):
        return f"rf_{self.res_frac:.2f}_" + "_".join([
            f"{par.name}_min_{par.bounds[0]:.2f}_max_"
            f"{par.bounds[1]:.2f}"
            for par in self.phys_model.params
        ])
