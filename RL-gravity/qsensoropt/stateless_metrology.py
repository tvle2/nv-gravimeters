"""Module containing the :py:obj:`~.StatelessMetrology`
class
"""

from typing import List, Callable, Optional
from itertools import product

from tensorflow import zeros, expand_dims, constant, concat, \
    reshape, sqrt, broadcast_to, eye, einsum, Tensor
from tensorflow.math import divide_no_nan, log
from tensorflow.linalg import diag_part, matmul, trace

from tensorflow.random import Generator

from .physical_model import PhysicalModel
from .particle_filter import ParticleFilter
from .utils import normalize
from .simulation_parameters import SimulationParameters
from .stateless_simulation import StatelessSimulation


class StatelessMetrology(StatelessSimulation):
    r"""Simulation class for the standard
    Bayesian metrological task, with a steless
    probe.

    This class describes a typical metrological task,
    where the estimator :math:`\hat{\vec{\theta}}`
    for the unknown parameters is the mean
    value of the parameters
    on the Bayesian posterior distribution,
    computed with the method
    :py:meth:`~.ParticleFilter.compute_mean`. The loss
    implemented in :py:meth:`~.StatelessMetrology.loss_function`
    is the mean square error (MSE) between this estimator
    and the true values of the unknowns
    used in the simulation.

    The input to the neural network,
    computed in the method
    :py:meth:`~.StatelessMetrology.generate_input`,
    is a `Tensor` obtained by
    concatenating at each step the estimators for the
    unknown parameters, their
    standard deviations, their correlation
    matrix, the total number of consumed resources,
    and the measurement step. All this
    variables are reparametrized/rescaled
    to make them fit in the `[-1, 1]` interval,
    this makes them more suitable to be
    the inputs of a neural network.

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
        Automatically generated size of the
        `input_strategy` `Tensor` returned by the
        method :py:meth:`~.StatelessMetrology.generate_input`.
        Given `d` the number of unknown parameters,
        the value of `input_size` is given by the
        following formula:

        ``input_size = d**2+2*d+2``
    input_name: List[str]
        List of names for each scalar entry of the
        `input_strategy` `Tensor` generate by
        :py:meth:`~.StatelessMetrology.generate_input`.
    simpars: :py:obj:`~.SimulationParameters`
        Parameter `simpars` passed to the
        class constructor.
    ns: int
        Maximum number of steps of the
        measurement loop in the
        :py:meth:`~.Simulation.execute` method.
        It is the :py:attr:`~.SimulationParameters.num_steps`
        attribute of `simpars`.
    cov_weight_matrix_tensor: Tensor
        `Tensor` version of the parameter
        `cov_weight_matrix` passed to the
        class constructor. It is `Tensor` of shape
        (`bs`, `d`, `d`) and of type `prec` that
        contains `bs` repetitions of the parameter
        `cov_weight_matrix` passed to the class
        constructor. In case no `cov_weight_matrix`
        is passed to the constructor this `Tensor`
        contains `bs` copies of the
        `d`-dimensional identity matrix.
    """

    def __init__(
            self, particle_filter: ParticleFilter,
            phys_model: PhysicalModel,
            control_strategy: Callable,
            simpars: SimulationParameters,
            cov_weight_matrix: Optional[List] = None,
    ):
        r"""Parameters passed to the constructor
        of the :py:obj:`~.StatelessMetrology` class.

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
            :py:meth:`~.StatelessPhysicalModel.perform_measurement`
            that simulates the measurement
            on the probe.
        control_strategy: Callable
            Callable object (normally a
            function or a lambda function) that
            computes the values of the controls
            for the next measurement from
            the `Tensor` `input_strategy`, which is
            produced by the method
            :py:meth:`~.StatelessSimulation.generate_input`
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
        simpars: :py:obj:`~.SimulationParameters`
            Contains the flags and parameters
            that regulate the stopping
            condition of the measurement loop
            and modify the loss function used in the
            training.
        cov_weight_matrix: List, optional
            Weight matrix :math:`G` appearing
            in the computation of the loss performed
            by the method
            :py:meth:`~.StatelessMetrology.loss_function`.
            When passed, this parameter must be a `List`
            with `d` rows and `d` columns,
            where `d` is the number of parameters
            to estimate. It must represent a positive
            semidefinite matrix.
            This matrix controls how the errors
            for the multiple parameters are weighted in
            the scalar loss used in the weight
            update step. It regulates whether a parameter is
            of interest or is a nuisance.
            If this parameter is not passed then the
            default weight matrix is the identity, i.e.
            :math:`G=\text{Id}`.
        """
        pars = simpars

        # Set the input size
        d = particle_filter.d
        input_size = d**2+2*d+2
        # Set the name of each column of the input
        input_name = [f"Mean_{par.name}" for par
                      in particle_filter.phys_model.params] + \
            [f"LogDev_{par.name}" for par
             in particle_filter.phys_model.params] + \
            [f"Corr_{par1.name}_{par2.name}" for par1, par2
             in product(particle_filter.phys_model.params,
                        particle_filter.phys_model.params)] + \
            ["StepOverMaxStep", "ResOverMaxRes", ]

        super().__init__(
            particle_filter, phys_model,
            control_strategy,
            input_size, input_name,
            simpars,
        )

        # Weight matrix or the loss
        if cov_weight_matrix is None:
            self.cov_weight_matrix_tensor = \
                eye(d, batch_shape=[self.bs], dtype=pars.prec)
        else:
            self.cov_weight_matrix_tensor = broadcast_to(
                expand_dims(constant(
                    cov_weight_matrix, dtype=pars.prec
                ), axis=0),
                (self.bs, self.pf.d, self.pf.d),
            )

        if not self.phys_model.state_specifics["type"] in \
                ("float64", "float32"):
            raise ValueError("The allowed values of \
                             prec are float32 and float64.")

    def generate_input(
        self, weights: Tensor, particles: Tensor,
        meas_step: Tensor, used_resources: Tensor,
        rangen: Generator,
    ) -> Tensor:
        r"""Returns the input tensor
        for `control_strategy` computed
        by concatenating the first
        moments of particle filter ensemble
        together with the used
        resources, and the measurement step.

        Parameters
        ----------
        weights: Tensor
            A `Tensor` of weights for the particle filter
            ensemble with shape (`bs`, `pf.np`)
            and type `prec`. Here, `pf`, `bs`, and `prec` are
            attributes of :py:obj:`~.Simulation`.
        particles: Tensor
            A `Tensor` with shape (`bs`, `pf.np`, `pf.d`)
            and type `prec` containing the particles of the
            ensemble. Here, `pf`, `bs`,
            and `prec` are
            attributes of :py:obj:`~.Simulation`.
        meas_step: Tensor
            The index of the current measurement on the probe system.
            The counting starts from zero.
            It is a `Tensor` of shape (`bs`, 1) and of type `int32`.
            `bs` is the attribute of the :py:obj:`~.Simulation` class.
        used_resources: Tensor
            A `Tensor` of shape (`bs`, 1) containing
            the total amount of consumed resources for
            each estimation in the batch up to the
            point this method is called.
            `bs` is the attribute of the
            :py:obj:`~.Simulation` class. The resources
            are counted according to the user defined method
            :py:meth:`~.PhysicalModel.count_resources`
            from the attribute `phys_model` of
            the :py:obj:`~.Simulation` class.
        rangen: Generator
            A random number generator from the module
            :py:mod:`tensorflow.random`.

        Returns
        -------
        input_strategy: Tensor
            `Tensor` of shape (`bs`, `input_size`),
            where `bs` and `input_size` are attributes
            of the :py:obj:`~.Simulation` class. This is
            the `Tensor` passed as a parameter
            to the call of
            `control_strategy`, which is an attribute
            of the :py:obj:`~.Simulation`
            class. From the elaboration of this input
            the function `control_strategy` (which
            is typically a wrapper for a neural network)
            produces the controls for the next measurement.

            `input_strategy` is composed on order of

            * Mean of unknown parameters computed
              on the Bayesian posterior distribution represented by the
              particle filter ensemble (the parameters `particles`
              and `weights`). It is compute calling the method
              :py:meth:`~.ParticleFilter.compute_mean` of the
              `pf` attribute. These mean values are normalized to
              lay in the interval `[-1, 1]`. This is possible since
              the extrema of the admissible values for each parameters
              are known and codified in each :py:obj:`~.Parameter` object.
              In the `input_name` list
              these inputs are called "Mean_param.name", where
              `param` is the corresponding :py:obj:`~.Parameter`
              object. These inputs are `d` scalars.

            * Standard deviations around the mean for each parameter
              computed from the Bayesian posterior distribution.
              The method
              :py:meth:`~.ParticleFilter.compute_covariance` is used
              to compute the covariance matrix of the particle filter
              ensemble. Calling this matrix :math:`C` the next `d` inputs
              for the control strategy are given by

              .. math::
                  I_j = -\frac{2}{10} \ln \sqrt{C_{jj}} - 1 \; ,

              being :math:`\sqrt{C_{jj}}` the said standard deviations.
              This time, since we do not know in advance the admissible
              values of the covariance matrix, we cannot cast the standard
              deviation exactly in `[-1, 1]`, but we can do it approximately
              for standard deviations in the range `1~1e-5`, through the above
              formula. In the `input_name` list
              these inputs are called "LogDev_param.name", where
              `param` is the corresponding :py:obj:`~.Parameter`
              object. These inputs are `d` scalars.

            * Correlation matrix between the parameters, computed as

              .. math::
                  \rho_{ij} = \frac{C_{ij}}{\sqrt{C_{ii} C_{jj}}} \; .

              This matrix doesn't need normalization, since its entries
              are already in the interval `[-1, 1]`.
              The matrix :math:`\rho_{ij}`
              is flattened and each entry is added to `input_strategy`,
              and called "Corr_param1.name_param2.name",
              where `param1` and `param2` are :py:obj:`~.Parameter`
              objects. These inputs are `d**2` scalars.

            * The index of the measurement step `meas_step` for
              each simulation in the batch, normalized in `[-1, 1]`.
              This input is called `StepOverMaxStep` and is one
              single scalar.

            * The amount of consumed resources, i.e. the
              parameter `used_resources`, normalized in `[-1, 1]`.
              This input is called `resOverMaxRes` and is one
              single scalar.

            Summing the total number of scalar inputs we get the formula
            for the attribute `input_size`, i.e.

            ``input_size = d**2+2*d+2``
        """
        pars = self.simpars
        # First two moments of the pf distribution
        mean = self.pf.compute_mean(weights, particles)
        cov = self.pf.compute_covariance(weights, particles)
        # Setting the size and the name of the input
        # Loading of the normalized mean
        mean_tensor = zeros((self.bs, 0), dtype=pars.prec)
        for i, par in enumerate(self.pf.phys_model.params):
            mean_part = expand_dims(
                normalize(mean[:, i], par.bounds), axis=1, name="mean_part",
            )
            mean_tensor = concat(
                [mean_tensor, mean_part], 1, name="mean_tensor",
            )
        dev_diag = sqrt(abs(diag_part(cov)), name="dev_diag")
        shift = 1e-300 if pars.prec == "float64" else 1e-35
        log_dev_diag = normalize(
            log(dev_diag+shift, name="log_dev_diag"), (-10.0, 0.0),
        )
        denominator_matrix = matmul(
            expand_dims(dev_diag, axis=2), expand_dims(dev_diag, axis=1),
            name="dev_out_dev",
        )
        # Correlation matrix of the parameters,
        rho = divide_no_nan(
            cov, denominator_matrix, name="corr_matrix",
        )
        scaled_step = normalize(meas_step, (0, self.simpars.num_steps))
        scaled_resources = normalize(
            used_resources, (0, self.simpars.max_resources))
        # Logarithm of the standard deviations, e^(-10) ~ 1e-5
        input_tensor = concat(
            [mean_tensor, log_dev_diag,
             reshape(rho, (self.bs, self.pf.d**2)),
             scaled_step, scaled_resources],
            1, name="input_tensor",
        )
        return input_tensor

    def loss_function(
        self, weights: Tensor, particles: Tensor,
        true_values: Tensor,
        used_resources: Tensor, meas_step: Tensor,
    ):
        r"""

        Parameters
        ----------
        weights: Tensor
            A `Tensor` of weights for the particle filter
            ensemble with shape (`bs`, `pf.np`)
            and type `prec`. Here, `pf`, `bs`, and `prec` are
            attributes of :py:obj:`~.Simulation`.
        particles: Tensor
            A `Tensor` with shape (`bs`, `pf.np`, `pf.d`)
            and type `prec` containing the particles of
            the ensemble. Here, `pf`, `bs`,
            and `prec` are
            attributes of :py:obj:`~.Simulation`.
        true_values: Tensor
            Contains the true values of the unknown
            parameters in the simulations.
            The loss is in general computed by comparing a
            suitable estimator to these values.
            It is a `Tensor` of shape
            (`bs`, 1, `pf.d`) and type `prec`, where
            `bs`, `pf`, and `prec` are
            attributes of the :py:obj:`~.Simulation` class.
        used_resources: Tensor
            A `Tensor` of shape (`bs`, 1) containing
            the total amount of consumed resources for
            each estimation in the batch up to the
            point this method is called.
            `bs` is the attribute of the
            :py:obj:`~.Simulation` class. The resources
            are counted according to the user defined method
            :py:meth:`~.PhysicalModel.count_resources`
            from the attribute `phys_model` of
            the :py:obj:`~.Simulation` class.
        meas_step: Tensor
            The index of the current measurement on the probe system.
            The counting starts from zero.
            It is a `Tensor` of shape (`bs`, 1) and of type `int32`.
            `bs` is the attribute of the :py:obj:`~.Simulation` class.

        Returns
        -------
        loss_values: Tensor
            `Tensor` of shape (`bs`, 1) and type `prec`
            containing the mean square error of the mean
            of the posterior estimator.
            `bs` and `prec` are attributes of the
            :py:obj:`~.Simulation` class.

        Returns for each estimation in the batch
        the mean square error
        contracted with the weight matrix :math:`G`,
        i.e. the attribute `cov_weight_matrix_tensor`

        The error on the metrological estimation task
        is:

        .. math::
            \mathcal{L} (\vec{\lambda}) = \text{tr}
            [ G \cdot \Sigma (\vec{\lambda})] \; ,
            :label: loss_G_stateless

        where :math:`\vec{\lambda}` are the trainable
        variables of the control
        strategy (the weights and biases of the neural
        network), :math:`\Sigma` is the mean error matrix
        of the estimator :math:`\hat{\vec{\theta}}` on the
        batch, i.e.

        .. math::
            \Sigma_{ij} = \sum_{k=1}^B (\hat{\vec{\theta}}
            - \vec{\theta})_i
            (\hat{\vec{\theta}} - \vec{\theta})_j \; ,

        while :math:`G` is a semi-positive matrix
        of shape (`d`, `d`) called
        the weight matrix
        that is used to obtain a scalar error in
        a multiparameter metrological problem.
        The integer `d` is the dimension
        of :math:`\vec{\theta}`.
        This matrix controls which errors contribute
        to the final loss and how much. It discriminates
        also
        between parameters of interest and nuisances,
        which do not contribute to the scalar loss
        :math:`\mathcal{L}(\vec{\lambda})`, because their
        corresponding entries in the :math:`G` matrix
        are null.

        The mean loss in :eq:`loss_G_stateless` can be expanded as

        .. math::
            \mathcal{L} (\vec{\lambda}) = \sum_{k=1}^B
            \text{tr} [G \cdot (\hat{\vec{\theta}}_k
            - \vec{\theta}_k)_i
            (\hat{\vec{\theta}}_k - \vec{\theta}_k)_j ] \; ,

        from which is clear what the loss
        :math:`\ell (\omega_k, \vec{\lambda})` for each single
        estimation in the batch should be:

        .. math::
            \ell (\omega_k, \vec{\lambda}) = \text{tr}
            [G \cdot (\hat{\vec{\theta}}_k
            - \vec{\theta}_k)_i
            (\hat{\vec{\theta}}_k - \vec{\theta}_k)_j ] \; ,

        with :math:`\omega_k = (\vec{y}_k, \vec{\theta}_k)`
        being a tuple that contains the string of observed
        measurement outcomes and the true values
        :math:`\vec{\theta}_k` for a particular
        estimation in the batch.
        This is the loss implemented in this method.
        """
        # Expected value of the parameters based on the posterior distribution
        mean = self.pf.compute_mean(weights, particles)
        diff = mean-true_values[:, 0, :]  # (bs, d)
        diff_tensor = einsum('ai,aj->aij', diff, diff)
        loss_value = expand_dims(
            trace(matmul(self.cov_weight_matrix_tensor,
                         diff_tensor)),
            axis=1,
            name="loss_value",
        )
        return loss_value
