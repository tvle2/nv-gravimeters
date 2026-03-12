"""Module containing the
:py:obj:`~.StatefulSimulation` class."""


from tensorflow import Tensor
from tensorflow.random import Generator

from .simulation import Simulation


class StatefulSimulation(Simulation):
    """Simulation class to be derived if the
    quantum probe is stateful.

    If the class chosen for
    the model description of the probe is
    :py:obj:`~.StatefulPhysicalModel`, then
    :py:obj:`~.StatefulSimulation`
    is the class to be derived for describing
    the simulation.

    **Achtung!** It is technically possible
    to have two different instances of the
    :py:obj:`~.PhysicalModel` class passed
    to the :py:obj:`~.ParticleFilter`
    and to the :py:obj:`~.Simulation` objects
    in their respective constructors,
    one being stateful
    and the other not. For example, one might have
    a stateful simulation with a stateful
    physical model object while the physical
    model of the particle filter is stateless.
    This means that the simulation is going
    to take into account the evolving state
    of the probe in sampling the measurement
    outcomes but the particle filter is unable
    to keep track of the states ensemble of the
    probe. This can be done to save memory in the
    training at the expense of precision.
    """

    def generate_input(
        self, weights: Tensor, particles: Tensor,
        state_ensemble: Tensor,
        meas_step: Tensor, used_resources: Tensor,
        rangen,
    ) -> Tensor:
        """Computes from the particle filter ensemble and the
        states ensemble the input to the `control_strategy`
        attribute of the :py:obj:`~.Simulation` class.

        **Achtung!** This method must be implemented
        by the user.

        Parameters
        ----------
        weights: Tensor
            A `Tensor` of weights for the particles and states
            ensembles with shape (`bs`, `pf.np`)
            and type `prec`. Here, `pf`, `bs`, and `prec` are
            attributes of :py:obj:`~.Simulation`.
        particles: Tensor
            A `Tensor` with shape (`bs`, `pf.np`, `pf.d`)
            and type `prec` containing the particles of the
            ensemble. Here, `pf`, `bs`,
            and `prec` are
            attributes of :py:obj:`~.Simulation`.
        state_ensemble: Tensor
            The state of the quantum probe associated
            with each entry of `particles`.
            Each entry of `state_ensemble` is
            the state of the probe computed as if the particles
            of the ensemble were the true values of the unknown
            parameters.
            It is a `Tensor` of
            shape (`bs`, `pf.np`, `pf.state_size`) and
            type `pf.state_type`,
            where `pf` and `bs` are
            attributes of :py:obj:`~.Simulation`.
        meas_step: Tensor
            The index of the current measurement on the probe system.
            The counting starts from zero.
            It is a `Tensor` of shape (`bs`, 1) and of type `prec`.
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

            .. image:: ../docs/_static/neural_network.png
                :width: 400
                :alt: neural_network
        """
        raise NotImplementedError("You should override this method!")

    def _generate_input(
        self, weights: Tensor, particles: Tensor,
        state_ensemble: Tensor,
        meas_step: Tensor, used_resources: Tensor,
        rangen: Generator,
    ):
        return self.generate_input(
            weights, particles, state_ensemble,
            meas_step, used_resources,
            rangen,
        )

    def loss_function(
        self, weights: Tensor, particles: Tensor,
        true_state: Tensor, state_ensemble: Tensor,
        true_values: Tensor, used_resources: Tensor,
        meas_step: Tensor,
    ) -> Tensor:
        """Returns the loss to minimize in the training.

        **Achtung!** This method has to be implemented
        by the user.

        The loss should be the error of the
        metrological task, so that minimizing it in
        the training means
        optimizing the precision of the sensor.
        This function
        must return a loss value for each estimation
        in the batch. The actual quantity that
        is differentiated in the
        the gradient update
        depends also on the flags of the
        `simpars` attribute of
        :py:obj:`~.Simulation`, which is an instance
        of :py:obj:`~.SimulationParameters`. Typically
        the loss define by the user in this method is
        mixed with the log-likelihood
        of the of the observed outcomes in the
        simulation in order to obtain a bias-free
        estimator for the gradient from the batch.

        Parameters
        ----------
        weights: Tensor
            A `Tensor` of weights for the particles and states
            ensembles with shape (`bs`, `pf.np`)
            and type `prec`. Here, `pf`, `bs`, and `prec` are
            attributes of :py:obj:`~.Simulation`.
        particles: Tensor
            A `Tensor` with shape (`bs`, `pf.np`, `pf.d`)
            and type `prec` containing the particles of
            the ensemble. Here, `pf`, `bs`,
            and `prec` are
            attributes of :py:obj:`~.Simulation`.
        true_states: Tensor
            The true unobservable state of the probe
            in the estimation,
            computed from the evolution determined
            (among other factors) by the encoding of
            the parameters `true_values`.
            This is a `Tensor` of shape
            (`bs`, 1, `pf.state_size`),
            where `bs` and `pf` are attributes of
            the :py:obj:`~.Simulation` class.
            Its type is `pf.state_type`.
        state_ensemble: Tensor
            The state of the quantum probe associated
            with each entry of `particles`.
            Each entry of `state_ensemble` is
            the state of the probe computed as if the particles
            of the ensemble were the true values of the unknown
            parameters.
            It is a `Tensor` of
            shape (`bs`, `pf.np`, `pf.state_size`) and
            type `pf.state_type`,
            where `pf` and `bs` are
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
            containing the metrological error defined by the
            used for each estimation in the batch.
            `bs` and `prec` are attributes of the
            :py:obj:`~.Simulation` class.

        """
        raise NotImplementedError("You should override this method!")

    def _loss_function(
        self, weights: Tensor, particles: Tensor, true_state: Tensor,
        state_ensemble: Tensor, true_values: Tensor,
        used_resources: Tensor, meas_step: Tensor,
    ) -> Tensor:
        return self.loss_function(
            weights, particles, true_state, state_ensemble,
            true_values, used_resources, meas_step,
        )
