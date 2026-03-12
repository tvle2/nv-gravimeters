"""Module containing the
:py:obj:`~.StatelessSimulation` class."""

from tensorflow import Tensor

from .simulation import Simulation


class StatelessSimulation(Simulation):
    """Simulation class to be derived if the
    quantum probe is stateless.

    If the class chosen for
    the model description of the probe is
    :py:obj:`~.StatelessPhysicalModel`, then
    :py:obj:`~.StatelessSimulation`
    is the class to be derived for describing
    the simulation.
    """

    def generate_input(
        self, weights: Tensor, particles: Tensor,
        meas_step: Tensor, used_resources: Tensor,
        rangen,
    ) -> Tensor:
        """Computes from the particle filter ensemble
        the input to the `control_strategy`
        attribute of the :py:obj:`~.Simulation` class.

        **Achtung!** This method has to be implemented
        by the user.

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
        rangen
    ):
        return self.generate_input(
            weights, particles, meas_step, used_resources,
            rangen,
        )

    def loss_function(
        self, weights: Tensor, particles: Tensor,
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
            containing the metrological error defined by the
            used for each estimation in the batches.
            `bs` and `prec` are attributes of the
            :py:obj:`~.Simulation` class.

        """
        raise NotImplementedError("You should override this method!")

    def _loss_function(
        self, weights: Tensor, particles: Tensor,
        true_state: Tensor,
        state_ensemble: Tensor, true_values: Tensor,
        used_resources: Tensor, meas_step: Tensor,
    ) -> Tensor:
        return self.loss_function(
            weights, particles, true_values,
            used_resources, meas_step,
        )
