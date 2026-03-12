"""Module containing the :py:obj:`~.InverseSqrtDecay`
class.
"""

from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow import Tensor, convert_to_tensor, cast, constant
from tensorflow.math import divide, add, sqrt


class InverseSqrtDecay(LearningRateSchedule):
    r"""Inverse square root decay
    of the learning rate.

    This class implements a learning rate
    schedule that allows
    the learning rate used in the training to be
    a function of the iteration number, that is
    the number of gradient descent updates of the
    training variables already performed. In
    particular, this class realize the
    following schedule for the learning rate
    :math:`l_r(i)` as function of the
    iteration number (starting from zero):

    .. math::
        l_r(i) := \frac{l_r(0)}{\sqrt{i+1}} \; ,

    being :math:`l_r(0)` the initial learning rate.
    The reason for wanting a decay of the
    learning rate is to let the neural network
    learn finer and finer details of
    the optimal control strategy as the training
    goes on, for which smaller and smaller
    updated steps are needed.
    """

    def __init__(
        self,
        initial_learning_rate: float,
        prec: str = "float64"
    ):
        """Constructor of the
        :py:obj:`~.InverseSqrtTimeDecay` class.

        Parameters
        ----------
        initial_learning_rate: float
            Learning rate use in the first iteration
            of the training cycle (the number zero iteration).
        prec: str
            Floating point precision of the variable to
            be trained. Can be either `float32` or `float64`.
        """
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.prec = prec

    def __call__(self, step) -> Tensor:

        initial_learning_rate = convert_to_tensor(
            self.initial_learning_rate,
            dtype=self.prec,
        )

        casted_step = cast(step, dtype=self.prec)

        denom = sqrt(add(
            casted_step,
            cast(constant(1), dtype=self.prec),
        ))

        return divide(
            initial_learning_rate, denom, name="learning_rate",
        )

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,

        }
        return config
