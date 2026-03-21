"""Module containing some utility functions
for training the control strategy and visualizing the
results.
"""

from typing import Tuple, TypedDict, Callable, \
    List

from tensorflow import GradientTape, reshape, \
    function, zeros, transpose, broadcast_to, \
    zeros_like, print, eye, Tensor, Variable
from tensorflow import reshape, broadcast_to, \
    linspace, Tensor, dtypes
from tensorflow import range as tfrange
from tensorflow.linalg import svd, diag, matmul
from tensorflow.math import exp, log, lgamma, abs, tanh, \
    reduce_mean
from tensorflow.math import sqrt as tfsqrt
from tensorflow.profiler.experimental import Trace, start, stop
from tensorflow.summary import trace_on, create_file_writer, \
    trace_export
from tensorflow.random import stateless_uniform, Generator
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from numpy import repeat, concatenate, savetxt, loadtxt, \
    linspace, inf
from numpy import sqrt as npsqrt
from numpy import floor as npfloor
from numpy import tanh as nptanh
from numpy import exp as npexp
import scipy.integrate as integrate
from pandas import DataFrame, Series
from os.path import join, isfile
from os import makedirs, rmdir, listdir, remove
from math import pi, floor
from datetime import datetime
from typing import Optional
from progressbar import progressbar

from .schedulers import InverseSqrtDecay


def normalize(
        input_tensor: Tensor,
        bounds: Tuple[float, float],
) -> Tensor:
    r"""Normalizes the entries of
    `input_tensor` in the interval `[-1, 1]`.
    The `input_tensor` entries must lay within the
    given tuple of upper and lower bounds.

    Parameters
    ----------
    input_tensor: Tensor
        `Tensor` of unnormalized values.
    bound: List[float, float]
        Min and max of the admissible entries
        of `input_tensor`.

    Returns
    -------
    Tensor:
        Version of `input_tensor` with entries
        normalized in the interval `[-1, 1]`.
        Each entry :math:`x` of the input becomes

        .. math::
            y = \frac{2(x-\text{bound}[0])}{\text{bound}[1]-
            \text{bound}[0]} - 1 \;

        in the returned `Tensor`, which has the same
        type and shape of `input_tensor`.
    """
    step_1 = (input_tensor-bounds[0])/(bounds[1]-bounds[0])
    return 2*step_1-1

def _keras_weights_path(file_name: str) -> str:
    return file_name if file_name.endswith(".weights.h5") else file_name + ".weights.h5"

def denormalize(
        input_tensor: Tensor,
        bounds: Tuple[float, float],
) -> Tensor:
    r"""Given `input_tensor` with values in `[-1, 1]`
    this function rescales it, so that its entries take
    values within the given tuple of extrema.

    Parameters
    ----------
    input_tensor: Tensor
        `Tensor` having entries normalized in `[-1, 1]`.
    bound: Tuple[float, float]
        Min and max of the admissible entries
        of the returned `Tensor`.

    Returns
    -------
    Tensor:
        Version of `input_tensor` normalized in
        the interval delimited by the new extrema.
        Each entry :math:`y` of `input_tensor` becomes

        .. math::
            x = \frac{y (\text{bound}[1]-
            \text{bound}[0])}{2}+\text{bounds}[0] \; .

        It is a `Tensor` with the same type and shape
        of `input_tensor`.
    """
    step_1 = (input_tensor+1)/2
    return (step_1*(bounds[1]-bounds[0]))+bounds[0]


def get_seed(
        random_generator: Generator,
) -> Tensor:
    """Generate a random seed from `random_generator`.

    Extracts uniformly a couple of integers from
    `random_generator` to be used as seed in the stateless
    functions of the :py:mod:`tensorflow.random` module.

    Parameters
    ----------
    random_generator: Generator
        Random number generator from the module
        :py:mod:`tensorflow.random`.

    Returns
    -------
    seed: Tensor
        `Tensor` of shape (2, ) and of type `int32`
        containing two random seeds.
    """
    return random_generator.uniform(
        [2, ], minval=0, maxval=dtypes.int32.max,
        dtype="int32", name="seed",
    )


def random_uniform(
        batchsize: int, prec: str,
        min_value: float, max_value: float,
        seed: Tensor,
):
    """Extracts `batchsize` random number of type `prec` between
    the `min_value` and the `max_value` from a uniform
    distribution.

    Parameters
    ----------
    batchsize: int
        Number of random values to be extracted.
    prec: str
        Type of the extracted numbers.
        It is typically `float32` or `float64`.
    min_values: float
        Lower extremum of the uniform distribution
        from which the values are extracted.
    max_values: float
        Upper extremum of the uniform distribution
        from which the values are extracted.
    seed: Tensor
        Seed of the random number generator used in this
        function. It is a `Tensor` of type `int32` and of shape
        (`2`, ). This is the kind of seed that is accepted by
        the stateless random
        function of the module :py:mod:`tensorflow.random`.
        It can be generated with the function
        :py:func:`~.utils.get_seed` from a
        :py:obj:`Generator` object.

    Returns
    -------
    Tensor:
        `Tensor` of shape (`batchsize`, ) and of
        type `prec` with uniformly extracted entries.
    """
    random_values = stateless_uniform(
        (batchsize, ), seed, maxval=1, dtype=prec,
        name="stateless_uniform",
    )
    return (max_value - min_value)*random_values + min_value


def train(
        simulation,
        optimizer: Optimizer,
        iterations: int,
        save_path: str,
        interval_save: int = 128,
        network: Optional[Model] = None,
        custom_controls: Optional[Variable] = None,
        load_best: bool = True,
        gradient_accumulation: int = 1,
        xla_compile: bool = True,
        rangen: Generator = Generator.from_seed(0xdeadd0d0),
):
    """Training routine for the neural network
    or the custom controls.

    This function contains a cycle that
    performs a gradient descent optimization
    of the training variables of the control strategy.

    Parameters
    ----------
    simulation: :py:obj:`~.Simulation`
        Contains the complete description of the
        estimation task, and how the various components
        of it (particle filter, probe system, and control)
        interact to produce the estimator and the loss.
    optimizer: :py:obj:`Optimizer`
        This object applies the gradient to the trainable
        variables of the control strategy and realized therefore
        the update of the gradient descent step.
        The default optimizer to be chosen
        is `Adam` with the variable learning
        rate described in the
        class :py:obj:`~.InverseSqrtDecay`.
    iterations: int
        Number of update steps in the training of the
        strategy. It is the number of complete executions
        of a batch of estimation tasks for training the controls.
    save_path: str
        Path containing the intermediate trained variables,
        that are saved each `interval_save` iterations,
        and the history of the loss at the end of the
        training.
    interval_save: int = 128
        Number of iterations of the training cycle
        between two consecutive saving of the training
        variables on disk.
    network: :py:obj:`Model`, optional
        Neural network model used in the `simulation`
        object to parametrize the control strategy.
        In `simulation` it is wrapped inside the
        `control_strategy` attribute and there the
        :py:obj:`~.Simulation` object has no direct access
        to its trainable variables. If no neural network
        is used in the estimation, for example
        because a static strategy is being trained,
        this parameter can be omitted.
    custom_controls: :py:obj:`Variable`, optional
        It is possible to train a control strategy that
        goes beyond what can be done with a neural network,
        by writing a custom `control_strategy` function
        which is not only a wrapper for the neural
        network. The trainable variables of this custom
        strategy are contained in the object object
        `custom_controls`.

        The typical application is that of a **static strategy**,
        that is a control strategy that depends only
        on the measurement step and not on the
        posterior distribution moments. This is a
        non-adaptive strategy, that therefore
        doesn't need to be
        computed in real time, but can be hard coded before
        the beginning of the estimation.
        Such approach will save memory, because the
        Bayesian analysis can be postponed
        and performed on a more powerful device,
        and it will also be faster, because no
        computation is necessary between two
        measurements, since the approach is
        non-adaptive. Of course one can't expect
        in general the same performances of the
        adaptive strategy.
    load_best: bool = True
        Keeps track of the loss of the control strategy
        during the training and picks the best model
        instead of the last. To pick the best model
        the loss is averaged over
        `interval_save` update steps.
    gradient_accumulation: int = 1
        This controls the "effective" batchsize of the
        training. In the training loop for each
        update of the variables the method
        :py:meth:`~.Simulation.execute` runs
        a number `gradient_accumulation` of times
        and the gradients computed from each run
        are averaged before the update step
        is performed.
        In this way, given `bs` the true batchsize of the
        `simulation` object, the "effective" batchsize
        of the training is
        `gradient_accumulation*bs`, which is the number of
        independent simulations that contribute to a single
        update step of the training variables. This feature
        can be used to increase the batchsize, when the
        memory on the machine is limited,
        since the maximum amount of memory required
        by the training is determine by
        `bs` and is independent on
        `gradient_accumulation`.
    xla_compile: bool = True
        Just-in-time XLA (Accelerated Linear Algebra)
        compilation for the strategy training.
        It should reduce the memory and training
        time on a GPU.


        **Achtung!** Not all the Tensorflow
        operations can be compiled with XLA.

        **Achtung!** It might not perform well on a CPU.
    rangen: Generator = Generator.from_seed(0xdeadd0d0)
        Random number generator from the module
        :py:mod:`tensorflow.random`.

    After the return of the :py:func:`~.utils.train`
    routine
    either `network` or the `custom_controls`
    will be trained
    and in `save_path` there will be
    a directory containing the partially trained
    variables (with name ending in "_weights")
    and a csv file with the history
    of the loss (with name ending in "_history.csv"),
    averaged over a window of
    `interval_save` iterations.

    """
    if network is not None:
        variables = network.trainable_variables
        custom = False
    elif custom_controls is not None:
        variables = [custom_controls]
        custom = True
    else:
        print("No NN or static_controls given!")
        return

    @function(jit_compile=xla_compile)
    def single_iteration(rangen):
        with GradientTape() as tape:
            loss_diff, loss = simulation.execute(rangen)
        grads = tape.gradient(loss_diff, variables)
        return grads, loss

    def update_nn(rangen):
        acc_loss = zeros((1, ), simulation.simpars.prec)
        acc_gradient = [zeros_like(this_var)
                        for this_var in variables]
        for _ in range(gradient_accumulation):
            grads, loss = single_iteration(rangen)
            acc_loss += loss
            acc_gradient = [(acc_grad+grad)
                            for acc_grad, grad in
                            zip(acc_gradient, grads)]
        acc_gradient = [this_grad /
                        gradient_accumulation for
                        this_grad in acc_gradient]
        optimizer.apply_gradients(zip(acc_gradient, variables))
        acc_loss /= gradient_accumulation
        return acc_loss

    loss_list = []
    for j in progressbar(range(iterations)):
        loss = update_nn(rangen)
        loss_list.append(loss.numpy())
        if (j+1) % interval_save == 0:  # Save the NN weights
            dir_name = join(save_path, str(simulation)) + \
                "_history_weights/"
            file_name = dir_name + str(floor((j+1)/interval_save))
            makedirs(dir_name, exist_ok=True)
            if custom:
                savetxt(file_name, custom_controls.numpy())
            else:
                network.save_weights(_keras_weights_path(file_name))
    # The last training steps,
    # that would not fit in the block
    # average are trimmed out.
    loss_array = concatenate(loss_list, axis=0)
    num_nn_saved = floor(iterations/interval_save)
    loss_df = DataFrame(
        {'Loss': loss_array[:num_nn_saved*interval_save]},
    )
    groups_block_mean = Series(
        repeat(range(num_nn_saved), interval_save),
    )
    mean_loss = loss_df.groupby(groups_block_mean).agg('mean')
    mean_loss.to_csv(
        join(save_path, str(simulation))+'_history.csv',
        index=False,
        float_format='%.4e',
    )
    if load_best:
        index_best = mean_loss.idxmin().values[0] + 1
        file_name = join(save_path, str(simulation)) + \
            "_history_weights/" + str(index_best)
        if custom:
            loaded_controls = loadtxt(file_name)
            if len(loaded_controls.shape) == 1:
                loaded_controls = loaded_controls[:, None]
            custom_controls.assign(loaded_controls)

        else:
            # Loading of the best model
            network.load_weights(_keras_weights_path(file_name))
            
    for filename in listdir(dir_name):
        file_path = join(dir_name, filename)
        if isfile(file_path):
            remove(file_path)  
    rmdir(dir_name)


def train_nn_graph(
        simulation,
        optimizer: Optimizer,
        data_dir: str,
        network,
        rangen=Generator.from_seed(0xdeadd0d0),
):
    """Produces a report of the tensor operations
    in the training that can be visualized in
    Tensorboard as a graph.

    **Achtung!** This function works only for
    the training of a neural network.

    Parameters
    ----------
    simulation: :py:obj:`~.Simulation`
        Contains the complete description of the
        estimation task, and how the various components
        of it (particle filter, probe system, and control)
        interact to produce the estimator and the loss.
    optimizer: :py:obj:`Optimizer`
        This object applies the gradient to the trainable
        variables of the control strategy and realized therefore
        the update of the gradient descent step.
        The default optimizer to be chosen
        is `Adam` with the variable learning
        rate described in the
        class :py:obj:`~.InverseSqrtDecay`.
    data_dir: str
        Directory in which the report containing
        the computational graph is saved.
    network: :py:obj:`Model`, optional
        Neural network model used in the `simulation`
        object to parametrize the control strategy.
        In `simulation` it is wrapped inside the
        `control_strategy` attribute and there the
        :py:obj:`~.Simulation` object has no direct access
        to its trainable variables. If no neural network
        is used in the estimation, for example
        because a static strategy is being trained,
        this parameter can be omitted.
    rangen: Generator = Generator.from_seed(0xdeadd0d0)
        Random number generator from the module
        :py:mod:`tensorflow.random`.
    """
    variables = network.trainable_variables

    @function
    def update_nn(rangen):
        with GradientTape() as tape:
            loss_diff, _ = simulation.execute(rangen)
        grads = tape.gradient(loss_diff, variables)
        optimizer.apply_gradients(zip(grads, variables))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir_graph = join(data_dir, f'logdir/func/{stamp}')
    writer_graph = create_file_writer(logdir_graph)

    trace_on(graph=True)
    _ = update_nn(rangen)

    with writer_graph.as_default():
        trace_export(
            name="update_nn", step=0,
        )


def train_nn_profiler(
        simulation,
        optimizer: Optimizer,
        data_dir: str,
        network: Model,
        iterations: int = 10,
        gradient_accumulation: int = 1,
        xla_compile: bool = True,
        rangen=Generator.from_seed(0xdeadd0d0),
):
    """
    Saves profiling information regarding the
    memory and time consumption of each operation
    in the simulation that can be visualized
    on Tensorboard.

    Parameters
    ----------
    simulation: :py:obj:`~.Simulation`
        Contains the complete description of the
        estimation task, and how the various components
        of it (particle filter, probe, ...) interact to
        produce the estimator and the loss.
    optimizer: :py:obj:`Optimizer`
        This object applies the gradient to the trainable
        variables of the control strategy and realized therefore
        the update of the gradient descent step.
        The default optimizer to be chosen
        is `Adam` with the variable learning
        rate described in the
        class :py:obj:`~.InverseSqrtDecay`.
    data_dir: str
        Directory in which the report on the
        profiling data is saved.
    network: :py:obj:`Model`, optional
        Neural network model used in the `simulation`
        object to parametrize the control strategy.
        In `simulation` it is wrapped inside the
        `control_strategy` attribute and there the
        :py:obj:`~.Simulation` object has no direct access
        to its trainable variables. If no neural network
        is used in the estimation, for example
        because a static strategy is being trained,
        this parameter can be omitted.
    iterations: int = 10
        Number of update steps in the training of the
        strategy. It is the number of complete executions
        of a batch of estimation tasks for training the controls.
    gradient_accumulation: int = 1
        This controls the "effective" batchsize of the
        training. In the training loop for each
        update of the variables the method
        :py:meth:`~.Simulation.execute` runs
        a number `gradient_accumulation` of times
        and the gradients computed from each run
        are averaged before the update step
        is performed.
        In this way, given `bs` the true batchsize of the
        `simulation` object, the "effective" batchsize
        of the training is
        `gradient_accumulation*bs`, which is the number of
        independent simulations that contribute to a single
        update step of the training variables. This feature
        can be used to increase the batchsize, when the
        memory on the machine is limited,
        since the maximum amount of memory required
        by the training is determine by
        `bs` and is independent on
        `gradient_accumulation`.
    xla_compile: bool = True
        Just-in-time XLA (Accelerated Linear Algebra)
        compilation for the strategy training.
        It should reduce the memory and training
        time on a GPU.


        **Achtung!** Not all the Tensorflow
        operations can be compiled with XLA.

        **Achtung!** It might not perform well on a CPU.
    rangen: Generator = Generator.from_seed(0xdeadd0d0)
        Random number generator from the module
        :py:mod:`tensorflow.random`.
    """
    variables = network.trainable_variables

    @function(jit_compile=xla_compile)
    def single_iteration(rangen):
        with GradientTape() as tape:
            loss_diff, loss = simulation.execute(rangen)
        grads = tape.gradient(loss_diff, variables)
        return grads, loss

    def update_nn(rangen):
        acc_loss = zeros((1, ), simulation.simpars.prec)
        acc_gradient = [zeros_like(this_var)
                        for this_var in variables]
        for _ in range(gradient_accumulation):
            print("Exec!")
            grads, loss = single_iteration(rangen)
            acc_loss += loss
            acc_gradient = [(acc_grad+grad)
                            for acc_grad, grad in
                            zip(acc_gradient, grads)]
        acc_gradient = [this_grad /
                        gradient_accumulation for
                        this_grad in acc_gradient]
        optimizer.apply_gradients(zip(acc_gradient, variables))
        acc_loss /= gradient_accumulation
        return acc_loss

    # The tracing is outside of the profiler...
    _ = update_nn(rangen)

    start(join(data_dir, 'logdir'))
    for j in progressbar(range(iterations)):
        with Trace('TraceContext', step_num=j, _r=1):
            _ = update_nn(rangen)
    stop(save=True)


def store_input_control(
    simulation,
    data_dir: str,
    iterations: int,
    xla_compile: bool = True,
    rangen=Generator.from_seed(0xdeadd0d0),
):
    """This routine collects data on the behavior of
    of the `control_strategy` attribute of the `simulation`
    parameter.

    Stores in a csv file all the `input_strategy`
    and the `controls`
    tensors, which are respectively the input and the
    output of the callable attribute `control_strategy`
    of `simulation`, which is a :py:obj:`~.Simulation`
    object. The inputs are returned by the
    :py:meth:`~.StatefulSimulation.generate_input` method.
    In the produced csv file, whose name ends in "_ext.csv"
    the `input_strategy` and `controls` tensors for
    each estimation in the batch, for a number
    `iterations` of executions of the
    :py:obj:`~.Simulation.execute` method.

    The first column of the produced csv file
    is `Estimation` and it is the index of the
    independent estimation to which the line is
    referring. It is incremented both across the
    batchsize and the iterations, so that it goes
    from `0` to `bs*iterations-1`. The next `d` columns
    are the true values of the unknown parameters,
    where `d` is the attribute of
    `simulation.ps`, i.e. the
    :py:obj:`~.ParticleFilter` attribute
    of `simulation`. The next `simulation.input_size`
    columns are the entries of
    `input_strategy` produced
    by :py:meth:`~.StatefulSimulation.generate_input`. The
    last `simulation.phys_model.controls_size` columns
    contains the controls produced by
    `simulation.control_strategy`.
    The name of the columns are taken
    from the name attributes of the corresponding
    objects and from `simulations.input_name`.

    **Achtung!** The lines with null
    consumed resources are not reported
    in the csv file.

    Parameters
    ----------
    simulation: :py:obj:`~.Simulation`
        Contains the complete description of the
        estimation task, and how the various components
        of it (particle filter, probe system, and control)
        interact to produce the estimator and the loss.
    data_dir: str
        Directory in which the csv file containing
        the
    iterations: int
        Number of update steps in the training of the
        strategy. It is the number of complete executions
        of a batch of estimation tasks for training the controls.
    xla_compile: bool = True
        Just-in-time XLA (Accelerated Linear Algebra)
        compilation for the simulation.
        It should reduce the used memory and time
        on a GPU.

        **Achtung!** Not all the Tensorflow
        operations can be compiled with XLA.

        **Achtung!** It might not perform well on a CPU.
    rangen: Generator = Generator.from_seed(0xdeadd0d0)
        Random number generator from the module
        :py:mod:`tensorflow.random`.
    """
    @function(jit_compile=xla_compile)
    def simulate_nn(rangen):
        true_values, history_input, history_control, \
            history_resources, _ = simulation.execute(
                rangen, deploy=True,
            )
        return true_values, history_input, \
            history_control, history_resources

    data = {"index": [], "values": [], "input": [],
            "control": [], "resources": []}
    for j in progressbar(range(iterations)):
        true_values, history_input, history_control, \
            history_resources = simulate_nn(
                rangen,
            )
        index_part = reshape(
            tfrange(j*simulation.bs, (j+1)*simulation.bs),
            (1, simulation.bs),
        )
        index_part = reshape(repeat(
            index_part, repeats=[simulation.simpars.num_steps],
            axis=0,
        ), (simulation.bs*simulation.simpars.num_steps, 1)
        )
        true_values = reshape(broadcast_to(
            transpose(true_values, (1, 0, 2)),
            (simulation.simpars.num_steps, simulation.bs,
             simulation.phys_model.d),),
            (simulation.bs*simulation.simpars.num_steps,
             simulation.phys_model.d)
        )
        history_input = reshape(
            history_input,
            (simulation.bs*simulation.simpars.num_steps,
             simulation.input_size)
        )
        history_control = reshape(
            history_control,
            (simulation.bs*simulation.simpars.num_steps,
                simulation.phys_model.controls_size)
        )
        history_resources = reshape(
            history_resources,
            (simulation.bs*simulation.simpars.num_steps, )
        )
        data["index"].append(index_part.numpy())
        data["values"].append(true_values.numpy())
        data["input"].append(history_input.numpy())
        data["control"].append(history_control.numpy())
        data["resources"].append(history_resources.numpy())

    index_array = concatenate(data["index"], axis=0)
    values_array = concatenate(data["values"], axis=0)
    input_array = concatenate(data["input"], axis=0)
    control_array = concatenate(data["control"], axis=0)
    resources_array = concatenate(data["resources"], axis=0)

    data_array = concatenate(
        [index_array, values_array, input_array, control_array],
        axis=1,
    )

    columns = ['Estimation', ]
    for par in simulation.phys_model.params:
        columns.append(par.name)
    columns += simulation.input_name
    for contr in simulation.phys_model.controls:
        columns.append(contr.name)

    # Loading of the true values, the inputs and the controls
    # in a pandas data frame
    data_df = DataFrame(data=data_array, columns=columns)
    # This is needed to remove the blank lines in data_tf
    resources_df = DataFrame(
        data=resources_array, columns=['Resources', ],
    )
    data_df = data_df[resources_df['Resources'] != 0]
    data_df.to_csv(
        join(data_dir, str(simulation))+'_ext.csv',
        index=False,
        float_format='%.4e',
    )


class FitSpecifics(TypedDict):
    """This dictionary specifies the hyperparameters of the
    precision fit operated by the
    function :py:func:`~.utils.performance_evaluation`."""
    num_points: int
    """After the fit the neural network
    representing the relation between the resources
    and the average precision is evaluated
    on `num_points` resources values
    equally spaced in the
    interval [0, `max_resources`], with
    `max_resources` being the attribute
    of :py:obj:`~.SimulationParameters`."""
    batchsize: int
    """Batchsize of the training of the
    neural network to fit the
    Precision/Resources relation. The
    data cloud is divided in minibatches
    and each of them is used for a sigle
    iteration of the training loop."""
    epochs: int
    """The number of trainings on the
    same data."""
    direct_func: Callable
    r"""Callable object that takes in input
    the precision and the consumed resources
    and outputs a values :math:`x`
    that is of order one. In symbols

    .. math::
        f(\text{Resources}, \text{Precision})
        = x \sim \mathcal{O}(1) \; .

    This requires having some knowledge of the
    expected precision given the resources, which
    could be for example a Cramér-Rao bound
    on the precision."""
    inverse_func: Callable
    r"""Inverse of the function defined
    by `direct_func`, that is

    .. math::
        g(\text{Resources}, x) = \text{Precision} \; ."""


def performance_evaluation(
    simulation,
    iterations: int,
    data_dir: str,
    xla_compile: bool = True,
    precision_fit: Optional[FitSpecifics] = None,
    delta_resources: Optional[float] = None,
    y_label: str = 'Precision',
    rangen=Generator.from_seed(0xdeadd0d0),
):
    """Computes the relation
    Precision/Resources obtained
    from the averaged data of a number `iterations`
    of runs of the :py:meth:`~.Simulation.execute` method.
    Such relation is saved on
    disk as as a list of points
    (`Resources`, `Precision`) in a csv file, whose
    name terminate with "_eval".

    The "resources" in the estimation task are
    defined in the method
    :py:meth:`~.PhysicalModel.count_resources`.
    Different executions of the task can only be compared
    when they refer to the same amount of
    resources.

    Since the estimation is a stochastic
    process this routine will produce the
    expected precision as a function
    of the consumed resources.

    Parameters
    ----------
    simulation: :py:obj:`~.Simulation`
        Contains the complete description of the
        estimation task, and how the various components
        of it (particle filter, probe system, and control)
        interact to produce the estimator and the loss.
    iterations: int
        Number of update steps in the training of the
        strategy. It is the number of complete executions
        of a batch of estimation tasks.
    data_dir: str
        Directory containing the csv file
        with the evaluated relation Precision/Resources.
    xla_compile: bool = True
        Just-in-time XLA (Accelerated Linear Algebra)
        compilation for the simulation.
        It should reduce the used memory and time
        on a GPU.

        **Achtung!** Not all the Tensorflow
        operations can be compiled with XLA.

        **Achtung!** It might not perform well on a CPU.
    precision_fit: :py:obj:`~.FitSpecifics`, optional
        The simulations produce a
        cloud of points (`Resources`, `Precision`)
        from which a simple relation between the
        expected precision and the resources
        must be obtained.
        One possibility is to fit this cloud of points
        with a neural network.

        .. image:: ../docs/_static/precision_nn.png
            :width: 600
            :alt: precision_nn

        When the dictionary
        `parameter_fit` is passed to the call
        of this routine, then the cloud of
        points is fitted, and the fitted curve is
        saved in the csv file.
    delta_resources: float, optional
        The simulations produce a
        cloud of points (`Resources`, `Precision`),
        and all the points inside a window of
        resources of size `delta_resources` are
        averaged in order to produce a single point
        containing the expected precision, which
        is practically the barycenter of the
        said window.

        .. image:: ../docs/_static/precision_average.png
            :width: 600
            :alt: precision_average

        The barycenters of all the consecutive windows are
        saved in the csv file produced by this routine.
    y_label: str = 'Precision'
        Label of the y-component of the data
        saved in the csv file. This column
        represent the precision of the estimation
        and more specifically it could be the
        the mean square error, the error probability,
        etc.
    rangen: Generator = Generator.from_seed(0xdeadd0d0)
        Random number generator from the module
        :py:mod:`tensorflow.random`.

    Notes
    -----
    The parameters `precision_fit` and `delta_resources`
    are not mutually exclusive. When both are passed
    to the call of the routine, first the cloud of
    (`Resources`, `Precision`) points is averaged
    according to the value of `delta_resources` and
    then the fewer points resulting from the
    averaging are fitted with a neural network.
    """
    @function(jit_compile=xla_compile)
    def simulate_nn(rangen):
        _, _, _, history_resources, history_precision = \
            simulation.execute(
                rangen, deploy=True,
            )
        return history_resources, history_precision

    precision_list = []
    resources_list = []
    for _ in progressbar(range(iterations)):
        history_resources, history_precision = simulate_nn(
            rangen,
        )
        history_resources = reshape(
            history_resources,
            (simulation.bs*simulation.simpars.num_steps, )
        )
        history_precision = reshape(
            history_precision,
            (simulation.bs*simulation.simpars.num_steps, )
        )
        resources_list.append(history_resources.numpy())
        precision_list.append(history_precision.numpy())

    resources_array = concatenate(resources_list, axis=0)
    precision_array = concatenate(precision_list, axis=0)
    # Loading of the resources and the precisions in a Numpy array
    prec_df = DataFrame(
        {'Resources': resources_array, y_label: precision_array},
    )

    # The rows with zero resources are removed
    prec_df = prec_df[prec_df['Resources'] != 0]
    if delta_resources:

        bin_index_floor = npfloor(
            prec_df['Resources'].values/delta_resources,
        )
        # The number of resources associated to each bin
        # is the mean number of resources in that bin
        prec_df = prec_df.groupby(bin_index_floor).mean()
    if precision_fit:
        # Parametrization of the function that returns the precision
        # as a function of the resources
        prec_df = prec_df.sample(frac=1).reset_index(drop=True)

        # Create model for fitting the Precision/Resources relation
        model = standard_model()
        model.compile(
            optimizer=Adam(learning_rate=InverseSqrtDecay(1e-2)),
            loss='mean_squared_error',
        )
        max_res = simulation.simpars.max_resources

        model.fit(
            prec_df['Resources']/max_res,
            precision_fit['direct_func'](
                prec_df['Resources'], prec_df[y_label]),
            batch_size=precision_fit['batchsize'],
            epochs=precision_fit['epochs'],
            verbose=1,
        )
        # Generate the prediction for the precision
        x_axis = linspace(0, max_res, precision_fit['num_points']+1)[1:]
        y_axis = precision_fit['inverse_func'](
            x_axis, model(x_axis/max_res)[:, 0])
        prec_df = DataFrame(
            {'Resources': x_axis, y_label: y_axis},
        )
    prec_df.to_csv(
        join(data_dir, str(simulation))+'_eval.csv',
        index=False,
        float_format='%.4e',
    )


def standard_model(
    input_size: int = 1,
    controls_size: int = 1,
    neurons_per_layer: int = 64,
    num_mid_layers: int = 5,
    prec: str = "float64",
    normalize_activation: bool = True,
    sigma_input: float = 0.33,
    last_layer_activation: str = "tanh",
) -> Model:
    """Returns a dense neural network to fit the optimal
    control strategy.

    Parameters
    ----------
    input_size: int = 1
        Number of scalars given in input to the neural
        network.
    controls_size: int = 1
        Number of scalars outputted by the neural network.
        It corresponds to the number of controls in the
        estimation task.
    neurons_per_layer: int = 64
        Number of units (neurons) in each intermediate
        layer of the neural network. The intermediate
        layers of the neural network
        are all identical.
    num_mid_layers: int = 5
        Number of layers between the input and the output
        of the neural network. Should be greater then one.
    prec: str = "float64"
        Type of the weights and biases of the neural network.
    normalize_activation: bool = True
        If this flag is active the activation function (`tanh`)
        is normalized to preserve the variance
        of the input after each layer of neurons. This
        should speed up the convergence.
    sigma_input: float = 0.33
        Approximate variance of the input of the neural
        network. The default value is the variance of
        a uniform distribution in `[-1, +1]`.
    last_layer_activation: str = "tanh"
        Activation function of the last neural
        network layer.
    Returns
    -------
    Model:
        Sequential model made up of
        `num_mid_layers` intermediate layers, each with
        `neurons_per_layer` units. The initialization
        of the weight and biases is done with
        *glorot uniform* and the activation function
        is `tanh`, possibly normalized to reduce
        the saturation.

        .. image:: ../docs/_static/nn_definition.png
            :width: 500
            :alt: definition_nn
    """
    # There should always be at least 2 intermediate layers.
    if num_mid_layers <= 1:
        raise ValueError("num_mid_layers must be > 1.")

    if normalize_activation:
        sigma_z_input = npsqrt(
            2*input_size/(input_size+neurons_per_layer))*sigma_input
        sigma_z_mid = sigma_input
        sigma_z_output = npsqrt(2*neurons_per_layer /
                                (controls_size+neurons_per_layer))*sigma_input

        C_input = integrate.quad(
            lambda x: nptanh(x)**2*npgauss_pdf(x, sigma_z_input), -inf, inf,
        )[0]
        C_mid = integrate.quad(
            lambda x: nptanh(x)**2*npgauss_pdf(x, sigma_z_mid), -inf, inf,
        )[0]
        C_output = integrate.quad(
            lambda x: nptanh(x)**2*npgauss_pdf(x, sigma_z_output), -inf, inf)[0]

        input_norm, mid_norm, output_norm = sigma_input/npsqrt(C_input), \
            sigma_input/npsqrt(C_mid), sigma_input/npsqrt(C_output)

    else:
        input_norm, mid_norm, output_norm = 1.0, 1.0, 1.0

    layer_list = [
        Dense(neurons_per_layer,
              activation=lambda x: input_norm*tanh(x),
              dtype=prec, input_shape=(input_size, )),
    ]

    for _ in range(num_mid_layers-2):
        layer_list.append(
            Dense(neurons_per_layer,
                  activation=lambda x: mid_norm*tanh(x),
                  dtype=prec),
        )

    layer_list.append(
        Dense(neurons_per_layer,
              activation=lambda x: output_norm*tanh(x),
              dtype=prec),
    )

    layer_list.append(
        Dense(controls_size,
              activation=last_layer_activation,
              dtype=prec)
    )

    return Sequential(layer_list)


def npgauss_pdf(x, dev):
    """Logarithm of a batch of
    1D-Gaussian probability densities (compatible with
    Numpy)"""
    return 1/(npsqrt(2*pi)*dev)*npexp(-0.5*x**2/dev**2)


def loggauss(x, mean, dev):
    r"""Logarithm of a batch of
    1D-Gaussian probability densities.

    Parameters
    ----------
    x: Tensor
        Values extracted from the Gaussian distributions.
        Must have the same type and size of `mean` and `dev`.
    mean: Tensor
        Means of the Gaussian distributions.
        Must have the same type and size of `x` and `dev`.
    dev:
        Standard deviation of the Gaussian distributions.
        Must have the same type and size of `mean` and `x`.

    Returns
    -------
    Tensor:
        Logarithm of the probability densities for
        extracting the entries of `x` from the Gaussian
        distributions defined by `mean` and `dev`. It has
        the same shape and type of `x`, `mean`, and `dev`.

        Calling :math:`x`, :math:`\mu`, and :math:`\sigma`
        respectively an entry of the tensor `x`, `mean`, and `dev`,
        the corresponding entry of the returned tensor is

        .. math::
            -\log \left( \sqrt{2 \pi} \sigma \right) -
            \frac{(x-\mu)^2}{2 \sigma^2} \; .

    """
    return -log(npsqrt(2*pi)*dev)+(-0.5*((x-mean)/dev)**2)


def logpoisson(mean, k):
    r"""Logarithm of the probability densities
    of a batch of Poissonian distributions.

    Parameters
    ----------
    mean: Tensor
        Mean values defining the Poissonian distributions.
        Must have the same type and shape of `k`.
    k: Tensor
        Observed outcomes of the sampling from the
        Poissonian distributions.
        Must have the same type and shape of `k`.

    Returns
    -------
    Tensor:
        `Tensor` having the same type and shape
        of `mean` and `k`, whose entries are defined
        by

        .. math::
            k \log (\mu) - \mu - \log (k !) \; ,

        where :math:`k` and :math:`\mu` are respectively
        the entries of `k` and `mean`.
    """
    return k*log(mean) - mean - lgamma(k + 1)


def sqrt_hmatrix(matrix: Tensor) -> Tensor:
    """Square root of the absolute value of
    a symmetric (hermitian) matrix.

    The default matrix square root algorithm
    implemented in Tensorflow [7]_
    doesn't work for matrices with very small entries,
    this implementation does, and must be
    always preferred.

    Parameters
    ----------
    matrix: Tensor
        Batch of symmetric (hermitian) square matrices.

    Returns
    -------
    Tensor:
        Matrix square root of `matrix`.

    Examples
    --------
    ``A = constant([[1e-16, 1e-15], [1e-15, 1e-16]],
    dtype="float64", )``

    ``print(tf.sqrtm(A))``

    Output:

    ``[[-nan -nan], [-nan -nan]]``

    While ``print(sqrt_hmatrix(A))`` outputs

    ``[[1.58312395e-09, 3.15831240e-08],
    [3.15831240e-08, 1.58312395e-09]]``

    .. [7] N. J. Higham, "Computing real square
           roots of a real matrix", Linear
           Algebra Appl., 1987.
    """
    s, _, v = svd(matrix)
    return matmul(
        matmul(v, diag(tfsqrt(abs(s)))), v, adjoint_b=True,
        name="square_root",
    )
