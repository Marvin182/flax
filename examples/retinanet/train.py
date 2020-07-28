from input_pipeline import prepare_data
from model import create_retinanet
from flax.training import checkpoints
from jax import numpy as jnp

import flax
import jax
import math


def cos_decay(global_step, decay_steps, alpha=0.):
  """Returns the value of the cosine decay for a particular step.

  Args:
    global_step: the current global step
    decay_steps: after this many steps, the decay remains constant at alpha
    alpha: the alpha constant in the decay formula

  Returns:
    The value of the decay
  """
  # Note that the phase ratio is a decimal number between [0, 1]
  phase_ratio = jnp.maximum(0., global_step / decay_steps)
  cosine_decay = .5 * (1 + jnp.cos(math.pi * phase_ratio))
  return (1 - alpha) * cosine_decay + alpha


def create_decay_fn(learning_rate, total_epochs, steps_per_epoch,
                    warmup_epochs=5, alpha=0.):
  """Creates a cosine decay function.

  More specifically, produces a function which takes in a single parameter,
  the current step in the training process, and yields its respective learning
  rate.

  Args:
    learning_rate: the base learning rate
    total_epochs: the total number of epochs for the training process
    steps_per_epoch: the number of batches per epoch
    warmup_epochs: the number of warmup epochs
    alpha: the constant alpha in the `cos_decay` function

  Returns:
    A single parameter function, which takes in a single parameter, the global
    step in the training process, and yields the current learning rate
  """
  def decay_fn(global_step):
    epoch = global_step / steps_per_epoch
    lr = learning_rate * cos_decay(epoch - warmup_epochs,
                                   total_epochs - warmup_epochs,
                                   alpha)

    # Linearly increase the learning rate during warmup
    return lr * jnp.minimum(1., epoch / warmup_epochs)

  return decay_fn


def create_model(rng, classes=1000, shape=(224, 224, 3), dtype=jnp.float32):
  """Creates a RetinaNet model.

  Args:
    rng: the Jax PRNG, which is used to instantiate the model weights
    classes: the number of classes in the object detection task
    shape: the shape of the image inputs, with the format HWC.
    dtype: the data type of the model

  Returns:
    The RetinaNet instance, and its state object
  """
  # The number of classes is increased by 1 since we add the background
  partial_module = create_retinanet(50, classes=classes + 1, dtype=dtype)

  # Since the BatchNorm has state, we'll need to use stateful here
  with flax.nn.stateful() as init_state:
    _, params = partial_module.init(rng, jnp.zeros((1,) + shape))

  return flax.nn.Model(partial_module, params), init_state


def create_optimizer(model, optimizer="momentum", **optimizer_params):
  """Create either a Momentum or an Adam optimizer.

  Args:
    model: a flax.nn.Model object, which encapsulates the neural network
    optimizer: this selects the optimizer, either `momentum` or `adam`
    **optimizer_params: extra kwargs for the created optimizer

  Returns:
    An optimizer which wraps the model
  """
  assert optimizer in ["momentum", "adam"], "The optimizer is not supported"

  if optimizer == "adam":
    optimizer_def = flax.optim.Adam(**optimizer_params)
  else:
    optimizer_def = flax.optim.Momentum(**optimizer_params)

  return optimizer_def.create(model)


@jax.vmap
def cross_entropy_loss(logits, label):
  """Implements the Cross Entropy (Log-loss).

  Args:
    logits: the logit array
    label: the ground truth

  Returns:
    The Log-loss
  """
  return -jnp.log(logits[label])


def compute_metrics(pred, labels):
  """Returns the accuracy and the cross entropy.

  Args:
    pred: the logits
    labels: the ground truths

  Returns:
    A dictionary containins the metrics
  """
  cross_entropy = jnp.mean(cross_entropy_loss(pred, labels))
  accuracy = jnp.mean(jnp.argmax(pred, axis=1) == labels)
  metrics = {
    "accuracy": accuracy,
    "cross_entropy": cross_entropy
  }
  return jax.lax.pmean(metrics, "device")


def eval(data, meta_state):
  """Evaluates the model.

  The evaluation is done against the Log-loss and the Accuracy
  metrics. Note, the model should be stateful.

  Args:
    data: the test data
    model: an instance of the CheckpointState class

  Returns:
    The accuracy and the Log-loss aggregated across multiple workers.
  """
  with flax.nn.stateful(meta_state.model_state, mutable=False):
    pred = meta_state.optimizer.target(data['image'], train=False)
  return compute_metrics(pred, data['label'])

def aggregate_evals(eval_array):
  vals = jnp.array(list(map(lambda x: list(x.values()), eval_array)))
  return dict(zip(eval_array[0].keys(), jnp.mean(vals, axis=0)))


@flax.struct.dataclass
class CheckpointState:
  """A dataclass which stores the state of the training loop.
  """
  # The state variable of the model
  model_state : flax.nn.Collection
  # The optimizer, which also holds the model
  optimizer : flax.optim.Optimizer
  # The global state of this checkpoint
  step : int = -1


def checkpoint_state(meta_state : CheckpointState, checkpoint_step : int,
                     checkpoint_dir="checkpoints"):
  """
  Checkpoints the training state.

  Args:
    meta_state: a `CheckpointState` object, which contains the state of
      the current training step
    checkpoint_step: a checkpoint step, used for versioning the checkpoint
    checkpoint_dir: the directory where the checkpoint is stored
  """
  if jax.host_id() == 0:
    meta_state = jax.device_get(jax.tree_map(lambda x: x[0], meta_state))
    checkpoints.save_checkpoint(checkpoint_dir, meta_state, checkpoint_step)


def restore_checkpoint(meta_state, checkpoint_dir="checkpoints"):
  """Restores the latest checkpoint.

  More specifically, either return the latest checkpoint from the
  `checkpoint_dir` or returns the `meta_state` object, if no such checkpoint
  exists.

  Args:
    meta_state: a `CheckpointState` object, used as last resort if no checkpoint
      does exist
    checkpoint_dir: the directory where the checkpoints are searched for

  Returns:
    Either the latest checkpoint, if it exists, or the `meta_state` object
  """
  return checkpoints.restore_checkpoint(checkpoint_dir, meta_state)


def sync_model_state(meta_state):
  """Synchronizes the model_state across devices.

  Args:
    meta_state: a `CheckpointState` object to be used towards synchronization

  Returns:
    A new CheckpointState object with an updated `model_state` field.
  """
  mean = jax.pmap(lambda x: jax.lax.pmean(x, 'axis'), 'axis')
  return meta_state.replace(model_state=mean(meta_state.model_state))


def create_step_fn(lr_function):
  """Creates a step function with a custom LR scheduler.

  Args:
    lr_function: function which takes in a single argument, the current step
      in the training process, and yields the learning rate

  Returns:
    A function responsible with carrying out a training step. The function takes
    in two arguments: the batch, and a `CheckpointState` object, which
    stores the current training state.
  """
  def take_step(data, meta_state: CheckpointState):
    """Trains the model on a batch and returns the updated model.

    Args:
      data: the batch on which the pass is performed
      meta_state: a `CheckpointState` object, which holds the current model

    Returns:
      The updated model as a `CheckpointState` object and the batch's loss
    """
    def _loss_fn(model, state):
      with flax.nn.stateful(state) as new_state:
        pred = model(data['image'])
      loss = jnp.mean(cross_entropy_loss(pred, data['label']))

      # Penalize large model weights via a decayed l2 norm
      weight_decay = 0.0001 * 0.5
      weights = jax.tree_leaves(model.params)
      weight_loss = weight_decay * sum([jnp.sum(x ** 2)
                                        for x in weights if x.ndim > 1])

      return loss + weight_loss, (new_state, pred)

    # flax.struct.dataclass is immutable, so unwrap it
    step = meta_state.step + 1

    # Compute the gradients
    aux, grads = jax.value_and_grad(_loss_fn, has_aux=True)(
      meta_state.optimizer.target, meta_state.model_state)
    new_model_state, pred = aux[1]
    metrics = compute_metrics(pred, data['label'])

    # Synchronize device model across devices
    grads = jax.lax.pmean(grads, "device")

    # Apply the gradients to the model
    updated_optimizer = meta_state.optimizer.apply_gradient(
      grads, learning_rate=lr_function(step))

    # Update the meta_state
    meta_state = meta_state.replace(step=step, model_state=new_model_state,
                                    optimizer=updated_optimizer)

    return meta_state, metrics

  return take_step


def train_retinanet_model(data, learning_rate=0.1, batch_size=64, epochs=5,
                       warmup_epochs=3, try_restore=True, half_precision=False,
                       checkpoint_period=1):
  """This method trains a RetinaNet instance.

  Args:
    data: a data dictionary, which takes on the following form:
      ```
      {
        "shape": (H, W, C),
        "train": {
            "count": <count>,
            "data": <train_data>
        },
        "test": {
            "count": <count>,
            "data": <test_data>
        }
      }
      ```
    learning_rate: the base learning rate for the training process
    batch_size: the batch size
    epochs: the number of epochs for training (this number should include the
      `warmup_epochs`)
    warmup_epochs: the number of warmup epochs
    try_restore: a boolean variable which indicates whether the latest
      checkpoint should be restored (in case it exists)
    half_precision: indicates whether half-precision floating point types should
      be used
    checkpoint_period: the frequency in epochs for checkpointing the model

  Returns:
    A `CheckpointState` object, which contains the trained model
  """
  assert warmup_epochs < epochs, "epochs must be greater than warmup_epochs"
  assert 0 == batch_size % jax.local_device_count(), "batch_size must be " \
                                                     "divisible by the number" \
                                                     " of local devices"
  # Set up the data pipeline
  train_data, val_data = prepare_data(data, batch_size=batch_size)
  train_iter = iter(train_data)

  # Crate the training parameters
  steps_per_epoch = int(math.ceil(data['train']['count'] / batch_size))
  steps_per_eval = int(math.ceil(data['test']['count'] / batch_size))
  total_step_count = epochs * steps_per_epoch

  # Set the correct dtype
  dtype = jnp.float32
  if half_precision:
    if jax.local_devices()[0].platform == 'tpu':
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float16

  # Create the training entities, and replicate the state
  model, model_state = create_model(jax.random.PRNGKey(0), classes=10,
                                    shape=data['shape'], dtype=dtype)
  optimizer = create_optimizer(model, nesterov=True)
  meta_state = CheckpointState(optimizer=optimizer, model_state=model_state)
  del model, model_state, optimizer  # Remove duplicate data

  # Try to restore the state of a previous run
  start_epoch = 0
  if try_restore:
    meta_state = restore_checkpoint(meta_state)
    start_epoch = max(meta_state.step, 0) // steps_per_epoch

  meta_state = flax.jax_utils.replicate(meta_state)  # Replicate the state

  # Prepare the LR scheduler
  # Note: increasing the learning rate eventually leads to inf or nan log loss
  learning_rate *= batch_size / 256
  learning_rate_fn = create_decay_fn(learning_rate, epochs, steps_per_epoch,
                                     warmup_epochs=warmup_epochs)

  # Prepare the training loop for distributed runs
  step_fn = create_step_fn(learning_rate_fn)
  p_step_fn = jax.pmap(step_fn, axis_name="device")
  p_eval_fn = jax.pmap(eval, axis_name="device")

  # Run the training loop
  for epoch in range(start_epoch, epochs):
    step_offset = epoch * steps_per_epoch

    # Run an epoch
    for step in range(steps_per_epoch):
      # Use ._numpy() to avoid copy.
      batch = jax.tree_map(lambda x: x._numpy(), next(train_iter))  # pylint: disable=protected-access
      meta_state, metrics = p_step_fn(batch, meta_state)
      if step % 10 == 0:
        single_metric = jax.tree_map(lambda x: x[0], metrics)
        print("(Train Step #{}) Log Loss: {}".format(
          step + step_offset, single_metric))

    # Sync up the model_state after every epoch
    meta_state = sync_model_state(meta_state)

    # Evaluate the model
    eval_results = []
    val_iter = iter(val_data)
    for _ in range(steps_per_eval):
      # Use ._numpy() to avoid copy.
      batch = jax.tree_map(lambda x: x._numpy(), next(val_iter))  # pylint: disable=protected-access
      res = p_eval_fn(batch, meta_state)
      eval_results.append(jax.tree_map(lambda x: x[0], res))
    eval_results = aggregate_evals(eval_results)
    print("(Epoch #{}) Evaluation results:\n".format(epoch), eval_results)

    if epoch % checkpoint_period == 0:
      checkpoint_state(meta_state, epoch)

  return meta_state
