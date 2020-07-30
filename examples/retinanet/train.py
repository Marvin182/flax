import math

from absl import logging
import flax
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

import input_pipeline
from model import create_retinanet


def create_scheduled_decay_fn(learning_rate: float, training_steps: int, 
                              warmup_steps: int, division_factor:float = 10.0,
                              division_schedule: list = None):
  """Creates a scheduled division based learning rate decay function.

  More specifically, produces a function which takes in a single parameter,
  the current step in the training process, and yields its respective learning
  rate based on a scheduled division based decay, which divides the previous
  learning rate by `division_factor` at the steps specified in the 
  `division_schedule`.

  Args:
    learning_rate: the base learning rate
    training_steps: the number of training steps 
    warmup_steps: the number of warmup steps 
    division_factor: the factor by which the learning rate is divided at the 
      training steps indicated by `division_schedule`
    division_schedule: a list which indicates the iterations at which the 
      learning rate should be divided by the `division_factor`. Note that
      the values in 

  Returns:
    A function, which takes in a single parameter, the global step in 
    the training process, and yields the current learning rate.
  """
  assert training_steps > 0, "training_steps must be greater than 0"
  assert warmup_steps >= 0, "total_steps must be greater than 0"
  assert division_factor > .0, "division_factor must be positive"

  # Get the default values for learning rate decay
  if division_schedule is None:
    division_schedule = [int(training_steps * .66), int(training_steps * .88)]

  # Adjust the schedule to not consider the warmup steps
  division_schedule = jnp.sort(jnp.unique(division_schedule)) + warmup_steps
  
  # Define the decay function
  def decay_fn(step):
    lr = lr / division_factor ** jnp.argmax(division_schedule > step)

    # Linearly increase the learning rate during warmup
    return lr * jnp.minimum(1., epoch / warmup_epochs)

  return decay_fn


def create_model(rng, depth=50, classes=1000, shape=(224, 224, 3)):
  """Creates a RetinaNet model.

  Args:
    rng: the Jax PRNG, which is used to instantiate the model weights
    depth: the depth of the basckbone network
    classes: the number of classes in the object detection task
    shape: the shape of the image inputs, with the format (N, H, W, C)

  Returns:
    The RetinaNet instance, and its state object
  """
  # The number of classes is increased by 1 since we add the background
  partial_module = create_retinanet(depth, classes=classes + 1)

  # Since the BatchNorm has state, we'll need to use stateful here
  with flax.nn.stateful() as init_state:
    _, params = partial_module.init_by_shape(rng, [(shape, jnp.float32)])

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


def train_and_evaluate(config, workdir: str):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  tf.io.gfile.makedirs(workdir)

  # Deterministic training, see go/deterministic training.
  rng = jax.random.PRNGKey(config.seed)

  if config.batch_size % jax.device_count():
    raise ValueError(f"Batch_size ({config.batch_size}) must be divisible by "
        f"the number of devices {jax.device_count}).")

  # Set up the data pipeline
  dataset_builder = tfds.builder("coco/2014")
  num_classes = dataset_builder.info.features["objects"]["label"].num_classes
  rng, data_rng = jax.random.split(rng)
  data = input_pipeline.read_data(data_rng)
  train_data, val_data = input_pipeline.prepare_data(data, per_device_batch_size=config.batch_size // jax.device_count())
  logging.info("Training data shapes: %s", train_data.element_spec)
  input_shape = list(train_data.element_spec["image"].shape)[1:]
  train_iter = iter(train_data)

  # Crate the training parameters
  steps_per_epoch = int(math.ceil(data['train']['count'] / config.batch_size))
  steps_per_eval = int(math.ceil(data['test']['count'] / config.batch_size))

  # Create the training entities, and replicate the state
  rng, model_rng = jax.random.split(rng)
  model, model_state = create_model(model_rng, shape=input_shape,
    classes=num_classes, depth=config.depth)
  optimizer = create_optimizer(model,  beta=0.9, weight_decay=0.0001)
  meta_state = CheckpointState(optimizer=optimizer, model_state=model_state, step=0)
  del model, model_state, optimizer  # Remove duplicate data

  # Try to restore the state of a previous run
  # meta_state = restore_checkpoint(meta_state) if try_restore else meta_state

  initial_step = int(meta_state.step) + 1

  # Replicate the state across devices
  meta_state = flax.jax_utils.replicate(meta_state)

  # Prepare the LR scheduler
  learning_rate = config.learning_rate * config.batch_size / 256
  learning_rate_fn = create_scheduled_decay_fn(
      learning_rate, config.num_train_steps, config.warmup_steps)

  # Prepare the training loop for distributed runs
  step_fn = create_step_fn(learning_rate_fn)
  p_step_fn = jax.pmap(step_fn, axis_name="device")
  p_eval_fn = jax.pmap(eval, axis_name="device")

  # Run the training loop
  for step in range(initial_step, config.num_train_steps + 1):
    # Use ._numpy() to avoid copy.
    batch = jax.tree_map(lambda x: x._numpy(), next(train_iter))  # pylint: disable=protected-access
    meta_state, metrics = p_step_fn(batch, meta_state)

    # Quick indication that training is happening.
    logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)

    if step > 4:
      return
    continue

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
