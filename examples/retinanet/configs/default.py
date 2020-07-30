

class ConfigDict(dict):
  """Dictionary with dot access to values."""

  def __getattr__(self, attr):
    return self[attr]

  def __setattr__(self, key, value):
    self[key] = value

  def __setitem__(self, key, value):
    super().__setitem__(key, value)
    self.__dict__.update({key: value})

  def __delattr__(self, item):
    self.__delitem__(item)

  def __delitem__(self, key):
    super().__delitem__(key)
    del self.__dict__[key]


def get_config():
  """Get the default hyperparameter configuration."""
  config = ConfigDict()

  config.learning_rate = 0.1
  config.batch_size = 64
  config.num_train_steps = 90_000,
  config.warmup_steps = 30_000

  # The number of layers in the RetinaNet backbone.
  config.depth = 50

  config.checkpoint_every_steps = 20_000

  config.seed = 42

  return config
