import flax
import jax
from jax import numpy as jnp, random
import tensorflow_datasets as tfds


class CustomCNN(flax.nn.Module):
    """
    This is Dan's example CNN. This was created just for training purposes.
    """

    def apply(self, data, out_size=10):
        # First Convolutional module
        x = flax.nn.Conv(data, features=32, kernel_size=(2, 2), strides=(1, 1), padding="SAME")
        x = flax.nn.relu(x)
        x = flax.nn.max_pool(x, (2, 2), strides=(2, 2), padding="SAME")

        # Flatten the input 
        x = x.reshape((x.shape[0], -1))

        # Produce the classifications
        x = flax.nn.Dense(x, features=out_size)
        x = flax.nn.softmax(x)

        return x


def create_model(rng, shape=(32, 32, 3)):
    """
    Helper function for creating a model.

    :param rng: the PRNG which will be used by Module.init() to initialize the params.
    :param shape: the shape of the input data. Defaults to CIFAR's (32, 32, 3).
    :return: the Model object, which wraps the Module.
    """
    _, params = CustomCNN.init(rng, jnp.zeros((1, ) + shape, dtype=jnp.float32))
    return flax.nn.Model(CustomCNN, params)
    

def create_optimizer(model):
    """
    Creates an optimizer for the specified model.
    
    :param model: the model which will be optimized.
    :return: the optimizer.
    """
    optimizer_def = flax.optim.Momentum(0.1)
    return optimizer_def.create(model)


@jax.vmap
def cross_entropy(logits, label):
    """
    This is the squared error. 

    :param y_true: the true label.
    :param t_pred: the predicted label.
    :return: the squared loss.
    """
    return -logits[label]


def optimizer_step(batch, optimizer):
    """
    Execute an optimization step at the batch granularity.

    :param batch: the batch of images+labels.
    :param model: the model object.
    :param optimizer: the optimizer object.
    :return: the updated model and optimizer.
    """
    def _loss_helper(model):
        y = model(batch['image'])
        loss = jnp.mean(cross_entropy(y, batch['label']))
        return loss

    # Curried function, which the computes the loss and gradient
    grad = jax.grad(_loss_helper)(optimizer.target)
    optimizer.apply_gradient(grad)
    return optimizer


def eval(data, model):
    """
    This method produces some performance metrics for the model.

    :param data: the data on which evaluation is performed.
    :parma model: the model being evaluated.
    :return: the Accuracy and the Cross-Entropy of the model.
    """
    y = model(data['image'])
    acc = jnp.sum(jnp.argmax(y, axis=1) == data['label']) / len(data['image'])
    ce = jnp.mean(cross_entropy(y, data['label']))
    return acc, ce

def main():
    # Get the CIFAR-10 data
    data = tfds.load('cifar10', split=tfds.Split.TRAIN)
    data = data.cache().shuffle(1024).batch(32)
    test_data = tfds.as_numpy(tfds.load('cifar10', split=tfds.Split.TEST, batch_size=-1))
    test_data['image'] = test_data['image'] / 255.0

    # Use PRNG for reproductibility 
    model = create_model(random.PRNGKey(0))
    optimizer = create_optimizer(model)
    
    # Train the model
    for epoch in range(5):
        for batch in tfds.as_numpy(data):
            batch['image'] = batch['image'] / 255.0
            optimizer = optimizer_step(batch, optimizer)
        acc, ce = eval(test_data, optimizer.target)
        print("> Batch #{}\n  - Accuracy: {}\n  - Cross Entropy: {}\n".format(epoch, acc, ce))

if __name__ == '__main__':
    main()

