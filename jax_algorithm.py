import jax.numpy as jnp
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax

from algorithm import Algorithm, ArrayDict, Info

init_random_params, predict = stax.serial(
    Dense(1024), Relu, Dense(1024), Relu, Dense(10), LogSoftmax
)


def loss(params: jnp.ndarray, batch: jnp.ndarray) -> jnp.ndarray:
    inputs, targets = batch
    preds = predict(params, inputs)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))


def accuracy(params: jnp.ndarray, batch: jnp.ndarray) -> jnp.ndarray:
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)


rng = random.PRNGKey(0)

step_size = 0.001
num_epochs = 10
batch_size = 128
momentum_mass = 0.9
log_interval = 100

opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)


@jit
def update(i: int, opt_state, batch: jnp.ndarray):
    params: jnp.ndarray = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)


class SupervisedMLP(Algorithm):
    def __init__(self):
        self.i = 0
        _, init_params = init_random_params(rng, (-1, 28 * 28))
        self.opt_state = opt_init(init_params)

    def update(self, inputs: ArrayDict, targets: ArrayDict, info: Info) -> Info:
        self.opt_state = update(self.i, self.opt_state, inputs)
        self.i += 1
        return info

    def infer(self, inputs: ArrayDict) -> ArrayDict:
        params = get_params(self.opt_state)
        return predict(params, inputs)
