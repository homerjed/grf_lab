# ====================================
# 1. Import statements
# ====================================

# Jax is what we use for the network architecture. PyTorch or Tensorflow would also work just as well
import jax
import jax.numpy as jnp
import jax.random as jr

# Equinox provides easier NN/ML architectures
# Diffrax is a tool for solving differential equations that involve NN/ML architectures
# Optax is a trainer for gradient descent optimization
import equinox as eqx
import diffrax as dfx
import optax

# This is just some standard python to get a new function where one parameter is fixed
from functools import partial

# Finally, these are just for notation and don't actually do anything else
from jaxtyping import Key, Array, Float
from typing import Callable, Tuple, Optional

# ====================================
# 2. The flow neural network
# ====================================
# To generate the flow velocity field for the CNF,
# we need to construct a neural network that can take in psi_t and t, and gives out the velocity
# People have subtly improved the architectures over the years
# While you could use a simple multi-layer-perceptron (MLP) as a dense NN,
# we use a SquashMLP: a network that takes in conditioning variables and uses them to weigh/rescale
# the actual linear MLP output before the activation function.
# It is built of ConcatSquash layers, which do this operation in each layer

class ConcatSquash(eqx.Module):
    lin1: eqx.nn.Linear; lin2: eqx.nn.Linear; lin3: eqx.nn.Linear
    dropout1: eqx.nn.Dropout; dropout2: eqx.nn.Dropout

    # How do we construct a ConcatSquash layer from primitive elements?
    def __init__(self, in_size, out_size, y_dim, dropout_rate, key):
        key1, key2, key3 = jr.split(key, 3)
        self.lin1 = eqx.nn.Linear(in_size, out_size, key=key1)
        self.lin2 = eqx.nn.Linear(y_dim, out_size, key=key2)
        self.lin3 = eqx.nn.Linear(y_dim, out_size, use_bias=False, key=key3)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    # How do we actually go through a ConcatSquash layer?
    # 1) We concatenate time t and conditioning variable y into a super-conditioning ty
    # 2) We use that super-conditioning in a linear network and sigmoid it
    # 3) These weights (between 0 and 1) then multiply the ACTUAL linear layer, which is lin1
    # 4) The result is our unbiased output v
    # 5) Finally, we use the super-conditioning to compute some bias u
    # 6) The result is sum of unbiased output v and bias u
    def __call__(self, x, t, y, key):
        key1, key2 = jr.split(key, 2)
        ty = jnp.concatenate([t, y])
        v = self.dropout1(self.lin1(x) * jax.nn.sigmoid(self.lin2(ty)), key=key1)
        u = self.dropout2(self.lin3(ty), key=key2)
        return v + u

# Now to build the actual overall network from these layers!
class SquashMLP(eqx.Module):
    layers: list[ConcatSquash]
    norms: list[eqx.nn.LayerNorm]
    activation: Callable

    # How do we construct the neural network?
    def __init__(
        self,
        in_size: int,
        width_size: int,
        depth: int,
        y_dim: int,
        dropout_rate: float = 0.,
        activation: Callable = jax.nn.tanh,
        key: Key = None
    ):
        keys = jr.split(key, depth + 1)
        layers = []
        norms = []
        # This part is just for legibility of the code. 
        # Of course it's possible to build a 1-layer SquashedMLP in principle
        if depth == 0: raise Exception("We don't want flat SquashedMLPs here!")
        # Start: boost from in_size to width_size (broadening)
        layers.append(
            ConcatSquash(
                in_size=in_size, 
                out_size=width_size, 
                y_dim=y_dim, 
                dropout_rate=dropout_rate,
                key=keys[0]
            )
        )
        # Internal: Stay at width_size
        for i in range(depth - 1):
            layers.append(
                ConcatSquash(
                    in_size=width_size, 
                    out_size=width_size, 
                    y_dim=y_dim, 
                    dropout_rate=dropout_rate,
                    key=keys[i + 1]
                )
            )
        # Final: Go back from width_size to out_size==in_size
        layers.append(
            ConcatSquash(
                in_size=width_size, 
                out_size=in_size, 
                y_dim=y_dim, 
                dropout_rate=dropout_rate,
                key=keys[-1]
            )
        )
        # Everything assembled, now store it!
        self.layers = layers
        self.norms = norms 
        self.activation = activation

    # How do we actually call the neural network?
    # Note that this generates a velocity field v_phi(psi_t, t),
    # which depends additionally on the conditioning variable y
    def __call__(self, psi_t, t, y, key):
        t = jnp.atleast_1d(t)
        for layer in self.layers[:-1]:
            key, _ = jr.split(key)
            # For each layer, pass through the ConcatSquash layer and an activation function
            psi_t = layer(psi_t, t, y, key=key)
            psi_t = self.activation(psi_t)
        # For the final layer, don't use an activation function (!)
        key, _ = jr.split(key)
        psi_t = self.layers[-1](psi_t, t, y, key=key)
        # Return the output as is :)
        return psi_t


# ====================================
# 3. Scaling datasets
# ====================================
# In most machine learning applications, we do not want to trust the
# input/output values just like that. They could be 10^(-100) for all we know
# So in order to put them into a 'reasonable' range, we scale them
# Typically, with the mean and standard deviation of the dataset
# This module here allows us to keep track of this without further hassle in the main code
# This module considers x = the actual data, q = any conditioning variables 
#  (though it treats both equally, it's just convenient to do both simultaneously)
class Scaler(eqx.Module):
    # Dimensionality, mean and standard deviation arrays
    x_dim: int; q_dim: Optional[int] = None
    mu_x: Array; std_x: Array
    mu_q: Array; std_q: Array

    # How to build it:
    def __init__(
        self, 
        X: Float[Array, "n x"] = None,
        Q: Float[Array, "n q"] = None
    ):
        self.x_dim = X.shape[-1]
        self.mu_x = X.mean(axis=0)
        self.std_x = X.std(axis=0)
        self.q_dim = Q.shape[-1] 
        self.mu_q = Q.mean(axis=0)
        self.std_q = Q.std(axis=0)

    # How to use it x' = (x-mu_x)/std_x
    def forward(
        self, 
        x: Float[Array, "{self.x_dim}"], 
        q: Optional[Float[Array, "{self.q_dim}"]] = None
    ) -> Tuple[Float[Array, "{self.x_dim}"], Float[Array, "{self.q_dim}"]]:
        x = (x - jax.lax.stop_gradient(self.mu_x)) / jax.lax.stop_gradient(self.std_x)
        q = (q - jax.lax.stop_gradient(self.mu_q)) / jax.lax.stop_gradient(self.std_q)
        return x, q

    # How to reverse it x = x' * std_x + mu_x
    def reverse(
        self, 
        x: Float[Array, "{self.x_dim}"], 
        q: Optional[Float[Array, "{self.q_dim}"]] = None
    ) -> Tuple[Float[Array, "{self.x_dim}"], Float[Array, "{self.q_dim}"]]:
        x = x * jax.lax.stop_gradient(self.std_x) + jax.lax.stop_gradient(self.mu_x)
        q = q * jax.lax.stop_gradient(self.std_q) + jax.lax.stop_gradient(self.mu_q)
        return x, q

# ====================================
# 3. The actual CNF
# ====================================
# Now that we have the velocity flow field for the CNF
# it is time to put the remaining training apparatus for a CNF

class CNF(eqx.Module):
    # The class parameters
    net: eqx.Module # The net for the flow velocity field
    x_dim: int; cond_dim: int
    dt: float; t1: float
    solver: dfx.AbstractSolver
    scaler: eqx.Module

    # How to build a CNF
    def __init__(
        self,
        data_dim: int,
        condition_dim: int,
        width_size: int, # Width of each layer of the flow network
        depth: int, # Depth of the flow network
        activation: Callable = jax.nn.tanh, # Activation of the flow network
        dt: float = 0.08, # Time stepping for ODE
        t1: float = 1.0,  # Total time for ODE
        dropout_rate: float = 0., # Optional dropout functionality
        solver: Optional[dfx.AbstractSolver] = dfx.Heun(), # Do you want a custom ODE solver?
        scaler: eqx.Module = None, # Scaling the input data?
        *,
        key: Key
    ):

        cond_dim = condition_dim + 1  # We condition on the conditioning variables (like cosm. parameters) + time

        self.net = SquashMLP(
            in_size=data_dim,
            width_size=width_size,
            depth=depth,
            y_dim=cond_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            key=key
        )

        self.x_dim = data_dim
        self.cond_dim = condition_dim 
        self.dt = dt
        self.t1 = t1
        self.solver = solver
        self.scaler = scaler

    # The complete ODE term, where we simultaneously evolve psi_t and logp (!)
    # Both of these are inside of 'y = [psi_t, logp]'
    # To be used mostly internally (!)
    def complete_ode_term(self,
        t: float | Float[Array, "1"], 
        y: Float[Array, "y"], 
        args: Tuple[Float[Array, "q"], eqx.Module, Key]
    ) -> Tuple[Float[Array, "y"], Float[Array, "1"]]:

        # 1) We decompose the ODE input into what we actually use internally
        # q = our conditioning variable
        psi_t, logp = y
        q, v_phi, key = args
        t = jnp.atleast_1d(t)

        # 2) We tell jax to interpret v_phi(psi_t, t) as a function of psi_t only for the purpose of the Jacobian
        fn = lambda psi_t: v_phi(psi_t, t, q, key=key)
        # 2.5) Now we get the v_phi evaluation == the derivative of psi_t and the jacobian
        dpsi_t, jacobian_v_phi = jax.vjp(fn, psi_t) 

        # 3) Compute trace of Jacobian == the divergence of the v_phi
        (size,) = psi_t.shape
        (dfdy,) = jax.vmap(jacobian_v_phi)(jnp.eye(size))
        dlogp = jnp.trace(dfdy)
        
        # 4) The derivative of the 'y' is 'dy = [dpsi_t, dlogp]'
        return dpsi_t, dlogp

    # The prior probability (Multivariate Gaussian)
    def prior_log_prob(self, z: Float[Array, "{self.x_dim}"]) -> Float[Array, ""]:
        # Mean = 0, covariance = diagonal of ones => All independent
        return jax.scipy.stats.multivariate_normal.logpdf(
            z, jnp.zeros(self.x_dim), jnp.eye(self.x_dim)
        )

    # Compute the log_probability of a given single given data sample
    # x = data sample
    # cond = conditioning value
    # This explicitly integrates the normalizing flow from t=T (where x is sitting) to t=0 (where z is sitting)
    def single_log_prob_x(
        self, 
        x: Float[Array, "{self.x_dim}"], 
        cond: Float[Array, "{self.cond_dim}"], 
        key: Optional[Key[jnp.ndarray, "..."]],
    ) -> Float[Array, ""]:

        # Scale x, cond if desired
        if self.scaler is not None:
            x, cond = self.scaler.forward(x, cond)

        # Set-up for the ODE
        args = (cond, self.net, key)
        term = dfx.ODETerm(self.complete_ode_term)

        # Actually solving ODE (input, solving, output)
        y1 = (x, 0.) # y at t=1 = [x=psi_T , 0]
        soln = dfx.diffeqsolve(term, self.solver, self.t1, 0., -self.dt, y1, args)
        (z,), (delta_log_likelihood,) = soln.ys

        # Final probability formula = prior(z) + integral (div of v_phi)
        log_prob = self.prior_log_prob(z) + delta_log_likelihood
        return log_prob
    
    # The corresponding loss is just the negative log likelihood loss
    @eqx.filter_jit
    def loss(
        self,
        x: Float[Array, "batch {self.x_dim}"],
        cond: Float[Array, "batch {self.cond_dim}"],
        key: Optional[Key[jnp.ndarray, "..."]] = None 
    ) -> Float[Array, ""]:
        # Create a copy of self where dropout is disabled
        train_model = eqx.nn.inference_mode(self, False)
        keys = jr.split(key, len(x))

        # Simultaneously solve all the ODEs for each data sample in x
        neg_loglikes = -jax.vmap(train_model.single_log_prob_x)(x=x, cond=cond, key=keys)

        # Negative log likelihood loss
        return jnp.mean(neg_loglikes)



    # Compute the value of x and a log_probability corresponding to a given latent sample z
    # If you do not provide z, it is going to be sampled randomly
    def single_sample_and_log_prob(
        self, 
        key: Key[jnp.ndarray, "..."], 
        cond: Float[Array, "{self.cond_dim}"],
        z: Float[Array, "{self.x_dim}"] = None,
        tmax: Float[Array, ""] = None
    ) -> Tuple[Float[Array, "{self.x_dim}"], Float[Array, ""]]:

        key_z, key_sample = jr.split(key, 2)

        # Set up for the ODE
        args = (cond, self.net, key_sample)
        term = dfx.ODETerm(self.complete_ode_term)

        # Latent sample only if we do not provide it
        if z is None:
          z = jr.normal(key_z, (self.x_dim,))
        tmax = (tmax if tmax is not None else self.t1)

        # Actually solving ODE (input, solving, output)
        y0 = (z, 0.) # y at t=0 = [z=psi_0 , 0]
        soln = dfx.diffeqsolve(term, self.solver, 0., tmax, self.dt, y0, args)
        (x,), (delta_log_likelihood,) = soln.ys

        # Final probability formula = prior(z) + integral (div of v_phi)
        log_prob = self.prior_log_prob(z) + delta_log_likelihood
        if self.scaler is not None:
            x, cond = self.scaler.reverse(x, cond)
        return x, log_prob

    # Compute the values of x and a log_probability corresponding to a given set of latent samples z
    # If you do not provide z, it is going to be sampled randomly. Otherwise, it needs to be n_samples long
    # You can pass cond either as a single array for all n_samples, or a batch of cond of length n_samples
    def multi_sample_and_log_prob(
        self, 
        key: Key[jnp.ndarray, "..."], 
        cond: Float[Array, "*batch {self.cond_dim}"], 
        n_samples: int = None, 
        z: Float[Array, "n_samples {self.x_dim}"] = None,
        tmax: Float[Array, ""] = None
    ) -> Tuple[Float[Array, "n_samples {self.x_dim}"], Float[Array, "n_samples"]]:

        # Detect if we want to broadcast a single value (None), or if we want the cond to be mapped over all samples (0)
        cond_axis = 0 if (cond.ndim > 1 and len(cond) == n_samples) else None
        # Same for z
        z_axis = 0 if (z is not None) else None
        
        assert (z is None) or (z.shape == (n_samples, self.x_dim)), "z shape mismatch"
        assert (cond_axis is None) or (cond.shape == (n_samples, self.cond_dim)), "cond shape mismatch"

        # Generate an object we can sample from
        _sampler = jax.vmap(
            self.single_sample_and_log_prob, 
            in_axes=(0, cond_axis, z_axis, None)
        )

        # Now do the actual sampling
        keys = jr.split(key, n_samples)
        samples, log_probs = _sampler(keys, cond, z, tmax)
        return samples, log_probs 


# ====================================
# 4. A general module trainer
# ====================================
# This class is a general trainer for any kind of module
class Trainer:
    # Building a general trainer
    def __init__(self, model: eqx.Module, lr: float = 1e-3):
        self.model = model
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        self.history = {"train": [], "val": []}

    # Performing a single step in the training loop
    # *args is just whatever arguments the model needs to compute its loss
    @eqx.filter_jit
    def _step(self, model, opt_state, key, *args):
        loss_fn = lambda m: m.loss(*args, key=key)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        
        updates, opt_state = self.optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # Just compute the model loss, small wrapper function
    @eqx.filter_jit
    def _val_loss(self, model, key, *args):
        return model.loss(*args, key=key)

    # Fitting to a dataset
    # The *data should be the inputs to the network loss (e.g. (x_samples, cond) for the CNF or (pi, x) for the MLP)
    def fit(self, *data, steps=10_000, batch_size=64, key=jr.PRNGKey(0), validation_split = 0.2, verbose=True):
        num_val = int(len(data[0]) * validation_split)
        train_data = tuple(d[num_val:] for d in data)
        val_data = tuple(d[:num_val] for d in data)

        if verbose:
          print(f"Training {type(self.model).__name__} | Number of training data: {len(train_data[0])} | Number of validation data: {num_val}")
        for i in range(steps):
            key, step_key = jr.split(key)

            # Slice all data arrays by the same indices
            idx = jr.randint(step_key, (batch_size,), 0, len(train_data[0]))
            batch = tuple(d[idx] for d in train_data)
            
            # Feed them into our training step
            self.model, self.opt_state, train_loss = self._step(
                self.model, self.opt_state, step_key, *batch
            )

            # Update some print statements
            val_key, step_key = jr.split(step_key)
            val_idx = jr.randint(val_key, (batch_size,), 0, len(val_data[0]))
            val_batch = tuple(d[val_idx] for d in val_data)
            val_loss = self._val_loss(self.model, val_key, *val_batch)
            
            self.history["train"].append(train_loss)
            self.history["val"].append(val_loss)
            if verbose and i % 100 == 0 or i == steps - 1:
                print(f"Step {i:4d} | Train loss: {train_loss:.4f} | Validation loss: {val_loss:.4f}")
        return self.model

# ====================================
# 5. Some simple architectures in jax if you need them
# ====================================
class SimpleMLP(eqx.Module):
    model: eqx.nn.MLP

    def __init__(self, in_size, out_size, width_size, depth, key, activation = jax.nn.relu):
        self.model = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key
        )

    def __call__(self, x):
        if x.ndim == 2:
            return jax.vmap(self.model)(x)
        return self.model(x)

    # Standard mean-squared error loss
    def loss(self, x, y, key=None):
        preds = jax.vmap(self.model)(x)
        return jnp.mean(jnp.square(preds - y))

class SimpleCNN(eqx.Module):
    layers: list

    def __init__(self, key, input_shape=(64, 64), out_size=2, kernel_size=3):
        # Architecture: 3 Conv blocks -> Flatten -> Dense

        keys = jr.split(key, 5)
        h, w = input_shape

        if h < 8 or w < 8:
            raise ValueError(f"Input shape {input_shape} is too small. "
                             f"A 3-layer stride-2 CNN requires at least 8x8 pixels.")

        # Calculate spatial reduction (Stride-2 halves each dimension)
        def get_next_dim(dim): return (dim + 1) // 2
        h1, w1 = get_next_dim(h), get_next_dim(w)     # After Layer 1
        h2, w2 = get_next_dim(h1), get_next_dim(w1)   # After Layer 2
        h3, w3 = get_next_dim(h2), get_next_dim(w2)   # After Layer 3

        # 64 is the number of filters in the final conv layer
        flattened_size = 64 * h3 * w3

        self.layers = [
            # Layer 1: Input (1 channel) -> 16
            eqx.nn.Conv2d(1, 16, kernel_size, stride=2, padding=1, key=keys[0]),
            jax.nn.relu,
            # Layer 2: 16 -> 32
            eqx.nn.Conv2d(16, 32, kernel_size, stride=2, padding=1, key=keys[1]),
            jax.nn.relu,
            # Layer 3: 32 -> 64
            eqx.nn.Conv2d(32, 64, kernel_size, stride=2, padding=1, key=keys[2]),
            jax.nn.relu,
            # Flatten 3D tensor to 1D vector
            jnp.ravel, 
            # Multi-Layer Perceptron (MLP) Head
            eqx.nn.Linear(flattened_size, 128, key=keys[3]),
            jax.nn.relu,
            eqx.nn.Linear(128, out_size, key=keys[4]),
        ]

    def __call__(self, x):
        if x.ndim == 2:
            x = x[jnp.newaxis, :, :]
        for layer in self.layers:
            x = layer(x)
        return x

    # Standard mean-squared error loss
    def loss(self, x, y, key=None):
        preds = jax.vmap(self.__call__)(x)
        return jnp.mean(jnp.square(preds - y))

class CNNEnsemble(eqx.Module):
    networks: list

    def __init__(self, key, n_models=20, **kwargs):
        # Generate a unique key for each sub-model
        keys = jr.split(key, n_models)
        # Create a list of independent SimpleCNN instances
        self.networks = [SimpleCNN(k, **kwargs) for k in keys]

    def __call__(self, x):
        # Get predictions from all models, take the average
        preds = jnp.stack([net(x) for net in self.networks])
        return jnp.mean(preds, axis=0)

    def loss(self, x, y, key=None):
        # x shape: (batch, h, w)
        preds = jax.vmap(self.__call__)(x)
        return jnp.mean(jnp.square(preds - y))
