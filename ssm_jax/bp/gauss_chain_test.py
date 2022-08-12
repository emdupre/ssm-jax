from jax import numpy as jnp
from jax import random as jr

#from ssm_jax.bp.gauss_chain import cliques_from_lgssm, gauss_chain_bp
from ssm_jax.lgssm.info_inference import lgssm_info_smoother
from ssm_jax.lgssm.info_inference_test import TestInfoFilteringAndSmoothing

lgssm = TestInfoFilteringAndSmoothing.lgssm
lgssm_info = TestInfoFilteringAndSmoothing.lgssm_info

key = jr.PRNGKey(111)
num_timesteps = 15
inputs = jnp.zeros((num_timesteps, input_size))
z, y = lgssm.sample(key, num_timesteps, inputs)

lgssm_info_posterior = lgssm_info_smoother(lgssm_info, y, inputs)

potentials = cliques_from_lgssm(lgssm_info, inputs)

bels = gauss_chain_bp(potentials,y)

K_down, h_down = bels_down
print(jnp.allclose(lgssm_info_posterior.smoothed_precisions,K_down,
                   rtol=1e-3))
print(jnp.allclose(lgssm_info_posterior.smoothed_etas,h_down,
                   rtol=1e-3))

