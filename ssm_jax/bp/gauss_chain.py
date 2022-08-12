from jax import vmap
from jax import numpy as jnp
from ssm_jax.bp.gauss_bp import (potential_from_conditional_linear_gaussian,
                                 info_condition,
                                 info_marginalise,
                                 info_marginalise_down,
                                 info_multiply,
                                 info_divide)


def cliques_from_lgssm(lgssm_params, inputs):
    """Construct pairwise latent and emission cliques from model.

    Args:
        lgssm_params: an LGSSMInfoParams instance.
        inputs: (T,D_in): array of inputs.

    Returns:
        prior_pot: A tuple of parameters representing the prior potential,
                    (Lambda0, eta0)
        lambda_pots: A tuple containing the parameters for each pairwise latent
                     clique potential - ((K11, K12, K22),(h1, h2)). The ith row
                     of each array contains parameters for the clique containing
                     the pair of latent states at times (i, i+1).
                     Arrays have shapes:
                         K11, K12, K22 - (T-1, D_hid, D_hid)
                         h1, h2 - (T-1, D_hid)
        emission_pots: A tuple containing the parameters for each pairwise
                     emission clique potential - ((K11, K12, K22),(h1, h2)).
                     Arrays have shapes:
                         K11 - (T, D_hid, D_hid)
                         K12 - (T, D_hid, D_obs)
                         K22 - (T, D_obs, D_obs)
                         h1 - (T, D_hid)
                         h2 - (T, D_obs)
    """
    B, b = lgssm_params.dynamics_input_weights, lgssm_params.dynamics_bias
    D, d = lgssm_params.emission_input_weights, lgssm_params.emission_bias
    latent_net_inputs = vmap(jnp.dot, (None, 0))(B, inputs) + b
    emission_net_inputs = vmap(jnp.dot, (None, 0))(D, inputs) + d

    F, Q_prec = lgssm_params.dynamics_matrix, lgssm_params.dynamics_precision
    H, R_prec = lgssm_params.emission_matrix, lgssm_params.emission_precision
    # Each of these is a tuple ((K11, K12, K22), (h1, h2)) where each element contains
    #  the clique parameters for different timepoints as stacked rows.
    latent_pots = vmap(potential_from_conditional_linear_gaussian, (None, 0, None))(F, latent_net_inputs[:-1], Q_prec)
    emission_pots = vmap(potential_from_conditional_linear_gaussian, (None, 0, None))(
        H, emission_net_inputs[1:], R_prec
    )

    Lambda0, mu0 = lgssm_params.initial_precision, lgssm_params.initial_mean
    prior_pot = (Lambda0, Lambda0 @ mu0)

    return prior_pot, emission_pots, latent_pots


def gauss_chain_bp(potentials, y):

    prior_pot, emission_pots, latent_pots = potentials
    init_emission_pot = jax.tree_map(lambda a: a[0], emission_pots)
    emission_pots_rest = jax.tree_map(lambda a: a[1:], emission_pots)

    # Absorb first emission message
    init_emission_message = info_blocks_condition(*init_emission_pot, y[0])
    init_carry = info_multiply((Lambda0, eta0), init_emission_message)

    def _forward_step(carry, x):
        prev_bel = carry
        latent_pot, emission_pot, y = x

        latent_pot = absorb_latent_message(prev_bel, latent_pot)
        latent_message = info_marginalise(*latent_pot)

        emission_message = info_blocks_condition(*emission_pot, y)
        bel = info_multiply(latent_message, emission_message)

        return bel, (bel, latent_message)

    # Message pass forwards along chain
    _, (bels, messages) = lax.scan(_forward_step,
                                   init_carry,
                                   (latent_pots, emission_pots_rest, y[1:]))
    # Append first belief
    bels_up = jax.tree_map(lambda h, t: jnp.row_stack((h[None, ...], t)), init_carry, bels)

    # Extract final belief
    init_carry = jax.tree_map(lambda a: a[-1], bels_up)
    bels_rest = jax.tree_map(lambda a: a[:-1], bels_up)

    def _backward_step(carry, x):
        prev_bel = carry
        bel, message_up, latent_pot = x

        # Divide out forward message    
        bel_minus_message_up = info_divide(prev_bel, message_up)
        # Absorb into joint potential
        latent_pot = info_multiply(latent_pot, ((0, 0, bel_minus_message_up[0]), (0, bel_minus_message_up[1])))
        message_down = info_marginalise_down(*latent_pot)

        bel = info_multiply(bel, message_down)
        return bel, bel

    # Message pass back along chain
    _, bels_down = lax.scan(_backward_step,
                            init_carry,
                            (bels_rest, messages, latent_pots),
                            reverse=True)
    # Append final belief
    bels_down = jax.tree_map(lambda h, t: jnp.row_stack((h, t[None, ...])), bels_down, init_carry)

    return bels_down
