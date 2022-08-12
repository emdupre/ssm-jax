import chex


@chex.dataclass
class CanonicalPotential:
    """Wrapper for the parameters of a canonical gaussian potential."""

    K11: chex.Array
    K12: chex.Array
    K22: chex.Array
    h1: chex.Array
    h2: chex.Array
