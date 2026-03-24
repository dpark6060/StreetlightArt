"""Numeric parameter with configurable min/max and probability distribution."""

import random
import typing as t


_SUPPORTED_DISTRIBUTIONS = ("uniform", "normal", "exponential")


class RangeParam:
    """Samples a float from a bounded range using a named distribution.

    All numeric parameter randomisation in the project goes through this
    class so that results remain reproducible when a seeded RNG is provided.

    Supported distributions:
        - "uniform":     Uniform sampling over [min_val, max_val].
        - "normal":      Gaussian with mean = (min+max)/2, sigma = (max-min)/4,
                         clamped to [min_val, max_val].
        - "exponential": Exponential distribution scaled and shifted to
                         [min_val, max_val].
    """

    def __init__(
        self,
        min_val: float,
        max_val: float,
        distribution: str,
        rng: random.Random,
    ) -> None:
        """Initialize a RangeParam.

        Args:
            min_val (float): Lower bound of the range (inclusive).
            max_val (float): Upper bound of the range (inclusive).
            distribution (str): Sampling distribution — one of "uniform",
                "normal", or "exponential".
            rng (random.Random): Seeded RNG instance for reproducible sampling.

        Raises:
            ValueError: If distribution is not one of the supported values,
                or if min_val >= max_val.
        """
        self._validate_args(min_val, max_val, distribution)
        self._min = min_val
        self._max = max_val
        self._distribution = distribution
        self._rng = rng

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def pick(self) -> float:
        """Sample a value from the configured distribution.

        Returns:
            float: A sampled value within [min_val, max_val].
        """
        if self._distribution == "uniform":
            return self._pick_uniform()
        if self._distribution == "normal":
            return self._pick_normal()
        return self._pick_exponential()

    # ------------------------------------------------------------------
    # Private sampling helpers
    # ------------------------------------------------------------------

    def _pick_uniform(self) -> float:
        """Sample uniformly over [min, max].

        Returns:
            float: Uniform sample.
        """
        return self._rng.uniform(self._min, self._max)

    def _pick_normal(self) -> float:
        """Sample from a Gaussian centered between min and max, clamped.

        Mean is the midpoint; sigma is one quarter of the range so that
        ~95% of samples fall within [min, max] before clamping.

        Returns:
            float: Clamped Gaussian sample.
        """
        mean = (self._min + self._max) / 2.0
        sigma = (self._max - self._min) / 4.0
        value = self._rng.gauss(mean, sigma)
        return max(self._min, min(self._max, value))

    def _pick_exponential(self) -> float:
        """Sample from an exponential distribution shifted to [min, max].

        Uses a rate lambda = 1 / (max - min) so the expected value of the
        raw exponential equals one range-width. The raw sample is then
        scaled and shifted into [min, max] via modulo wrapping so the
        result always stays in bounds.

        Returns:
            float: Exponential sample mapped to [min, max].
        """
        span = self._max - self._min
        rate = 1.0 / span
        raw = self._rng.expovariate(rate)
        # Wrap into [0, span] so the result never exceeds the bounds
        scaled = raw % span
        return self._min + scaled

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_args(min_val: float, max_val: float, distribution: str) -> None:
        """Raise ValueError if arguments are invalid.

        Args:
            min_val (float): Proposed lower bound.
            max_val (float): Proposed upper bound.
            distribution (str): Proposed distribution name.

        Raises:
            ValueError: If min_val >= max_val or distribution is unsupported.
        """
        if min_val >= max_val:
            raise ValueError(
                f"min_val ({min_val}) must be strictly less than max_val ({max_val})."
            )
        if distribution not in _SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"Unsupported distribution '{distribution}'. "
                f"Choose from: {_SUPPORTED_DISTRIBUTIONS}."
            )
