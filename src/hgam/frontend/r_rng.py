"""R's default random number stream, as far as ``discrete.mf`` (bam.r) draws from it.

``bam(discrete=TRUE)`` shuffles each compressed covariate and pads the shorter ones with
``sample()`` draws under ``temp.seed(8547)``, which is ``set.seed(8547)`` with
``RNGkind("default", "default")``: Mersenne-Twister, Inversion, and (R >= 3.6) Rejection
sampling. Which values pad a covariate decides the rows its sum-to-zero constraint is taken over
(``smoothCon``'s ``colMeans(X)`` on the padded frame), so the stream is ported exactly.

These are R's own C routines, not mgcv's: ``RNG_Init``, ``MT_genrand`` and ``fixup`` (R's
``src/main/RNG.c``), ``rbits`` and ``R_unif_index`` (``RNG.c``) and ``do_sample`` (R's
``src/main/random.c``), read from R 4.x. Integer arithmetic is unsigned 32-bit, as there.
"""

import math

import numpy as np

from hgam.errors import HgamValidationError

__all__ = ["RRandom"]

_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF
_MASK32 = 0xFFFFFFFF
_I2_32M1 = 2.328306437080797e-10  # 1 / (2^32 - 1), fixup's constant


class RRandom:
    """R's ``set.seed(seed)`` stream with the default kinds (Mersenne-Twister, Rejection)."""

    def __init__(self, seed: int) -> None:
        """R counterpart: ``set.seed`` -> ``RNG_Init`` (RNG.c), Mersenne-Twister.

        Fifty rounds of ``seed = 69069 * seed + 1`` scramble the seed, then each of the 625
        integers of ``.Random.seed`` (``mti`` first, then ``mt``) takes the next round;
        ``FixupSeeds`` sets ``mti = 624`` so the first draw regenerates the state."""
        s = int(seed) & _MASK32
        for _ in range(50):
            s = (69069 * s + 1) & _MASK32
        state = []
        for _ in range(_N + 1):
            s = (69069 * s + 1) & _MASK32
            state.append(s)
        self._mt = state[1:]
        self._mti = _N
        if not any(self._mt):
            raise HgamValidationError("RRandom: the seed gave an all-zero Mersenne-Twister state.")

    def _genrand(self) -> float:
        """R counterpart: ``MT_genrand`` (RNG.c): MT19937, one tempered word times 2^-32."""
        mt = self._mt
        if self._mti >= _N:
            for kk in range(_N):
                y = (mt[kk] & _UPPER_MASK) | (mt[(kk + 1) % _N] & _LOWER_MASK)
                mt[kk] = mt[(kk + _M) % _N] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            self._mti = 0
        y = mt[self._mti]
        self._mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return (y & _MASK32) * 2.3283064365386963e-10

    def unif_rand(self) -> float:
        """R counterpart: ``unif_rand`` (RNG.c): ``fixup(MT_genrand())``, in (0, 1)."""
        x = self._genrand()
        if x <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - x <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return x

    def _rbits(self, bits: int) -> int:
        """R counterpart: ``rbits`` (RNG.c): 16 bits per ``unif_rand``, masked to ``bits``."""
        v = 0
        for _ in range(0, bits + 1, 16):
            v = 65536 * v + math.floor(self.unif_rand() * 65536)
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: int) -> int:
        """R counterpart: ``R_unif_index`` (RNG.c), Rejection: a uniform integer in ``[0, dn)``."""
        if dn <= 0:
            return 0
        bits = math.ceil(math.log2(dn))
        while True:
            dv = self._rbits(bits)
            if dv < dn:
                return dv

    def sample_int(self, n: int, size: int, *, replace: bool = False) -> np.ndarray:
        """R counterpart: ``sample.int(n, size, replace)`` -> ``do_sample`` (random.c), no ``prob``.

        Returns R's 1-based draws. Without replacement it is ``do_sample``'s partial shuffle;
        ``sample.int``'s hash branch (``n > 1e7``) is not ported."""
        if n < 0 or size < 0 or (not replace and size > n):
            raise HgamValidationError(f"sample_int: cannot take {size} of {n} (replace={replace}).")
        if n > 10_000_000 and not replace and size <= n / 2:
            raise NotImplementedError("sample_int: sample.int's hash branch (n > 1e7) is not ported.")
        out = np.empty(size, dtype=np.int64)
        if replace or size < 2:
            for i in range(size):
                out[i] = self.unif_index(n) + 1
            return out
        pool = list(range(n))
        remaining = n
        for i in range(size):
            j = self.unif_index(remaining)
            out[i] = pool[j] + 1
            remaining -= 1
            pool[j] = pool[remaining]
        return out
