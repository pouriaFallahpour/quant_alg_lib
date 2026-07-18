# An informal library for quantum algorithms

This library contains the code for implementing some quantum algorithms and is supposed to be extended.

# List of algorithms
- The coherent and fixed-point amplitude amplification of [YLC14].
- The coherent Hamiltonian simulation of [LMR14]. 
- The amplitude amplifcation using copies of the state instead of the creator operator
    - It is a nice combination of [YLC14] and [LMR14]. **To see a cool application for Euclidean lattice problems, send me an email.**

[YLC14] Theodore J. Yoder, Guang Hao Low, and Isaac L. Chuang. Fixed-point quantum search with an optimal
number of queries. Phys. Rev. Lett., 2014.

[LMR14] Seth Lloyd, Masoud Mohseni, and Patrick Rebentrost. Quantum principal component analysis. Nature
Physics, 2014.

# To run the code

Install uv and clone the repo. Then run the following commands to see the tests

```
uv sync
uv run ylc_amplifier.py
uv run lmr_simulation.py
uv run ylc_plus_lmr.py
```


