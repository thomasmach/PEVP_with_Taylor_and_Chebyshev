# PEVP_with_Taylor_and_Chebyshev

This repository contains MATLAB code to solve parametric eigenvalue problems
with a Taylor or Chebyshev expansion. The code has been used for the numerical
experiments in the paper “Solving the Parametric Eigenvalue Problem by Taylor
Series and Chebyshev Expansion” by Thomas Mach and Melina Freitag.

A parametric eigenvalue problem has the form

	A(mu) v(mu) = lambda(mu) v(mu),

with A(mu) given. The code computes approximations for eigenpairs (lambda(mu),v(mu)).

The code was tested with MATLAB R2020b and the latest release of Chebfun
(v.5.7.0). The Chebfun package is
[available on github under https://github.com/chebfun/chebfun](https://github.com/chebfun/chebfu.n).
Chebfun is used to represent the matrix A(mu) in some of the examples.


## Authors

- [Thomas Mach](https://sites.google.com/site/thomasmach/), University of Potsdam
- [Melina Freitag](https://sites.google.com/view/melina-freitag), University of Potsdam


## Publication

* Thomas Mach, Melina Freitag: Solving the Parametric Eigenvalue Problem by Taylor
Series and Chebyshev Expansion, in preparation.


## Installation
