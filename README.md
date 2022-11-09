# Statistical static timing analysis via modern optimization lens

We formulate statistical static timing analysis (SSTA) as a mixed-
integer program and as a geometric program, utilizing histogram approxima-
tions of the random variables involved. The geometric-programming approach
scales linearly with the number of gates and quadratically with the number
of bins in the histogram. This translates, for example, to solving the SSTA
for a circuit of 400 gates with 30 bins per each histogram approximation of a
random variable in 440 seconds

### Set up: ###
`pip install -r requirements.txt`  
`pip install .`  
`python3 setup.py clean`

### For more details, please see: ###

Our [arXiv draft](https://arxiv.org/abs/2211.02981)

Full documentation at: [docs/_build/html](https://htmlpreview.github.io/?https://github.com/bosakad/GP-Optimization/blob/development/docs/_build/html/index.html)

If you like the concept, please cite our draft:

```
@misc{Bosak2022,
  doi = {10.48550/ARXIV.2211.02981},  
  url = {https://arxiv.org/abs/2211.02981},
  author = {Bosak, Adam and Mishagli, Dmytro and Marecek, Jakub},
  title = {Statistical timing analysis via modern optimization lens},
  publisher = {arXiv},
  year = {2022},
}
```
