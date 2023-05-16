# Statistical static timing analysis via modern optimization lens : I. Histogram–based approach


Statistical static timing analysis (SSTA) is studied from the mathematical optimization point of view. We give two formulations of the problem of finding the critical path delay distribution that were not known before: (i) a formal mathematical formulation of the SSTA problem using Binary–Integer Programming and (ii) a practical formulation using Geometric Programming. For simplicity, we use histogram approximation of the distributions. Scalability of the approaches are studied and possible generalizations are discussed.

### Set up: ###
`pip install -r requirements.txt`  
`pip install .`  
`python3 setup.py clean`

### For more details, please see: ###

Our [arXiv draft](https://arxiv.org/abs/2211.02981)

Full documentation at: [docs/_build/html](https://htmlpreview.github.io/?https://github.com/bosakad/GP-Optimization/blob/development/docs/_build/html/index.html)

Examples at: [examples/](https://github.com/bosakad/GP-Optimization/tree/main/examples)

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
