#Statistical static timing analysis via modern optimization lens

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

### Other info: ###
Full documentation is located at: docs/_build/html
