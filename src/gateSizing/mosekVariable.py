import mosek
import sys
import numpy as np

"""
  This module includes same functions as module 'cvxpyVariable'. Is coded in MOSEK API for speed-up purposes.
  Unary variable is represented by numpy [n, m], where n is number of bins, m is number of unaries. For a MOSEK 
  variable bins is represented by [n, m] list of indices.
"""


class RandomVariableMOSEK:
    """
    Class representing a random variable given by histogram represented as MOSEK variable.
    Information about each variable is kept by their indices.

    Class includes:
      bins: numpy list of lists of integer indices to MOSEK variable list
      edges: len n+1 of histogram edges, dtype: 1-D np.array
    """


    def __init__(self, bins: np.array, edges: np.array):
        self.bins = bins
        self.edges = edges




    def convolution_UNARY_DIVIDE(self, secondVariable, withSymmetryConstr=False, asMin=False):
        """ Calculates convolution of 2 PDFs of random variable. Works only for 2 identical edges. Is computed
        using the unary representation of bins - M 0/1-bins for each bin. Unarization is kept using the divison.
        Is in MOSEK environment.

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :param asMin: boolean, true for minimization problem, false for maximization problem
        :param withSymmetryConstr: boolean whether a symmetry constraints should be included
        :return convolutionClass:  class RandomVariableCVXPY with cvxpy slack variables
        :return ConvConstraints: python array with inequalities - for computing the convolution
        """

        print('ahoj')

        return None

    def maximum_QUAD_UNARY_DIVIDE(self, secondVariable, withSymmetryConstr=False, asMin=False):
        """
        Calculates maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the 'quadratic' algorithm and unary representation of bins - M 0/1-bins for each bin.
        Unarization is kept using the divison.
        Is in MOSEK environment.

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :param asMin: boolean, true for minimization problem, false for maximization problem
        :param withSymmetryConstr: boolean whether a symmetry constraints should be included
        :return maximumClass: class RandomVariableCVXPY with cvxpy slack variables (1, 1)
        :return MaxConstraints: python array with inequalities - for computing the maximum
        """

        return None























if __name__ == "__main__":

  # call the main function
  try:
      main()
  except mosek.MosekException as msg:
      #print "ERROR: %s" % str(code)
      if msg is not None:
          print("\t%s" % msg)
          sys.exit(1)
  except:
      import traceback
      traceback.print_exc()
      sys.exit(1)