import parser
import numpy as np


def parserTest1(name: str) -> None:

    actual = parser.getIncidenceMatrixFromNetlist(name)

    desired = np.array(   [[0, 0, 1, 0, 0, 0],
                           [1, 1, 0, 0, 0, 0],
                           [-1, 0, 0, 1, 1, 0],
                           [0, -1, 0, 0, 0, 1],
                           [0, 0, -1, -1, 0, 0],
                           [0, 0, 0, 0, -1, -1]] )

    np.testing.assert_almost_equal(actual, desired)

    return None



if __name__ == "__main__":
    parserTest1(sys.argv[1:])

    print("All tests passed!")
