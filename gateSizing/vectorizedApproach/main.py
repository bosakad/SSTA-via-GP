import numpy as np
import cvxpy as cp




m = 7   # number of cells
n = 8   # number of edges
A = np.zeros((m,n))

A[0][0] = 1
A[1][1] = 1
A[1][2] = 1
A[2][3] = 1
A[2][7] = 1
A[3][0] = -1
A[3][1] = -1
A[3][4] = 1
A[3][5] = 1
A[4][2] = -1
A[4][3] = -1
A[4][6] = 1
A[5][4] = -1
A[6][5] = -1
A[6][6] = -1
A[6][7] = -1


# problem constants
f = np.array([1, 0.8, 1, 0.7, 0.7, 0.5, 0.5])
e = np.array([1, 2, 1, 1.5, 1.5, 1, 2])
Cout6 = 10
Cout7 = 10

a     = np.ones(m)
alpha = np.ones(m)
beta  = np.ones(m)
gamma = np.ones(m)

# maximum area and power specification
Amax = 25
Pmax = 50


Aout = np.where(A<=0, 0, A)
Ain  = np.where(A>=0, 0, 1)

# optimization variables
x = cp.Variable(m, pos=True)         # sizes
t = cp.Variable(m, pos=True)         # arrival times

# input capacitance is an affine function of sizes
cin = alpha + beta*x

# load capacitance is the input capacitance times the fan-out matrix
# given by Fout = Aout*Ain'
cload = (Aout @ Ain.T) * cin

# Create an array of constraints
constr_cload = []
# Make your requirement a constraint
constr_cload.append(cload[5]==Cout6)
constr_cload.append(cload[6]==Cout7)


# delay is the product of its driving resistance R = gamma/x and cload
d = cload * gamma/x

#total power
power = (f*e) * x

# total area
area = a * x

# constraints
constr_x = x >= 1 # all sizes greater than 1 (normalized)

# create timing constraints
# these constraints enforce t_j + d_j <= t_i over all gates j that drive gate i
constr_timing = Aout.T*t + Ain.T*d <= Ain.T*t
# for gates with inputs not connected to other gates we enforce d_i <= t_i
input_timing  = d[0:2] <= t[0:2]



# objective is the upper bound on the overall delay
# and that is the max of arrival times for output gates 6 and 7
D = cp.atoms.elementwise.maximum.maximum(t[5],t[6])

# collect all the constraints
constraints = [power <= Pmax, area <= Amax, constr_timing, input_timing, constr_x] + constr_cload


# formulate the GP problem and solve it
objective = cp.Minimize(D)

problem = cp.Problem(objective, constraints)
problem.solve(gp=True)




