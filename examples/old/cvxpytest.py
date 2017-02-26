from cvxpy import *

import pywren
import subprocess
import sys
import numpy as np

def trivial_cvx():
    def trivial(x):

        # Create two scalar optimization variables.
        x = Variable()
        y = Variable()

        # Create two constraints.
        constraints = [x + y == 1,
                       x - y >= 1]

        # Form objective.
        obj = Minimize(square(x - y))

        # Form and solve problem.
        prob = Problem(obj, constraints)
        prob.solve()  # Returns the optimal value.
        print "status:", prob.status
        print "optimal value", prob.value
        print "optimal var", x.value, y.value
        return prob.value

    wrenexec = pywren.default_executor()
    fut = wrenexec.call_async(run_command, 0)
    print fut.callset_id

    res = fut.result() 
    print res


def kalman_exp():
    n = 2000 # number of timesteps
    T = 50 # time will vary from 0 to T with step delt
    ts, delt = np.linspace(0,T,n,endpoint=True, retstep=True)
    gamma = .05 # damping, 0 is no damping


    ## Set up dynamical system
    A = np.zeros((4,4))
    B = np.zeros((4,2))
    C = np.zeros((2,4))

    A[0,0] = 1
    A[1,1] = 1
    A[0,2] = (1-gamma*delt/2)*delt
    A[1,3] = (1-gamma*delt/2)*delt
    A[2,2] = 1 - gamma*delt
    A[3,3] = 1 - gamma*delt

    B[0,0] = delt**2/2
    B[1,1] = delt**2/2
    B[2,0] = delt
    B[3,1] = delt

    C[0,0] = 1
    C[1,1] = 1


    ## generate syntehtic data

    sigma = 20
    p = .20
    np.random.seed(6)

    x = np.zeros((4,n+1))
    x[:,0] = [0,0,0,0]
    y = np.zeros((2,n))

    # generate random input and noise vectors
    w = np.random.randn(2,n)
    v = np.random.randn(2,n)

    # add outliers to v
    np.random.seed(0)
    inds = np.random.rand(n) <= p
    v[:,inds] = sigma*np.random.randn(2,n)[:,inds]

    # simulate the system forward in time
    for t in range(n):
        y[:,t] = C.dot(x[:,t]) + v[:,t]
        x[:,t+1] = A.dot(x[:,t]) + B.dot(w[:,t])

    x_true = x.copy()
    w_true = w.copy()

    # solve

    def solve_robust_kf((tau, rho)):
        x = Variable(4,n+1)
        w = Variable(2,n)
        v = Variable(2,n)


        obj = sum_squares(w)
        obj += sum(tau*huber(norm(v[:,t]),rho) for t in range(n))
        obj = Minimize(obj)

        constr = []
        for t in range(n):
            constr += [ x[:,t+1] == A*x[:,t] + B*w[:,t] ,
                        y[:,t]   == C*x[:,t] + v[:,t]   ]

        Problem(obj, constr).solve(verbose=True,solver='ECOS', max_iters=1000)

        rkf_x = np.array(x.value)
        w = np.array(w.value)

        return np.linalg.norm(rkf_x - x_true)
    wrenexec = pywren.default_executor()
    fut = wrenexec.call_async(run_command, 0)
    print fut.callset_id

    res = fut.result() 
    print res
    
if __name__ == "__main__":
    trivial_cvx()

    
