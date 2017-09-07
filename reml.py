#! /usr/bin/env python
import math

import numpy as np
import numpy.linalg as la
import sympy

from scipy.linalg import clarkson_woodruff_transform as cwt


def logdet(X):
    sign, ld = la.slogdet(X)
    if sign == 1:
        return ld
    else:
        return ld * -1


def get_terms(X, V):
    if X is not None:
        Vinv = la.inv(V)
        XtVinvX = la.multi_dot([X.T, Vinv, X])
        P = Vinv - la.multi_dot([Vinv, X, la.inv(XtVinvX), X.T, Vinv])
    else:
        Vinv = la.inv(V)
        P = Vinv
        XtVinvX = None

    return (P, XtVinvX)


# the log likelihood equation for the linear mixed model
def ll(y, X, P, V, XtVinvX):
    if X is not None:
        return -0.5 * (logdet(XtVinvX) + logdet(V) + la.multi_dot([y.T, P, y]))
    else:
        return -0.5 * (logdet(V) + la.multi_dot([y.T, P, y]))


def theory_se(A, P, Var):
    """ Method as described in ``Visscher et al 2014, Statistical Power to detect
        Genetic (Co)Variance of Complex Traits Using SNP Data in Unrelated Samples''.

        I think the formulated this be estimating the true population variance for pair-wise GRM...
        Not sure how well this holds up in real data.
    """
    n, n = A[0].shape
    return 10E5 / n**2


def delta_se(A, P, Var):
    """ Use the delta method to estimate the sample variance
    """
    r = len(A)
    S = np.zeros((r, r))
    N, N = A[0].shape

    # Calculate matrix for standard errors (same as AI matrix but w/out y)
    for i in range(r):
        for j in range(r):
            S[i, j] = np.trace(la.multi_dot([P, A[i], P, A[j]]))

    S = 0.5 * S
    Sinv = la.inv(S)
    
    SE = np.zeros(r)
    xs = ["x{}".format(i) for i in range(r)]
    vars = sympy.Matrix(sympy.symbols(" ".join(xs)))
    sub=dict([(vars[i],Var[i]) for i in range(r)])
    
    for i in range(r):
        x = vars[i]
        exprs = [sympy.diff(vars[i] / sum(vars), x) for x in vars]
        grad = np.array([expr.evalf(subs=sub) for expr in exprs])
        SE[i] = la.multi_dot([grad.T, Sinv / float(N), grad])

    return [np.sqrt(SE), Sinv]


def emREML(A, y, Var, X=None, calc_se=True, bounded=False, max_iter=100, verbose=False):
    """ Computes the ML estimate for variance components via the EM algorithm.
    A = List of variance components (A1, ..., Ak, R)
    y = phenotype
    Var = array of initial estimates
    X = the design matrix for covariates

    if calc_se is true this returns
        var = variance estimates for each component
        se = the sample variance for each component's variance estimate
        sinv = the variance covariance matrix for estimates
    otherwise this returns
        var = variance estimates for each component
    """
    N = float(len(y))

    # Add matrix of residuals to A
    r = len(A)

    y_var = np.var(y, ddof=1)
    Var = y_var * Var

    V = sum(A[i] * Var[i] for i in range(r))

    P, XtVinvX = get_terms(X, V)
    logL = ll(y, X, P, V, XtVinvX)
    if verbose:
        print 'LogLike', 'V(G)', 'V(e)'

    l_dif = 10
    it = 0
    while it < max_iter and ( math.fabs(l_dif) >= 10E-4 or (math.fabs(l_dif) < 10E-2 and l_dif < 0) ):
        for i in range(r):
            vi2 = Var[i]
            vi4 = vi2**2
            Ai = A[i]
            Var[i] = (vi4 * la.multi_dot([y.T, P, Ai, P, y]) + np.trace(vi2 * np.eye(N) - vi4 * P.dot(Ai)) ) / N
        V = sum(A[i] * Var[i] for i in range(r))

        P, XtVinvX = get_terms(X, V)
        new_logL = ll(y, X, P, V, XtVinvX)
        l_dif = new_logL - logL
        logL = new_logL
        it += 1

        if verbose:
            print logL, Var[0], Var[1]

    if not calc_se:
        final = Var
    else:
        SE, Sinv = delta_se(A, P, Var)
        final = [Var, SE, Sinv]

    return final


def aiREML(A, y, Var, X=None, sketch=0, calc_se=True, bounded=False, max_iter=100, verbose=False):
    """ Average Information method for computing the REML estimate of variance components.
    A = List of variance components (A1, ..., Ak, R)
    y = phenotype
    Var = array of initial estimates
    X = the design matrix for covariates
    if calc_se is true this returns
        var = variance estimates for each component
        se = the sample variance for each component's variance estimate
        sinv = the variance covariance matrix for estimates
    otherwise this returns
        var = variance estimates for each component
    """

    N = len(y)
    r = len(A)

    AI = np.zeros((r, r))
    s = np.zeros((r, 1))

    l_dif = 1e100
    it = 0

    y_var = np.var(y, ddof=1)
    Var /= sum(Var)

    Var = y_var * Var

    """
    if X is None:
        X = np.c_[np.ones(N)]
    else:
        X = np.concatenate([X, np.c_[np.ones(N)]], axis=1)
    """

    # Perform a single iteration of EM-based REML to initiate parameters
    V = np.sum(A[i] * Var[i] for i in range(r))

    P, XtVinvX = get_terms(X, V)
    logL = ll(y, X, P, V, XtVinvX)

    for i in range(r):
        vi2 = Var[i]
        vi4 = vi2**2
        Ai = A[i]
        Var[i] = (vi4 * la.multi_dot([y.T, P, Ai, P, y]) + np.trace(vi2 * np.eye(N) - vi4 * P.dot(Ai)) ) / float(N)

    V = sum(A[i] * Var[i] for i in range(r))

    P, XtVinvX = get_terms(X, V)
    logL = ll(y, X, P, V, XtVinvX)
    if verbose:
        print 'LogLike', 'V(G)', 'V(e)'
        print logL, Var[0], Var[1]

    # Iterate AI REML until convergence
    I = np.eye(r)
    Arry = np.concatenate(A, axis=0)
    while it < max_iter and ( math.fabs(l_dif) >= 10E-4 or (math.fabs(l_dif) < 10E-2 and l_dif < 0) ):
        it = it + 1

        ytP = y.T.dot(P)
        # Average information matrix
        if sketch == 0:
            AI = la.multi_dot([np.kron(I, ytP), Arry, P, Arry.T, np.kron(I, ytP).T])
        else:
            L = la.cholesky(P)
            #S = np.random.normal(size=(N, sketch))
            #AI = la.multi_dot([np.kron(I, ytP), Arry, L, S, S.T, L.T, Arry.T, np.kron(I, ytP).T])
            PL = cwt(L.T, sketch).T
            sub = la.multi_dot([np.kron(I, ytP), Arry, PL]) 
            AI = sub.dot(sub.T)

        AI = 0.5 * AI

        # Vector of first derivatives of log likelihood function
        for i in range(r):
            Ai = A[i]
            s[i, 0] = np.trace(P.dot(Ai)) - la.multi_dot([ytP, Ai, ytP.T ])

        s = -0.5 * s

        # New variance components from AI and likelihood
        if l_dif > 1:
            # can't we just do line search here?
            Var_tmp = (Var + 0.316 * la.inv(AI).dot(s).T)[0]
        else:
            Var_tmp = (Var + la.inv(AI).dot(s).T)[0]

        if bounded:
            constrain = np.zeros(r)
            delta = 0.0
            for i in range(r):
                if Var_tmp[i] < 0:
                    delta += y_var * 1e-6 - Var_tmp[i]
                    Var_tmp[i] = y_var * 1e-6
                    constrain[i] = True

            delta /= (r - sum(constrain))
            for i in range(r):
                if not constrain[i] and Var_tmp[i] > delta:
                    Var_tmp[i] -= delta
            if sum(constrain) > r / 2:
                if verbose:
                     print "More than half of the components are constrained! Cannot reliably estimate variance parameters."
                if not calc_se:
                    final = Var
                else:
                    SE, Sinv = delta_se(A, P, Var)
                    final = [Var, SE, Sinv]
                return final

        Var = Var_tmp
        V = sum(A[i] * Var[i] for i in range(r))

        # Re-calculate V and P matrix
        P, XtVinvX = get_terms(X, V)

        # Likelihood of the MLM
        new_logL = ll(y, X, P, V, XtVinvX)
        l_dif = new_logL - logL
        logL = new_logL

        if verbose:
            print logL, Var[0], Var[1]

    final = None
    if not calc_se:
        final = Var
    else:
        SE, Sinv = delta_se(A, P, Var)
        final = [Var, SE, Sinv]

    return final
