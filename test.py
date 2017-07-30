#! /usr/bin/env python
import argparse as ap
import os
import time
import sys

import numpy as np
import reml


def main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("-o", "--output", type=ap.FileType("w"), default=sys.stdout)

    args = argp.parse_args(args)

    n = 1000 # sample size
    p = 100  # snps
    h2g = np.random.beta(10, 10) # mean = 0.5

    X = np.random.normal(size=(n, 2))

    Z = np.random.binomial(n=2, p=0.1, size=(n, p)).astype(float)
    Z -= Z.mean(axis=0)
    Z /= Z.std(axis=0)

    beta = np.random.normal(0, np.sqrt(h2g / p), size=p)
    alpha = np.random.normal(0, 1, size=2)
    G = Z.dot(beta)
    s2g = G.var(ddof=1)
    s2e = s2g * ((1 / h2g) - 1)
    e = np.random.normal(0, np.sqrt(s2e), size=n)

    #y = X.dot(alpha) + G + e
    y = G + e

    y -= y.mean()
    y /= y.std()

    A = Z.dot(Z.T) / float(p)

    #res = reml.aiREML([A, np.eye(n)], y, np.array([0.5, 0.5]), X=X, verbose=True)
    t0 = time.time()
    res = reml.aiREML([A, np.eye(n)], y, np.array([0.5, 0.5]), verbose=True)
    t1 = time.time() - t0

    #res = reml.aiREML([A, np.eye(n)], y, np.array([0.5, 0.5]), X=X, verbose=True, sketch=500)
    t2 = time.time()
    res = reml.aiREML([A, np.eye(n)], y, np.array([0.5, 0.5]), verbose=True, sketch=500)
    t3 = time.time() - t2

    print t1, t3

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
