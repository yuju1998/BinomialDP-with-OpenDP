from scipy.optimize import OptimizeResult
import numpy as np

def custmin(fun, bracket, args=(), maxfev=None, stepsize=0.01,
        maxiter=500, callback=None, **options):
    bestx = (bracket[1] + bracket[0]) / 2.0
    besty = fun(bestx)
    funcalls = 1
    niter = 0
    improved = True
    stop = False

    while improved and not stop and niter < maxiter:
        improved = False
        niter += 1
        for testx in [bestx - stepsize, bestx + stepsize]:
            testy = fun(testx, *args)
            funcalls += 1

            if testy < besty:
                besty = testy
                bestx = testx
                improved = True
        if callback is not None:
            callback(bestx)
        if maxfev is not None and funcalls >= maxfev:
            stop = True
            break

    return OptimizeResult(fun=besty, x=bestx, nit=niter,
                          nfev=funcalls, success=(niter > 1))