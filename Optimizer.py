from scipy.optimize import OptimizeResult
import numpy as np
from pvalue import pvalLeft, pvalRight

def custmin(fun, bracket, args=(), 
            maxfev=None, stepsize=1e-3, maxiter=500, callback=None, **options):
    #print("binary search, stepsize = ", 1e-3)
    lower = bracket[0]
    upper = bracket[1]
    
    funcalls = 1
    niter = 0
    
    mid = (lower + upper) / 2.0
    bestx = mid
    besty = fun(mid, *args)
    min_diff = 1e-6
    
    while lower <= upper:
        mid = (lower + upper) / 2.00
        # print("low: ", lower, "up: ", upper)
        # print("mid: ", mid)
        # print("diff: ", fun(mid, *args))
        funcalls += 1
        niter += 1
        if fun(mid, *args) == 0:
            # print("diff = 0")
            besty = fun(mid, *args)
            bestx = mid
            return OptimizeResult(fun=besty, x=bestx, nit=niter,
                          nfev=funcalls, success=(niter > 1))
        elif abs(fun(mid, *args)) <= min_diff:
            # print("diff <= min_diff")
            besty = fun(mid, *args)
            bestx = mid
            return OptimizeResult(fun=besty, x=bestx, nit=niter,
                          nfev=funcalls, success=(niter > 1))
        elif fun(mid, *args) > 0:      # mid > alpha
            # print("diff > 0")
            upper = mid-stepsize
        elif fun(mid, *args) < 0:      # mid < alpha
            # print("diff < 0")
            lower = mid+stepsize
            
    bestx = mid
    besty = fun(mid, *args)
    # print("while loop break")
    # print("low and up: ", lower, upper)
    return OptimizeResult(fun=besty, x=bestx, nit=niter,
                    nfev=funcalls, success=(niter > 1))

    
    
# def custmin(fun, bracket, args=(), maxfev=None, stepsize=0.001,
#         maxiter=500, callback=None, **options):
#     #print("linear search, stepsize = ",1e-3)
#     bestx = (bracket[1] + bracket[0]) / 2.0
#     besty = fun(bestx)
#     funcalls = 1
#     niter = 0
#     improved = True
#     stop = False

#     while improved and not stop and niter < maxiter:
#         improved = False
#         niter += 1
#         for testx in [bestx - stepsize, bestx + stepsize]:
#             testy = fun(testx, *args)
#             funcalls += 1

#             if testy < besty:
#                 besty = testy
#                 bestx = testx
#                 improved = True
#         if callback is not None:
#             callback(bestx)
#         if maxfev is not None and funcalls >= maxfev:
#             stop = True
#             break

#     return OptimizeResult(fun=besty, x=bestx, nit=niter,
#                           nfev=funcalls, success=(niter > 1))