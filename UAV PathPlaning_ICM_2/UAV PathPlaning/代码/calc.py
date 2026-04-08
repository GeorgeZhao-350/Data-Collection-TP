import math

exam = 78
total = 83

for alpha in range(1, 10):
    alpha *= 0.1
    beta = (1-alpha)
    x = (60 - 50 * alpha) / beta
    print("alpha: %.3f, beta: %.3f, x: %.3f" % (alpha, beta, x))

