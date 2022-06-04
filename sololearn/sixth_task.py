import math

w1, w2, b, x1, x2 = [float(x) for x in input().split()]


x = w1*x1 + w2*x2 + b

def f (x):
    return 1 / (1 + math.exp(-x))


print(round(f(x), 4))