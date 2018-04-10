# Multi Objective Functions

# Lis & Eiben (1997)

# Set one (-5 <= x <= 10)

def lis_and_eiben1(x, y):
    return (x**2 + y**2)**(1/8)

def lis_and_eiben2(x):
    return ((x-0.5)**2 + (y-0.5)**2)**1/4

# Set two (-100 <= x <= 100)

def lis_and_eiben3(x):
    if x <= 1:
        return -x
    if x <= 3:
        return -2 + x
    if x <= 4:
        return 4-x
    if x > 4:
        return x - 4

def lis_and_eiben4(x):
    return (x - 5) **2

# Zitzler (1999/2000)

def zitzler1(x):
    def f1(x):
       return x[0]
    def f2(x):
       return g(x) * h(f1(x), g(x))
    def g(x):
        return 1 + 9/29 * sum[x[i] for i in range(1,30)]
    def h(y, z): #f1(x), g(x)
        return 1 - (y/z)**0.5

def zitzler2(x):
    def f1(x):
       return x[0]
    def f2(x):
       return g(x) * h(f1(x), g(x))
    def g(x):
        return 1 + 9/29 * sum[x[i] for i in range(1,30)]
    def h(y, z): #f1(x), g(x)
        return 1 - (y/z)**2
