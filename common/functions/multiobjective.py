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



