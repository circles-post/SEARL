from sympy import symbols, solve

# Define variables
a, b, c, d, e, f, g, h, i = symbols('a b c d e f g h i', positive=True, integer=True)

# Equation: abc + def + ghi = 1665
abc = 100*a + 10*b + c
def2 = 100*d + 10*e + f
ghi = 100*g + 10*h + i
result = abc + def2 + ghi
result == 1665

# Solve the equation
values = solve(result == 1665, (a, b, c, d, e, f, g, h, i))
values