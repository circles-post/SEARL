@mcp.tool(name='quadratic_solver')
def quadratic_solver(a, b, c):
    """
    Solves ax²+bx+c=0
    Solve quadratic equation ax² + bx + c = 0
    Returns roots as numpy array
    Args:
        a,b,c (float)
    
    Returns:
        roots list
    """
    try:
        import numpy as np
        discriminant = b**2 - 4*a*c

        if discriminant > 0:
            root1 = (-b + np.sqrt(discriminant)) / (2*a)
            root2 = (-b - np.sqrt(discriminant)) / (2*a)
            return [root1, root2] # Return list for easier JSON serialization
        elif discriminant == 0:
            root = -b / (2*a)
            return [root, root]
        else:
            # For simplicity in JSON, return string representation of complex numbers or just handled logic
            return "Complex roots not supported in this test JSON output
     except Exception as e:
	    print(e)    
