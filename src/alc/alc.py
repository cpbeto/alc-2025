import numpy as np

def error(x, y):
    """
    Recibe dos numeros x e y, y calcula el error de aproximar x usando y en float64
    """
    x = np.float64(x)
    y = np.float64(y)

    return np.abs(x - y)

def error_relativo(x, y):
    """
    Recibe dos numeros x e y, y calcula el error relativo de aproximar x usando y en float64
    """
    x = np.float64(x)
    y = np.float64(y)

    return np.abs(x - y) / x


def matricesIguales(A, B, rtol=1e-5, atol=1e-8):
    """
    Devuelve True si ambas matrices son iguales y False en otro caso.
    Considerar que las matrices pueden tener distintas dimensiones, ademas de distintos valores.
    Refs:
      https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
      https://docs.python.org/3/library/math.html#math.isclose
    """
    if np.shape(A) != np.shape(B):
        return False
    
    n, m = np.shape(A)

    for i in range(n):
        for j in range(m):
            # Esta definición es equivalente a la definición de numpy.allclose
            # con la diferencia que es simétrica respecto de A y B
            tol = atol + rtol * np.maximum(np.abs(A[i, j]), np.abs(B[i, j]))
            if error(A[i, j], B[i, j]) > tol:
                return False

    return True