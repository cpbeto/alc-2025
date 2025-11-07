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


def matricesIguales(A, B):
    """
    Devuelve True si ambas matrices son iguales y False en otro caso.
    Considerar que las matrices pueden tener distintas dimensiones, ademas de distintos valores.
    """
    if np.shape(A) != np.shape(B):
        return False
    
    n, m = np.shape(A)

    for i in range(n):
        for j in range(m):
            if not np.allclose(A[i, j], B[i, j]):
                return False

    return True