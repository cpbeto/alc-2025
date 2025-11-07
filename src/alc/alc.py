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

def rota(theta):
    """
    Recibe un ángulo theta y retorna una matriz de 2x2
    que rota un vector dado en un ángulo theta
    """
    A = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return A

def escala(s):
    """
    Recibe una tira de números s y retorna una matriz cuadrada de
    n x n, donde n es el tamaño de s.
    La matriz escala la componenete i de un vector de Rn
    en un factor s[i].
    """
    n = len(s)
    
    A = np.zeros((n,n))
    for i in range(n):
        A[i, i] = s[i]

    return A

def rota_y_escala(theta, s):
    """
    Recibe un ángulo theta y una tira de números s
    y retorna una matriz de 2x2 que rota el vector en un ángulo theta
    y luego lo escala en un factor s.
    """
    A = rota(theta)
    B = escala(s)
    return B@A # Primero rota, luego escala

def afin(theta, s, b):
    """
    Recibe un ángulo theta, una tira de números s (en R2) y un vector b en R2.
    Retorna una matriz de 3x3 que rota el vector en un ángulo theta,
    luego lo escala en un factor s y por último lo mueve en un valor fijo b.
    """
    b = np.asarray(b).reshape(-1, 1)
    C = np.zeros((3, 3))

    C[0:2, 0:2] = rota_y_escala(theta, s)
    C[0:2, 2:] = b
    C[2, 2] = 1

    return C

def trans_afin(v, theta, s, b):
    """
    Recibe un vector v (en R2), un ángulo theta,
    una tira de números s (en R2) y un vector b en R2.
    Retorna el vector 2 resultante de aplicar la transformación afín a v.
    """
    v = np.asarray(v).reshape(-1, 1)
    
    x = np.zeros((3, 1))
    x[0:2, 0:] = v
    x[2, 0] = 1

    w = afin(theta, s, b) @ x
    
    return w[0:2, 0:].reshape(1, -1)