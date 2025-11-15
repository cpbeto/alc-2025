import numpy as np
from collections.abc import Iterable
from functools import reduce

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
    
    return w[0:2, 0:].reshape(-1)

def norma(x, p):
    """
    Devuelve la norma p del vector x.
    """
    n, = x.shape
    p_inf = (p == 'inf')
    p = p if not p_inf else 1

    # Se necesita casteo explícito acá
    x = x.astype(np.float64)

    for i in range(n):
        x[i] = np.power(np.abs(x[i]), p)

    if p_inf:
        max = 0
        for i in range(n):
            if x[i] > max:
                max = x[i]
        return max
    
    sum = 0
    for i in range(n):
        sum += x[i]

    return np.power(sum, 1/p)

def normaliza(X, p):
    """
    Recibe X, una lista de vectores no vacíos y un escalar p.
    Devuelve una lista donde cada elemento corresponde a normalizar
    los elementos de X con la norma p.
    """
    result = []

    for i in range(len(X)):
        n, = X[i].shape
        norma_x = norma(X[i], p)

        y = np.zeros_like(X[i], dtype=np.float64)

        for j in range(n):
            y[j] = X[i][j] / norma_x

        result.append(y)
    
    return result

def normaMatMC(A, q, p, Np):
    """
    Devuelve la norma ||A||(q,p) y el vector x en el cual se alcanza
    el máximo.
    """
    _, n = A.shape

    x_max = None
    norma_max = 0
    for _ in range(Np):
        x = normaliza([np.random.rand(n)], p)[0]
        x = x.reshape(-1, 1)

        y = A@x
        y = y.reshape(-1)

        norma_A_x = norma(y, q)

        if norma_A_x > norma_max:
            norma_max = norma_A_x
            x_max = x
    
    return norma_max, x_max

def normaExacta(A, p=[1, 'inf']):
    """
    Devuelve una lista con las normas 1 e infinito de una matriz A,
    usando las expresiones del enunciado (2.c).
    """
    if p == 1:
        p = [1]
    elif p == 'inf':
        p = ['inf']
    elif not isinstance(p, Iterable) or list(p) != [1, 'inf']:
        return None
    
    m, n = A.shape
    
    norma_1 = None
    if 1 in p:
        max = 0
        for j in range(n):
            sum = 0
            for i in range(m):
                sum += np.abs(A[i, j])

            if sum > max:
                max = sum

        norma_1 = max

    norma_inf = None
    if 'inf' in p:
        max = 0
        for i in range(m):
            sum = 0
            for j in range(n):
                sum += np.abs(A[i, j])

            if sum > max:
                max = sum

        norma_inf = max

    if p == [1]:
        return norma_1
    elif p == ['inf']:
        return norma_inf

    return norma_1, norma_inf

def condMC(A, p, Np=10000):
    """
    Devuelve el número de condici+on de A usando la norma inducida p.
    """
    A_inv = np.linalg.inv(A)
    norma_A = normaMatMC(A, p, p, Np)
    norma_A_inv = normaMatMC(A_inv, p, p, Np)

    return norma_A[0] * norma_A_inv[0]

def condExacto(A, p):
    """
    Que devuelve el número de condición de A a partir de la fórmula
    de la ecuación (1) usando la norma p.
    """
    if p not in (1, 'inf'):
        return None
    
    idx = 0 if p == 1 else 1

    A_inv = np.linalg.inv(A)
    norma_A = normaExacta(A)[idx]
    norma_A_inv = normaExacta(A_inv)[idx]

    return norma_A * norma_A_inv

def calculaLU(A):
    """
    Calcula la factorización LU de la matriz A y retorna las matrices L y U,
    junto con el número de operaciones realizadas. En caso de que la matriz
    no pueda factorizarse retorna None.
    """
    if A is None:
        return None, None, 0
    
    m, n = A.shape
    if m != n:
        # raise NotImplementedError
        return None, None, 0 # Solo matrices cuadradas
    
    L = np.zeros((m, m))
    U = A.copy()

    operaciones = 0

    for fila in range(m):
        # Diagonal de L siempre es 1
        L[fila, fila] = 1

        pivote = U[fila, fila]
        # Requiere pivoteo
        if np.allclose(np.abs(pivote), 0):
            # raise NotImplementedError
            return None, None, 0

        for i in range(fila+1, m):
            factor = U[i, fila] / pivote
            operaciones += 1

            L[i, fila] = factor
            U[i, fila] = 0 # Elemento debajo del pivote queda en 0
            for j in range(fila+1, m):
                U[i, j] -= factor * U[fila, j]
                operaciones += 2
                
    return L, U, operaciones