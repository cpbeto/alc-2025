import numpy as np
from collections.abc import Iterable
from functools import reduce

# ------------------------------------------------------------
# Utilidades para operar con vectores y matrices
# ------------------------------------------------------------

def vector_unidimensional(v):
    return v.reshape(-1)

def vector_fila(v):
    return v.reshape(1, -1)

def vector_columna(v):
    return v.reshape(-1, 1)

def multiplicar(A, B):
    m_A, n_A = A.shape
    m_B, n_B = B.shape

    if n_A != m_B:
        raise ValueError
        # return None
    
    result = np.zeros((m_A, n_B))
    for i in range(m_A):
        for j in range(n_B):
            suma = 0
            for k in range(m_B):
                suma += A[i, k] * B[k, j]
            result[i, j] = suma
    
    return result

def producto_interno(u, v):
    return multiplicar(vector_fila(u), vector_columna(v)).item()

def producto_exterior(u, v):
    return multiplicar(vector_columna(u), vector_fila(v))

# ------------------------------------------------------------
# Laboratorio 1
# ------------------------------------------------------------

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

# --- Utilidad para comparar resultados en precisión flotante, etc. ---
def escalares_iguales(a, b, rtol=1e-5, atol=1e-8):
    """
    Implementación propia de np.allclose
    con la diferencia que es simétrica respecto de a y b.
    Refs:
      https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
      https://docs.python.org/3/library/math.html#math.isclose
    """ 
    tol = atol + rtol * np.maximum(np.abs(a), np.abs(b))
    return error(a, b) <= tol

def matricesIguales(A, B, rtol=1e-5, atol=1e-8):
    """
    Devuelve True si ambas matrices son iguales y False en otro caso.
    Considerar que las matrices pueden tener distintas dimensiones,
    ademas de distintos valores.
    """
    if np.shape(A) != np.shape(B):
        return False
    
    m, n = np.shape(A)

    for i in range(m):
        for j in range(n):
            if not escalares_iguales(A[i,j], B[i,j], rtol=rtol, atol=atol):
                return False

    return True

# ------------------------------------------------------------
# Laboratorio 2
# ------------------------------------------------------------

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
    La matriz escala la componente i de un vector de Rn
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
    return multiplicar(B, A) # Primero rota, luego escala

def afin(theta, s, b):
    """
    Recibe un ángulo theta, una tira de números s (en R2) y un vector b en R2.
    Retorna una matriz de 3x3 que rota el vector en un ángulo theta,
    luego lo escala en un factor s y por último lo mueve en un valor fijo b.
    """
    b = vector_columna(np.asarray(b))
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
    v = vector_columna(np.asarray(v))
    
    x = np.zeros((3, 1))
    x[0:2, 0:] = v
    x[2, 0] = 1

    w = multiplicar(afin(theta, s, b), x)
    
    return vector_unidimensional(w[0:2, 0:])

# ------------------------------------------------------------
# Laboratorio 4
# ------------------------------------------------------------

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
    U = np.array(A, dtype=np.float64)

    operaciones = 0

    for fila in range(m):
        # Diagonal de L siempre es 1
        L[fila, fila] = 1

        pivote = U[fila, fila]
        # Requiere pivoteo
        if escalares_iguales(pivote, 0):
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

def res_tri(L, b, inferior=True):
    """
    Resuelve el sistema Lx = b, donde L es triangular. Se puede indicar
    si es triangular inferior o superior usando el parámetro inferior
    (por default asumir que es triangular inferior).
    """
    m, n = L.shape
    y = np.zeros(n) 
    if inferior:
        # Resolución de sistema triangular inferior: L y = b
        for fila in range(m):
            suma = 0
            for i in range(fila): # Solo los anteriores a la diagonal
                suma += L[fila, i] * y[i]
            y[fila] = (b[fila] - suma) / L[fila, fila]
    else:
        # Resolución de sistema triangular superior: U y = b
        for fila in range(n-1, -1, -1):
            suma = 0
            for i in range(fila + 1, n): # Sólo los posteriores a la diagonal
                suma += L[fila, i] * y[i]
            y[fila] = (b[fila] - suma) / L[fila, fila]

    return y

def inversa(A):
    """
    Calcula la inversa de A empleando la factorización LU
    y las funciones que resuelven sistemas triangulares.
    """
    m, n = A.shape
    if m != n:
        # raise ValueError
        return None # Solo matrices cuadradas
    
    L, U, _ = calculaLU(A)
    if L is None or U is None:
        # raise ArithmeticError
        return None  # No LU-factorizable

    I = np.eye(n)
    A_inv = np.zeros((m,m))

    for i in range(m):
        e_i = I[:, i]
        # Resolver L y = e_i (sistema triangular inferior)
        y = res_tri(L, e_i, inferior=True)
        # Resolver U x = y (sistema triangular superior)
        x = res_tri(U, y, inferior=False)
        # Guardo la columna x en la matriz inversa.
        A_inv[:, i] = x

    return A_inv

def calculaLDV(A):
    """
    Calcula la factorización LDV de la matriz A, de forma tal que A = LDV,
    con L triangular inferior, D diagonal y V triangular superior. En caso de que
    la matriz no pueda factorizarse retorna None.
    """
    m, n = A.shape
    L = np.eye(m)
    V = np.eye(m)
    D = np.zeros((m, m))
    A = np.array(A, dtype=np.float64)

    for i in range(m):
        if escalares_iguales(A[i, i], 0):
            return None

        D[i, i] = A[i, i]

        # Calcular factores de L y V
        for j in range(i+1, m):
            L[j, i] = A[j, i] / D[i, i]
            V[i, j] = A[i, j] / D[i, i]

        # Actualizar el subbloque restante
        for j in range(i+1, m):
            for k in range(i+1, m):
                A[j, k] = A[j, k] - L[j, i] * D[i, i] * V[i, k]

    return L, D, V

def esSDP(A, atol=1e-8):
    """
    Chequea si la matriz A es simétrica definida positiva (SDP)
    usando la factorización LDV.
    """
    m, n = A.shape
    if n != m:
        # raise ValueError
        return False # Solo matrices cuadradas
        
    # Simetría
    for i in range(m):
        for j in range(n):
            if not escalares_iguales(A[i,j], A[j,i], atol=atol):
                return False
        
    LDV = calculaLDV(A)
    if LDV == None:
        return False

    _, D, _ = LDV
    m, n = D.shape
    for d in range(m):
        if D[d,d] <= atol: # Pivote no positivo
            return False

    return True

# ------------------------------------------------------------
# Laboratorio 3
# ------------------------------------------------------------

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
        x = vector_columna(x)

        y = multiplicar(A, x)
        y = vector_unidimensional(y)

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
    A_inv = inversa(A)
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

    A_inv = inversa(A)
    norma_A = normaExacta(A)[idx]
    norma_A_inv = normaExacta(A_inv)[idx]

    return norma_A * norma_A_inv

# ------------------------------------------------------------
# Laboratorio 5
# ------------------------------------------------------------

def QR_con_GS(A, tol = 1e-12, retorna_nops=False):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R.
    retorna_nops permite (opcionalmente) retornar el número de operaciones realizadas
    Retorna matrices Q y R calculadas con Gram-Schmidt (y como tercer argumento opcional,
    el número de operaciones).
    Si la matriz A no es cuadrada, retorna None.
    """
    m, n = A.shape
    if m != n:
        return None # Solo matrices cuadradas
    
    Q = np.zeros((m, n)) 
    Q[:, 0] = A[:, 0] / norma(A[:, 0], p=2)

    for col in range(1, n):
        Q[:, col] = A[:, col]
        for i in range(col):
            num = producto_interno(Q[:, col], Q[:, i])
            den = producto_interno(Q[:, i], Q[:, i])
            Q[:, col] = Q[:, col] - (num / den) * Q[:, i]
        Q[:, col] = Q[:, col] / norma(Q[:, col], p=2)
    
    R = multiplicar(Q.T, A)
    return Q, R

def QR_con_HH (A, tol=1e-12):
    """
    A una matriz de m x n (m >= n)
    tol la tolerancia con la que se filtran elementos nulos en R.
    Retorna matrices Q y R calculadas con reflexiones de Householder.
    Si la matriz A no cumple m >= n, retorna None.
    """
    m, n = A.shape
    if m < n:
        return None
    
    Q = np.eye(m)
    R = np.array(A, dtype=np.float64)

    for col in range(n):
        # Construyo vector u de la subcolumna
        u = np.zeros(m)
        for i in range(col, m):
            u[i] = R[i][col]
        norma = np.linalg.norm(u)
        if norma < tol:
            continue
        
        signo = np.sign(R[col][col]) if R[col][col] != 0 else 1
        e = np.zeros(m)
        e[col] = 1
        for i in range(m):
            u[i] += signo * norma * e[i]

        uuT = producto_exterior(u, u)
        uTu = producto_interno(u, u)
        H_local = np.eye(m)
        for i in range(m):
            for j in range(m):
                H_local[i][j] -= 2 * uuT[i][j] / uTu

        # Construyo H definitiva
        H = np.eye(m)
        for i in range(col, m):
            for j in range(col, m):
                H[i][j] = H_local[i][j]

        R = multiplicar(H, R)
        Q = multiplicar(Q, H)

    return Q, R

def calculaQR (A, metodo, tol=1e-12, retorna_nops=False):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R.
    metodo = ['RH', 'GS'] usa reflectores de Householder o Gram-Schmidt, respectivamente,
    para realizar la factorización QR.
    Retorna matrices Q y R calculadas con el método indicado (y como tercer argumento opcional,
    el número de operaciones si se usa Gram-Schmidt).
    """
    m, n = A.shape
    if m != n:
        return None # Solo matrices cuadradas
    
    if metodo == 'RH':
        return QR_con_HH(A, tol)
    if metodo == 'GS':
        return QR_con_GS(A, tol)
    else:
        return None
