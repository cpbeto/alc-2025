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

def producto_externo(u, v):
    return multiplicar(vector_columna(u), vector_fila(v))

def transpuesta(A):
    m, n = A.shape
    AT = np.zeros((n, m))
    for i in range(m):
        for j in range(n):
            AT[j, i] = A[i, j]
    return AT

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

# --- Utilidad para verificar simetría ---
def simetrica(A, atol=1e-8):
    """
    Chequea si la matriz A es simétrica.
    """
    m, n = A.shape
    if m != n:
        # raise ValueError
        return False # Solo matrices cuadradas
        
    for i in range(m):
        for j in range(n):
            if not escalares_iguales(A[i,j], A[j,i], atol=atol):
                return False

    return True

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
    if not simetrica(A, atol=atol):
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
    x = vector_unidimensional(x)

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

def QR_con_GS(A, atol=1e-12, retorna_nops=False):
    """
    A una matriz de n x n
    atol la tolerancia con la que se filtran elementos nulos en R (no utilizado).
    retorna_nops permite (opcionalmente) retornar el número de operaciones realizadas
    Retorna matrices Q y R calculadas con Gram-Schmidt (y como tercer argumento opcional,
    el número de operaciones).
    Si la matriz A no es cuadrada, retorna None.
    """
    m, n = A.shape
    if m < n:
        # raise ValueError
        return None # Solo matrices cuadradas
    
    operaciones = 0

    Q = np.zeros((m, n))
    R = np.zeros((m, n))

    # Iterar sobre las columnas de A
    for j in range(n):
        v = A[:, j].copy()

        # Ortogonalizar contra las columnas anteriores de Q
        for i in range(j):
            R[i, j] = producto_interno(v, Q[:, i])
            operaciones += 2*m - 1  # Producto interno

            v -= R[i, j] * Q[:, i]
            operaciones += 2*m  # Multiplicación y resta elemento a elemento

        R[j, j] = norma(v, p=2)
        Q[:, j] = v / R[j, j]
        operaciones += m + 2*m  # División elemento a elemento + cálculo de norma
        
    if retorna_nops:
        return Q, R, operaciones
    return Q, R

def QR_con_HH(A, atol=1e-12):
    """
    A una matriz de m x n (m >= n)
    atol la tolerancia con la que se filtran elementos nulos en R.
    Retorna matrices Q y R calculadas con reflexiones de Householder.
    Si la matriz A no cumple m >= n, retorna None.
    """
    m, n = A.shape
    if m < n:
        return None

    Q = np.eye(m)
    A = np.array(A, dtype=np.float64) # Copia modificable de la matriz

    for k in range(n):
        # Construir el reflector de Householder
        z = A[k:m, k]
        v = np.zeros_like(z)
        v[0] = np.sign(z[0]) * norma(z, p=2)
        v += z
        v /= norma(v, p=2)

        # Aplicar la reflexión a cada columna de A y Q
        for j in range(k, n):            
            A[k:m, j] = A[k:m, j] - 2 * producto_interno(v, A[k:m, j]) * v
        for i in range(m):
            Q[i, k:m] = Q[i, k:m] - 2 * producto_interno(v, Q[i, k:m]) * vector_fila(v)

    return Q, A

def calculaQR(A, metodo, atol=1e-12, retorna_nops=False):
    """
    A una matriz de n x n
    atol la tolerancia con la que se filtran elementos nulos en R.
    metodo = ['RH', 'GS'] usa reflectores de Householder o Gram-Schmidt, respectivamente,
    para realizar la factorización QR.
    Retorna matrices Q y R calculadas con el método indicado (y como tercer argumento opcional,
    el número de operaciones si se usa Gram-Schmidt).
    """
    m, n = A.shape
    if m != n:
        return None # Solo matrices cuadradas
    
    if metodo == 'RH':
        return QR_con_HH(A, atol)
    if metodo == 'GS':
        return QR_con_GS(A, atol)
    else:
        return None

# ------------------------------------------------------------
# Laboratorio 6
# ------------------------------------------------------------

def metpot2k(A: np.array, atol=1e-15, K=1000):
    """
    A una matriz de n x n
    atol la tolerancia en la diferencia entre un paso y el siguiente
    de la estimación del autovector.
    K el número máximo de iteraciones a realizarse.
    Retorna el autovector v, el autovalor lambda y número de iteraciones realizadas.
    """
    m, n = A.shape
    if m != n:
        return None # Solo matrices cuadradas
    
    # Genero un vector aleatorio de norma 1
    u = np.random.rand(m)
    u = vector_columna(u) / norma(u, p=2)
    
    iteraciones = 0
    lambda1 = 0

    for _ in range(K):
        v = multiplicar(A, u)
        v = v / norma(v, p=2)

        # Chequeo de convergencia
        if norma(v - u, p=2) < atol:
            break

        u = v
        iteraciones += 1

    # Calculo lambda con el último v obtenido
    Av = multiplicar(A, v)
    lambda1 = producto_interno(v, Av)

    return v, lambda1, iteraciones

def diagRH(A: np.array, atol=1e-15, K=1000):
    """
    A una matriz simétrica de n x n
    atol la tolerancia en la diferencia entre un paso y el siguiente
    de la estimación del autovector.
    K el número máximo de iteraciones a realizarse.
    Retorna matriz de autovectores S y matriz de autovalores D, tal que A = S D S.T.
    Si la matriz A no es simétrica, retorna None.
    """
    m, n = A.shape
    
    if not simetrica(A, atol=atol):
        return None

    # Caso base n = 1
    if m == 1:
        return np.eye(1), A

    # Obtener autovector y autovalor dominante
    v1, lambda1, _ = metpot2k(A, atol, K)

    # Construir vector columna e1
    e1 = np.zeros((m, 1))
    e1[0, 0] = 1

    # Construir la reflexión de Householder
    u = e1 - v1
    if norma(u, p=2) < atol:
        Hv1 = np.eye(n)
    else:
        u = u / norma(u, p=2)
        Hv1 = np.eye(n) - 2 * producto_externo(u, u)

    # Caso base n = 2
    if n == 2:
        S = Hv1
        D = multiplicar(multiplicar(transpuesta(S), A), S)
        return S, D

    # Reducir el problema
    B = multiplicar(multiplicar(transpuesta(Hv1), A), Hv1)
    A_tilde = B[1:, 1:]

    # Paso recursivo
    S_tilde, D_tilde = diagRH(A_tilde, atol, K)

    # Reconstruir D
    D = np.zeros((n, n))
    D[0, 0] = lambda1
    D[1:, 1:] = D_tilde

    # Reconstruir S
    S = np.eye(n)
    S[1:, 1:] = S_tilde
    S = multiplicar(Hv1, S)

    # Verificar tolerancia
    return S, D

# ------------------------------------------------------------
# Laboratorio 7
# ------------------------------------------------------------

def transiciones_al_azar_continuas(n):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas y con entradas
    al azar en el intervalo [0, 1).
    """
    T = np.random.rand(n, n)
    for j in range(n):
        suma = 0
        for i in range(n):
            suma += T[i, j]
        for i in range(n):
            T[i, j] /= suma
    return T

def transiciones_al_azar_uniforme(n, threshold):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    threshold probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas.
    El elemento i,j es distinto de cero si el número generado al azar
    es menor o igual a threshold.
    Todos los elementos de la columna j son iguales y suman 1.
    """
    T = np.random.rand(n, n)
    for j in range(n):
        for i in range(n):
            T[i, j] = 1 if T[i, j] <= threshold else 0
        suma = 0
        for i in range(n):
            suma += T[i, j]

        if suma == 0:
            # Evitar columnas nulas
            for i in range(n):
                T[i, j] = 1 / n
        else:
            for i in range(n):
                T[i, j] /= suma
    return T

def nucleo(A, atol=1e-15):
    """
    A una matriz de m x n
    atol la tolerancia para considerar que un vector está en el núcleo.
    Calcula el núcleo de la matriz A diagonilazado A.T @ A, usando el método diagRH.
    El núcleo corresponde a los autovectores de autovalor <= atol.
    Retorna los autovectores como una matriz de n x k, siendo k la dimensión del núcleo.
    """
    m, n = A.shape

    # Generar matriz cuadrada
    cuadrada = multiplicar(transpuesta(A), A)

    # Aplicar diagonalización
    S, D = diagRH(cuadrada, atol, K=10000)

    autovectores = [S[:,i] for i in range(n) if D[i, i] <= atol]
    k = len(autovectores)

    result = np.zeros((n, k))
    for i in range(k):
        result[:, i] = autovectores[i]

    return result

def crea_rala(listado, m, n, atol=1e-15):
    """
    Recibe listado, con tres elementos: lista con índices i, lista con índices j
    y lista con valores A_ij de la matriz A.
    Recibe las dimensiones de la matriz resultante m x n.
    Los elementos menores a atol se descartan.
    Idealmente, el listado debe incluir únicamente posiciones
    correspondientes a valores distintos de cero.
    Retorna una lista con:
        - Diccionario {(i, j): A_ij} con los valores no nulos de la matriz rala.
        - Tupla (m, n) con las dimensiones de la matriz rala.
    """
    if not listado:
        listado = [[], [], []]
        
    filas, columnas, valores = listado

    A = {}
    for i, j, Aij in zip(filas, columnas, valores):
        if np.abs(Aij) > atol:
            A[(i, j)] = Aij

    return A, (m, n)

def multiplica_rala_vector(A, v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v.
    Retorna un vector w resultado de A @ v.
    """
    (m, n) = A[1]
    w = np.zeros(m)

    for (i, j), Aij in A[0].items():
        w[i] += Aij * v[j]

    return w

# ------------------------------------------------------------
# Laboratorio 8
# ------------------------------------------------------------

def svd_reducida(A, k='max', atol=1e-15):
    """
    A una matriz de m x n
    k el número de componentes singulares a calcular (por default 'max' calcula el máximo posible)
    atol la tolerancia para considerar un valor singular igual a cero.
    Retorna las matrices hatU (m x k), hatSigma (vector de k valores singulares)
    y hatV (n x k).
    """
    m, n = A.shape

    # Generar matriz cuadrada
    cuadrada = multiplicar(transpuesta(A), A)
    
    # Aplicar una diagonalización para obtener autovaloes y autovectores asociados:
    S, D = diagRH(cuadrada, atol, K=10000)

    # Cuadrada es simétrica, por lo tanto, ortogonalmente diagonalizable. S es ortogonal.
    # ... pero los autovalores no están ordenados por sus valores absolutos.

    # Armar lista de tuplas ordenadas (lambda, v), descartando autovalores nulos
    tuplas = [(S[:,i], D[i, i]) for i in range(n) if D[i, i] > atol]
    tuplas = sorted(tuplas, key=lambda x: x[1], reverse=True)

    L = len(tuplas) if k == 'max' else min(k, len(tuplas))

    # Armar matrices V y Sigma de tamaño L x L
    VT = np.zeros((L, n))
    Sigma = np.zeros(L)

    for i in range (L):
        Sigma[i] = np.sqrt(tuplas[i][1])
        VT[i] = tuplas[i][0]
    V = transpuesta(VT)
    
    # Calcular la matriz U sabiendo que A*v = s*u => u = (A*v) / s

    U = np.zeros((m, L))
    for i in range(L):
        if Sigma[i] > atol:
            Av = multiplicar(A, vector_columna(V[:, i]))
            U[:, i] = vector_unidimensional(Av / Sigma[i])

    return U, Sigma, V