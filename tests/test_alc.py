import numpy as np
from alc.alc import error, error_relativo, matricesIguales
from alc.alc import rota, escala, rota_y_escala, afin, trans_afin
from alc.alc import norma, normaliza, normaMatMC, normaExacta, condMC, condExacto
from alc.alc import calculaLU, res_tri, inversa, calculaLDV, esSDP
from alc.alc import QR_con_GS, QR_con_HH, calculaQR
from alc.alc import metpot2k, diagRH
from alc.alc import transiciones_al_azar_continuas, transiciones_al_azar_uniforme, nucleo, crea_rala, multiplica_rala_vector
from alc.alc import svd_reducida

# ------------------------------------------------------------
# Laboratorio 1
# ------------------------------------------------------------

def test_error():
    def sonIguales(x, y, atol=1e-8):
        return np.allclose(error(x,y), 0, atol=atol)

    assert(not sonIguales(1, 1.1))
    assert(sonIguales(1, 1 + np.finfo('float64').eps))
    assert(not sonIguales(1, 1 + np.finfo('float32').eps))
    assert(not sonIguales(np.float16(1), np.float16(1) + np.finfo('float32').eps))
    assert(sonIguales(np.float16(1), np.float16(1) + np.finfo('float16').eps, atol=1e-3))

    assert(np.allclose(error_relativo(1, 1.1), 0.1))
    assert(np.allclose(error_relativo(2,1), 0.5))
    assert(np.allclose(error_relativo(-1,-1), 0))
    assert(np.allclose(error_relativo(1,-1), 2))

def test_matrices_iguales():
    assert(matricesIguales(np.diag([1,1]), np.eye(2)))
    assert(matricesIguales(np.linalg.inv(np.array([[1,2],[3,4]])) @ np.array([[1,2],[3,4]]), np.eye(2)))
    assert(not matricesIguales(np.array([[1,2],[3,4]]).T, np.array([[1,2],[3,4]])))

# ------------------------------------------------------------
# Laboratorio 2
# ------------------------------------------------------------

def test_theta():
    assert(np.allclose(rota(0), np.eye(2)))
    assert(np.allclose(rota(np.pi/2), np.array([[0,-1],[1,0]])))
    assert(np.allclose(rota(np.pi), np.array([[-1,0],[0,-1]])))

def test_escala():
    assert(np.allclose(escala([2, 3]), np.array([[2,0],[0,3]])))
    assert(np.allclose(escala([1, 1, 1]), np.eye(3)))
    assert(np.allclose(escala([0.5, 0.25]), np.array([[0.5,0],[0,0.25]])))

def test_rota_y_escala():
    assert(np.allclose(rota_y_escala(0, [2, 3]), np.array([[2,0], [0,3]])))
    assert(np.allclose(rota_y_escala(np.pi/2, [1, 1]), np.array([[0,-1], [1,0]])))
    assert(np.allclose(rota_y_escala(np.pi, [2, 2]), np.array([[-2,0], [0,-2]])))
    assert(np.allclose(rota_y_escala(np.pi/2, [3, 2]), np.array([[0,-3], [2,0]])))

def test_afin():
    assert(np.allclose(
        afin(0, [1,1], [1,2]),
        np.array([
            [1, 0, 1],
            [0, 1, 2],
            [0, 0, 1]
        ])
    ))

    assert(np.allclose(
        afin(np.pi/2, [1,1], [0,0]),
        np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    ))

    assert(np.allclose(
        afin(0, [2,3], [1,1]),
        np.array([
            [2, 0, 1],
            [0, 3, 1],
            [0, 0, 1]
        ])
    ))

def test_trans_afin():
    assert(np.allclose(
        trans_afin(np.array([1,0]), np.pi/2, [1,1], [0,0]),
        np.array([0,1])
    ))

    assert(np.allclose(
        trans_afin(np.array([1,1]), 0, [2,3], [0,0]),
        np.array([2,3])
    ))
    
    assert(np.allclose(
        trans_afin(np.array([1,0]), np.pi/2, [3,2], [4,5]),
        np.array([4,7])
    ))

# ------------------------------------------------------------
# Laboratorio 3
# ------------------------------------------------------------

def test_norma():
    assert(np.allclose(norma(np.array([0,0,0,0]), 1), 0))
    assert(np.allclose(norma(np.array([4,3,-100,-41,0]), 'inf'), 100))
    assert(np.allclose(norma(np.array([1,1]), 2),np.sqrt(2)))
    assert(np.allclose(norma(np.array([1]*10), 2), np.sqrt(10)))
    
    # Rigorous dtype test
    assert(np.allclose(norma(np.array([2,2]), 1.5), np.power(2 * np.power(2, np.float64(3)/2), np.float64(2)/3)))

def test_normaliza():
    # result = normaliza([np.array([0,0,0,0])], 2)
    # assert(len(result) == 1)
    # assert(np.allclose(result[0], np.array([0,0,0,0])))

    result = normaliza([np.array([1]*k) for k in range(1,11)], 2)
    assert(len(result) != 0)
    for x in result:
        assert(np.allclose(norma(x, 2), 1))

    result = normaliza([np.array([1]*k) for k in range(2,11)], 1)
    assert(len(result) != 0)
    for x in result:
        assert(np.allclose(norma(x, 1), 1))

    result = normaliza([np.random.rand(k) for k in range(1,11)], 'inf')
    assert(len(result) != 0)
    for x in result:
        assert(np.allclose(norma(x, 'inf'), 1))

def test_normaMatMC():
    result = normaMatMC(A=np.eye(2), q=2, p=1, Np=10000)
    assert(np.allclose(result[0], 1, atol=1e-3))
    assert(np.allclose(np.abs(result[1][0]), 1, atol=1e-3) or np.allclose(np.abs(result[1][1]), 1, atol=1e-3))
    assert(np.allclose(np.abs(result[1][0]), 0, atol=1e-3) or np.allclose(np.abs(result[1][1]), 0, atol=1e-3))

    result = normaMatMC(A=np.eye(2), q=2, p='inf', Np=10000)
    assert(np.allclose(result[0],np.sqrt(2), atol=1e-3))
    assert(np.allclose(np.abs(result[1][0]), 1, atol=1e-3) and np.allclose(np.abs(result[1][1]), 1, atol=1e-3))

    A = np.array([[1,2],[3,4]])
    result = normaMatMC(A=A, q='inf', p='inf', Np=10000)
    assert(np.allclose(result[0], normaExacta(A)[1], rtol=1e-1))

def test_normaExacta():
    # Default tuple behavior
    assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]))[0], 2))
    assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]))[1], 2))
    assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]))[0], 6))
    assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]))[1], 7))
    
    # Single value behavior
    assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]), 1), 6))
    assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]), 'inf'), 7))
    assert(normaExacta(np.array([[1,-2],[-3,-4]]), 2) is None)    

def test_condMC():
    A = np.array([[1,1],[0,1]])
    A_inv = np.linalg.solve(A, np.eye(A.shape[0]))
    norma_A = normaMatMC(A, 2, 2, 10000)
    norma_A_inv = normaMatMC(A_inv, 2, 2, 10000)
    condA = condMC(A, 2)
    assert(np.allclose(norma_A[0]*norma_A_inv[0], condA, atol=1e-2))

    A = np.array([[3,2],[4,1]])
    A_inv = np.linalg.solve(A, np.eye(A.shape[0]))
    norma_A = normaMatMC(A, 2, 2, 10000)
    norma_A_inv = normaMatMC(A_inv, 2, 2, 10000)
    condA = condMC(A, 2)
    assert(np.allclose(norma_A[0]*norma_A_inv[0], condA, atol=1e-2))

def test_condExacto():
    A = np.random.rand(10,10)
    A_inv = np.linalg.solve(A,np.eye(A.shape[0]))
    norma_A = normaExacta(A)[0]
    norma_A_inv = normaExacta(A_inv)[0]
    condA = condExacto(A,1)
    assert(np.allclose(norma_A*norma_A_inv, condA))

    A = np.random.rand(10,10)
    A_inv = np.linalg.solve(A,np.eye(A.shape[0]))
    norma_A = normaExacta(A)[1]
    norma_A_inv = normaExacta(A_inv)[1]
    condA = condExacto(A,'inf')
    assert(np.allclose(norma_A*norma_A_inv, condA))

# ------------------------------------------------------------
# Laboratorio 4
# ------------------------------------------------------------

def test_calculaLU():
    L0 = np.array([[1,0,0],
                [0,1,0],
                [1,1,1]])

    U0 = np.array([[10,1,0],
                [0,2,1],
                [0,0,1]])
    A =  L0 @ U0
    L, U, _ = calculaLU(A)
    assert(np.allclose(L, L0))
    assert(np.allclose(U, U0))


    L0 = np.array([[1,0,0],
                [1,1.001,0],
                [1,1,1]])

    U0 = np.array([[1,1,1],
                [0,1,1],
                [0,0,1]])
    A =  L0 @ U0
    L, U, n_ops = calculaLU(A)
    assert(not np.allclose(L, L0))
    assert(not np.allclose(U, U0))
    assert(np.allclose(L, L0, atol=1e-3))
    assert(np.allclose(U, U0, atol=1e-3))
    assert(n_ops == 13)

    L0 = np.array([[1,0,0],
                [1,1,0],
                [1,1,1]])

    U0 = np.array([[1,1,1],
                [0,0,1],
                [0,0,1]])

    A =  L0 @ U0
    L, U, nops = calculaLU(A)
    assert(L is None)
    assert(U is None)
    assert(nops == 0)

    assert(calculaLU(None) == (None, None, 0))

    assert(calculaLU(np.array([[1,2,3],[4,5,6]])) == (None, None, 0))

def test_res_tri():
    A = np.array([
        [1,0,0],
        [1,1,0],
        [1,1,1]
    ])

    b = np.array([1,1,1])
    assert(np.allclose(res_tri(A, b), np.array([1,0,0])))

    b = np.array([0,1,0])
    assert(np.allclose(res_tri(A, b), np.array([0,1,-1])))

    b = np.array([-1,1,-1])
    assert(np.allclose(res_tri(A, b), np.array([-1,2,-2])))

    b = np.array([-1,1,-1])
    assert(np.allclose(res_tri(A, b, inferior=False), np.array([-1,1,-1])))

    A = np.array([[3,2,1], [0,2,1], [0,0,1]])
    b = np.array([3,2,1])
    assert(np.allclose(res_tri(A, b, inferior=False), np.array([1/3,1/2,1])))

    A = np.array([[1,-1,1], [0,1,-1], [0,0,1]])
    b = np.array([1,0,1])
    assert(np.allclose(res_tri(A, b, inferior=False), np.array([1,1,1])))

def test_inversa():
    def esSingular(A):
        try:
            np.linalg.inv(A)
            return False
        except:
            return True

    # No siempre es invertible entonces hacemos varios tests
    for i in range(30):
        A = np.random.random((4,4))
        A_inv = inversa(A)
        if not esSingular(A):
            A_inv_np = np.linalg.inv(A)
            assert(A_inv is not None)
            assert(np.allclose(A_inv_np, A_inv))
        else: 
            assert(A_inv is None)

    # Matriz singular devería devolver None
    A = np.array([[1,2,3],[4,5,6],[7,8,9]])
    assert(inversa(A) is None)

def test_calculaLDV():
    L0 = np.array([[1,0,0],[1,1.,0],[1,1,1]])
    D0 = np.diag([1,2,3])
    V0 = np.array([[1,1,1],[0,1,1],[0,0,1]])
    A =  L0 @ D0 @ V0
    L,D,V = calculaLDV(A)
    assert(np.allclose(L, L0))
    assert(np.allclose(D, D0))
    assert(np.allclose(V, V0))


    L0 = np.array([[1,0,0],[1,1.001,0],[1,1,1]])
    D0 = np.diag([3,2,1])
    V0 = np.array([[1,1,1],[0,1,1],[0,0,1.001]])
    A =  L0 @ D0 @ V0
    L,D,V = calculaLDV(A)
    assert(np.allclose(L, L0, 1e-3))
    assert(np.allclose(D, D0, 1e-3))
    assert(np.allclose(V, V0, 1e-3))

def test_esSDP():
    L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
    D0 = np.diag([1,1,1])
    A = L0 @ D0 @ L0.T
    assert(esSDP(A))

    D0 = np.diag([1,-1,1])
    A = L0 @ D0 @ L0.T
    assert(not esSDP(A))

    D0 = np.diag([1,1,1e-16])
    A = L0 @ D0 @ L0.T
    assert(not esSDP(A))

    L0 = np.array([
        [1,0,0],
        [1,1,0],
        [1,1,1]
    ])
    D0 = np.diag([1,1,1])
    V0 = np.array([
        [1,0,0],
        [1,1,0],
        [1,1+1e-3,1]
    ]).T
    A = L0 @ D0 @ V0
    assert(esSDP(A, 1e-3))

# ------------------------------------------------------------
# Laboratorio 5
# ------------------------------------------------------------

A2 = np.array([
    [1., 2.],
    [3., 4.]
])

A3 = np.array([
    [1., 0., 1.],
    [0., 1., 1.],
    [1., 1., 0.]
])

A4 = np.array([
    [2., 0., 1., 3.],
    [0., 1., 4., 1.],
    [1., 0., 2., 0.],
    [3., 1., 0., 2.]
])

def check_QR(Q, R, A, atol=1e-10):
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=atol)
    assert np.allclose(Q @ R, A, atol=atol)
    
def test_QR_con_GS():
    Q2, R2 = QR_con_GS(A2)
    check_QR(Q2, R2, A2)

    Q3, R3 = QR_con_GS(A3)
    check_QR(Q3, R3, A3)

    Q4, R4 = QR_con_GS(A4)
    check_QR(Q4, R4, A4)

def test_QR_con_HH():
    Q2, R2 = QR_con_HH(A2)
    check_QR(Q2, R2, A2)

    Q3, R3 = QR_con_HH(A3)
    check_QR(Q3, R3, A3)

    Q4, R4 = QR_con_HH(A4)
    check_QR(Q4, R4, A4)

def test_calculaQR():
    Q2, R2 = calculaQR(A2, metodo='RH')
    check_QR(Q2, R2, A2)

    Q3, R3 = calculaQR(A3, metodo='GS')
    check_QR(Q3, R3, A3)

    Q4, R4 = calculaQR(A4, metodo='RH')
    check_QR(Q4, R4, A4)

# ------------------------------------------------------------
# Laboratorio 6
# ------------------------------------------------------------

def test_metpot2k():
    S = np.vstack([
        np.array([2,1,0])/np.sqrt(5),
        np.array([-1,2,5])/np.sqrt(30),
        np.array([1,-2,1])/np.sqrt(6)
    ]).T

    # Tasa de muestreo (30 es el número mágico según alguna literatura)
    N = 30

    # Pedimos que pase el 95% de los casos
    exitos = 0
    for i in range(N):
        D = np.diag(np.random.random(3) + 1) * 100
        A = S@D@S.T
        _, lambda1, _ = metpot2k(A, 1e-15, 10000)
        if np.allclose(lambda1, np.max(D), atol=1e-8):
            exitos += 1
    assert exitos/N > 0.95


    # Test con HH
    exitos = 0
    for i in range(N):
        v = np.random.rand(9)
        idx = np.argsort(-np.abs(v))
        D = np.diag(v[idx]) # Matriz diagonal con autovalores en orden decreciente
        I = np.eye(9)
        H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2) # Matriz de HouseHolder

        A = H@D@H.T
        _, lambda1, _ = metpot2k(A, 1e-15, 10000)
        if np.allclose(lambda1, D[0,0]):
            exitos += 1
    assert exitos/N > 0.95

def test_diagRH():
    D = np.diag([1,0.5,0.25])
    S = np.vstack([
        np.array([1,-1,1])/np.sqrt(3),
        np.array([1,1,0])/np.sqrt(2),
        np.array([1,-1,-2])/np.sqrt(6)
    ]).T

    A = S@D@S.T
    SRH, DRH = diagRH(A, atol=1e-15, K=10000)
    assert np.allclose(D,DRH)
    assert np.allclose(np.abs(S.T@SRH), np.eye(A.shape[0]), atol=1e-7)

    # Tasa de muestreo (30 es el número mágico según alguna literatura)
    N = 30

    # Pedimos que pase el 95% de los casos
    exitos = 0
    for i in range(N):
        A = np.random.random((5,5))
        A = 0.5*(A+A.T)
        S,D = diagRH(A, atol=1e-15, K=10000)
        ARH = S@D@S.T
        e = normaExacta(ARH-A, p='inf')
        if e < 1e-5: 
            exitos += 1
    assert exitos/N >= 0.95

# ------------------------------------------------------------
# Laboratorio 7
# ------------------------------------------------------------

def es_markov(T, atol=1e-6):
    m, n = T.shape
    if m != n:
        return False
    
    for i in range(n):
        for j in range(n):
            if T[i,j] < 0:
                return False
    
    for j in range(n):
        suma = sum(T[:,j])
        if not np.allclose(suma, 1, atol=atol):
            return False
        
    return True

def es_markov_uniforme(T, atol=1e-6):
    """
    T una matriz cuadrada.
    atol la tolerancia para asumir que una entrada es igual a cero.
    Retorna True sii T es una matriz de transición de Markov uniforme
    (entradas iguales a cero o iguales entre si en cada columna, y columnas que suman 1 dentro de la tolerancia).
    """
    if not es_markov(T, atol):
        return False
    
    # Cada columna debe tener entradas iguales entre si o iguales a cero
    m, n = T.shape
    for j in range(n):
        non_zero = T[:,j][T[:,j] > atol]
        # All close
        close = all(np.abs(non_zero - non_zero[0]) < atol)
        if not close:
            return False
    return True

def test_transiciones_al_azar_continuas():
    # Tasa de muestreo (30 es el número mágico según alguna literatura)
    N = 30

    for n in range(N):
        T = transiciones_al_azar_continuas(n)
        assert es_markov(T)

def test_transiciones_al_azar_uniforme():
    # Tasa de muestreo (30 es el número mágico según alguna literatura)
    N = 30

    for n in range(N):
        T = transiciones_al_azar_uniforme(n, threshold=0.3)
        print(T)
        assert es_markov_uniforme(T)
        T = transiciones_al_azar_uniforme(n, threshold=0.01)
        assert es_markov_uniforme(T)

def esNucleo(A, S, atol=1e-5):
    for col in S.T:
        res = A @ col
        if not np.allclose(res, np.zeros(A.shape[0]), atol=atol):
            return False
    return True

def test_nucleo():
    A = np.eye(3)
    S = nucleo(A)
    assert S.shape == (3,0)

    A[1,1] = 0
    S = nucleo(A)
    assert esNucleo(A,S)
    assert S.shape == (3,1)
    assert abs(S[2,0]) < 1e-2
    assert abs(S[0,0]) < 1e-2

    v = np.random.random(5)
    v = v / np.linalg.norm(v)
    H = np.eye(5) - np.outer(v, v) # Proyección ortogonal
    S = nucleo(H)
    assert S.shape == (5,1)
    v_gen = S[:,0]
    v_gen = v_gen / np.linalg.norm(v_gen)
    assert np.allclose(v, v_gen) or np.allclose(v, -v_gen)

def test_crea_rala():
    listado = [[0,17],[3,4],[0.5,0.25]]
    A_rala, dims = crea_rala(listado, 32, 89)
    assert dims == (32,89)
    assert A_rala[(0,3)] == 0.5
    assert A_rala[(17,4)] == 0.25
    assert len(A_rala) == 2

    listado = [[32,16,5],[3,4,7],[7,0.5,0.25]]
    A_rala, dims = crea_rala(listado, 50, 50)
    assert dims == (50,50)
    assert A_rala.get((32,3)) == 7
    assert A_rala[(16,4)] == 0.5
    assert A_rala[(5,7)] == 0.25

    listado = [[1,2,3],[4,5,6],[1e-20,0.5,0.25]]
    A_rala, dims = crea_rala(listado, 10, 10)
    assert dims == (10,10)
    assert (1,4) not in A_rala
    assert A_rala[(2,5)] == 0.5
    assert A_rala[(3,6)] == 0.25
    assert len(A_rala) == 2

    # caso borde: lista vacia. Esto es una matriz de 0s
    listado = []
    A_rala, dims = crea_rala(listado, 10, 10)
    assert dims == (10,10)
    assert len(A_rala) == 0

def test_multiplica_rala_vector():
    listado = [[0,1,2],[0,1,2],[1,2,3]]
    A_rala = crea_rala(listado, 3, 3)
    v = np.random.random(3)
    v = v / np.linalg.norm(v)
    res = multiplica_rala_vector(A_rala, v)
    A = np.array([[1,0,0],[0,2,0],[0,0,3]])
    res_esperado = A @ v
    assert np.allclose(res, res_esperado)

    A = np.random.random((5,5))
    A = A * (A > 0.5)
    listado = [[],[],[]]
    for i in range(5):
        for j in range(5):
            listado[0].append(i)
            listado[1].append(j)
            listado[2].append(A[i,j])
            
    A_rala = crea_rala(listado,5,5)
    v = np.random.random(5)
    assert np.allclose(multiplica_rala_vector(A_rala,v), A @ v)

# ------------------------------------------------------------
# Laboratorio 8
# ------------------------------------------------------------

def test_svd_reducida():
    def genera_matriz_para_test(m, n=2, dimension_nucleo=0):
        if dimension_nucleo == 0:
            A = np.random.random((m, n))
        else:
            A = np.random.random((m, dimension_nucleo))
            A = np.hstack([A,A])
        return(A)

    def test_svd_reducida_mn(A, atol=1e-12):
        m, n = A.shape
        hU, hS, hV = svd_reducida(A, atol=atol)
        nU, nS, nVT = np.linalg.svd(A)

        r = len(hS) + 1
        assert np.all(np.abs(np.abs(np.diag(hU.T @ nU))-1) < (10**r)*atol), 'Revisar calculo de hat U en ' + str((m,n))
        assert np.all(np.abs(np.abs(np.diag(nVT @ hV))-1) < (10**r)*atol), 'Revisar calculo de hat V en ' + str((m,n))
        assert len(hS) == len(nS[np.abs(nS) > atol]), 'Hay cantidades distintas de valores singulares en ' + str((m,n))
        assert np.all(np.abs(hS-nS[np.abs(nS) > atol]) < (10**r)*atol), 'Hay diferencias en los valores singulares en ' + str((m,n))

    # Tasa de muestreo (30 es el número mágico según alguna literatura)
    N = 5 # El tiempo de test se dispara con N>1

    for m in [2,5,10,20]:
        for n in [2,5,10,20]:
            for _ in range(N):
                A = genera_matriz_para_test(m, n)
                test_svd_reducida_mn(A)


    # Matrices con núcleo
    m = 12
    for dimension_nucleo in [2,4,6]:
        for _ in range(10):
            A = genera_matriz_para_test(m, dimension_nucleo=dimension_nucleo)
            test_svd_reducida_mn(A)

    # Tamaños de las reducidas
    A = np.random.random((8,6))
    for k in [1,3,5]:
        hU, hS, hV = svd_reducida(A, k=k)
        assert hU.shape[0] == A.shape[0], 'Dimensiones de hU incorrectas (caso a)'
        assert hV.shape[0] == A.shape[1], 'Dimensiones de hV incorrectas(caso a)'
        assert hU.shape[1] == k, 'Dimensiones de hU incorrectas (caso a)'
        assert hV.shape[1] == k, 'Dimensiones de hV incorrectas(caso a)'
        assert len(hS) == k, 'Tamaño de hS incorrecto'