import pytest
import numpy as np
from alc.alc import error, error_relativo, matricesIguales
from alc.alc import rota, escala, rota_y_escala, afin, trans_afin
from alc.alc import norma, normaliza, normaMatMC, normaExacta, condMC, condExacto

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