import pytest
import numpy as np
from alc.alc import error, error_relativo, matricesIguales
from alc.alc import rota, escala, rota_y_escala, afin, trans_afin

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