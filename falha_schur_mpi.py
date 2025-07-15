from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
tamanho = comm.Get_size()

def ler_matriz_de_arquivo(caminho):
    with open(caminho, 'r') as arquivo:
        linhas = arquivo.readlines()
        matriz = [list(map(float, linha.replace(';', ' ').split())) for linha in linhas]
    return np.array(matriz, dtype=np.float64)

def multiplicar_matrizes_paralelo(A, B):
    """
    Multiplica matrizes A e B em paralelo dividindo A por linhas entre os processos.
    """
    if rank == 0:
        n, m = A.shape
        p = B.shape[1]
    else:
        n = m = p = None

    n = comm.bcast(n, root=0)
    m = comm.bcast(m, root=0)
    p = comm.bcast(p, root=0)

    linhas_por_processo = n // tamanho
    sobras = n % tamanho
    contagens = [linhas_por_processo + 1 if i < sobras else linhas_por_processo for i in range(tamanho)]
    deslocamentos = [sum(contagens[:i]) for i in range(tamanho)]

    parte_local_A = np.empty((contagens[rank], m), dtype=np.float64)
    if rank == 0:
        comm.Scatterv([A.ravel(), np.array(contagens) * m, np.array(deslocamentos) * m, MPI.DOUBLE], parte_local_A.ravel(), root=0)
    else:
        comm.Scatterv([None, np.array(contagens) * m, np.array(deslocamentos) * m, MPI.DOUBLE], parte_local_A.ravel(), root=0)

    B = comm.bcast(B if rank == 0 else None, root=0)

    parte_local_resultado = np.dot(parte_local_A, B)

    if rank == 0:
        resultado = np.empty((n, p), dtype=np.float64)
    else:
        resultado = None

    comm.Gatherv(parte_local_resultado.ravel(), [resultado.ravel() if rank == 0 else None, np.array(contagens) * p, np.array(deslocamentos) * p, MPI.DOUBLE], root=0)

    return resultado

def inverter_matriz_paralelo(matriz):
    """
    Calcula a inversa de uma matriz dividindo a matriz identidade entre os processos,
    cada processo resolve o sistema linear para suas colunas, e o resultado é reunido.
    """
    n = matriz.shape[0]

    if rank == 0:
        identidade = np.eye(n, dtype=np.float64)
    else:
        identidade = None

    # Todos recebem a matriz completa
    matriz_local = matriz.copy() if rank == 0 else np.empty((n, n), dtype=np.float64)
    comm.Bcast(matriz_local, root=0)

    # Dividir as colunas da identidade entre os processos
    colunas_por_processo = n // tamanho
    sobras = n % tamanho
    contagens = [colunas_por_processo + 1 if i < sobras else colunas_por_processo for i in range(tamanho)]
    deslocamentos = [sum(contagens[:i]) for i in range(tamanho)]

    parte_local_identidade = np.empty((n, contagens[rank]), dtype=np.float64)

    if rank == 0:
        comm.Scatterv([identidade.ravel(), np.array(contagens) * n, np.array(deslocamentos) * n, MPI.DOUBLE],
                      parte_local_identidade.ravel(), root=0)
    else:
        comm.Scatterv([None, np.array(contagens) * n, np.array(deslocamentos) * n, MPI.DOUBLE],
                      parte_local_identidade.ravel(), root=0)

    # Cada processo resolve o sistema linear
    parte_local_inversa = np.linalg.solve(matriz_local, parte_local_identidade)

    # Reunir as partes da inversa no processo 0
    if rank == 0:
        inversa = np.empty((n, n), dtype=np.float64)
    else:
        inversa = None

    comm.Gatherv(parte_local_inversa.ravel(),
                 [inversa.ravel() if rank == 0 else None, np.array(contagens) * n, np.array(deslocamentos) * n, MPI.DOUBLE],
                 root=0)

    return inversa

def determinante_via_schur(M):
    n = M.shape[0]
    meio = n // 2

    A = M[:meio, :meio].copy()
    B = M[:meio, meio:].copy()
    C = M[meio:, :meio].copy()
    D = M[meio:, meio:].copy()

    if rank == 0:
        print("\nSubmatriz A:\n", A)
        print("Submatriz B:\n", B)
        print("Submatriz C:\n", C)
        print("Submatriz D:\n", D)
        print("\nCalculando a inversa de A...")

    inversa_A = inverter_matriz_paralelo(A)

    if rank == 0:
        print("\nInversa de A:\n", inversa_A)
        print("\nCalculando C * A_inv...")
    CA_inv = multiplicar_matrizes_paralelo(C, inversa_A)

    if rank == 0:
        print("\nC * A_inv:\n", CA_inv)
        print("\nCalculando C * A_inv * B...")
    CA_inv_B = multiplicar_matrizes_paralelo(CA_inv, B)

    if rank == 0:
        print("\nC * A_inv * B:\n", CA_inv_B)
        schur = D - CA_inv_B
        print("\nComplemento de Schur (S = D - C * A_inv * B):\n", schur)

        det_A = np.linalg.det(A)
        det_S = np.linalg.det(schur)
        det_M = det_A * det_S

        print(f"\ndet(A) = {det_A}")
        print(f"det(S) = {det_S}")
        print(f"det(M) pela fórmula de Schur = {det_M}")
        print(f"det(M) pelo numpy = {np.linalg.det(M)}")

        return det_M
    else:
        return None


if __name__ == "__main__":
    if rank == 0:
        try:
            matriz_M = ler_matriz_de_arquivo("matriz.txt")
            print("Matriz M (lida do arquivo):\n", matriz_M)
            if matriz_M.shape[0] != matriz_M.shape[1]:
                raise ValueError("Matriz deve ser quadrada")
            if matriz_M.shape[0] % 2 != 0:
                raise ValueError("Ordem da matriz deve ser par")
        except Exception as e:
            print("Erro lendo matriz:", e)
            matriz_M = None
    else:
        matriz_M = None

    matriz_M = comm.bcast(matriz_M, root=0)

    determinante_via_schur(matriz_M)

''' 
Matriz M (lida do arquivo):
 [[2. 3. 1. 0.]
 [1. 4. 2. 1.]
 [5. 1. 3. 2.]
 [0. 2. 1. 1.]]

Submatriz A:
 [[2. 3.]
 [1. 4.]]
Submatriz B:
 [[1. 0.]
 [2. 1.]]
Submatriz C:
 [[5. 1.]
 [0. 2.]]
Submatriz D:
 [[3. 2.]
 [1. 1.]]

Calculando a inversa de A...

Inversa de A:
 [[ 0.8 -0.2]
 [-0.6  0.4]]

Calculando C * A_inv...

C * A_inv:
 [[ 3.4 -0.6]
 [-1.2  0.8]]

Calculando C * A_inv * B...

C * A_inv * B:
 [[ 2.2 -0.6]
 [ 0.4  0.8]]

Complemento de Schur (S = D - C * A_inv * B):
 [[0.8 2.6]
 [0.6 0.2]]

det(A) = 5.000000000000001
det(S) = -1.4000000000000004
det(M) pela f¾rmula de Schur = -7.000000000000003
det(M) pelo numpy = 8.999999999999998 
'''