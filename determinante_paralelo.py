# determinante_paralelo.py

import numpy as np
from mpi4py import MPI
import sys

# Verificar se um número é potência de 2
def is_power_of_two(n):
    """Verifica se um número inteiro n é uma potência de 2."""
    # Um número n é potência de 2 se for > 0 e a operação bitwise (n & (n-1)) for 0.
    if n <= 0:
        return False
    return (n & (n - 1)) == 0

def print_matrix(mat, name, precision=2):
    """Função auxiliar para imprimir uma matriz NumPy de forma legível."""
    with np.printoptions(precision=precision, suppress=True):
        print(f"Matriz {name}:\n{mat}\n")

# --- Inicialização do MPI ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- Lógica do Processo Raiz (Coordenador) ---
if rank == 0:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except TypeError:
        pass

    if size < 2:
        print("Erro: Este programa requer pelo menos 2 processos (1 coordenador e 1+ trabalhadores).")
        comm.Abort()

    print(f"Executando com {size} processos.\n")

    filename = "matriz.txt"
    try:
        M = np.loadtxt(filename)
        print(f"Matriz carregada com sucesso do arquivo '{filename}'.\n")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{filename}' não foi encontrado.")
        comm.Abort()
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo '{filename}': {e}")
        comm.Abort()
    
    print_matrix(M, "M (Original)")
    
    n = M.shape[0]
    
    # Verifica se é quadrada E se a dimensão é uma potência de 2.
    if M.shape[0] != M.shape[1] or not is_power_of_two(n):
        print("Erro: A matriz deve ser quadrada e sua dimensão (N) deve ser uma potência de 2 (2, 4, 8, etc.).")
        comm.Abort()

    n2 = n // 2

    # 1. Dividir M em blocos A, B, C, D
    A = M[:n2, :n2]
    B = M[:n2, n2:]
    C = M[n2:, :n2]
    D = M[n2:, n2:]

    print_matrix(A, "A")
    print_matrix(B, "B")
    print_matrix(C, "C")
    print_matrix(D, "D")
    
    # 2. Calcular det(A) e A⁻¹ (Verificação de A inversível)
    try:
        detA = np.linalg.det(A)
        if np.isclose(detA, 0):
             raise np.linalg.LinAlgError("Matriz A singular")
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError as e:
        print(f"Erro: A matriz A eh singular (determinante proximo de zero), impossivel continuar.")
        comm.Abort()

    print(f"det(A) = {detA:.2f}\n")
    print_matrix(A_inv, "A inversa")

    # 3. Distribuir o cálculo de T
    num_workers = size - 1
    rows_to_calculate = np.array_split(np.arange(n2), num_workers)
    
    for i in range(num_workers):
        worker_rank = i + 1
        indices = rows_to_calculate[i]
        C_chunk = C[indices, :]
        comm.send({'c_chunk': C_chunk, 'indices': indices}, dest=worker_rank, tag=1)
    
    comm.bcast({'A_inv': A_inv, 'B': B}, root=0)

    # 4. Coletar os resultados
    T = np.zeros((n2, n2))
    for i in range(num_workers):
        worker_rank = i + 1
        result_data = comm.recv(source=worker_rank, tag=2)
        T_partial = result_data['t_partial']
        indices = result_data['indices']
        T[indices, :] = T_partial

    print_matrix(T, "T (calculado C @ A inversa @ B)")

    # 5. Calcular o Complemento de Schur e o determinante final
    S = D - T
    print_matrix(S, "S (D - T)")
    detS = np.linalg.det(S)
    detM = detA * detS

    print(f"det(S) = {detS:.2f}\n")
    print("------------------------------------------")
    print("Resultado Final (det(A) * det(S))")
    print(f"det(M) = {detA:.2f} * {detS:.2f} = {detM:.2f}")
    print("------------------------------------------")

    # Verificação do determinante de M
    if np.isclose(detM, 0):
        print("VERIFICATION FINAL: determinante da matriz M nulo (matriz singular).")
    else:
        print("VERIFICATION FINAL: determinante da matriz M diferente de zero (matriz != singular).")
    print("------------------------------------------")

# --- Lógica dos Processos Trabalhadores ---
else:
    data_chunk = comm.recv(source=0, tag=1)
    C_chunk = data_chunk['c_chunk']
    indices = data_chunk['indices']
    
    common_data = comm.bcast(None, root=0)
    A_inv = common_data['A_inv']
    B = common_data['B']
    
    T_partial = C_chunk @ A_inv @ B
    
    comm.send({'t_partial': T_partial, 'indices': indices}, dest=0, tag=2)