import numpy as np
from mpi4py import MPI
import sys
import time 

# ======================================================
# Cálculo de determinante de matriz com Schur
# ======================================================

# ======================================================================================================================================
# REMOVIDO: A função is_power_of_two() não é mais necessária:
# Para o método de Schur a única condição necessária é que a matriz possa ser dividida em quatro submatrizes quadradas de tamanho igual.
# Ele funciona para qualquer matriz quadrada que seja dividida em blocos, desde que o bloco A seja inversível. 

# REMOVIDO: o sys(0) da verificação if np.isclose(detS, 0):
# O programa apenas faz uma verificação de se o det(M) é 0 e determina se a matriz é singular ou não.
# ======================================================================================================================================

# Exec: mpiexec -n 3 python determinante_paralelo_geral.py

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

    # Verificação inicial do número mínimo de processos
    if size < 2:
        print("ERRO: Este programa requer pelo menos 2 processos (1 coordenador e 1+ trabalhadores).", flush=True)
        comm.Abort()

    print(f"\nExecutando com {size} processos.\n")

    filename = "matriz.txt"
    try:
        M = np.loadtxt(filename)
        print(f"Matriz carregada com sucesso do arquivo '{filename}'.\n")
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{filename}' nao foi encontrado.", flush=True)
        comm.Abort()
    except Exception as e:
        print(f"ERRO: Ocorreu um erro ao ler o arquivo '{filename}': {e}", flush=True)
        comm.Abort()
    
    print_matrix(M, "M (Original)")
    
    n = M.shape[0]
    
    # Verifica se é quadrada E se a dimensão é um número par.
    if M.shape[0] != M.shape[1] or n % 2 != 0:
        print("ERRO: A matriz deve ser quadrada e sua dimensao (N) deve ser um número par.", flush=True)
        comm.Abort()

    n2 = n // 2

    # VERIFICAÇÃO DE PROCESSOS EXCEDENTES 
    num_workers = size - 1
    if num_workers != n2:
        print("ERRO: Configuracao de processos invalida para o tamanho do problema.", flush=True)
        print(f"      A quantidade de trabalhadores ({num_workers}) deve ser igual a dimensao da submatriz ({n2}).", flush=True)
        print(f"      Por favor, execute novamente com {n2 + 1} processos (mpiexec -n {n2 + 1} ...).", flush=True)
        comm.Abort()

    # 1. Dividir M em blocos
    A = M[:n2, :n2]
    B = M[:n2, n2:]
    C = M[n2:, :n2]
    D = M[n2:, n2:]

    print_matrix(A, "A")
    print_matrix(B, "B")
    print_matrix(C, "C")
    print_matrix(D, "D")
    
    # 2. Calcular det(A) e A⁻¹
    try:
        detA = np.linalg.det(A)
        if np.isclose(detA, 0):
            raise np.linalg.LinAlgError("Matriz A singular")
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError as e:
        print(f"ERRO: A submatriz A eh singular (determinante proximo de zero), impossivel continuar.", flush=True)
        comm.Abort()

    print(f"det(A) = {detA:.2f}\n")
    print_matrix(A_inv, "A inversa")
    
    print("Iniciando calculo paralelo...")
    start_time = time.perf_counter()

    # 3. Distribuir o cálculo de T
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

    # 5. Calcular o Complemento de Schur
    S = D - T
    print_matrix(S, "S (D - T)")
    
    detS = np.linalg.det(S)
    detM = detA * detS
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"det(S) = {detS:.2f}\n")
    print("------------------------------------------------------------------------------------")
    print(f"Tempo total de paralelismo: {elapsed_time:.6f} segundos")
    print("------------------------------------------------------------------------------------")
    print("Resultado Final (det(A) * det(S))")
    print(f"det(M) = {detA:.2f} * {detS:.2f} = {detM:.2f}")
    print("------------------------------------------------------------------------------------")
    print(f"det(M) pelo numpy = {np.linalg.det(M):.2f}")
    print("------------------------------------------------------------------------------------")
    
    # Verificação final sobre o resultado
    if np.isclose(detM, 0):
        print("VERIFICACAO FINAL: determinante da matriz M eh nulo (matriz singular).")
    else:
        print("VERIFICACAO FINAL: determinante da matriz M diferente de zero (matriz nao singular).")
    print("------------------------------------------------------------------------------------")

# --- Lógica dos Processos Trabalhadores ---
else:
    # Como o mestre aborta antes de enviar qualquer mensagem se a configuração for inválida,
    # os trabalhadores serão encerrados pelo comm.Abort() sem receberem tarefas.
    
    data_chunk = comm.recv(source=0, tag=1)
    C_chunk = data_chunk['c_chunk']
    indices = data_chunk['indices']
    
    common_data = comm.bcast(None, root=0)
    A_inv = common_data['A_inv']
    B = common_data['B']
    
    T_partial = C_chunk @ A_inv @ B
    
    comm.send({'t_partial': T_partial, 'indices': indices}, dest=0, tag=2)