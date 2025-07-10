# determinante_paralelo.py

import numpy as np
from mpi4py import MPI
import sys # Usado para garantir a codificação correta da saída

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
    # Garante que o console do Windows exiba caracteres especiais corretamente
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except TypeError:
        # Em alguns ambientes (não-Windows), essa reconfiguração pode falhar.
        # Não é um problema crítico, então apenas ignoramos o erro.
        pass

    if size < 2:
        print("Erro: Este programa requer pelo menos 2 processos (1 coordenador e 1+ trabalhadores).")
        comm.Abort()

    print(f"Executando com {size} processos.\n")

    # ***** INÍCIO DA ALTERAÇÃO *****
    # Define o nome do arquivo a ser lido
    filename = "matriz.txt"
    try:
        # Carrega a matriz do arquivo de texto
        M = np.loadtxt(filename)
        print(f"Matriz carregada com sucesso do arquivo '{filename}'.\n")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{filename}' não foi encontrado.")
        print("Por favor, crie o arquivo com a matriz ou verifique o nome e o local.")
        comm.Abort()
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo '{filename}': {e}")
        comm.Abort()
    # ***** FIM DA ALTERAÇÃO *****
    
    print_matrix(M, "M (Original)")
    
    n = M.shape[0]
    # Verifica se a matriz é quadrada
    if M.shape[0] != M.shape[1] or n % 2 != 0:
        print("Erro: A matriz deve ser quadrada e ter uma dimensão par (2x2, 4x4, 6x6, etc.).")
        comm.Abort()

    n2 = n // 2 # Tamanho das submatrizes

    # 1. Dividir M em blocos A, B, C, D
    A = M[:n2, :n2]
    B = M[:n2, n2:]
    C = M[n2:, :n2]
    D = M[n2:, n2:]

    print_matrix(A, "A")
    print_matrix(B, "B")
    print_matrix(C, "C")
    print_matrix(D, "D")
    
    # 2. Calcular det(A) e A⁻¹ (serialmente no processo raiz)
    try:
        detA = np.linalg.det(A)
        if np.isclose(detA, 0):
             raise np.linalg.LinAlgError("Matriz A é singular")
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError as e:
        print(f"Erro: A matriz A é singular (determinante próximo de zero). Não é possível continuar.")
        print(e)
        comm.Abort()

    print(f"det(A) = {detA:.2f}\n")
    
    # Usando a representação correta da inversa
    print_matrix(A_inv, "A⁻¹")

    # 3. Distribuir o cálculo de T = C @ A⁻¹ @ B para os trabalhadores
    num_workers = size - 1
    rows_to_calculate = np.array_split(np.arange(n2), num_workers)
    
    for i in range(num_workers):
        worker_rank = i + 1
        indices = rows_to_calculate[i]
        C_chunk = C[indices, :]
        comm.send({'c_chunk': C_chunk, 'indices': indices}, dest=worker_rank, tag=1)
    
    comm.bcast({'A_inv': A_inv, 'B': B}, root=0)

    # 4. Coletar os resultados dos trabalhadores
    T = np.zeros((n2, n2))
    for i in range(num_workers):
        worker_rank = i + 1
        result_data = comm.recv(source=worker_rank, tag=2)
        T_partial = result_data['t_partial']
        indices = result_data['indices']
        T[indices, :] = T_partial

    print_matrix(T, "T (calculado C @ A⁻¹ @ B)")

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


# --- Lógica dos Processos Trabalhadores ---
else:
    # A lógica dos trabalhadores não precisa ser alterada,
    # pois eles recebem os dados do processo raiz.
    data_chunk = comm.recv(source=0, tag=1)
    C_chunk = data_chunk['c_chunk']
    indices = data_chunk['indices']
    
    common_data = comm.bcast(None, root=0)
    A_inv = common_data['A_inv']
    B = common_data['B']
    
    T_partial = C_chunk @ A_inv @ B
    
    comm.send({'t_partial': T_partial, 'indices': indices}, dest=0, tag=2)