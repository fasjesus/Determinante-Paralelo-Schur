import numpy as np
import time

def determinante_laplace(matrix):
    """
    Calcula o determinante de uma matriz usando a expansão de Laplace (recursiva).
    """
    # Verifica se a matriz é quadrada
    n = matrix.shape[0]
    if n != matrix.shape[1]:
        raise ValueError("A matriz de entrada deve ser quadrada.")

    # Caso base: se a matriz for 1x1, o determinante é o próprio elemento.
    if n == 1:
        return matrix[0, 0]

    # Caso base: se a matriz for 2x2, calcula diretamente para otimizar.
    if n == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

    # Passo recursivo: expansão pela primeira linha
    total = 0
    for j in range(n):
        # Cria a submatriz removendo a primeira linha e a coluna j
        sub_matrix = np.delete(matrix, 0, axis=0)  # Remove a linha 0
        sub_matrix = np.delete(sub_matrix, j, axis=1) # Remove a coluna j

        # Pega o elemento e calcula o sinal do cofator
        element = matrix[0, j]
        sign = (-1) ** j  # Alterna entre +1 e -1

        # Soma o termo atual ao total
        total += sign * element * determinante_laplace(sub_matrix)

    return total

# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    filename = "matriz.txt"
    print(f"--- Calculando o Determinante pela Forma Convencional (Laplace) ---")
    
    try:
        M = np.loadtxt(filename)
        print(f"Matriz carregada com sucesso do arquivo '{filename}'.\n")
        print("Matriz M (Original):")
        print(M, "\n")
    except Exception as e:
        print(f"Não foi possível ler o arquivo '{filename}'. Erro: {e}")
        exit()

    # --- Cálculo com a implementação de Laplace ---
    print("Calculando...")
    start_time = time.time()
    try:
        det_laplace = determinante_laplace(M)
        end_time = time.time()
        print(f"Resultado do Determinante: {det_laplace}")
        print(f"Tempo de execução: {end_time - start_time:.6f} segundos\n")
    except ValueError as e:
        print(f"Erro: {e}")
        exit()