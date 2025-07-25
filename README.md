# Calculadora de Determinante de Matrizes em Paralelo - 
## Documentação para o script determinante_paralelo.py que implementa Schur paralelo com alguns requisitos.

Este projeto contém um script em Python que calcula o determinante de uma matriz quadrada de forma paralela, utilizando a biblioteca MPI (Message Passing Interface).

## Para que serve?

O objetivo principal deste programa é demonstrar o conceito de computação paralela aplicado a um problema matemático. Ele implementa o cálculo do determinante utilizando a fórmula do **Complemento de Schur**, que permite dividir a matriz em blocos e distribuir o trabalho computacional entre múltiplos processos.

As principais características são:
- **Leitura de Arquivo**: A matriz de entrada é lida de um arquivo de texto (`matriz.txt`), tornando o programa flexível.
- **Processamento Paralelo**: Utiliza a biblioteca `mpi4py` para orquestrar a comunicação e a divisão de tarefas entre um processo "coordenador" e múltiplos processos "trabalhadores".
- **Algoritmo Eficiente**: A parte mais custosa do cálculo (multiplicação de matrizes) é distribuída, acelerando o processo em comparação com uma execução puramente sequencial.

**Restrição**: O algoritmo implementado funciona com matrizes quadradas de dimensão par (ex: 2x2, 4x4, 6x6, etc.) e com dimensão N sendo potência de 2.

## Requisitos

Para executar este programa, você precisará ter os seguintes softwares instalados:

1.  **Python**: Versão 3.7 ou superior.
2.  **Implementação de MPI**:
    - **Windows**: [Microsoft MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
    - **Linux/macOS**: [OpenMPI](https://www.open-mpi.org/) ou [MPICH](https://www.mpich.org/)
3.  **Bibliotecas Python**: `NumPy` e `mpi4py`. Você pode instalá-las com o seguinte comando:
    ```bash
    pip install numpy mpi4py
    ```
## Observação

O arquivo determinante-laplace.py calcula o determinante da matriz de forma convencional e serve apenas como objeto de comparação para os resultados. Para executar, escreva no terminal:
```bash
    python determinante_laplace.py
```
## Como Executar

Siga os passos abaixo para configurar e rodar o programa.

### 1. Preparar os Arquivos

Certifique-se de que os dois arquivos a seguir estão na mesma pasta:
- `determinante_paralelo.py` (o script principal)
- `matriz.txt` (o arquivo com a matriz de entrada)

### 2. Configurar a Matriz de Entrada

Edite o arquivo `matriz.txt` para conter a matriz que você deseja calcular. O formato é simples:
- Os números em cada linha são separados por espaços.
- Cada linha do arquivo representa uma linha da matriz.

**Exemplo de `matriz.txt` para uma matriz 4x4:**
```bash
2 3 1 0
1 4 2 1
5 1 3 2
0 2 1 1
```

### 3. Executar o Comando

Abra um terminal (CMD, PowerShell, Terminal do Linux, etc.) e navegue até a pasta onde salvou os arquivos. Use o seguinte comando para executar o script:

```bash
mpiexec -n <numero_de_processos> python determinante_paralelo.py
```
Onde:

<numero_de_processos> é o número total de processos deve ser igual a {1 (coordenador) + ds (dimensão das submatrizes)}, sendo ds = N / 2 (N é a dimensão da matriz).

Exemplo prático com 3 processos (1 coordenador e 2 trabalhadores):

```bash
mpiexec -n 3 python determinante_paralelo.py
```

### 4. Resultado Esperado

```bash

Executando com 3 processos.

Matriz carregada com sucesso do arquivo 'matriz.txt'.

Matriz M (Original):
[[2. 1. 1. 0.]
 [0. 1. 0. 1.]
 [3. 0. 5. 2.]
 [0. 4. 1. 3.]]

Matriz A:
[[2. 1.]
 [0. 1.]]

Matriz B:
[[1. 0.]
 [0. 1.]]

Matriz C:
[[3. 0.]
 [0. 4.]]

Matriz D:
[[5. 2.]
 [1. 3.]]

det(A) = 2.00

Matriz A inversa:
[[ 0.5 -0.5]
 [ 0.   1. ]]

Iniciando calculo paralelo...
Matriz T (calculado C @ A inversa @ B):
[[ 1.5 -1.5]
 [ 0.   4. ]]

Matriz S (D - T):
[[ 3.5  3.5]
 [ 1.  -1. ]]

det(S) = -7.00

------------------------------------------------------------------------------------
Tempo total de paralelismo: 0.002110 segundos
------------------------------------------------------------------------------------
Resultado Final (det(A) * det(S))
det(M) = 2.00 * -7.00 = -14.00
------------------------------------------------------------------------------------
det(M) pelo numpy = -14.00
------------------------------------------------------------------------------------
VERIFICACAO FINAL: determinante da matriz M diferente de zero (matriz nao singular).
------------------------------------------------------------------------------------
```

rank = 2:

C_chunk = [[13, 14]]
indices = [1]
A_inv = [[-1.5, 0.5], [1.25, -0.25]]
B = [[3, 4], [7, 8]]

T_partial = [[13, 14]] @ [[-1.5, 0.5], [1.25, -0.25]] @ [[3, 4], [7, 8]]

T_partial = [[15, 16]]

{'t_partial': [[15, 16]], 'indices': [1]}
