import networkx as nx
import numpy as np
import pandas as pd
import math
import queue
import time


class Point(tuple):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __new__(self, x, y):
        return tuple.__new__(self, (x, y))

    def __str__(self):
        print('[' + str(self.x) + ', ' + str(self.y) + ']')

## calcular distancias
def euclidean(p1, p2):  
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

def manhattan(p1, p2):
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)

def generate_points(lower= -10, upper= 10, size = 4):
    '''
    Funcao para gerar os pontos
    '''
    points = []
    for i in range(2**size):
        r1 = np.random.randint(lower, upper) 
        r2 = np.random.randint(lower, upper)
        points.append(Point(r1, r2))
    return points

def fun_cost(G, order):
    '''
    Funcao utilizada para calcular o custo de um caminho no grafo
    '''
    i = 0
    custo = 0
    while (i < len(order)-1):
        custo += (G.get_edge_data(order[i], order[i+1]))['weight']
        i += 1
    return custo


def create_graph(points):
    '''
    Funcao utilizada para criar grafos completos com distancia euclidiana e distancia de manhattan
    '''
    G_euc = nx.Graph()
    G_man = nx.Graph()
    for i in range(len(points)):
        for j in range(len(points)):
            if (i != j):
                d_euc = euclidean(points[i], points[j])
                d_man = manhattan(points[i], points[j])
                G_euc.add_edge(i,j,weight=d_euc)
                G_man.add_edge(i,j,weight=d_man)
            
    return G_euc, G_man


class Node:
    '''
    Classe utilizada para criar os nodes do heap utilizado para o branch-and-bound
    '''
    def __init__(self, bound, level, cost, S):
        self.bound = bound
        self.level = level
        self.s = S 
        self.cost = cost 

    # Comparação entre Nodes
    def __lt__(self, other):
        return self.bound < other.bound


def fun_bound(G, V):
    '''
    Funcao que calcula a estimativa para cada node
    
    G: grafo original
    V: arestas que devem fazer parte do subproblema

    Return: estimativa (bound)

    '''

    cost = 0
    e2 = 0
    for i in G.nodes():
        edges = [(G.get_edge_data(j, k))['weight'] for j, k in G.edges(i)]
        edges = sorted(edges)

        if i in V and len(V) > 1:   #analisa arestas que ja devem fazer parte da solucao
            for j in range(len(V)):
                if V[j] == i:
                    if j != len(V) - 1:
                        e = (G.get_edge_data(V[j], V[j+1]))['weight']
                        if j != 0:
                            e2 = (G.get_edge_data(V[j], V[j-1]))['weight']
                    else:
                        e = (G.get_edge_data(V[j], V[j-1]))['weight']
            if e2 != 0:
                cost += e + e2 
            elif edges[0] != e:
                cost += edges[0] + e
            else:
                cost += edges[1] + e
            e2 = 0
        else:
            cost += edges[0] + edges[1]
            
    return math.ceil(cost/2)


def branch_and_bound(G, limit=None):
    '''
    Funcao que executa o algoritmo de branch-and-bound

    G: grafo original
    limit: limite de tempo dado pelo usuario 

    Return: conjunto solucao, custo da solucao, tempo de execucao

    '''

    bound = fun_bound(G,[0])
    root = Node(bound, 0, 0, [0])
    listNodes = queue.PriorityQueue()  
    listNodes.put(root)
    
    best = np.inf
    sol = []
    
    start = time.time()
    duration = 0
    C = 0
    while not listNodes.empty():

        if limit != None:  # verifica se estourou o tempo
            if (time.time() - start) > limit:
                print('*** Time limit excedeed ***')
                break
            
        node = listNodes.get()
        
        if node.level == len(G.nodes())-1:  # chegamos numa folha, analisa se a solucao eh melhor que a melhor
            total_cost = node.cost + G[node.s[-1]][0]['weight']
            if best > total_cost:
                best = total_cost
                sol = node.s +[0]
                
        elif node.bound < best: 
            if node.level < len(G.nodes()):
                for k in range(1, len(G.nodes())):
                    v = node.s[-1]

                    if k not in node.s:
                        new_sol = node.s + [k]
                        bound = fun_bound(G, new_sol)

                        if bound < best: # adiciona no heap se a estimativa do node for inferior a melhor solucao encontrada
                            new_cost = node.cost + G[v][k]['weight']
                            new_node = Node(bound, node.level+1, new_cost, new_sol)
                            listNodes.put(new_node)
     
    return sol, best, time.time() - start



def twice_around_the_tree(G):
    '''
    Funcao que executa o algortimo twice-around-the-tree

    G: grafo original

    Return: conjunto solucao e custo da solucao
    '''

    root = 0
    MST = nx.minimum_spanning_tree(G, algorithm='prim')  # computar mst a partir de 0 como raiz
    preorder = list(nx.dfs_preorder_nodes(MST, 0)) # computa caminhamento em pre ordem na arvore
    preorder.append(0)
    C = fun_cost(G, preorder)
    return preorder, C


def create_multigraph(MST, match, G):
    '''
    Funcao utilizada para criar um multigrafo a partir das arestas selecionadas

    '''

    M = nx.MultiGraph()
    for i in MST.edges():
        w = MST.get_edge_data(i[0], i[1])
        w = w['weight']
        M.add_edge(i[0], i[1], weight=w)
    for i in match:
        w = G.get_edge_data(i[0], i[1])
        w = w['weight']
        M.add_edge(i[0], i[1], weight=w)
        
    odd_list = []
    for i in range(len(M)):
        if M.degree[i]%2 == 1:
            odd_list.append(i)   # numero de vertices com grau impar eh par
  
    i = 0
    while (i < len(odd_list)):
        M.add_edge(odd_list[i], odd_list[i+1])  
        i+=2
    return M



def christofides(G):
    '''
    Funcao que executa o algoritmo de Christofides

    G: grafo original

    Return: conjunto solucao e custo da solucao
    '''

    MST = nx.minimum_spanning_tree(G, algorithm='kruskal') # computar mst
    odd_degree = []
    
    # selecionar vertices de grau impar da mst
    for i in range(len(MST)):
        if MST.degree[i]%2 == 1:
            odd_degree.append(i)
            
    S = nx.Graph()
    S = G.subgraph(odd_degree)
    
    # encontra aresta de peso maximo
    max_weight = 0
    for u, v in S.edges:
        weight = G.get_edge_data(u, v)['weight']
        if (weight > max_weight):
            max_weight = weight

    # modifica arestas para utilizar a funcao de matching perfeito de peso maximo
    # para encontrar o matching perfeito de peso minimo
    S_copy = S.copy()
    for (a, b) in S.edges:
        S_copy[a][b]['weight'] = max_weight - S_copy[a][b]['weight']  
    
    match = nx.max_weight_matching(S_copy, maxcardinality=True)  #  minimum-weight perfect matching 

    M = create_multigraph(MST, match, G) # cria multigrafo com arestas da MST e o matching perfeito de peso minimo
    
    eulerCircuit = [u for (u, v) in nx.eulerian_circuit(M, source=0)]  # faz um circuito euleriano no grafo

    hamiltonian = []
    for node in eulerCircuit:
        if node not in hamiltonian:
            hamiltonian.append(node)

    hamiltonian.append(0)  
    C = fun_cost(G, hamiltonian)

    return hamiltonian, C