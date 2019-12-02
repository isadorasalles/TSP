from tsp import *

import pandas as pd
import matplotlib.pyplot as plt

with open('analise2.csv', 'w') as f:
    f.write('Algorithm,Length,Cost,Time,Dist_Type\n')

    for i in range(15):
            print('*** Fazendo instância de tamanho 2^{} ***'.format(i))
            points = generate_points(size=4)
            G_euc, G_man = create_graph(points)
            
            print('*** Calculando para distância Euclidiana (Twice around the tree) ***')
            start = time.time()
            _, cost = twice_around_the_tree(G_euc)
            end = time.time() - start
            f.write('{},{},{},{},{}\n'.format('Twice around the tree', 16, cost, end, 'Euclidean'))
            
            print('*** Calculando para distância Manhattan (Twice around the tree) ***')
            start = time.time()
            _, cost = twice_around_the_tree(G_man)
            end = time.time() - start
            f.write('{},{},{},{},{}\n'.format('Twice around the tree', 16, cost, end, 'Manhattan'))

            print('*** Calculando para distância Euclidiana (Christofides) ***')
            start = time.time()
            _, cost = christofides(G_euc)
            end = time.time() - start
            f.write('{},{},{},{},{}\n'.format('Christofides', 16, cost, end, 'Euclidean'))
    
            print('*** Calculando para distância Manhattan (Christofides) ***')
            start = time.time()
            _, cost = christofides(G_man)
            end = time.time() - start
            f.write('{},{},{},{},{}\n'.format('Christofides', 16, cost, end, 'Manhattan'))
            
            print('*** Calculando para distância Euclidiana (Branch-and-Bound) ***')
            _, cost, t = branch_and_bound(G_euc, 1200)
            f.write('{},{},{},{},{}\n'.format('Branch-and-Bound', 16, cost, t, 'Euclidean'))
            
            print('*** Calculando para distância Manhattan (Branch-and-Bound) ***')
            _, cost, t = branch_and_bound(G_man, 1200)
            f.write('{},{},{},{},{}\n'.format('Branch-and-Bound', 16, cost, t, 'Manhattan'))
