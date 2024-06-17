#!pip install pandas scikit-learn

import pandas as pd

# Carregar os dados
edge_global_deltas = pd.read_csv('edge_global_deltas.csv')
local_deltas = pd.read_csv('local_deltas.csv')
valid_tuples = pd.read_csv('valid_tuples.csv')
disconnects = pd.read_csv('disconnects.csv')
vertex_global_deltas = pd.read_csv('vertex_global_deltas.csv')

# Exibir as primeiras linhas dos dados para verificação
print(edge_global_deltas.head())
print(local_deltas.head())
print(valid_tuples.head())
print(disconnects.head())
print(vertex_global_deltas.head())
