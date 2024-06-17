#!pip install pandas scikit-learn

import pandas as pd

def load_data():
    # Carregar os dados
    edge_global_deltas = pd.read_csv('edge_global_deltas.csv')
    local_deltas = pd.read_csv('local_deltas.csv')
    valid_tuples = pd.read_csv('valid_tuples.csv')
    vertex_global_deltas = pd.read_csv('vertex_global_deltas.csv')

    # Combinando dados relevantes
    data = valid_tuples.copy()
    data['edge_delta'] = edge_global_deltas['DELTA']
    data['local_delta'] = local_deltas['DELTA']
    data['vertex_global_delta'] = vertex_global_deltas['DELTA']

    return data

def prepare_features(data):
    # Definir as variáveis de entrada (features) e saída (target)
    X = data[['edge_delta', 'local_delta', 'vertex_global_delta']]
    # criar coluna/criterio de criticidade
    y = data['criticidade']  
    return X, y

if __name__ == "__main__":
    data = load_data()
    X, y = prepare_features(data)
    print(X.head())
    print(y.head())
