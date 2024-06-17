from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # Para salvar o modelo
import data_preparation  # Importar funções do data_preparation.py

def train_model(X, y):
    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Fazer previsões
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    return model

def save_model(model, filename):
    joblib.dump(model, filename)

if __name__ == "__main__":
    data = data_preparation.load_data()
    X, y = data_preparation.prepare_features(data)
    model = train_model(X, y)
    save_model(model, 'model.pkl')
