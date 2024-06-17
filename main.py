import data_preparation
import model_training

def main():
    # Carregar e preparar os dados
    data = data_preparation.load_data()
    X, y = data_preparation.prepare_features(data)

    # Treinar e salvar o modelo
    model = model_training.train_model(X, y)
    model_training.save_model(model, 'model.pkl')

if __name__ == "__main__":
    main()
