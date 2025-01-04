import torch
import torch.nn as nn

# Definindo uma rede neural simples
class RedeNeuralSimples(nn.Module):
    def __init__(self):
        super(RedeNeuralSimples, self).__init__()
        # Primeira camada: entrada de 1 neurônio, saída de 10 neurônios
        self.fc1 = nn.Linear(1, 100)
        # Segunda camada: entrada de 10 neurônios, saída de 1 neurônio
        self.fc2 = nn.Linear(100, 1)

    # # Forward pass
    def forward(self, x):
    #     # Passar pela primeira camada e ativar a função de ativação  
        x = torch.relu(self.fc1(x))
        # x = self.fc1(x)
        # Passar pela segunda camada, sem função de ativação (ex: última camada de regressão)
        x = self.fc2(x)
        return x
    
rede_neural = RedeNeuralSimples()

# Carregando os pesos do treinamento
rede_neural.load_state_dict(torch.load('modelo_treinado.pth', weights_only=True))
print('Modelo carregado com sucesso!')

# Colocar em modo avaliação
rede_neural.eval()

# Entrada 
entrada_nova = torch.tensor([[5.0], [3.0]])

# Inciar predição
predi = rede_neural(entrada_nova)

# Imprimir resultados
print(predi.squeeze().tolist())