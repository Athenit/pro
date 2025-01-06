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

    # Forward pass
    def forward(self, x):
        # Passar pela primeira camada e ativar a função de ativação  
        x = torch.relu(self.fc1(x))
        # Passar pela segunda camada, sem função de ativação (ex: última camada de regressão)
        x = self.fc2(x)
        return x

rede_neural = RedeNeuralSimples() 
    
# Função de perda (MSE Loss para regressão)
criterio = nn.MSELoss()

# Otimizador (SGD - Stochastic Gradient Descent)
otimizador = torch.optim.SGD(rede_neural.parameters(), lr=0.01)

# Dados fictícios de exemplo (inputs e targets)
entrada = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
target = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Treinamento
num_epocas = 5000  # Número de vezes que os dados vão passar pela rede

for epoca in range(num_epocas):
    # Forward pass: gerar a predição
    predicao = rede_neural(entrada)

    # Calcular a perda
    perda = criterio(predicao, target)

    # Backward pass: zerar os gradientes, calcular novos gradientes e ajustar os pesos
    otimizador.zero_grad() # Zera os gradientes acumulados
    perda.backward() # Calcula os gradientes
    otimizador.step() # Atualiza os pesos

    # Exibir a perda a cada (n) épocas
    if (epoca+1) % 500 == 0:
        print(f'Época[{epoca+1}/{num_epocas}], Perda: {perda.item():.4f}')
        print(f'Predições: {predicao.squeeze().tolist()}')


print("\nSaídas finais do modelo (predições):")
print(predicao.squeeze().tolist())

torch.save(rede_neural.state_dict(), 'modelo_base_teste.pth')
print('Modelo salvo com sucesso')


