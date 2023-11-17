import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from matplotlib import colors
df = pd.DataFrame(pd.read_csv("stroke_prediction_dataset.csv")).dropna()

def sigmoid(X):
   return 1/(1+np.exp(-X))

# Colunas a ser usadas : [ X ]
# Age - OK
# Gender - OK
# Hypertension - OK
# Heart Disease - OK
# Average Glucose Level - OK
# Smoking Status - OK
# Physical Activity - OK
# Stroke History - OK
# Family History - OK
# Stress Levels - OK

#Tratando colunas


df['Gender'].replace(['Male', 'Female'], [1,-1], inplace=True) # 1 = Homem | -1 = Mulher
df['Smoking Status'].replace(['Non-smoker', 'Formerly Smoked', 'Currently Smokes'], [1,2,3], inplace=True) # 1 = ñ fuma | 2 = fumava casualmente | 3 = fuma
df['Physical Activity'].replace(['Low', 'Moderate', 'High'], [1,2,3], inplace=True) # 1 = baixo | 2 = medio | 3 = alto
df['Family History of Stroke'].replace(['Yes', 'No'], [1,0], inplace=True) # 1 = sim | 0 = nao
df['Diagnosis'].replace(['Stroke', 'No Stroke'], [1,-1], inplace=True) # 1 = AVC | -1 = ñ AVC

numEpocas = 100    # Número de épocas.
q = len(df)                # Número de padrões.

eta = 0.01        # Taxa de aprendizado ( é interessante alterar para avaliar o comportamento)
m =  10                # Número de neurônios na camada de entrada (Age, Gender... Family hist, Stress levels)
N = 5                 # Número de neurônios na camada escondida.
L = 1                 # Número de neurônios na camada de saída. (0 = No Stroke E 1 = Stroke)

# Carrega os dados de treinamento (Atributos)
ages = df["Age"].to_numpy()
genders = df["Gender"].to_numpy()
hypertensions = df["Hypertension"].to_numpy()
heart_diseases = df["Heart Disease"].to_numpy()
average_glucose_level = df["Average Glucose Level"].to_numpy()
smoking_statuses = df["Smoking Status"].to_numpy()
physical_activities = df["Physical Activity"].to_numpy()
stroke_histories = df["Stroke History"].to_numpy()
family_histories = df["Family History of Stroke"].to_numpy()
stress_levels = df["Stress Levels"].to_numpy()

# Vetor de classificação desejada.
d = df['Diagnosis'].to_numpy()


# Inicia aleatoriamente as matrizes de pesos.
# Inicializando com m, N e L nos dá a liberdade de diferentes arquiteturas (só alterando as linhas 17,18 e 19)
W1 = np.random.random((N, m + 1)) #dimensões da Matriz de entrada
W2 = np.random.random((N, N + 1)) #dimensões da Matriz de entrada
W3 = np.random.random((N, N + 1)) #dimensões da Matriz de entrada
W4 = np.random.random((L, N + 1)) #dimensões da Matriz de saída

        
E = np.zeros(q)
Etm = np.zeros(numEpocas) #Etm = Erro total médio ==> serve para acompanharmos a evolução do treinamento da rede
# bias
bias = 1
# Entrada do Perceptron.
X = np.vstack((ages, genders, hypertensions, heart_diseases, average_glucose_level, smoking_statuses, physical_activities, stroke_histories, family_histories, stress_levels))
        




for i in range(numEpocas): #repete o numero de vezes terminado, no caso 20
    for j in range(q): #repete o numero de "dados" existentes (nesse exemplo 13)
            
            # Insere o bias no vetor de entrada (apresentação do padrão da rede)
        Xb = np.hstack((bias, X[:,j])) #empilhamos pelo hstack junto ao bias e ficamos 
                                        #com unico vetor [bias peso PH]

        o1 = sigmoid(W1.dot(Xb))  # Saída da primeira camada oculta
        o1b = np.insert(o1, 0, bias)
        o2 = sigmoid(W2.dot(o1b))  # Saída da segunda camada oculta
        o2b = np.insert(o2, 0, bias)
        o3 = sigmoid(W3.dot(o2b))  # Saída da terceira camada oculta
        o3b = np.insert(o3, 0, bias)

        Y = sigmoid(W4.dot(o3b))  # Saída da camada de saída
            
        e = d[j] - Y                        # Equação (5).

            # Erro Total.
        E[j] = (e.transpose().dot(e))/2     # Equação de erro quadrática.
            
            # Imprime o número da época e o Erro Total.
        print('i = ' + str(i) + '   E = ' + str(E))
    
            # Backpropagation
            # Camada de Saída
        delta4 = np.diag(e).dot(1 - Y*Y)
        vdelta4 = (W4.transpose()).dot(delta4)
        delta3 = np.diag(1 - o3b*o3b).dot(vdelta4)
        # Camada Oculta 3
        vdelta3 = (W3.transpose()).dot(delta3[1:]) #
        delta2 = np.diag(1 - o2b*o2b).dot(vdelta3)
        # Camada Oculta 2
        vdelta2 = (W2.transpose()).dot(delta2[1:]) #
        delta1 = np.diag(1 - o1b*o1b).dot(vdelta2)

        # Atualização dos pesos
        W1 = W1 + eta * (np.outer(delta1[1:], Xb))
        W2 = W2 + eta * (np.outer(delta2[1:], o1b))
        W3 = W3 + eta * (np.outer(delta3[1:], o2b))
        W4 = W4 + eta * (np.outer(delta4, o3b))
        
        #Calculo da média dos erros
        Etm[i] = E.mean()
        print(Etm[i])
    

plt.xlabel("Épocas")
plt.ylabel("Erro Médio")
plt.plot(Etm, color='b')
plt.plot(Etm)
plt.show()
    
 
Error_Test = np.zeros(q)
for i in range(q):
    # Insere o bias no vetor de entrada.
    Xb = np.hstack((bias, X[:,i]))

    # Saída da Camada Escondida.
    o1 = sigmoid(W1.dot(Xb))            # Equações (1) e (2) juntas.      
    #print(o1)
        
    # Incluindo o bias. Saída da camada escondida é a entrada da camada
    # de saída.
    o1b = np.insert(o1, 0, bias)
    o2 = sigmoid(W2.dot(o1b))
    o2b = np.insert(o2, 0, bias)
    o3 = sigmoid(W3.dot(o2b))
    o3b = np.insert(o3, 0, bias)

    # Neural network output
    Y = sigmoid(W4.dot(o3b))            # Equações (3) e (4) juntas.
    print(Y)

        

        
    Error_Test[i] = d[i] - (Y)
    
print(Error_Test)
vet_erros = np.round(Error_Test) - d
print(vet_erros) #aqui se ela acertou todas o vetor tem que estar zerado

idade = int(input("Digite a sua idade: "))
genero = int(input("Voce eh Homem(1) ou Mulher(-1)?:  "))
hipertensao = int(input("Voce ja teve hipertensao? (1)s (0)n: "))
doenca_coracao = int(input("Voce ja teve alguma doenca do coracao? (1)s (0)n: "))
nivel_glucose = float(input("Digite o seu nivel de glucose: "))
status_fumo = int(input("Voce nao fuma(1), fuma casualmente(2) ou fuma com frequencia(3)?: "))
frequencia_treino = int(input("O seu nivel de treino eh: baixo(1), medio(2) ou alto(3)?: "))
hist_avc = int(input("Voce ja teve um AVC previamente? (1)s (0)n: "))
hist_familiar_avc = int(input("Voce tem um historico familiar de AVC? (1)s (0)n"))
nivel_stress = float(input("Digite o seu nivel de stresse: "))

input_usuario = []
input_usuario.append(1) #bias
input_usuario.append(idade)
input_usuario.append(genero)
input_usuario.append(hipertensao)
input_usuario.append(doenca_coracao)
input_usuario.append(nivel_glucose)
input_usuario.append(status_fumo)
input_usuario.append(frequencia_treino)
input_usuario.append(hist_avc)
input_usuario.append(hist_familiar_avc)
input_usuario.append(nivel_stress)
print(input_usuario)



# Saída da Camada Escondida.
o1 = sigmoid(W1.dot(Xb))            # Equações (1) e (2) juntas.      
#print(o1)
        
# Incluindo o bias. Saída da camada escondida é a entrada da camada
# de saída.
o1b = np.insert(o1, 0, bias)
o2 = sigmoid(W2.dot(o1b))
o2b = np.insert(o2, 0, bias)
o3 = sigmoid(W3.dot(o2b))
o3b = np.insert(o3, 0, bias)

# Neural network output
Y = sigmoid(W4.dot(o3b))            # Equações (3) e (4) juntas.
print(Y)
if Y < 0:
    print("Você não corre risco de ter um AVC.")
else:
    print("Você corre risco de ter um AVC.")


