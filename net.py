from layer import Layer
from link import Link
from numpy import dot
from numpy import transpose

class Net:
    def __init__(self, topology_array, p_coeficient_array): #vetor que mostra a quantidade de neurônios em cada camada
        
        ## Verificações
        
        if len(topology_array) != 3:
            raise Exception('This version suports only 3 layers')
        for number in topology_array:
            if type(number) is not int or number < 1:
                raise ValueError('The numbes on topology_array must be integers greater than 0')

        ##
        
        self.layers = [] #Contém todos os layers da rede
        self.links = [] #Comtém todos os links da rede

        i = 0
        for number in topology_array:#Alocação dos layers de acordo com a topologia da rede
            self.layers.append(Layer(position_id = i, bias=1))
            self.layers[i].create_neurons(number, p_coeficient_array[i])
            i = i + 1


        for i in range(len(topology_array) - 1):# Criação dos links
            self.links.append(Link(self.layers[i], self.layers[i + 1]))

    ## Funções de interação com a classe

    def net_run(self, in_array, d_array = []):# Retorna a saída da rede e se inserida uma saída esperada são calculados os gradientes locais de cada neurônio
        
        ## Verificações

        if len(in_array) != self.layers[0].layer_len:
            raise Exception('The number of inputs must be equal to the number of input neurons')

        if len(d_array) != self.layers[-1].layer_len and d_array != []:
            raise Exception('The number of desired outputs must be equal to the number of output neurons')

        ##

        ## Determinação da saída da rede

        for i in range(len(self.layers)):
            if i == 0:
                for j in range(len(in_array)):
                    self.layers[0].neuro_vec[j].set_sum(in_array[j])
            
            else:
                for j in range(self.layers[i].layer_len):
                    self.layers[i].neuro_vec[j].set_sum(dot(self.layers[i - 1].get_layer_out(), transpose(self.links[i - 1].weights_array)[j]) + self.layers[i].bias)
        
        out_array = self.layers[-1].get_layer_out()

        ##

        ## Cálculo dos gradientes locais (versão para três camadas)
        
        if d_array != []:
            e_array = []
            for i in range(self.layers[-1].layer_len):#Cálculo do vetor erro (erro em cada saída)
                e_array.append(d_array[i] - out_array[i])

            for i in range(self.layers[-1].layer_len):#Cálculo do gradiente local dos neurônios da camada de saída
                self.layers[-1].neuro_vec[i].lc = e_array[i] * self.layers[-1].neuro_vec[i].get_d_out()

            lc_vec = self.layers[-1].get_layer_lc()#Cálculo do gradiente local dos neurônios da camada oculta
            for i in range(self.layers[-2].layer_len):
                self.layers[-2].neuro_vec[i].lc = self.layers[-2].neuro_vec[i].get_d_out()*dot(lc_vec, self.links[-1].weights_array[i])
        
        ##

        s = 0
        for e in e_array:
            s = s + e**2

        q_error = s/2

        return [out_array, q_error]
    
    def __season(self, input_data, d_output_data, learning_rate):
        if len(input_data) != len(d_output_data):
            raise Exception('The number of input arrays must be equal to the number of desired output arrays')

        if learning_rate <= 0:
            raise ValueError('The learning rate must be more than zero')
        
        s = 0
        for n in range(len(input_data)):
            s = s + self.net_run(input_data[n], d_output_data[n])[1]
            for l in [-1, -2]:
                for i in range(self.layers[l-1].layer_len):
                    for j in range(self.layers[l].layer_len):
                        self.links[l].weights_array[i][j] = self.links[l].weights_array[i][j] + learning_rate * self.layers[l].neuro_vec[j].lc * self.layers[l-1].get_layer_out()[i]

        MSE = s/len(input_data)
        return MSE

    def __calculate_validation_MSE(self, input_data, d_output_data):
        if len(input_data) != len(d_output_data):
            raise Exception('The number of input arrays must be equal to the number of desired output arrays')

        s = 0
        for n in range(len(input_data)):
            s = s + self.net_run(input_data[n], d_output_data[n])[1]

        MSE = s/len(input_data)
        return MSE

    
    ##

        
##  Teste pra ver se o código está sem erros

rede = Net([2,2,1],[[1,1],[1,1],[1,1]])

print('____________II_________________')
print(rede.links[0].weights_array)
print('_____________________________')
print(rede.links[1].weights_array)
print('_____________________________')
print(rede.net_run([0.2, 0.3], [5]))

##