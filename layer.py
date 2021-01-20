from neuron import Neuron
from random import random
class Layer:
    def __init__(self, position_id, bias = 0): #position id identifica qual a posição da camada (entrada, oculta ou saída) nesseprojeto mais simples, de três camadas, seria 0, 1 ou 2

        ## Verificações

        if position_id < 0 or position_id > 2 or (type(position_id) is not int): #mudar essa parte se forem adicionadas mais funções internas
            raise ValueError('The position id must be between 0 and 2 and an integer')

        if not (type(bias) is int or type(bias) is float):
            raise TypeError('Bias must be numeric.')

        ##

        ## Atributos
        
        if position_id != 0:# Atribuição do bias. Se a camada é a camada de entrada ela não possui bias
            self.bias = bias

        self.pos_id = position_id #Como explicado acima, identifica a posição da camada
        self.layer_len = 0 #Quantidade de neurônios na camada
        self.neuro_vec = [] #Lista contendo todos os neurônios existentes na camada
        self.__layer_out = [] #Lista que armazena as saídas de cada neurônio

        ##
    
    ## Funções de interação com a classe

    def create_neurons(self, number_of_neurons, list_of_p_coeficients = []):

        ## Verificações

        if number_of_neurons < 1 or (type(number_of_neurons) is not int):
            raise ValueError('the number of neurons Must be more than 1 and an integer')

        if self.pos_id < 0:
            if len(list_of_p_coeficients) != number_of_neurons:
                raise Exception('The number of p coeficients must be equal to the number of neurons')

        if len(self.neuro_vec) != 0:
            raise Exception('Already existent neurons in this layer. The neurons must be created all at the same time in this version. If you want to create new neurons you may use "self.delete_neurons" to reset the vector and then create new ones.')

        ##

        neu_n = number_of_neurons
        p_list = list_of_p_coeficients
        inf_id = []

        ## Definição do valor das funções de ativação de acordo com a posição da camada

        if self.pos_id == 0:# Se a camada é a de entrada, a função de ativação é a função identidade
            p_list = []
            for i in range(neu_n):#Desta forma p = 1 para todos os Neurônios
                p_list.append(1)
            inf_id = 0

        if self.pos_id == 1:#Se a camada é a oculta, utiliza-se a sigmoide
            inf_id = 1

        if self.pos_id == 2:#Se a camada é a de saída utiliza-se a função linear
            inf_id = 0

        ##

        for i in range(neu_n):#São alocados os neurônios no vetor
            self.neuro_vec.append(Neuron(inf_id, p_list[i]))

        self.layer_len = neu_n

    def delete_neurons(self):
        self.neuro_vec = []

    def get_layer_out(self):# Retorna a saída de cada neurônio na camada
        self.__layer_out = []

        for neuron in self.neuro_vec:
            self.__layer_out.append(neuron.get_out())
        
        return self.__layer_out

    def get_layer_lc(self):# Retorna o valor de gradiente local de cada neurônio na camada
        lc_vector = []
        for neuron in self.neuro_vec:
            lc_vector.append(neuron.lc)
        return lc_vector

    ##