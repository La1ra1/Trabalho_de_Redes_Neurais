from neuron import Neuron
from random import random
class Layer:
    def __init__(self, position_id, bias = 0): #position id identifica qual a posição da camada (entrada, oculta ou saída) nesseprojeto mais simples de três camadas seria 0, 1 ou 2

        if position_id < 0 or position_id > 2 or (type(position_id) is not int): #mudar essa parte se forem adicionadas mais funções internas
            raise ValueError('The position id must be between 0 and 2 and an integer')

        if not (type(bias) is int or type(bias) is float):
            raise TypeError('Bias must be numeric.')
        
        if position_id == 0:
            self.bias = 0
        else:
            self.bias = bias

        self.pos_id = position_id
        self.layer_len = 0
        self.neuro_vec = [] #list of neurons
        self.__layer_out = []
    
    def create_neurons(self, number_of_neurons, list_of_p_coeficients = []):
        if number_of_neurons < 1 or (type(number_of_neurons) is not int):
            raise ValueError('the number of neurons Must be more than 1 and an integer')

        if self.pos_id < 0:
            if len(list_of_p_coeficients) != number_of_neurons:
                raise Exception('The number of p coeficients must be equal to the number of neurons')

        if len(self.neuro_vec) != 0:
            raise Exception('Already existent neurons in this layer. The neurons must be created all at the same time in this version. If you want to create new neurons you may use "self.delete_neurons" to reset the vector and then create new ones.')

        neu_n = number_of_neurons
        p_list = list_of_p_coeficients
        inf_id = 5

        if self.pos_id == 0:
            p_list = []
            for i in range(neu_n):
                p_list.append(1)
            inf_id = 0

        if self.pos_id == 1:
            inf_id = 1

        if self.pos_id == 2:
            inf_id = 0

        for i in range(neu_n):
            self.neuro_vec.append(Neuron(inf_id, p_list[i]))

        self.layer_len = neu_n

    def delete_neurons(self):
        self.neuro_vec = []

    def get_layer_out(self):
        self.__layer_out = []

        for neuron in self.neuro_vec:
            self.__layer_out.append(neuron.get_out())
        
        return self.__layer_out

    def get_layer_lc(self):
        lc_vector = []
        for neuron in self.neuro_vec:
            lc_vector.append(neuron.lc)
        return lc_vector