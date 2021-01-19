from layer import Layer
from link import Link
from numpy import dot
from numpy import transpose

class Net:
    def __init__(self, topology_array, p_coeficient_array): #vetor que mostra a quantidade de neurônios em cada camada
        if len(topology_array) != 3:
            raise Exception('This version suports only 3 layers')
        for number in topology_array:
            if type(number) is not int or number < 1:
                raise ValueError('The numbes on topology_array must be integers greater than 0')
        
        self.layers = []
        self.links = []

        i = 0
        for number in topology_array:
            self.layers.append(Layer(position_id = i, bias=1))
            self.layers[i].create_neurons(number, p_coeficient_array[i])
            i = i + 1


        for i in range(len(topology_array) - 1):
            self.links.append(Link(self.layers[i], self.layers[i + 1]))

    def net_run(self, in_array, d_array = []):
        if len(in_array) != self.layers[0].layer_len:
            raise Exception('The number of inputs must be equal to the number of input neurons')

        if len(d_array) != self.layers[-1].layer_len and d_array != []:
            raise Exception('The number of desired outputs must be equal to the number of output neurons')

        for i in range(len(self.layers)):
            if i == 0:
                for j in range(len(in_array)):
                    self.layers[0].neuro_vec[j].set_sum(in_array[j])
            
            else:
                for j in range(self.layers[i].layer_len):
                    self.layers[i].neuro_vec[j].set_sum(dot(self.layers[i - 1].get_layer_out(), transpose(self.links[i - 1].weights_array)[j]) + self.layers[i].bias)
        
        out_array = self.layers[-1].get_layer_out()
        
        if d_array != []:
            e_array = []
            for i in range(self.layers[-1].layer_len):
                e_array.append(d_array[i] - out_array[i])

            for i in range(self.layers[-1].layer_len):
                self.layers[-1].neuro_vec[i].lc = e_array[i]* self.layers[-1].neuro_vec[i].get_d_out()

        return out_array

rede = Net([2,2,1],[[1,1],[1,1],[1,1]])

print('____________II_________________')
print(rede.links[0].weights_array)
print('_____________________________')
print(rede.links[1].weights_array)
print('_____________________________')
print(rede.net_run([0.2, 0.3], [5]))
