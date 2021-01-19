from math import exp

class Neuron:
    def __init__(self, inner_function_id, p_coeficient):

        if inner_function_id < 0 or inner_function_id > 1 or (type(inner_function_id) is not int): #mudar essa parte se forem adicionadas mais funções internas
            raise ValueError('The inner function id must be between 0 and 1 and an integer')

        if p_coeficient < 0:
            raise ValueError('The p coeficient must be more than zero')

        self._inf_id = inner_function_id #função de ativação
        self.p = p_coeficient
        self.__sum = 0 #Combinação linear das saídas dos neurônios da camada anterior
        self.lc = 0 #Gradiente local
        self.g = [] #Vetor qu recebe a função interna do neurônio e sua derivada

        if self._inf_id == 0:
            self.g.append(self.__linear_function)
            self.g.append(self.__linear_function_derivative)

        if self._inf_id == 1:
            self.g.append(self.__sigmoid_function)
            self.g.append(self.__sigmoid_function_derivative)
    
    def get_out(self):
        return self.g[0](self.__sum)

    def get_d_out(self):
        return self.g[1](self.__sum)

    def set_sum(self, sum):
        self.__sum = sum

    def __linear_function(self, x):
        return x*self.p

    def __linear_function_derivative(self, x):
        return self.p

    def __sigmoid_function(self, x):  
        return 1/(1+exp(-1*self.p*x))

    def __sigmoid_function_derivative(self, x):
        return self.p*exp(-1*self.p*x)/(exp(-1*self.p*x) + 1)**2