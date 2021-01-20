from math import exp

class Neuron:
    def __init__(self, inner_function_id, p_coeficient):

        ## Verificações

        if inner_function_id < 0 or inner_function_id > 1 or (type(inner_function_id) is not int): #Mudar essa parte se forem adicionadas mais funções internas
            raise ValueError('The inner function id must be between 0 and 1 and an integer')

        if p_coeficient < 0:
            raise ValueError('The p coeficient must be more than zero')

        ##

        ## Atributos

        self._inf_id = inner_function_id #função de ativação
        self.p = p_coeficient #Coeficiente que altera as funções de ativação
        self.__sum = 0 #Combinação linear das saídas dos neurônios da camada anterior + bias
        self.lc = 0 #Gradiente local
        self.g = [] #Vetor qu recebe a função interna do neurônio e sua derivada

        ##

        ## Atribuição das funções de ativação. Dependendo do id a lista g recebe diferentes funções

        if self._inf_id == 0:
            self.g.append(self.__linear_function)
            self.g.append(self.__linear_function_derivative)

        elif self._inf_id == 1:
            self.g.append(self.__sigmoid_function)
            self.g.append(self.__sigmoid_function_derivative)
    
        ##

    ## Funções de interação com a classe

    def get_out(self):#Retorna a saída do neurônio
        return self.g[0](self.__sum)

    def get_d_out(self):#Retorna o valor da derivada da função de ativação
        return self.g[1](self.__sum)

    def set_sum(self, sum):#Altera o valor de __sum
        self.__sum = sum

    ## Funções de ativação e suas derivadas

    def __linear_function(self, x):
        return x*self.p

    def __linear_function_derivative(self, x):
        return self.p

    def __sigmoid_function(self, x):  
        return 1/(1+exp(-1*self.p*x))

    def __sigmoid_function_derivative(self, x):
        return self.p*exp(-1*self.p*x)/(exp(-1*self.p*x) + 1)**2
        
    ##