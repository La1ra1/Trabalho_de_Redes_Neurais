from layer import Layer
from random import random

class Link:
   def __init__(self, first_layer, second_layer):

      ## Verificações

      if type(first_layer) is not Layer or type(second_layer) is not Layer:
         raise TypeError('The two parameters must be a Layer type class')
      
      if first_layer.pos_id < second_layer.pos_id:#Também define qual o vetor de origem e o vetor de chegada das ligações; assim como o id da camada
         if second_layer.pos_id - first_layer.pos_id ==1:
            self.link_id = [first_layer.pos_id, second_layer.pos_id]
            self.ol_len = first_layer.layer_len
            self.dl_len = second_layer.layer_len
         else:
            raise Exception('the two layers must be consecutive layers')
      else:
         if first_layer.pos_id - second_layer.pos_id == 1:
            self.link_id = [second_layer.pos_id, first_layer.pos_id]
            self.ol_len = second_layer.layer_len
            self.dl_len = first_layer.layer_len
         else:
            raise Exception('the two layers must be consecutive layers')

      ##

      ## Criação da matriz de pesos
      ## Linhas: Vetor de origem
      ## Colunas: Vetor de chegada

      self.weights_array = []

      for i in range(self.ol_len):
         self.weights_array.append([])
         for j in range(self.dl_len):
            self.weights_array[i].append((random() - 0.5)*2)

      ##