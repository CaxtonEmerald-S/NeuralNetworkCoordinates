import numpy as np
from tensorflow import keras

class NeuronCoordinates:
  def __init__(self, inputMatrix:np.ndarray, weightVector:np.ndarray, bias:float, activation, neuron:int):

    weightVector = np.reshape(weightVector, (weightVector.shape))
    self.__neuron = neuron;

    self.__scaledVector = self.__scale(inputMatrix, weightVector);
    self.__translatedVector = self.__translate(bias);
    self.__activatedVector = activation(self.__translatedVector);

    self.__individualScaleMatrix = self.__calculateIndividualScaleCoordinates(weightVector, inputMatrix);
    self.__cumulativeScaleMatrix = self.__calculateCumulativeScaleCoordinates();

  def __scale(self, inputMatrix, weightVector):
    return np.matmul(weightVector.T, inputMatrix);

  def __translate(self, bias):
    return self.__scaledVector + bias;

  def __calculateIndividualScaleCoordinates(self, weightVector, inputMatrix):
    return np.multiply(np.reshape(weightVector, (np.shape(weightVector)[0], 1)), inputMatrix);

  def __calculateCumulativeScaleCoordinates(self):
    return np.cumsum(self.__individualScaleMatrix, axis=0);

  def getNeuron(self):
    return self.__neuron;

  def getScaledVector(self):
    return self.__scaledVector;

  def getTranslatedVector(self):
    return self.__translatedVector;

  def getActivatedVector(self):
    return self.__activatedVector;

  def getIndividualScaleMatrix(self):
    return self.__individualScaleMatrix;

  def getCumulativeScaleMatrix(self):
    return self.__cumulativeScaleMatrix;

class LayerCoordinates:
  def __init__(self, layer:int, inputMatrix:np.ndarray, weightMatrix:np.ndarray, biasMatrix:np.ndarray, activation):
    self.__layer = layer;
    self.__layerInputMatrix = np.array(inputMatrix);
    self.__neuronCoordinatesList = [None]*np.size(biasMatrix);

    for neuron in range(np.size(biasMatrix)):
      self.__neuronCoordinatesList[neuron] = (NeuronCoordinates(self.__layerInputMatrix, weightMatrix[:,neuron], biasMatrix[neuron], activation, neuron));

  def getLayer(self):
    return self.__layer;

  def getInputMatrix(self):
    return self.__layerInputMatrix;

  def getNeuronCoordinatesList(self):
    return self.__neuronCoordinatesList;

  def getNeuronCoordinates(self, neuron:int):
    if neuron >= 0 and neuron < np.size(self.__neuronCoordinatesList):
      return self.__neuronCoordinatesList[neuron];
    return None;

class NetworkCoordinates:
  def __init__(self, model:keras.Model, initialInput:np.ndarray=[]):
    self.__model = model;
    if self.__model is not None:
      self.__layerCoordinatesList = [None]*np.size(self.__model.layers);
      self.__initialInput = np.array(initialInput);
      if self.__initialInput.size == 0:
        self.__initialInput = np.linspace(-1*np.ones(self.__model.layers[0].get_weights()[0].shape[0]), np.ones(self.__model.layers[0].get_weights()[0].shape[0]), 100).T;

  def __addLayerCoordinatesToList(self, index:int, layerCoordinates:LayerCoordinates):
    if layerCoordinates is not None:
      self.__layerCoordinatesList[index] = layerCoordinates;

  def getLayerCoordinatesList(self):
    return self.__layerCoordinatesList;

  def getLayerCoordinates(self, layer:int):
    if layer >=0 and layer <= np.size(self.__model.layers):
      return self.__layerCoordinatesList[layer];

  def getModel(self):
    return self.__model;

  def setModel(self, model:keras.Model):
    if model is not None:
      self.__model = model;
      return True;
    return False;

  def getInitialInput(self):
    return self.__initialInput;

  def setInitialInput(self, initialInput:np.ndarray):
    if initialInput is not None and np.array(initialInput).size != 0:
      self.__initialInput = initialInput;
      return True;
    return False;

  def prepareCoordinates(self):
    layer = 0;
    self.__addLayerCoordinatesToList(layer, LayerCoordinates(layer, self.getInitialInput(), self.getModel().layers[layer].get_weights()[0], self.getModel().layers[layer].get_weights()[1], self.getModel().layers[layer].activation));
    for layer in range(1, np.size(self.getModel().layers)):
      neuron = 0;
      layerInputMatrix = self.getLayerCoordinates(layer-1).getNeuronCoordinates(neuron).getActivatedVector();
      for neuron in range(1, self.getModel().layers[layer].get_weights()[0].shape[0]):
        layerInputMatrix = np.vstack((layerInputMatrix, self.getLayerCoordinates(layer-1).getNeuronCoordinates(neuron).getActivatedVector()));
      self.__addLayerCoordinatesToList(layer, LayerCoordinates(layer, layerInputMatrix, self.getModel().layers[layer].get_weights()[0], self.getModel().layers[layer].get_weights()[1], self.getModel().layers[layer].activation));
