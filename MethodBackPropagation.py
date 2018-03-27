import numpy as np
class  EnglishGrammarNN(object):
    def __init__(self,learning_rate=0.1):
        self.weights_0_1 = np.random.uniform(0.0,0.5,(6,2))
        self.weights_1_2 = np.random.uniform(0.0,0.5,(3,5))
        self.weights_2_3 = np.random.uniform(0.0,0.5,(12,7))
        self.weights_3_4 = np.random.uniform(0.0,0.5,(1,13))
        self.weights_bias = np.random.uniform(0.0,0.5,(3,1))
        self.sigmoid_mapper = np.vectorize(self.sigmoid)
        self.learning_rate = np.array([learning_rate])

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def predict(self,inputs,inputsTimeAndBias,inputsSubjectAndBias):
        inputs_1 = np.dot(self.weights_0_1,inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2,np.vstack(outputs_1,inputsTimeAndBias))
        outputs_2 = self.sigmoid_mapper(inputs_2)

        inputs_3 = np.dot(self.weights_2_3,np.vstack(outputs_2,inputsSubjectAndBias))
        outputs_3 = self.sigmoid_mapper(inputs_3)

        inputs_4 = np.dot(self.weights_3_4,outputs_3)
        outputs_4 = self.sigmoid_mapper(inputs_4)

        return outputs_4

    def train(self,inputs,inputsTimeAndBias,inputsSubjectAndBias,expected_predict):
        #actual_predict = predict(inputs,inputsTimeAndBias,inputsSubjectAndBias)
        #actual_predict = actual_predict[0]
        actual_predict = (predict(inputs,inputsTimeAndBias,inputsSubjectAndBias))[0]
        error_layer_4 = np.array([(actual_predict -expected_predict])
        gradient_layer = actual_predict*(1 - actual_predict)
        weights_delta_layer = error_layer_4 * gradient_layer
        self.weights_3_4 -= (np.dot(weights_delta_layer,outputs_3.reshape(1,len(outputs_3))))*self.learning_rate

        error_layer_3 = self.weights_3_4 *  weights_delta_layer
        gradient_layer_1 = outputs_2 * (1 - outputs_2)
        weights_delta_layer_1 = error_layer_3 * gradient_layer_1
        self.weights_2_3 -= np.dot(weights_delta_layer_1,outputs_2.reshape(1,len(outputs_2)))*self.learning_rate

        error_layer_2 = self.weights_2_3 * weights_delta_layer_1
        gradient_layer_2 = outputs_1 * (1 - outputs_1)
        weights_delta_layer_2 = error_layer_2 * gradient_layer_2
        self.weights_1_2 -= np.dot(weights_delta_layer_2,outputs_1.reshape(1,len(outputs_1)))*self.learning_rate

        error_layer_3 = self.weights_1_2 * weights_delta_layer_2
        gradient_layer_3 = outputs_1 * (1 - outputs_1)
        weights_delta_layer_3 = error_layer_3 * gradient_layer_3
        self.weights_0_1 -= np.dot(inputs.reshape(len(inputs),1),weights_delta_layer_3).T * self.learning_rate
