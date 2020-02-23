import numpy as np
import math

class KFoldCrossValidation:
    def __init__(self, k):
        self.num_folds = k
    
    def run_cross_validation(self, model, data_x, data_y):
        """
        Parameters:
            model (object) :
                model is either LDA or LR object. Calls to functions fit
                and predict will be used with model.
            data_x :
                the x input as a 2D array 
            data_y :
                column array, the class indicator for each row in data_x
        Returns:
            method returns a float calculating the average error among all the validation sets
        """

        avg_percentage_correct = 0
        indices = self.get_indices(data_x.shape[0])
        for i in range(0 , len(indices) - 1):
            training_set_x, validation_set_x = self.get_sets(data_x, indices[i], indices[i+1])
            training_set_y, validation_set_y = self.get_sets(data_y, indices[i], indices[i+1])

            model.fit(training_set_x, training_set_y)
            prediction = model.predict(validation_set_x)
            avg_percentage_correct += eva(validation_set_y, prediction)
        
        avg_percentage_correct /= self.num_folds
        return avg_percentage_correct


    def get_sets(self, data, curr_index, next_index):
        """
        Parameters:
            curr_index:
                the current index tracking the validation set
            next_index:
                the next index for the validation set
        Returns:
            a tuple containing the training sets and validation sets where the validation set lies
            between the curr_index and next_index
        """
        if(curr_index==0):
            return data[next_index:], data[curr_index:next_index]
        elif(next_index):
            return data[:curr_index], data[curr_index:next_index]
        else:
            return np.concatenate([data[:curr_index], data[next_index:]]), data[curr_index:next_index]


    def get_indices(self, length):
        """
        Parameters:
            length:
                length of the list to make indices from
        Returns:
            all the indices that are used to split the data set into k (the number of folds) equal chunks
        """
        
        indices = [x for x in range(0, length) if x % (length//self.num_folds) == 0]

        if len(indices) > self.num_folds :
            del indices[-1]

        indices.append(length)
        print(indices)
        return indices



def eva(y_true, y_target):
    correct = 0
    total = 0
    prediction = y_target.flatten()
    validation = y_true.values
    for i in range(len(prediction)):
        if prediction[i] == validation[i]:
            correct += 1
        total += 1
    return (correct/total) * 100