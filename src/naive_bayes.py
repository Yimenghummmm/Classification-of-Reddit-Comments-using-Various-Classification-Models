import numpy as np

class BernoulliNaiveBayes:
    def __init__(self, prior=1/2):
        self.smoothing = prior
        return
    def fit(self, X, Y):
        """
        Parameters
        X : numpy array (n instances, m features)
        Y : numpy array (n instances)
        """
        #count number of classes and instances per class
        #class_count is # of examples where y = k for class k
        class_names, class_count = np.unique(Y, return_counts=True)
        # print("Class names: ", class_names)
        self.classes = class_names
        
        num_instances, _ = X.shape

        #prob class is P(y = k)
        self.theta_k = class_count/Y.shape[0]

        y1_indicator = np.zeros((len(class_names), num_instances))
        for i in range(len(class_names)):
            y1_indicator[i] = np.where(Y == class_names[i], 1, 0)

        #theta is (#classes, #features)
        theta_y = y1_indicator @ X + self.smoothing
        

        self.theta_k_j = (theta_y.T/class_count).T
        
        
    def get_classes(self):
        return self.classes
    
    def predict(self, X):
        #1-X does not work; subtraction of a sparse matrix by a scalar is not supported. 
        #Therefore, we will have (1-X)log(1-theta) = log(1-theta)-(X * log(1-theta))
        log_neg = np.log10(1-self.theta_k_j)
        #print(log_neg)
#         class_prob = X @ (np.log10(self.theta_k_j) - log_neg).T
#         class_prob += np.log(self.theta_k) + log_neg.sum(axis=1)
        class_prob = X @ (np.log10(self.theta_k_j)).T + np.ones(X.shape)@log_neg.T - X @ log_neg.T
        class_prob += np.log(self.theta_k)

        return self.classes[np.argmax(class_prob, axis=1)]
