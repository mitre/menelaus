
import numpy as np
import pandas as pd
from menelaus.drift_detector import BatchDetector
from sklearn.cluster import KMeans
from scipy.stats import f, chi2


class DetectA(BatchDetector):
    """
    
    Initialize by setting reference 
    Update with 


    Args:
        BatchDetector (_type_): _description_

    Returns:
        _type_: _description_
    """

    input_type = "batch"

    def __init__(self, alpha):

        super().__init__() #total batches = 0, batches since reset = 0

        # Initialize parameters
        self._alpha = alpha 


    def set_reference(self, X, y_true, y_pred = None):
        """
        only use when you have labels 

        # super set reference 

        # validate 
        X, y_true, y_pred = super()._validate_input(X, y_true, y_pred) #ensures columns match, saves self._input_col_dim, saves feature names 

        # self._k = # of classes in y 

        # self._k_types = unique y 

        # self._ref_n

        # call reset (X, y_true)

        Args:
            X (_type_): _description_
            y_true (_type_): _description_
            y_pred (_type_, optional): _description_. Defaults to None.
        """

        # super set reference 
        super().set_reference(X, y_true, y_pred)

        # validate 
        X, y_true, y_pred = super()._validate_input(X, y_true, y_pred) #ensures columns match, saves self._input_col_dim, saves feature names 

        # Set attributes
        self._labels = np.unique(y_true)
        self._k = self._labels.shape[0]
        self._ref_n = X.shape[0]

        self.reset(X, y_true)


    def reset(self, X, y_true = None):

        # cluster and initialize reference attributes
        self._ref_data = self._cluster(X, y_true)
        self._ref_means = self._mean_vector(self._ref_data)
        self._ref_cov = self._cov_matrix(self._ref_data)

        if self.drift_state == None:
            super().reset()

    def update(self, X, y_true = None, y_pred = None):
        """_summary_

        save test_n
        validate data
        if no drift and y_true is passed in, verify that labels match reference labels
        if drift: reset data using reference data and ref_y?? what about new y 

        reset mean and cov dictionaries to store drift 

        if this is the second test batch passed in, set ref statistics to store last test batch stats 

        cluster test data
        compute conditional means and cov matrix for test data
        find t2 and c test statistics
        compare to appropriate distributions to detect drift 
        
        if labels are passed in, save as reference y
        save testn as new refn 
        save test data as new reference data 

        Args:
            X (_type_): _description_
            y_true (_type_, optional): _description_. Defaults to None.
            y_pred (_type_, optional): _description_. Defaults to None.
        """

        X, y_true, y_pred = super()._validate_input(X, y_true, y_pred)

        self._test_n = X.shape[0]

        if self.drift_state == None and y_true != None:
            #TODO implement verify labels match ref_y 

        if self.drift_state: 
            self.reset(self._ref_data, self._ref_y) # TODO how does this work if y_true not passed in 

        super.update() #total_batches += 1, b_since reset +=1 

        # reset to store drift for each class 
        mean_drift = {}
        cov_drift = {}

        if self.batches_since_reset > 1:
            self._ref_means = self._test_means 
            self._ref_cov = self._test_cov 

        data = self._cluster(X, y_true)

        self._test_means = self._mean_vector(data)
        self._test_cov = self._cov_matrix(data)

        # T2 statistic
        T2 = self._T2stat(self._ref_means, self._test_means, self._ref_cov, self._test_cov)

        # f threshold for conditional mean drift 
        f_threshold = f.ppf(self._alpha, self.input_col_dim, ((self._ref_n+self._test_n)- self.input_col_dim -1)) #TODO self.alpha or 1-alpha

        # C statistic #TODO getting inf and nans
        C = self._C_stat(self._ref_cov, self._test_cov)

        # chi squared threshold for covariance matrix drift 
        df = 0.5*self.input_col_dim*(self.input_col_dim + 1)
        chi_threshold = chi2.ppf(self._alpha, df) #TODO same thing with alpha here 

        for k in self._labels:

            if T2[k] > f_threshold: 
                mean_drift[k] = True 
                self.drift_state = True
            else: 
                mean_drift[k] = False 

            if C[k] > chi_threshold:
                cov_drift[k] = True 
                self.drift_state = True
            else: 
                cov_drift[k] = False 
        
        if y_true != None:
            self._ref_y = y_true 
        self._ref_n = self._test_n 
        self._ref_data = data #TODO data or X? 

        

    def _mean_vector(self, data):
        """
        Computes conditional mean vector.

        Args:
            X (numpy.ndarray): input data, wiht first column (index 0) being labels
            y (numpy.ndarray): label from input data

        Returns:
            Dictionary containing mean vector for each class, indexed by class
        
        both X and y are input as arrays, X is multidimensional, y is one dimensional 
        concantenated togehter with Y (label) becoming first column 
        
        Returns:
            conditional mean vector as a dictionary, each key is a class, value is a list of conditional means
        
        """
        cond_mean_vec = {}

        # grouping by unique labels
        for k in self._labels:
            k_array = data[data[:,0] == k][:,1:]
            cond_mean_vec[k] = [np.mean(k_array[:,j]) for j in range(self._input_col_dim)]
            
        return cond_mean_vec


    def _cov_matrix(self, data):
        """
                # grouping by unique labels
                        # select data assigned label k, excluding label column 
                        compute covariance matrix

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """

        cov_matrix = {}

        for k in self._labels:
            k_array = data[data[:,0] == k][:,1:]
            cov_matrix[k] = np.cov(k_array, rowvar = False) #TODO double check correct use of rowvar
            
        return cov_matrix


    def _cluster(self, X, y_true = None):
        """
        
        if labels are passed in, find conditional mean vectors of labels and use as initial centroids 

        if no labels, use reference conditional mean vector 

        if drift but no labels are passed in, perform unsupervised clustering 

        combine assigned labels from kmeans with data and return dataset, labels will be added as 0 column in numpy matrix

        Args:
            X (_type_): _description_
            y_true (_type_, optional): _description_. Defaults to None.
        """

        if y_true != None:
            # TODO verify that y_true matches self._labels
            data = np.concatenate((y_true, X), axis = 1)
            centroids = self._mean_vector(data)

        # TODO figure out condition and how to do unsupervised clustering w no initial centroids

        else: 
            centroids = self._ref_means

        init_centroids = list(centroids.values())
        kmeans = KMeans(n_clusters = self._k, init = init_centroids).fit(X)  #TODO convert to mahalanobis
        new_labels = kmeans.predict(X)
        new_labels = super()._validate_y(new_labels)
        data = np.concatenate((new_labels, X), axis = 1)

        return data

    def _T2stat(self, ref_means, test_means, ref_cov,  test_cov):

        t2_dict = {}

        for k in self._labels:

            # compute test statistic
            mean_vec_diff = np.subtract(ref_means[k],test_means[k])
            cov_inv = np.linalg.inv(np.add((ref_cov[k] / self._ref_n),(test_cov[k] / self._test_n)))
            t2 = np.transpose(mean_vec_diff) @ cov_inv @ mean_vec_diff

            # scale by degrees of freedom
            t2_scaled = (((self._ref_n + self._test_n) - self.input_col_dim - 1) /((self._ref_n + self._test_n - 2)*self.input_col_dim))*t2 

            t2_dict[k] = t2_scaled 
        
        return t2_dict 

    def _C_stat(self,ref_cov, test_cov):
    
        C_dict = {}

        for k in self._labels:

            # pooled covariance matrix
            S_pool = ((self._ref_n - 1)*ref_cov[k] + (self._test_n-1)*test_cov[k]) / (self._ref_n - 1 + self._test_n - 1)

            # determinant 
            det = ((np.linalg.det(ref_cov[k]) / np.linalg.det(S_pool))**((self._ref_n-1)/2)) * ((np.linalg.det(test_cov[k]) / np.linalg.det(S_pool))**((self._test_n-1)/2))
            M = -2*np.log(det)
            u = ((1/(self._ref_n -1)) + (1/(self._test_n -1)) - (1/(self._ref_n-1 + self._test_n-1)))*((2*self.input_col_dim**2 + 3*self.input_col_dim - 1)/(6*(self.input_col_dim+1)))
            C = (1-u)*M

            C_dict[k] = C

        return C_dict 
