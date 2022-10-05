
import numpy as np
import pandas as pd
from menelaus.drift_detector import BatchDetector
from sklearn.cluster import KMeans
from scipy.stats import f, chi2


class DetectA(BatchDetector):
    """
    
    Initialize by setting reference 
    Update with 

    """

    input_type = "batch"

    def __init__(
        self, 
        alpha = 0.95
    ):
        """
        Args:
            alpha (float): statistical significance used to identify
                threshold for f and X2 distributions. Defaults to 0.95.
        """

        super().__init__() #total batches = 0, batches since reset = 0

        # Initialize parameters
        self._alpha = alpha 


    def set_reference(self, X, y_true, k = None, y_pred = None):
        """Initialize detector with a reference batch. User should pass in true labels to be used as initial centroids when clustering. 
        If labels are unavailable, user must specify y_true = None and k (the number of labels) so clustering can be performed.

        The unique types and number of labels passed to set_reference must be maintained if labels are passed to proceeding update calls. 
        If user wishes to change the types and number of labels, must specify new reference batch using this method and restart run.

        Args:
            X (pandas.DataFrame): initial baseline dataset
            y_true (numpy.array): true labels for dataset. Defaults to None. 
            y_pred (numpy.array): predicted labels for dataset, not used by DetectA. Defaults to None.  
        """

        super().set_reference(X, y_true, y_pred)
        X, y_true, y_pred = super()._validate_input(X, y_true, y_pred) #ensures columns match, saves self._input_col_dim, saves feature names 

        # Set attributes
        if y_true != None:
            self._labels = np.unique(y_true)
            self._k = self._labels.shape[0]
        else:
            if k == None:
                raise ValueError(
                        "If y_true is none, k must be passed in."
                    )
            else: 
                self._k = k
        self._ref_n = X.shape[0]
        self.reset(X, y_true)


    def reset(self, X, y_true = None):
        """Cluster reference data and initialize reference mean and covariance statistics. Intended for use
        after ``drift_state == 'drift'``.

        Args:
            X (pandas.DataFrame): initial baseline dataset.
            y_true (numpy.array): true labels for dataset. Defaults to None. 
        """

        # cluster and initialize reference statistics attributes
        self._ref_data = self._cluster(X, y_true)
        self._pred_labels = np.unique(self._ref_data[:,[0]])
        self._ref_means = self._mean_vector(self._ref_data)
        self._ref_cov = self._cov_matrix(self._ref_data)

        if self.drift_state == None:
            super().reset()

    def update(self, X, y_true = None, y_pred = None):
        """Update the detector with a new test batch. On next update call, this test batch becomes the new reference batch. 
        If user wants to use labels to cluster data, whether drift is detected or not, labels must be passed in with test batch. New
        labels passed in must match unique types initial labels. 
        If drift is detected and labels were not passed on prior update call, clustering is performed without initial centroids.  
        If drift is not detected on prior update call and labels are not passed in, reference conditional mean vector used as initial centroids in clustering.

        Args:
            X (pandas.DataFrame): initial baseline dataset
            y_true (numpy.array): true labels for dataset. Defaults to None. 
            y_pred (numpy.array): predicted labels for dataset, not used by DetectA. Defaults to None.  
        """

        X, y_true, y_pred = super()._validate_input(X, y_true, y_pred)

        if y_true != None:
            if np.array_equal(self._labels,y_true) is False:
                raise ValueError(
                        "Y_true labels must match initial y_true labels passed to set_reference method."
                    )

        if self.drift_state: 
            self.reset(self._ref_data, self._prior_y) 

        super.update() 
        
        # Update attribtues
        self._ref_n = self._test_n 
        self._ref_data = self._test_data 
        self._test_n = X.shape[0]
        if self.batches_since_reset > 1:
            self._ref_means = self._test_means 
            self._ref_cov = self._test_cov 

        # Statistics to be monitored
        self._test_data = self._cluster(X, y_true)
        self._test_means = self._mean_vector(self._test_data)
        self._test_cov = self._cov_matrix(self._test_data)

        # Statistics for detecting drift 
        T2 = self._T2stat(self._ref_means, self._test_means, self._ref_cov, self._test_cov)
        f_threshold = f.ppf(self._alpha, self.input_col_dim, ((self._ref_n+self._test_n)- self.input_col_dim -1)) #TODO self.alpha or 1-alpha
        C = self._C_stat(self._ref_cov, self._test_cov) #TODO getting inf and nans
        df = 0.5*self.input_col_dim*(self.input_col_dim + 1)
        chi_threshold = chi2.ppf(self._alpha, df) #TODO same thing with alpha here 

        # Detect drift
        mean_drift = {}
        cov_drift = {}
        for k in self._pred_labels:

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
        
        # Save labels for next update call
        if y_true != None:
            self._prior_y = y_true 
        else:
            self._prior_y = None

        

    def _mean_vector(self, data):
        """Computes conditional mean vector.

        Args:
            data (numpy.ndarray): input data, with column at index 0 containing assigned labels from clustering 

        Returns:
            dict: dictionary containing mean vector for each class: key is class, value is array of conditional means
        """
        cond_mean_vec = {}
        for k in self._pred_labels: 
            k_array = data[data[:,0] == k][:,1:]
            cond_mean_vec[k] = [np.mean(k_array[:,j]) for j in range(self._input_col_dim)]
            
        return cond_mean_vec


    def _cov_matrix(self, data):
        """Computes conditional covariance matrix.

        Args:
            data (numpy.ndarray): input data, with column at index 0 containing assigned labels from clustering 

        Returns:
            dict: dictionary containing covariance matrix: key is class, value is covariance matrix for features.
        """

        cov_matrix = {}
        for k in self._pred_labels:
            k_array = data[data[:,0] == k][:,1:]
            cov_matrix[k] = np.cov(k_array, rowvar = False) #TODO double check correct use of rowvar
            
        return cov_matrix


    def _cluster(self, X, y_true = None):
        """Clusters data using kmeans. 

        If labels are passed to update call, uses conditional mean vectors of labels as initial centroids to speed up convergence.
        If labels are not passed in and drift was detected on prior batch, performs kmeans without initial centroids. 
        If labels are not passed in and drift was not detected, uses reference mean vector as initial centroids. 
        
        Args:
            X (pandas.DataFrame): initial baseline dataset
            y_true (numpy.array): true labels for dataset. Defaults to None. 

        Returns:
            numpy.ndarray: input data, with column at index 0 containing assigned labels from clustering 
        """

        # unsupervised clustering when no labels passed in 
        if y_true == None and self.drift_state == True:
            kmeans = KMeans(n_clusters = self._k).fit(X)

        else: 

            if y_true != None:
                data = np.concatenate((y_true, X), axis = 1)
                centroids = self._mean_vector(data)

            else: 
                centroids = self._ref_means

            init_centroids = list(centroids.values())
            kmeans = KMeans(n_clusters = self._k, init = init_centroids).fit(X)  #TODO convert to mahalanobis
        
        new_labels = kmeans.predict(X)
        new_labels = super()._validate_y(new_labels)
        data = np.concatenate((new_labels, X), axis = 1)

        return data

    def _T2stat(self, ref_means, test_means, ref_cov,  test_cov):
        """Computes T2 statistic, used to monitor drift in conditional mean vector.

        Args:
            ref_means (dict): Dictionary containing conditional mean vector, indexed by class, for reference data
            test_means (dict): Dictionary containing conditional mean vector, indexed by class, for test data
            ref_cov (dict): Dictionary containing conditional covariance matrix, indexed by class, for reference data
            test_cov (dict): Dictionary containing conditional covariance matrix, indexed by class, for test data

        Returns:
            dict: dictionary containing T2 statistic for each class.
        """

        t2_dict = {}

        for k in self._pred_labels:

            # compute test statistic
            mean_vec_diff = np.subtract(ref_means[k],test_means[k])
            cov_inv = np.linalg.inv(np.add((ref_cov[k] / self._ref_n),(test_cov[k] / self._test_n)))
            t2 = np.transpose(mean_vec_diff) @ cov_inv @ mean_vec_diff

            # scale by degrees of freedom
            t2_scaled = (((self._ref_n + self._test_n) - self.input_col_dim - 1) /((self._ref_n + self._test_n - 2)*self.input_col_dim))*t2 

            t2_dict[k] = t2_scaled 
        
        return t2_dict 

    def _C_stat(self,ref_cov, test_cov):
        """Computes C statistic, used to monitor drift in conditional covariance matrix. 

        Args:
            ref_cov (dict): Dictionary containing conditional covariance matrix, indexed by class, for reference data
            test_cov (dict): Dictionary containing conditional covariance matrix, indexed by class, for test data

        Returns:
            dict: Dictionary containing C statistic for each class.
        """
    
        C_dict = {}

        for k in self._pred_labels:

            # pooled covariance matrix
            S_pool = ((self._ref_n - 1)*ref_cov[k] + (self._test_n-1)*test_cov[k]) / (self._ref_n - 1 + self._test_n - 1)

            # determinant 
            det = ((np.linalg.det(ref_cov[k]) / np.linalg.det(S_pool))**((self._ref_n-1)/2)) * ((np.linalg.det(test_cov[k]) / np.linalg.det(S_pool))**((self._test_n-1)/2))
            M = -2*np.log(det)
            u = ((1/(self._ref_n -1)) + (1/(self._test_n -1)) - (1/(self._ref_n-1 + self._test_n-1)))*((2*self.input_col_dim**2 + 3*self.input_col_dim - 1)/(6*(self.input_col_dim+1)))
            C = (1-u)*M

            C_dict[k] = C

        return C_dict 
