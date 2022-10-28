
import numpy as np
import pandas as pd
from menelaus.drift_detector import BatchDetector
from sklearn.cluster import KMeans
from scipy.stats import f, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DetectA(BatchDetector):
    """
    
    # TODO 

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

        super().__init__() 

        # Initialize parameters
        self._alpha = alpha 


    def set_reference(self, X, y_true, k = None, y_pred = None):
        """Initialize detector with a reference batch. User should pass in true labels to be used as initial centroids when clustering. 
        If labels are unavailable, user must specify y_true = None and k (the number of labels) so clustering can be performed.

        The unique types and number of labels passed to set_reference must be maintained if labels are passed to proceeding update calls. 
        If user wishes to change the types and number of labels, must specify new reference batch using this method and restart run.

        Args:
            X (pandas.DataFrame): initial baseline dataset
            y_true (pandas.Series): true labels for dataset. If user cannot obtain, input as 'None' but must input k parameter
            k (int): the number of categories of labels. User should provide if y_true = 'None' to assist with clustering 
            y_pred (pandas.Series): predicted labels for dataset, not used by DetectA. Defaults to None.  
        """

  
        X, y_true, y_pred = super()._validate_input(X, y_true, y_pred)  
        X = pd.DataFrame(
            X, columns=self._input_cols
        )
        
        # Set attributes
        if y_true is not None:
            y_true = y_true.flatten()
            self._init_labels = np.unique(y_true)
            self._k = self._labels.shape[0]
        else:
            self._init_labels = None
            if k == None:
                raise ValueError(
                        "If y_true is None, k must be passed in."
                    )
            else: 
                self._k = k
        self._ref_n = X.shape[0]
        self.reset(X, y_true)
     

        return 


    def reset(self, X, y_true = None):
        """Cluster reference data and initialize reference mean and covariance statistics. Internally called after
        set_reference or ``drift_state == 'drift'``.

        Args:
            X (pandas.DataFrame): initial baseline dataset.
            y_true (pandas.Series): true labels for dataset. Defaults to None. 
        """

        # cluster and initialize reference statistics attributes
        
        X_scaled = StandardScaler().fit_transform(X)
        self._pca = PCA(n_components = self._input_col_dim, whiten = True).fit(X_scaled)
        X_pca = self._pca.transform(X_scaled)
        if y_true is not None: 
             centroids = self._mean_vector(pd.DataFrame(X_pca), y_true)
        else:
            centroids = None
        y_pred = self._cluster(X_pca, centroids) 
        self._ref_means = self._mean_vector(X, y_pred)
        self._ref_cov = self._cov_matrix(X, y_pred)

        if self.drift_state: 
            super().reset()

    def update(self, X, y_true = None, y_pred = None):
        """Update the detector with a new test batch. On next update call, this test batch becomes the new reference batch. 
        If user wants to use labels to cluster data, whether drift is detected or not, labels must be passed in with test batch. New
        labels passed in must match unique types initial labels. 
        If drift is detected and labels were not passed on prior update call, clustering is performed without initial centroids.  
        If drift is not detected on prior update call and labels are not passed in, reference conditional mean vector used as initial centroids in clustering.

        Args:
            X (pandas.DataFrame): initial baseline dataset
            y_true (pandas.Series): true labels for dataset. Defaults to None. 
            y_pred (pandas.Series): predicted labels for dataset, not used by DetectA. Defaults to None.  
        """

        X, y_true, y_pred = super()._validate_input(X, y_true, y_pred)
        X = pd.DataFrame(
            X, columns=self._input_cols
        )


        if y_true is not None and self._init_labels is not None:
            if np.array_equal(self._init_labels,np.unique(y_true)) is False:
                raise ValueError(
                        "Y_true labels must match initial y_true labels passed to set_reference method."
                    )

        if self.drift_state: 
            self.reset(self._prior_X, self._prior_y) 
        
        # Update attributes
        super().update() 
        if self.batches_since_reset > 1:
                self._ref_n = self._test_n 
                self._ref_means = self._test_means 
                self._ref_cov = self._test_cov 
        self._test_n = X.shape[0]

        # Statistics to be monitored
        X_scaled = StandardScaler().fit_transform(X)
        X_pca = self._pca.transform(X_scaled)
        y_pred  = self._cluster(X_pca) 
        self._test_means = self._mean_vector(X, y_pred)
        self._test_cov = self._cov_matrix(X, y_pred)

        # Statistics for detecting drift  
        T2 = self._T2stat(self._ref_means, self._test_means, self._ref_cov, self._test_cov)
        f_threshold = f.ppf(self._alpha, self.input_col_dim, ((self._ref_n+self._test_n)- self.input_col_dim -1)) 
        C = self._C_stat(self._ref_cov, self._test_cov) 
        df = 0.5*self.input_col_dim*(self.input_col_dim + 1)
        chi_threshold = chi2.ppf(self._alpha, df) 

        # Detect drift
        mean_drift = {}
        cov_drift = {}
        for k in np.unique(y_pred):

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
        
        # Save labels for next update call, if drift 
        if self.drift_state:
            self._prior_X = X
            if y_true is not None:
                self._prior_y = y_true 
            else:
                self._prior_y = None
        

    def _mean_vector(self, X, y):
        """Computes conditional mean vector.

        Args:
            X (pandas.DataFrame): input data
            y (pandas.Series): labels, true or assigned

        Returns:
            numpy.ndarray: Multidimensional array containing conditional covariance matrix, indexed by class
        """
        X['_label_'] = y
        return np.asarray(X.groupby('_label_').mean())


    def _cov_matrix(self, X, y):

        """Computes conditional covariance matrix.

        Args:
            X (pandas.DataFrame): input data
            y (pandas.Series): labels, true or assigned
        Returns:
            numpy.ndarray: Multidimensional array containing conditional covariance matrix, indexed by class
        """
        cov = []
        X['_label_'] = y 

        for k in np.unique(y): 
            # select data assigned label k, excluding label column 
            k_df = X[X['_label_'] == k].iloc[:,0:-1]
            cov.append(np.cov(k_df, rowvar = False))

        return cov


    def _cluster(self, X, centroids = None):
        """Clusters data using kmeans. 

        If labels are passed to update call, uses conditional mean vectors of labels as initial centroids to speed up convergence.
        If labels are not passed in and drift was detected on prior batch, performs kmeans without initial centroids. 
        If labels are not passed in and drift was not detected, clusters test data using kmeans clustering fit to reference data. 
        
        Args:
            X (pandas.DataFrame): PCA transformed dataset
            centroids (numpy.ndarray): Multidimensional array containing initial centroids for clustering. Defaults to None. 

        Returns:
            numpy.ndarray: assigned labels from clustering 
        """
        if centroids is None and self._batches_since_reset == 0: #test for update
            self._kmeans = KMeans(n_clusters = self._k
                ).fit(X)
        elif centroids is not None:
            self._kmeans = KMeans(n_clusters = self._k, 
                init = np.array(centroids)
                ).fit(X)
        y_pred = self._kmeans.predict(X)
        return y_pred

    def _T2stat(self, ref_means, test_means, ref_cov,  test_cov):
        """Computes T2 statistic, used to monitor drift in conditional mean vector.

        Args:
            ref_means (numpy.ndarray): Multidimensional array containing conditional mean vector, indexed by class, for reference data
            test_means (numpy.ndarray): Multidimensional array containing conditional mean vector, indexed by class, for test data
            ref_cov (numpy.ndarray): Multidimensional array containing conditional covariance matrix, indexed by class, for reference data
            test_cov (numpy.ndarray): Multidimensional array containing conditional covariance matrix, indexed by class, for test data

        Returns:
            dict: dictionary containing T2 statistic for each class.
        """

        t2_dict = {}

        for k in range(self._k):

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
            ref_cov (numpy.ndarray): Multidimensional array containing conditional covariance matrix, indexed by class, for reference data
            test_cov (numpy.ndarray): Multidimensional array containing conditional covariance matrix, indexed by class, for test data

        Returns:
            dict: Dictionary containing C statistic for each class.
        """
    
        C_dict = {}

        for k in range(self._k):

            # pooled covariance matrix
            S_pool = ((self._ref_n - 1)*ref_cov[k] + (self._test_n-1)*test_cov[k]) / (self._ref_n - 1 + self._test_n - 1)

             # determinant: log-transform to avoid floating-point errors
            log_S_pool_det = np.log(np.linalg.det(S_pool))
            log_ref_cov_det = np.log(np.linalg.det(ref_cov[k]))
            log_test_cov_det = np.log(np.linalg.det(test_cov[k]))

            M = ((self._ref_n - 1)/2) * (log_ref_cov_det - log_S_pool_det) + ((self._test_n - 1)/2) * (log_test_cov_det - log_S_pool_det)
            M = -2 * M
            u = ((1/(self._ref_n -1)) + (1/(self._test_n -1)) - (1/(self._ref_n-1 + self._test_n-1)))*((2*self.input_col_dim**2 + 3*self.input_col_dim - 1)/(6*(self.input_col_dim+1)))
            C = (1-u)*M

            C_dict[k] = C

        return C_dict 
