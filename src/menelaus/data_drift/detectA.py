
import numpy as np
import pandas as pd
from menelaus.drift_detector import BatchDetector
from sklearn.cluster import KMeans
from scipy.stats import f, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DetectA(BatchDetector):
    """
    Implements the DetectA drift detection algorithm. Inherits from
    ``BatchDetector`` (see docs). DetectA assumes the data follows a
    Multivariate Normal distribution. If this assumption is not verified, then
    the probability of a false positive or false negative drift detection
    increases. 

    DetectA is a drift detection algorithm monitoring for abrupt drift in the
    mean vectors and covariance matrices of each class. It operates by:
        #. labeling the data using kmeans clustering with Mahalanobis distance
        #. computing conditional mean vectors and conditional covariance
           matrices 
        #. detecting drift in both the mean vectors and covariance matrices
           using multivariate hypothesis tests
    
    The reference and test batches are updated by sliding each batch forward.
    After each update call, the most recent test batch becomes the new reference
    batch. Whether or not drift is detected, this allows for the current test
    batch to become the assumed standard, or reference, distribution to which
    the next test batch will be compared. 

    DetectA is designed to operate as an unsupervised drift detection algorithm.
    When specifying the reference batch (set_referece), it is recommended to
    provide the true labels for the data to increase both the computational
    speed and accuracy of the kmeans clustering. At each update step, the user
    has the option to provide the true labels, though it is not required. If the
    true labels are provided at each update step, those will be used as the
    initial centroids in the clustering, as opposed to the centroids from the
    clustering of the reference batch.

    When drift is detected, the algorithm is designed to proceed without
    requiring labels for the test batch containing drift. To increase accuracy
    of drift detection, the user may provide the true labels. This can be done
    in two ways: 
        #. The user can pass the true labels for the test batch to the update
           method when passing in the test batch. This requires the labels to be
           passed in proactively, prior to drift being detected on the current
           batch. 
        #. If the user would like to pass in the true labels retroactively,
           after being alerted to drift in the most recent batch, the user must
           call set_reference, passing in both the test batch containing drift
           and its true labels. 
        This will manually initialize the current test batch as the new
        reference batch.

    In order to use the sklearn Kmeans package to comptue clusters, transforms
    are applied to the data. The  sklearn KMeans package uses Euclidean distance
    to compute clusters.  Mahalanobis distance is equivalent to Euclidean
    distance after the data is standardized to zero mean and unit variance and a
    whitening transformation, such as principal component analysis, is applied
    to the covariance matrix. Principal component analysis transforms the data
    into a new set of variables such that the covariance matrix is uncorrelated
    and each component has unit variance. To cluster the data using Kmeans with
    Mahalanobis as the distance metric, the clusters are assigned using the
    principal components extracted from standardized data. Once the labels are
    obtained for each observation, they are assigned to their respective
    untransformed observations. This enables monitoring of the unstandardized,
    untransformed means and covariances of each class. 

    To detect drift in the conditional mean vectors, DetectA computes hottelings
    T2 statistic using each class's mean vector and compares it to a threshold
    derived from the F distribution. To detect drift in the conditional
    covariance matrix, DetectA computes the Box-M statistic using each class's
    covariance matrix and compares it to a threshold derived from the X2
    distribution.This enables drift in the mean and covariance of each class to
    be detected individually.

    Ref. #TODO 

    Attributes:
        init_labels (list): unique categories of true labels passed to
        set_reference. 
        k (int): number of unique labels 
        ref_means (numpy.ndarray): conditional mean vector for reference data, 
        indexed by class 
        test_means (numpy.ndarray): conditional mean vector for test data,
        indexed by class 
        ref_cov (numpy.ndarray): conditional covariance matrix for reference 
        data, indexed by class 
        test_cov (numpy.ndarray): conditional covariance matrix for test data, 
        indexed by class 
        mean_drift (dict): for each predicted cluster from kmeans, indicates if 
        drift detected in conditional mean vector 
        cov_drift (dict): for each predicted cluster from kmeans, indicates if 
        drift detected in conditional covariance matrix 
        y_pred (list): contains assigned label from kmeans for each observation 
        in test data

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
        """Initialize detector with a reference batch. User should pass in true
        labels to be used as initial centroids when clustering. 
        
        If labels are unavailable, user must specify y_true = None and k (the
        number of labels) so clustering can be performed.

        The unique types and number of labels passed to set_reference must be
        maintained if labels are passed to proceeding update calls. If user
        wishes to change the types and number of labels, must specify new
        reference batch using this method and restart run.

        Args:
            X (pandas.DataFrame): initial baseline dataset 
            y_true (pandas.DataFrame): true labels for dataset. If user cannot 
            obtain, input as 'None' but must input k parameter 
            k (int): the number of categories of labels. User should provide
            if y_true = 'None' to assist with clustering 
            y_pred (pandas.DataFrame): predicted labels for dataset, not used 
            by DetectA. Defaults to None.  
        """

  
        X, y_true, y_pred = super()._validate_input(X, y_true, y_pred)  
        X = pd.DataFrame(
            X, columns=self._input_cols
        )
        
        # Set attributes
        if y_true is not None:
            y_true = y_true.flatten()
            self.init_labels = np.unique(y_true)
            self.k = self.init_labels.shape[0]
        else:
            self.init_labels = None
            if k == None:
                raise ValueError(
                        "If y_true is None, k must be passed in."
                    )
            else: 
                self.k = k
        self._ref_n = X.shape[0]
        self.reset(X, y_true)
     

        return 


    def reset(self, X, y_true = None):
        """Cluster reference data and initialize reference mean and covariance
        statistics. Internally called after set_reference or ``drift_state == 'drift'``.

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
        self.y_pred = self._cluster(X_pca, centroids) 
        self.ref_means = self._mean_vector(X, self.y_pred)
        self.ref_cov = self._cov_matrix(X, self.y_pred)

        if self.drift_state == 'drift': 
            super().reset()

    def update(self, X, y_true = None, y_pred = None):
        """Update the detector with a new test batch. On next update call, this
        test batch becomes the new reference batch. 
        
        If user wants to use labels to cluster data, whether drift is detected
        or not, labels must be passed in with test batch. New labels passed in
        must match unique types initial labels. If drift is detected and labels
        were not passed on prior update call, clustering is performed without
        initial centroids. If drift is not detected on prior update call and
        labels are not passed in, reference conditional mean vector used as
        initial centroids in clustering.

        Args:
            X (pandas.DataFrame): initial baseline dataset 
            y_true (pandas.DataFrame): true labels for dataset. Defaults to None. 
            y_pred (pandas.DataFrame): predicted labels for dataset, not used by 
            DetectA. Defaults to None.  
        """

        X, y_true, y_pred = super()._validate_input(X, y_true, y_pred)
        X = pd.DataFrame(
            X, columns=self._input_cols
        )


        if y_true is not None and self.init_labels is not None:
            if np.array_equal(self.init_labels,np.unique(y_true)) is False:
                raise ValueError(
                        "Y_true labels must match initial y_true labels passed to set_reference method."
                    )

        if self.drift_state == 'drift': 
            self.reset(self._prior_X.drop(columns = ['_label_']), self._prior_y) 
        
        # Update attributes
        super().update(X, y_true, y_pred)
        if self.batches_since_reset > 1:
                self._ref_n = self._test_n 
                self.ref_means = self.test_means 
                self.ref_cov = self.test_cov 
        self._test_n = X.shape[0]

        # Statistics to be monitored
        X_scaled = StandardScaler().fit_transform(X)
        X_pca = self._pca.transform(X_scaled)
        self.y_pred  = self._cluster(X_pca) 
        self.test_means = self._mean_vector(X, self.y_pred)
        self.test_cov = self._cov_matrix(X, self.y_pred)

        # Statistics for detecting drift  
        self._T2 = self._T2stat(self.ref_means, self.test_means, self.ref_cov, self.test_cov)
        self._f_threshold = f.ppf(self._alpha, self._input_col_dim, ((self._ref_n+self._test_n)- self._input_col_dim -1)) 
        self._C = self._C_stat(self.ref_cov, self.test_cov) 
        df = 0.5*self._input_col_dim*(self._input_col_dim + 1)
        self._chi_threshold = chi2.ppf(self._alpha, df) 

        # Detect drift
        self.mean_drift = {}
        self.cov_drift = {}
        for k in np.unique(self.y_pred):

            if self._T2[k] > self._f_threshold: 
                self.mean_drift[k] = 'drift'
                self.drift_state = 'drift'
            else: 
                self.mean_drift[k] = None

            if self._C[k] > self._chi_threshold:
                self.cov_drift[k] = 'drift'
                self.drift_state = 'drift'
            else: 
                self.cov_drift[k] = None
        
        # Save labels for next update call, if drift 
        if self.drift_state == 'drift':
            self._prior_X = X
            if y_true is not None:
                self._prior_y = y_true 
            else:
                self._prior_y = None
        

    def _mean_vector(self, X, y):
        """Computes conditional mean vector.

        Args:
            X (pandas.DataFrame): input data 
            y (pandas.Series): labels, true or
            assigned

        Returns:
            numpy.ndarray: Multidimensional array containing conditional mean
            vector, indexed by class
        """
        X['_label_'] = y
        return np.asarray(X.groupby('_label_').mean())


    def _cov_matrix(self, X, y):
        """Computes conditional covariance matrix.

        Args:
            X (pandas.DataFrame): input data 
            y (pandas.Series): labels, true or
            assigned
        Returns:
            numpy.ndarray: Multidimensional array containing conditional
            covariance matrix, indexed by class
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

        If labels are passed to update call, uses conditional mean vectors of
        labels as initial centroids to speed up convergence. If labels are not
        passed in and drift was detected on prior batch, performs kmeans without
        initial centroids. If labels are not passed in and drift was not
        detected, clusters test data using kmeans clustering fit to reference
        data. 
        
        Args:
            X (pandas.DataFrame): PCA transformed dataset 
            centroids (numpy.ndarray): Multidimensional array containing initial 
            centroids for clustering. Defaults to None. 

        Returns:
            numpy.ndarray: assigned labels from clustering 
        """
        if centroids is None and self._batches_since_reset == 0: #test for update
            self._kmeans = KMeans(n_clusters = self.k
                ).fit(X)
        elif centroids is not None:
            self._kmeans = KMeans(n_clusters = self.k, 
                init = np.array(centroids)
                ).fit(X)
        y_pred = self._kmeans.predict(X)
        return y_pred

    def _T2stat(self, ref_means, test_means, ref_cov,  test_cov):
        """Computes T2 statistic, used to monitor drift in conditional mean
        vector.

        Args:
            ref_means (numpy.ndarray): Multidimensional array containing
            conditional mean vector, indexed by class, for reference data
            test_means (numpy.ndarray): Multidimensional array containing
            conditional mean vector, indexed by class, for test data 
            ref_cov (numpy.ndarray): Multidimensional array containing 
            conditional covariance matrix, indexed by class, for reference 
            data 
            test_cov (numpy.ndarray): Multidimensional array containing 
            conditional covariance matrix, indexed by class, for test data

        Returns:
            dict: dictionary containing T2 statistic for each class.
        """

        t2_dict = {}

        for k in range(self.k):

            # compute test statistic
            mean_vec_diff = np.subtract(ref_means[k],test_means[k])
            cov_inv = np.linalg.inv(np.add((ref_cov[k] / self._ref_n),(test_cov[k] / self._test_n)))
            t2 = np.transpose(mean_vec_diff) @ cov_inv @ mean_vec_diff

            # scale by degrees of freedom
            t2_scaled = (((self._ref_n + self._test_n) - self._input_col_dim - 1) /((self._ref_n + self._test_n - 2)*self._input_col_dim))*t2 

            t2_dict[k] = t2_scaled 
        
        return t2_dict 

    def _C_stat(self,ref_cov, test_cov):
        """Computes C statistic, used to monitor drift in conditional covariance
        matrix. 

        Args:
            ref_cov (numpy.ndarray): Multidimensional array containing
            conditional covariance matrix, indexed by class, for reference data
            test_cov (numpy.ndarray): Multidimensional array containing
            conditional covariance matrix, indexed by class, for test data

        Returns:
            dict: Dictionary containing C statistic for each class.
        """
    
        C_dict = {}

        for k in range(self.k):

            # pooled covariance matrix
            S_pool = ((self._ref_n - 1)*ref_cov[k] + (self._test_n-1)*test_cov[k]) / (self._ref_n - 1 + self._test_n - 1)

             # determinant: log-transform to avoid floating-point errors
            log_S_pool_det = np.log(np.linalg.det(S_pool))
            log_ref_cov_det = np.log(np.linalg.det(ref_cov[k]))
            log_test_cov_det = np.log(np.linalg.det(test_cov[k]))

            M = ((self._ref_n - 1)/2) * (log_ref_cov_det - log_S_pool_det) + ((self._test_n - 1)/2) * (log_test_cov_det - log_S_pool_det)
            M = -2 * M
            u = ((1/(self._ref_n -1)) + (1/(self._test_n -1)) - (1/(self._ref_n-1 + self._test_n-1)))*((2*self._input_col_dim**2 + 3*self._input_col_dim - 1)/(6*(self._input_col_dim+1)))
            C = (1-u)*M

            C_dict[k] = C

        return C_dict 
