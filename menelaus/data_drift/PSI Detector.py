from menelaus.detector import BatchDetector
import pandas as pd
import numpy as np
import copy


class PSI_Detector(BatchDetector):
    
    input_type = "batch"

    def __init__(
        self,
        
    ):

        # This initializes batch detector's parent class
        super().__init__() 
        
        # Initialize any parameters to be used in this algorithm


    def set_reference(self, X, y_true=None, y_pred=None):
       
        # leave this, it uses the parent class to validate input
        # and sets the self.reference variable to refer to the reference dataset 
     
        X, _, _ = super()._validate_input(X, None, None)
        X = pd.DataFrame(
            X, columns=self._input_cols
        )  

        # Initialize reference dataset
        self.reference = copy.deepcopy(X)

        self.reset() 
        
        
    def reset(self):


        super().reset()

        
    def update(self, X, by_feature=True, X_by_feature=None, y_true=None, y_pred=None):
        
        # this function will update the detector with the test batch 
        eps = 1e-4
        # create a variable to store psi values for each feature 
        feature_psi = []
        o = pd.DataFrame(
            columns=["Column", 'Moderate population change', 'Significant population change',"PSI"])
        z = 0
        # 1. iterate through each feature in reference and test data, identify minimum and maximum value 
        for column in self.reference.columns:
            min_val = min(min(self.reference[column]), min(X[column]))
            max_val = max(max(self.reference[column]), max(X[column]))
            
             # 2. use _bin_data function to bucketize reference, append to reference buckets array
            bins = self._bin_data(self.reference[column],min_val,max_val)
            bins_initial = pd.cut(self.reference[column], bins = bins, labels = range(1,len(bins)))
            df_initial = pd.DataFrame({'initial': self.reference[column], 'bin': bins_initial})
            grp_initial = df_initial.groupby('bin').count()
            grp_initial['percent_initial'] = grp_initial['initial'] / sum(grp_initial['initial'])
             # 3. use _bin_data function to bucketize test, append to reference test array 
            bins_new = pd.cut(X[column], bins = bins, labels = range(1,len(bins)))
            df_new = pd.DataFrame({'new': X[column], 'bin': bins_new})
            grp_new = df_new.groupby('bin').count()
            grp_new['percent_new'] = grp_new['new'] / sum(grp_new['new'])
            # 4. Call PSI function to calculate PSI on test and reference bucket representation,
            psi_value = self._PSI(grp_initial,grp_new)
            feature_psi.append([column,psi_value])
            # store PSI for each feature in feature_psi array 
            if psi_value >= 0.1 and psi_value <= 0.2:
                o.loc[z] = [column,'Yes','No',psi_value]
                z += 1
            elif psi_value > 0.2:
                o.loc[z] = [column,'No','Yes',psi_value]
                z += 1
        # 5. Aggregate PSI values to determine if dataset is drifting
        if o.any()['Column'] == True:
            self.drift_state == 'drift'
            return feature_psi, o
        else:
            return 'no drift detected',feature_psi
        # If PSI indicates drift, set self.drift_state == 'drift'
        
        # Create a dictionary to store if each individual feature is drifting 

        # Update self.reference dataset to refer to test data, X 
        self.reference = X

        

    def _bin_data(self, feature, min, max):
        eps = 1e-4
        if len(feature.unique()) < 10:
            bins = [min + (max - min)*(i)/len(feature.unique()) for i in range(len(feature.unique())+1)]
            bins[0] = min - eps # Correct the lower boundary
            bins[-1] = max + eps # Correct the higher boundary
            return bins
        else:
            bins = [min + (max - min)*(i)/10 for i in range(10+1)]
            bins[0] = min - eps # Correct the lower boundary
            bins[-1] = max + eps # Correct the higher boundary
            return bins
            # return an array containing the sample counts within each bucket 


    def _PSI(self, reference_feature, test_feature):
        eps = 1e-4
        # Compare the bins to calculate PSI
        psi_df = reference_feature.join(test_feature, on = "bin", how = "inner")
    
        # Add a small value for when the percent is zero
        psi_df['percent_initial'] = psi_df['percent_initial'].apply(lambda x: eps if x == 0 else x)
        psi_df['percent_new'] = psi_df['percent_new'].apply(lambda x: eps if x == 0 else x)
    
        # Calculate the psi
        psi_df['psi'] = (psi_df['percent_initial'] - psi_df['percent_new']) * np.log(psi_df['percent_initial'] / psi_df['percent_new'])
    
        # Return the mean of psi values
        return np.mean(psi_df['psi'])

