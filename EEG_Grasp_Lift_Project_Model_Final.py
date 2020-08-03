#!/usr/bin/env python
# coding: utf-8

# # Decoding Force Intended to be Exerted to Lift an Object of Unknown Weight from EEG Data

# ### Data Set Description:
# !2 Participants engaged in a grasp-lift movement task where they were asked to repeat a series of trials that consisted of reaching for the an object, lifting it up, holding it in place, and then putting it back down. The weight of the object was unknown to the participant, and data was recorded using an EEG cap on the participant's head as well as force sensors on the object being lifted.

# ### Project Objective
# It is a well known fact when lifting an object of unknown weight, the previous trial heavily influences how much force you will exert on the new object since that is your only point of reference. Using EEG data from the time at which the subject starts to reach for the object to when they start to pick up it, we attempt to decode whether they intend to lift a heavy or light object

# ### Summary of Results
# 
# We implemented a binary logistic regression to classify eeg data as either intending to lift a heavy or light object. 
# 
# <b> Time x Voltage Model </b>
# This method of analysis had too many features (time points) compared to samples (channels) which caused the model to overfit on training data. We used PCA to reduce the dimesnions, but the accuracy was still below chance.
# 
# <b> Spectral Model </b>
# We converted our time series data into time-frequency domain and averaged across the known frequency ranges of alpha, beta, theta, and delta waves. This method helped reduce the number of features of our model, yielding an accuracy of 81%

# ## Libraries

# In[5]:


from __future__ import division
import numpy as np
import scipy.io as scipy
import matplotlib.pyplot as plt
get_ipython().system('pip install mne')
import mne
from numpy import *
from numpy.fft import *
import scipy.signal as signal
from matplotlib.pyplot import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import sklearn


# ## Helper Functions

# ### Reading .mat Files

# In[6]:


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], scipy.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, scipy.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = scipy.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


# ### MNE Preprocessing

# In[7]:


def preprocess_trial_data( data, info):
    # Create raw array object
    mne_rarray = mne.io.RawArray( data, info)
    
    # Common-average referencing of data
    rrf_rarray = mne_rarray.copy().set_eeg_reference( ref_channels='average')
    
    # Application of FIR bandpass filter
    #fil_length = str(((441-1)/(2*500))*1000) + "ms"
    flt_rarray = rrf_rarray.copy().filter( l_freq = .1, h_freq = 30)
   
    #################################################################################
    # TODO: Determine if we want to return an ica object instead of the filtered data
    #     ica = mne.preprocessing.ICA( n_components = 32)
    #     ica.fit( flt_rarray)
    #     return ica
    #################################################################################
    
    return flt_rarray


# ## EEG Setup Information

# In[8]:


# Electrode names from data decription file
#'Fp1','Fp2','F7','F3','Fz','F4','F8',,'P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10'
ch_names = ['FC5',
            'FC1','FC2','FC6','T7','C3','Cz','C4','T8',
            'TP9','CP5','CP1','CP2','CP6','TP10']

# Directory of channel locations
'''"Fp1": [-2.7, 8.6, 3.6],"Fp2": [2.7, 8.6, 3.6],
          "F7": [-6.7, 5.2, 3.6],"F3": [-4.7, 6.2, 8],
          "Fz": [0, 6.7, 9.5],"F4": [4.7, 6.2, 8],
          "F8": [6.7, 5.2, 3.6],"P7": [-6.7, -5.2, 3.6],
          "P3": [-4.7, -6.2, 8],
          "Pz": [0, -6.7, 9.5],"P4": [4.7, -6.2, 8],
          "P8": [6.7, -5.2, 3.6],"PO9": [-4.7, -6.7, 0],
          "O1": [-2.7, -8.6, 3.6],"Oz": [0, -9, 3.6],
          "O2": [2.7, -8.6, 3.6],"PO10": [4.7, -6.7, 0],
          '''
ch_pos = {
          "FC5": [-5.5, 3.2, 6.6],
          "FC1": [-3, 3.3, 11],"FC2": [3, 3.3, 11],
          "FC6": [5.5, 3.2, 6.6],"T7": [-7.8, 0, 3.6],
          "C3": [-6.1, 0, 9.7],"Cz": [0, 0, 12],
          "C4": [6.1, 0, 9.7],"T8": [7.8, 0, 3.6],
          "TP9": [-7.3, -2.5, 0],"CP5": [-7.2, -2.7, 6.6],
          "CP1": [-3, -3.2, 11],"CP2": [3, -3.2, 11],
          "CP6": [7.2, -2.7, 6.6], "TP10": [7.3, -2.5, 0]}


# Channel montage
chan_locs = mne.channels.make_dig_montage(ch_pos)

# Sampling frequency in Hz
s_freq = 500

# Must specify channel type to enable filtering
info = mne.create_info( ch_names, s_freq, ch_types = "eeg")
info = info.set_montage(montage = chan_locs , verbose = None)
type(info)


# ## Loading Data For A Single Participant

# In[90]:


#########################################################
# TODO: This section could be better off as a function(?)
#########################################################

# Extract relevant data from Participant 1
#p1 = loadmat('P1/WS_P1_S9.mat')

# Load AllLifts for P1
p1_all = loadmat('P1/P1_AllLifts.mat')
print( "All Lifts loaded")

# series with weight changing only
weight_series = [1, 4, 7, 8, 9]
trialStartIdx = [ 1, 97, 193, 227, 261]

i = 0

eeg_data_raw = []
weight_class = []

for s in weight_series:
    # load data for series
    series_filename = 'P1/WS_P1_S{series}.mat'.format(series = s)
    p_series = loadmat(series_filename)
    print("Series data loaded")

    tStart = trialStartIdx[i]
    # Increment i for next start index
    i += 1
    # trial counter
    trial = 0

    for t in range(tStart, tStart + 33):
        # retrieve time from LED flash to picking up the object
        tLEDon = 2.0
        tStartLoadPhase = np.array(p1_all['P']['AllLifts'])[t, 17]

        # slice EEG window to desired time interval (currently startLoad - 1:startLoad)
        #idxStart = int((tStartLoadPhase - 1) // 0.002)
        #highest accuracy idxEnd = +500, idxStart = -1500
        
        idxEnd = int(tStartLoadPhase // 0.002) + 500
        idxStart = idxEnd - 500
        eeg_trial = np.array(p_series['ws']['win'][trial]['eeg'])[idxStart:idxEnd, 7:22].T
        #print(eeg_trial.shape)
        
        # slice force data to the same time interval
        #loadforce = np.array(p_series['ws']['win'][trial]['kin'])[idxLEDon:idxStartLoad, 38]
        
        # Get previous weight class (intended weight)
        prev_weight = np.array(p1_all['P']['AllLifts'])[t, 5]
        
        if prev_weight == 2:
            continue
        else:
            eeg_data_raw.append(eeg_trial)
            weight_class.append(prev_weight)
            
           
        
        # Increment trial number for accessing EEG data
        trial += 1

eeg_data_raw = np.asarray(eeg_data_raw)
weight_class = np.asarray(weight_class)


# In[91]:


print(eeg_data_raw.shape)


# ## Preprocessing Pipeline For All Relevant Trials

# In[92]:


trials = np.arange( 0, len( eeg_data_raw))

flt_rarray_data = []
flt_data = []

for trial in trials:
    trial_data = eeg_data_raw[ trial]
    flt_rarray_data.append(preprocess_trial_data( trial_data, info))
    flt_data.append( preprocess_trial_data( trial_data, info).get_data())
    
flt_data = np.asarray( flt_data)


# In[93]:



flt_data.shape
flt_data_reshaped = flt_data.reshape((110,15*500 ))
flt_data_reshaped.shape


# In[94]:


flt_data.shape


# ## Visualized Filtered Data

# In[95]:


flt_rarray_data[ 0].plot( n_channels = 15, scalings = "auto")
flt_rarray_data[ 0].plot_psd( fmin = 0.1, fmax = 70)


# In[96]:


psd, freqs = mne.time_frequency.psd_array_welch(flt_data, sfreq= 500,fmin = 1, fmax = 30)


# In[97]:


freqs.shape


# In[98]:


print(freqs)


# In[99]:


psd.shape


# In[100]:


plt.plot(freqs, psd[0,4,:])


# In[101]:


all_trials_avg = []
for i in range(len(psd)):
    trial_psd = psd[i,:,:]
    delta_avg = np.mean(trial_psd[:,freqs <= 2], axis = 1)
    theta_avg = np.mean(trial_psd[:,(freqs <= 8) & (freqs >2)], axis = 1)
    alpha_avg = np.mean(trial_psd[:,(freqs <= 12) & (freqs >8)], axis = 1)
    beta_avg = np.mean(trial_psd[:,(freqs <= 30) & (freqs >12)], axis = 1)
    
    trial_avg = np.hstack((delta_avg, theta_avg, alpha_avg, beta_avg))
    all_trials_avg.append(trial_avg)
    
all_trials_avg = np.asarray(all_trials_avg)
    
    


# In[102]:


all_trials_avg.shape


# ## Time Frequency Model

# In[103]:


X_train_psd, X_test_psd, y_train_psd, y_test_psd = sklearn.model_selection.train_test_split(all_trials_avg, weight_class, test_size=0.1, random_state=42)


# In[104]:


X_train_psd.shape


# In[105]:


log_reg_psd = LogisticRegression(penalty = 'none', max_iter = 1000)

#Then fit it to data
log_reg_psd.fit(X_train_psd, y_train_psd)


# In[106]:


y_pred_psd = log_reg_psd.predict(X_test_psd)
print(y_pred_psd)


# In[107]:


print(y_test_psd)


# In[108]:


print(sklearn.metrics.r2_score(y_test_psd, y_pred_psd))


# ### Accuracy: 81%

# In[109]:


score_psd = log_reg_psd.score(X_test_psd, y_test_psd)
print(score_psd)


# In[110]:


accuracies = cross_val_score(LogisticRegression(penalty="none",max_iter = 500), all_trials_avg, weight_class, cv=8)
print(accuracies)


# ## Time By Voltge Model below

# In[111]:


# 3 seconds before they start to lift, -1 second from reach to lift, 1 second after lift
#narrow down electrodes (13 electrodes)
#1-2 delta, 3-8 theta, 9-12 alpha, 13-30 beta
#for each electrode in each trial, bin by 4 frequencies (165x(13*4)
# take average power for theta, alpha, and beta across the time window


# In[112]:


y = weight_class
print(y)
weight_class.shape


# In[113]:


X = flt_data_reshaped
#X = flt_data
X.shape


# In[114]:


y.shape


# In[115]:


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=42)


# In[116]:


pca_x_train = PCA(n_components = 0.9)
pca_x_train.fit(X_train)
X_train_reduced = pca_x_train.transform(X_train)


# In[117]:


X_train_reduced.shape


# In[118]:


#pca_x_test = PCA(n_components = 0.9)
#pca_x_train.fit(X_test)
X_test_reduced = pca_x_train.transform(X_test)
X_test_reduced.shape


# In[119]:


y_train.shape


# In[120]:


# First define the model
log_reg = LogisticRegression(penalty="none", max_iter = 500)

#Then fit it to data
log_reg.fit(X_train_reduced, y_train)


# In[121]:


y_pred = log_reg.predict(X_test_reduced)
print(y_pred)


# In[122]:


print(y_test)


# In[123]:


print(sklearn.metrics.r2_score(y_test, y_pred))


# ### Accuracy: 36%

# In[124]:


score = log_reg.score(X_test_reduced, y_test)
print(score)


# In[125]:


accuracies = cross_val_score(LogisticRegression(penalty='none'), X, y, cv=8)
print(accuracies)


# ### SVM Classification

# In[126]:


from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train_reduced)
X_train = scaler.transform(X_train_reduced)
X_test  = scaler.transform(X_test_reduced)
clf = SVC()
y_train = y_train.reshape((len(y_train),1))
y_test = y_test.reshape((len(y_test),1))
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
acc = np.mean(y_predict == y_test)
print(acc)


# In[ ]:




