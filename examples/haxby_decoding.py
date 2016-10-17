from sklearn.cross_validation import LeavePLabelOut
import time_decoding.decoding as de
import pandas as pd
import numpy as np

# Parameters
subject_list = np.arange(1, 7)
tr = 2.5
k = 10000

# GLM parameters
logistic_window = 10

scores, subjects = [], []
for subject in subject_list:
    # Read data
    fmri, stimuli, onsets, conditions = de.read_data_haxby(subject)
    session_id_fmri = [[session] * len(fmri[session])
                       for session in range(len(fmri))]
    durations = [[24] * 8] * len(fmri)
    design = [de.design_matrix(len(fmri[session]), tr, onsets[session],
                               conditions[session],
                               durations=durations[session])
              for session in range(len(fmri))]

    # Stack the BOLD signals and the design matrices
    fmri = np.vstack(fmri)
    design = np.vstack(design)
    stimuli = np.vstack(stimuli)
    session_id_fmri = np.hstack(session_id_fmri)

    lplo = LeavePLabelOut(session_id_fmri, p=2)
    for train_index, test_index in lplo:
        # Split into train and test sets
        fmri_train, fmri_test = fmri[train_index], fmri[test_index]
        design_train, design_test = design[train_index], design[test_index]
        stimuli_train, stimuli_test = stimuli[train_index], stimuli[test_index]

        # Feature selection
        fmri_train, fmri_test = de.feature_selection(
            fmri_train, fmri_test, np.argmax(stimuli_train, axis=1))

        # Fit a ridge regression to predict the design matrix
        prediction_train, prediction_test, score = de.fit_ridge(
            fmri_train, fmri_test, design_train, design_test)

        # Fit a logistic regression for deconvolution
        accuracy = de.logistic_deconvolution(
            prediction_train, prediction_test, stimuli_train,
            stimuli_test, logistic_window)

        scores.append(accuracy)
        subjects.append(subject)

    print('finished subject ' + str(subject))

dict = {}
dict['subject'] = subjects
dict['accuracy'] = scores

data = pd.DataFrame(dict)
print(np.mean(data['accuracy']))
