from sklearn.feature_selection import SelectKBest, f_classif
from nistats.design_matrix import make_design_matrix
from nilearn.input_data import NiftiMasker
from sklearn import linear_model, metrics
from nilearn.datasets import fetch_haxby
from nilearn.image import load_img
import pandas as pd
import numpy as np


def read_data_haxby(subject, tr=2.5):
    """
    Applies anova feature selection to fmri data using classification
    accuracy on stimuli data as measure of performance

    Parameters
    ----------

    subject: int from 1 to 6
        subject from which to load haxby dataset data

    Returns
    -------

    fmri: numpy array of shape [n_scans, n_voxels]
        feature-selected train data from the fmri sessions

    stimuli: numpy array of shape [n_scans, n_categories]
        time series of the stimuli with one-hot encoding

    onsets: array of shape [n_sessions, n_stimuli]
        onset times for stimuli

    conditions: array of shape [n_sessions, n_stimuli]
        labels for stimuli

    """
    haxby_dataset = fetch_haxby(subjects=[subject])

    # Load fmri data
    fmri_filename = haxby_dataset.func[0]
    fmri = load_img(fmri_filename)
    # mask = haxby_dataset.mask_vt[0]
    masker = NiftiMasker(mask_strategy='epi', standardize=True, detrend=True,
                         high_pass=0.01, t_r=tr, smoothing_fwhm=5)
    fmri = masker.fit_transform(fmri)
    fmri = fmri.reshape(12, -1, fmri.shape[-1])

    # Load stimuli data
    classes = np.array(['rest', 'face', 'house', 'bottle', 'cat', 'chair',
                        'scissors', 'shoe', 'scrambledpix'])
    labels = np.recfromcsv(
        haxby_dataset.session_target[0], delimiter=" ")['labels'].reshape(
            12, -1)
    stimuli, onsets, conditions = (np.zeros((
        12, len(labels[0]), len(classes))), [], [])
    stimuli[0, 0] = 1
    for session in range(12):
        onsets.append([])
        conditions.append([])
        for scan in range(1, len(fmri[session])):
            if (labels[session][scan - 1] == 'rest' and
                labels[session][scan] != 'rest'):
                label = labels[session][scan]
                stimuli[session, scan, np.where(classes == label)[0][0]] = 1
                conditions[session].append(label)
                onsets[session].append(scan * tr)
            else:
                stimuli[session, scan, 0] = 1

    # Remove ninth run for subject 5 (corrupted)
    if subject == 5:
        fmri = np.vstack((fmri[:8], fmri[9:]))
        stimuli = np.vstack((stimuli[:8], stimuli[9:]))
        onsets = np.vstack((onsets[:8], onsets[9:]))
        conditions = np.vstack((conditions[:8], conditions[9:]))
    onsets = np.asarray(onsets)
    conditions = np.asarray(conditions)

    return fmri, stimuli, onsets, conditions


def feature_selection(fmri_train, fmri_test, stimuli_train, k=10000):
    """
    Applies anova feature selection to fmri data using classification
    accuracy on stimuli data as measure of performance

    Parameters
    ----------

    fmri_train: numpy array of shape [n_scans_train, n_voxels]
        train data from the fmri sessions

    fmri_test: numpy array of shape [n_scans_test, n_voxels]
        test data from the fmri sessions

    stimuli_train: numpy array of shape [n_scans_train, n_categories]
        time series of the train stimuli with one-hot encoding

    k: int, optional
        number of features to use. Defaults to 10000.

    Returns
    -------

    fmri_train: numpy array of shape [n_scans_train, k]
        feature-selected train data from the fmri sessions

    fmri_test: numpy array of shape [n_scans_test, k]
        feature-selected test data from the fmri sessions

    """

    # Fit the anova feature selection
    anova = SelectKBest(f_classif, k=k)
    fmri_train = anova.fit_transform(fmri_train, stimuli_train)

    # Transform the given test set
    fmri_test = anova.transform(fmri_test)

    return fmri_train, fmri_test


def design_matrix(n_scans, tr, onsets, conditions, durations=None,
                  hrf_model='spm', drift_model='cosine'):
    """
    Fits a Ridge regression on the data, using cross validation to choose the
    value of alpha.

    Parameters
    ----------

    n_scans: int
        number of scans in the session

    tr: float
        repetition time for the BOLD data

    onsets: array of shape [n_stimuli]
        onset times for stimuli in the session

    conditions: array of shape [n_stimuli]
        labels for stimuli in the session

    durations: array of shape [n_stimuli], optional
        durations for stimuli in the session

    hrf_model: {'spm', 'spm + derivative', 'spm + derivative + dispersion',
                'glover', 'glover + derivative',
                'glover + derivative + dispersion', 'fir'}
        HRF model to be used for creating the design matrix

    drift_model: {'polynomial', 'cosine', 'blank'}
        drift model to be used for creating the design matrix

    Returns
    -------

    design: numpy array of size [n_scans, n_regressors]
        design matrix for the given stimuli

    """
    frame_times = np.arange(n_scans) * tr
    paradigm = {}
    paradigm['onset'] = onsets
    paradigm['trial_type'] = conditions
    if durations is not None:
        paradigm['duration'] = durations
    paradigm = pd.DataFrame(paradigm)

    design = make_design_matrix(frame_times, paradigm, hrf_model=hrf_model,
                                drift_model=drift_model)

    return design


def fit_ridge(fmri_train, fmri_test, stimuli_train, stimuli_test, n_alpha=5):
    """
    Fits a Ridge regression on the data, using cross validation to choose the
    value of alpha.

    Parameters
    ----------

    fmri_train: numpy array of shape [n_scans_train, n_voxels]
        train data from the fmri sessions

    fmri_test: numpy array of shape [n_scans_test, n_voxels]
        test data from the fmri sessions

    stimuli_train: numpy array of shape [n_scans_train, n_categories]
        time series of the train stimuli with one-hot encoding

    stimuli_test: numpy array of shape [n_scans_test, n_categories]
        time series of the test stimuli with one-hot encoding

    n_alpha: int
        number of alphas to test (logarithmically distributed around 1).
        Defaults to 5.

    Returns
    -------

    prediction_train: numpy array of size [n_categories, n_train_scans]
        model prediction for the train fmri data

    prediction_test: numpy array of size [n_categories, n_test_scans]
        model prediction for the test fmri data

    score: numpy array of size [n_categories]
        prediction r2 score for each category
    """

    # Create alphas and initialize ridge estimator
    alphas = np.logspace(- n_alpha / 2, n_alpha - (n_alpha / 2), num=n_alpha)
    ridge = linear_model.RidgeCV(alphas=alphas)

    # Fit and predict
    ridge.fit(fmri_train, stimuli_train)
    prediction_train = ridge.predict(fmri_train)
    prediction_test = ridge.predict(fmri_test)

    # Compute score
    score = metrics.r2_score(stimuli_test, prediction_test,
                             multioutput='raw_values')

    return prediction_train, prediction_test, score


def logistic_deconvolution(estimation_train, estimation_test, stimuli_train,
                           stimuli_test, logistic_window, delay=0):
    """
    Learn a deconvolution filter for classification given a time window
    using logistic regression.

    Parameters
    ----------

    estimation_train: numpy array of shape [n_scans_train, n_categories]
        estimation of the categories time series for the train data

    estimation_test: numpy array of shape [n_scans_test, n_categories]
        estimation of the categories time series for the test data

    stimuli_train: numpy array of shape [n_scans_train, n_categories]
        time series of the train stimuli with one-hot encoding

    stimuli_test: numpy array of shape [n_scans_test, n_categories]
        time series of the test stimuli with one-hot encoding

    logistic_window: int
        size of the time window to be used for creating train and test data

    delay: int, optional
        delay between time series and stimuli to be applied to the data.
        Defaults to 0.

    Returns
    -------

    score: numpy array of size [n_categories]
        prediction r2 score for each category
    """

    log = linear_model.LogisticRegressionCV()

    # Add a delay between time series and stimuli if needed
    if delay != 0:
        estimation_train, estimation_test = (
            estimation_train[delay:], estimation_test[delay:])
        stimuli_train, stimuli_test = (
            stimuli_train[:-delay], stimuli_test[:-delay])

    # Create train and test masks for the stimuli (i.e. no 'rest' category)
    train_mask = np.sum(
        stimuli_train[:, 1:], axis=1).astype(bool)
    test_mask = np.sum(
        stimuli_test[:, 1:], axis=1).astype(bool)

    # Create train and test time windows
    time_windows_train = [
        estimation_train[scan: scan + logistic_window].ravel()
        for scan in xrange(len(estimation_train) - logistic_window + 1)
        if train_mask[scan]]
    time_windows_test = [
        estimation_test[scan: scan + logistic_window].ravel()
        for scan in xrange(len(estimation_test) - logistic_window + 1)
        if test_mask[scan]]

    # Create train and test stimuli labels
    stimuli_train = np.argmax(stimuli_train[train_mask], axis=1)
    stimuli_test = np.argmax(stimuli_test[test_mask], axis=1)

    # Fit logistic regression
    log.fit(time_windows_train, stimuli_train)
    accuracy = log.score(time_windows_test, stimuli_test)

    return accuracy
