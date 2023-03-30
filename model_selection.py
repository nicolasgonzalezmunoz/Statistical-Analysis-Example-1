"""
This module is focused in models' scores and inferential tests.

Functions
---------
    get_polynomial_regression_scores(X, y, X_test=None, y_test=None, min_degree=1, max_degree=5, cv=10, fit_intercept=True, n_jobs=None, sample_weight=None, interaction_only=False)
    get_KMeans_scores(X, X_test=None, min_n_clusters=2, max_n_clusters=10, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=None, sample_weight=None, metric='euclidean')
    feature_significance(linear_model, X, target, test_features=-1, transformer=None, iterative=True, recursive=False, confidence=0.95, significance={}, level=1, last_features_dropped=None, copy_test_features=None, use_copy=False)
    model_significance(model1, model2, X, target, transformer1=None, transformer2=None, confidence=0.95)
    white_test(residuals, data)
    bptest(residuals, data)
"""

import numpy as np
import seaborn as sns

from scipy.stats import f, chi2

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_samples, silhouette_score

def get_polynomial_regression_scores(X, y, X_test=None, y_test=None, min_degree=1, max_degree=5, cv=10, fit_intercept=True, n_jobs=None, sample_weight=None, interaction_only=False):
    """
    Returns arrays of R2 scores for each polynomial regressor with
    degree between min_degree and max_degree.

    Note: R2_cv_scores.shape == (cv, n_iterations)

    Parameters
    ----------
        X : 2d array)
            Fitting features
        y : 1d array
            Fitting target
        X_test : 2d array
            Testing features (default: None)
        y_test : 1d array
            Testing target (default: None)
        min_degree : int
            A positive integer (default: 1)
        max_degree : int
            A positive integer greater than min_degree (default: 5)
        cv : int
            Cross-validation parameter
        fit_intercept, n_jobs
            LinearRegression parameters
        sample_weight
            LinearRegression fitting parameter
        interaction_only
            PolynomialFeatures parameter
        
    Returns
    -------
        R2_scores : 1d array
            Array of R2 scores for all the iterated degrees
        R2_test_scores : 1d array
            Array of R2 scores for the test set, if provided
        R2_cv_scores : 2d array
            Array of R2 cross-validation scores
    """
    R2_scores = []
    R2_cv_scores = []
    if not (X_test is None or y_test is None):
        R2_test_scores = []

    for degree in np.arange(min_degree, max_degree+1, 1):
        poly_features = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=interaction_only)
        poly_features.fit(X)
        poly_train = poly_features.transform(X)
        poly_model = LinearRegression(fit_intercept=fit_intercept, n_jobs=n_jobs)
        poly_model.fit(X=poly_train, y=y, sample_weight=sample_weight)
        R2_train = poly_model.score(X=poly_train, y=y)
        R2_cv = cross_val_score(estimator=poly_model, X=poly_train, y=y, cv=cv)
        if degree -min_degree > 0:
            condition = R2_train < R2_scores[-1]*0.75 or np.mean(R2_cv)< np.mean(np.array(R2_cv_scores)[-1,:])*0.75
        if not (X_test is None or y_test is None):
            poly_test = poly_features.transform(X_test)
            R2_test = poly_model.score(X=poly_test, y=y_test)
            if degree -min_degree > 0:
                condition = condition or R2_test < R2_test_scores[-1]*0.75
        if degree - min_degree == 0:
            condition = False
        if condition:
            break
        R2_scores.append(R2_train)
        R2_cv_scores.append(R2_cv)
        if not (X_test is None or y_test is None):
            R2_test_scores.append(R2_test)
    R2_scores = np.array(R2_scores)
    R2_cv_scores = np.array(R2_cv_scores)
    if not (X_test is None or y_test is None):
        R2_test_scores = np.array(R2_test_scores)
        return R2_scores, R2_test_scores, R2_cv_scores.T

    return R2_scores, R2_cv_scores.T

def get_KMeans_scores(X, X_test=None, min_n_clusters=2, max_n_clusters=10, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=None, sample_weight=None, metric='euclidean', with_labels=True, with_centroids=True):
    """
    Returns an array of scores for each KMeans model with n_clusters between
    min_n_clusters and max_n_clusters. If X_test is entered, it also returns
    the scores for the test data.

    Note: The returns that are 2d array have a shape of
    (n_instances, n_iterations). In the case of centroids_list, each array
    contained in it is in the format (n_clusters, n_features)

    Parameters
    ----------
        X : 1d or 2d array 
            Fitting features.
        X_test : 1d or 2d array 
            Testing features (default: None).
        min_n_clusters : int 
            A positive integer (default: 1).
        max_n_clusters : int 
            A positive integer (default: 5).
        init, n_init, max_iter, tol, random_state 
            KMeans parameters.
        sample_weight 
            KMeans fitting parameter.
        metric : str
            silhouette_samples parameter.
        with_labels : bool 
            Defines if the labels of each cluster is returned (default: True).
        with_centroids : bool
            Defines if the centroids for each iteration are returned
            (default: True).

            
    Returns
    -------
        inertia : 1d array 
            Array of KMeans inertia score for each n_cluster iteration.
        sil_scores : 1d array
            Array of silhouette_score computed for each n_cluster iteration.
        sil_samples : 2d array
            Array of silhouette_samples for each n_cluster iteration.
        homogeneity_scores : 1d array
            Array of metrics that measures clustering variability.
        sil_test_scores : 1d array
            If X_test is not None: Array of silhouette_score computed
            for each n_cluster iteration on test data.
        sil_test_samples : 2d array
            If X_test is not None: Array of silhouette_samples for
            each n_cluster iteration on test data.
        homogeneity_test_scores : 1d array
            If X_test is not None: Array of metrics that measures
            clustering variability on test data.
        labels_array : 2d array
            If with_labels is set to True: numpy array with labels
            obtained by KMeans.
        centroids_list : list of arrays
            If with_centroids is set to True: list of numpy arrays
            with the centroids obtained by KMeans.
    """

    std_scal = StandardScaler()
    if X.size == X.shape[0]:
        X_std = std_scal.fit_transform(X.reshape(-1, 1))
        if not X_test is None:
            X_test_std = std_scal.transform(X_test.reshape(-1, 1))
    else:
        X_std = std_scal.fit_transform(X)
        if not X_test is None:
            X_test_std = std_scal.transform(X_test)
        
    inertia = []   
    sil_scores = []
    sil_samples = []
    homogeneity_scores = []
    if not X_test is None:
        sil_test_scores = []
        sil_test_samples = []
        homogeneity_test_scores = []

    flag = 0
    labels_array = []
    centroids_list = []
    for k in np.arange(min_n_clusters, max_n_clusters+1, 1):
        kmeans = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state)
        kmeans.fit(X_std, sample_weight=sample_weight)
        labels = kmeans.predict(X_std)
        labels_array.append(labels)
        centroids_list.append(kmeans.cluster_centers_)
        

        n_on_cluster = np.array([np.sum(labels == label) for label in labels])
        mean_n_on_cluster = np.mean(n_on_cluster)
        homogeneity = np.var(np.array([np.sum(labels == label)-mean_n_on_cluster for label in labels])/X_std.size)

        if not X_test is None:
            labels_test = kmeans.predict(X_test_std)
            n_on_cluster_test = np.array([np.sum(labels_test == label) for label in labels])
            mean_n_on_cluster_test = np.mean(n_on_cluster_test)
            homogeneity_test = np.var(np.array([np.sum(labels_test == label)-mean_n_on_cluster_test for label in labels])/X_test_std.size)

        if k - min_n_clusters > 0:
            sil_samp = silhouette_samples(X_std, labels, metric=metric)
            sil = silhouette_score(X_std, labels, metric=metric, random_state=random_state)
            if k - min_n_clusters > 1:
                condition = sil < sil_scores[-1]*0.75 or np.unique(kmeans.labels_).size < k
            if not X_test is None:
                sil_test_samp = silhouette_samples(X_test_std, labels_test, metric=metric)
                sil_test = silhouette_score(X_test_std, labels_test, metric=metric, random_state=random_state)
                if k - min_n_clusters > 1:
                    condition = condition or sil_test < sil_test_scores[-1]*0.75
            if k - min_n_clusters == 1:
                condition = False
            if condition:
                flag =1
                break
            sil_scores.append(sil)
            sil_samples.append(sil_samp)
            if not X_test is None:
                sil_test_scores.append(sil_test)
                sil_test_samples.append(sil_test_samp)
        if flag == 1:
            break
        homogeneity_scores.append(homogeneity)
        inertia.append(kmeans.inertia_)
        if not X_test is None:
            homogeneity_test_scores.append(homogeneity_test)
    
    labels_array = np.array(labels_array).T
    centroids_list = np.array(centroids_list)
    inertia = np.array(inertia).T
    sil_scores = np.array(sil_scores)
    sil_samples = np.array(sil_samples).T
    homogeneity_scores = np.array(homogeneity_scores)
    if not X_test is None:
        sil_test_scores = np.array(sil_test_scores)
        sil_test_samples = np.array(sil_test_samples).T
        homogeneity_test_scores = np.array(homogeneity_test_scores)

        if with_labels and with_centroids:
            return inertia, sil_scores, sil_test_scores, sil_samples, sil_test_samples, homogeneity_scores, homogeneity_test_scores, labels_array, centroids_list
        elif with_labels:
            return inertia, sil_scores, sil_test_scores, sil_samples, sil_test_samples, homogeneity_scores, homogeneity_test_scores, labels_array
        elif with_centroids:
            return inertia, sil_scores, sil_test_scores, sil_samples, sil_test_samples, homogeneity_scores, homogeneity_test_scores, centroids_list
        return inertia, sil_scores, sil_test_scores, sil_samples, sil_test_samples, homogeneity_scores, homogeneity_test_scores

    return inertia, sil_scores, sil_samples, homogeneity_scores

def get_VIF(X):
    """
    Calculates the Variance Inflation Factor for each feature in the array X.

    Parameters
    ----------
        X : 1d or 2d array
            matrix with the instances.
    
    Returns
    -------
        vif : 1d array
            Array with the VIF of each feature.
    """
    vif = np.ones(X.shape[1])
    for i in np.arange(X.shape[1]):
        data = X[:,np.arange(X.shape[1])!=i]
        target = X[:,i]
        linear_model = LinearRegression()
        linear_model.fit(data, target)
        R2 = linear_model.score(data, target)
        if R2 == 1:
            vif[i] = np.inf
        else:
            vif[i] = 1/(1-R2)
    return vif


def feature_significance(linear_model, X, target, test_features=-1, transformer=None, iterative=True, recursive=False, confidence=0.95, significance={}, level=1, last_features_dropped=None, copy_test_features=None, use_copy=False):
    """
    Apply an F-test to the model on the features to test.

    If iterative is set to True, the test is performed to each test feature
    separately. When set to False, the test is applied on all the features
    simultaneously.

    If recursive is set to True, the algorithm drops the features that are not
    significant, and keeps testing with the remaining features until all
    features are considered significant or there are no more features to test.
    This parameter is ignored if iterative is set to False.

    Parameters
    ----------
        linear_model : scikit learn linear regressor
            Model on which the test is performed.
        X : 1d or 2d array
            Regressor features.
        target : 1d array
            True values of the dependent variable.
        test_features : int or 1d array
            Array of indexes of features to test. If want to test on the
            intercept, include the value -1.
        transformer 
            Any scikit learn class with the method fit_transform.
        iterative : bool
            Wether or not you want to perform the test individually in each
            feature iteratively.
        recursive : bool
            Wether or not you want to perform the test recursively.
        confidence : float
            Level of confidence of the test. If the p_value is smaller than
            1-confidence, then the feature(s) are considered significant.
        significance : dict
            This is an internal parameter should not be set by the user,
            since it is used in iterative and recursive mode.
        level : int
            Internal parameter used in iterative and recursive mode.
        last_features_dropped : int or list
            Internal variable that keeps a record of the features dropped
            for the previous iteration.
        last_features_dropped : int or list
            Internal variable that keeps a record of the features dropped
            for the current iteration.
        copy_test_features : list
            Internal parameter that keeps a record of the original features
            to test and the ones dropped for the level.
        use_copy : bool
            Internal variable to decide wether or not to use copy_test_features
            instead of test_features.

    Returns
    -------
        significance : dict
            dict of nested dicts with the info on feature significance at
            each iteration level.
    """

    # Create array to keep track of the dropped features
    if X is None or test_features is None:
        return significance
    if last_features_dropped is None:
        current_features_dropped = []
    else:
        current_features_dropped = list(last_features_dropped)

    unrest_fit_intercept = linear_model.fit_intercept
    if X.size == X.shape[0]:
        X = X.reshape(-1, 1)

    if np.issubdtype(type(test_features), np.integer):
        test_features = np.array([test_features])
    
    # Keep track of the features that remain to be tested
    test_features = np.unique(test_features)

    if test_features.size > test_features.shape[0]:
        raise Exception('Expected 1d array on test_features, received '+str(len(test_features.shape))+'d array instead')
    if test_features[0] < -1 or test_features[-1] > X.shape[1] - 1:
        raise Exception('test_features values are out of range')

    if copy_test_features is None:
        copy_test_features = test_features.tolist()
    if use_copy:
        if len(copy_test_features) < 2:
            test_features = np.array([copy_test_features])
        else:
            test_features = np.array(copy_test_features)
    
    # Keep track of the features will be tested on this iteration
    is_tested = []
    flag = 0

    # Check if intercept has to be tested
    if test_features[0] == -1:
        is_tested.append(-1)
        fit_intercept = not unrest_fit_intercept
        if test_features.size > 1:
            test_features = np.delete(test_features, 0, axis=0)
        else:
            test_features = None
        flag = 1
    else:
        fit_intercept = unrest_fit_intercept
    
    # Check the features that have to be tested (if any)
    if iterative and not flag:
        if test_features is None:
            index_to_drop = None
        else:
            index_to_drop = np.array([test_features[0]])
            if test_features.size == 1:
                test_features = None
            else:
                test_features = np.delete(test_features, 0, axis=0)
    elif not iterative:
        index_to_drop = np.array(test_features)
        test_features = None
    else:
        index_to_drop = None
    
    # Get the data for the restricted model
    if not (index_to_drop is None or index_to_drop == []):
        if index_to_drop[index_to_drop>-1].size >= X.shape[1]:
            X_rest = None
        else:
            X_rest = np.delete(X, index_to_drop[index_to_drop>-1], axis=1)
    else:
        X_rest = X
    if not X_rest is None:
        if X_rest.size == X_rest.shape[0]:
            X_rest = X_rest.reshape(-1, 1)
    # Get list of tested features
    if not (index_to_drop is None):
        for i in np.arange(index_to_drop.size):
            is_tested.append(index_to_drop[i])

    # Apply polynomial features if necessary
    if not (transformer is None):
        X_feat = transformer.fit_transform(X)
        if not (X_rest is None):
            X_rest_feat = transformer.fit_transform(X_rest)
        else:
            X_rest_feat = None
    else:
        X_feat = X
        X_rest_feat = X_rest
    
    # Compute the residual sum of squares of the restricted and unrestricted model
    linear_model.fit(X_feat, target)
    unrest_pred = linear_model.predict(X_feat)

    if not (X_rest_feat is None):
        linear_model.fit_intercept = fit_intercept
        linear_model.fit(X_rest_feat, target)
        rest_pred = linear_model.predict(X_rest_feat)
    else:
        rest_pred = np.mean(target)
    linear_model.fit_intercept = unrest_fit_intercept

    rrsu = np.var(target- unrest_pred)
    rrsr = np.var(target- rest_pred)

    # Compute the F statistic and its p-value
    dfd = unrest_pred.size-X_feat.shape[1]
    if not (X_rest_feat is None):
        dfn = X_feat.shape[1]-X_rest_feat.shape[1]
        if is_tested[0] == -1:
            dfn += 1
        if X_rest_feat.shape[1] < X_feat.shape[1] and flag:
            dfn = 1
            dfd = 1
    else:
        dfn = X_feat.shape[1]
    f_statistic = (rrsr-rrsu)*dfd/(rrsr*dfn)
    p_value = 1-f.cdf(f_statistic, dfn, dfd)
    
    is_significant = p_value < 1 - confidence

    # Initialize the dict for the current level
    level_key = 'Level ' + str(level)
    if level_key not in significance.keys():
        significance[level_key] = {}

    # Get the nested keys for the level.  There is one key indicating
    # which features have been dropped in previous loops, and a key
    # indicating if the feature tested is significant or not.
    # The loop tracks the features to their initial position, before
    # any feature was dropped.  Also, append the features dropped at
    # this level to the corresponding list
    for feature in is_tested:
        tested_feature = feature
        if level > 1:
            n_dropped_features = len(last_features_dropped)
            if n_dropped_features == 1:
                dropped_key = 'When Dropping Feature '
            else:
                dropped_key = 'When Dropping Features '
            
            for feat in np.arange(n_dropped_features - 1, -1, -1):
                dropped_feature = last_features_dropped[feat]
                if tested_feature >= dropped_feature and dropped_feature >= 0:
                    tested_feature += 1
                dropped_key += str(last_features_dropped[n_dropped_features - feat - 1])+', '
        else:
            dropped_key = None

        tested_key = 'Feature ' + str(tested_feature)
        current_features_dropped.append(tested_feature)
        if not (dropped_key is None):
            if dropped_key not in significance[level_key].keys():
                significance[level_key][dropped_key] = {}
        msg = 'Significant'
        if not is_significant:
            msg = 'Not ' + msg
        if dropped_key is None:
            significance[level_key][tested_key] = {}
            significance[level_key][tested_key]['Significance'] = msg
            significance[level_key][tested_key]['F-Statistic'] = f_statistic
            significance[level_key][tested_key]['p-value'] = p_value
        else:
            significance[level_key][dropped_key][tested_key] = {}
            significance[level_key][dropped_key][tested_key]['Significance'] = msg
            significance[level_key][dropped_key][tested_key]['F-Statistic'] = f_statistic
            significance[level_key][dropped_key][tested_key]['p_value'] = p_value
    
    # If there is no features to test, then end the recursion
    if test_features is None:
        return significance
    
    # If iterative is True, call this function recursively on the features
    # that remain to be tested at the current level.  If additionaly recursive
    # is True, then call this function again, but dropping the non significant
    # features.
    if iterative:
        use_copy = False
        significance = feature_significance(linear_model, X, target, test_features=test_features, transformer=transformer, iterative=iterative, recursive=recursive, confidence=confidence, significance=significance, level=level, last_features_dropped=last_features_dropped, copy_test_features=copy_test_features, use_copy=use_copy)
        if recursive and not is_significant:
            level = level + 1
            copy_index_to_drop = [i for i in np.arange(len(copy_test_features)) if copy_test_features[i] == current_features_dropped[-1]][0]
            copy_test_features.pop(copy_index_to_drop)

            if is_tested[0] > -1:
                index_to_translate = test_features >= is_tested[0]
                test_features[index_to_translate] -= 1

                for i in np.arange(copy_index_to_drop, len(copy_test_features), 1):
                    copy_test_features[i] -= 1
            else:
                linear_model.fit_intercept = fit_intercept
            
            use_copy = True
            significance = feature_significance(linear_model, X_rest, target, test_features=test_features, transformer=transformer, iterative=iterative, recursive=recursive, confidence=confidence, significance=significance, level=level, last_features_dropped=current_features_dropped, copy_test_features=copy_test_features, use_copy=use_copy)
    
    return significance

def model_significance(model1, model2, X, target, transformer1=None, transformer2=None, confidence=0.95):
    """
    Computes wether or not a model is statisticaly better than other model.

    It firts transforms the data using the transformers, if given, then
    identify the complex model as the model with lesser residual variance 
    and compares it with the simpler model using an F-test.

    Parameters
    ----------
        model1, model2
            scikit learn regressors to compare. Both of them have to
            support have to support the fit_predict method and posses the 
            fit_intercept parameter.
        X : 1d or 2d array
            Data to use for fitting
        target : 1d array
            Target data to use for fitting.
        transformer1, transformer2
            Preprocessing step corresponding to model1 and model2,
            respectively. Supports any scikit learn class with the
            fit_transform method.
        confidence : float
            Confidence of the test.
    Returns
    -------
        is_significant : bool
            If the complex model is statisticaly better than than the simpler
            one, returns True. Else, returns False
    """

    if X.size == X.shape[0]:
        X = X.reshape(-1, 1)
    
    # Transform data
    if not (transformer1 is None):
        X1 = transformer1.fit_transform(X)
    else:
        X1 = X
    if not (transformer2 is None):
        X2 = transformer2.fit_transform(X)
    else:
        X2 = X
    
    # Verify the value of the fit_intercept parameter
    fit_intercept1 = model1.fit_intercept
    fit_intercept2 = model2.fit_intercept

    y_pred1 = model1.fit_predict(X1, target)
    y_pred2 = model2.fit_predict(X2, target)

    rrs1 = np.var(target - y_pred1)
    rrs2 = np.var(target - y_pred2)
    
    # Get parameters of the F-test
    if rrs1 > rrs2:
        complex_model = 'model 2'
        simple_model = 'model 1'
        rrsu = rrs2
        rrsr = rrs1
        dfn = X1.shape[1]-X2.shape[1]
        dfd = y_pred2.size-X2.shape[1]
        if fit_intercept1:
            dfn += 1
        if fit_intercept2:
            dfn -= 1
            dfd -= 1
    else:
        complex_model = 'model 1'
        simple_model = 'model 2'
        rrsu = rrs1
        rrsr = rrs2
        dfn = X2.shape[1]-X1.shape[1]
        dfd = y_pred1.size-X1.shape[1]
        if fit_intercept1:
            dfn -= 1
            dfd -= 1
        if fit_intercept2:
            dfn += 1
    
    if dfn == 0:
        dfn = 1
        dfd = 1
    
    # Perform F-test
    f_statistic = (rrsr-rrsu)*dfd/(rrsr*dfn)
    p_value = 1-f.cdf(f_statistic, dfn, dfd)
    is_significant = p_value < 1 - confidence

    # Print a message with the results
    msg = 'The more complex model, ' + complex_model + 'is '
    if not is_significant:
        msg += 'not '
    msg += 'statisticaly better than \nthe simpler model, ' + simple_model + '\n'
    print(msg)
    return is_significant

def white_test(residuals, data):
    """
    Performs a White test for homoscedasticity. Also prints a message
    with the results.

    Parameters
    ----------
        residuals : 1d array
            Residuals of a regression.
        data : 1d or 2d array
            Data with the features.
        
    Returns
    -------
        chi_stat : float
            Value of the statistic.
        p_value : float
            p-value of the test.
    """
    sqrd_residuals = np.square(residuals)
    poly_feat = PolynomialFeatures(degree=2, include_bias=False)
    transformed_data = poly_feat.fit_transform(data)
    aux_model = LinearRegression()
    aux_model.fit(X=transformed_data, y=sqrd_residuals)
    residual_R2 = aux_model.score(X=transformed_data, y=sqrd_residuals)
    chi_stat = residual_R2*residuals.size
    dof = 2
    p_value = 1-chi2.cdf(chi_stat, dof)
    print('Chi-statistic = ', chi_stat, ' p_value = ', p_value)
    return chi_stat, p_value

def bptest(residuals, data):
    """
    Performs a Breusch-Pagan test for homoscedasticity. Also prints a message
    with the results.

    Parameters
    ----------
        residuals : 1d array
            Residuals of a regression.
        data : 1d or 2d array
            Data with the features.
        
    Returns
    -------
        f_stat : float
            Value of the statistic.
        p_value : float
            p-value of the test.
    """
    sqrd_residuals = np.square(residuals)
    transformed_residuals = sqrd_residuals*residuals.size/np.var(residuals)
    aux_model = LinearRegression()
    aux_model.fit(X=data, y=transformed_residuals)
    res_pred = aux_model.predict(X=data)
    chi_stat = (np.var(transformed_residuals - 1) - np.var(res_pred))/2
    dof = 2
    p_value = 1-chi2.cdf(chi_stat, dof)
    print('Chi-statistic = ', chi_stat, ' p_value = ', p_value)
    return chi_stat, p_value