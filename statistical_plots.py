"""
This module is focused in routinary statistical plotting tasks.

Note: The centroids data is always assumed to be in the format (n_features, n_clusters)
or (n_features, n_clusters, n_iterations). The data itself to be plotted is assumed
to be in the format (n_instances, n_features) or (n_instances, ) if n_features == 1.
Any other variable is entered in the format (n_features, n_iterations) or
(n_features, ) in the case n_features == 1.

Functions
---------
    get_plot_grid(n_axes)
    get_figsize(ax=None, scale=(6.4, 4.8))
    plot_one_vs_each(data, labels, pivot=0, n_timestamps=None, scale=(6.4, 4.8), suptitle='', style='ro', ms=1, sharex=True)
    plot_each_vs_each(data, labels, n_timestamps=None, scale=(6.4, 4.8), suptitle='', style='ro', ms=1)
    plot_correlation_matrix(corr_coef, labels, scale=(6.4, 4.8), suptitle='', triangular=True, with_colorbar=False, cmap_style='jet', annotation_color=None, set_bad='w')
    qq_plot(data, x_label='Randomly generated data', y_label='Errors', data_style='bo', random_style='r-', scale=(6.4, 4.8), suptitle='', standar=False)
    plot_histogram(data, x_label='Errors', y_label='Density', scale=(6.4, 4.8), suptitle='', normal=False, standar=False, plot_random_data=True)
    plot_scores(train_score, test_score=None, cross_val_score=None, x_array=None, labels=None, x_label='Degrees', y_label='R2', scale=(6.4, 4.8), suptitle='', styles=None, plot_full=False)
    plot_cross_val_scores(scores, x_array=None, labels=None, x_label='Degrees', y_label='R2', scale=(6.4, 4.8), suptitle='', styles=None, with_means=True)
    plot_silhouette_samples(samples, cluster_labels, ks=None, x_labels=None, y_label=None, scale=(6.4, 4.8), suptitle='', with_means=True, share_x=True)
    plot_clusters(data, cluster_labels, centroids=None, centroid_color='r', centroid_marker='*', centroid_size=250, features_to_plot=None, ks=None, x_labels=None, y_label=None, scale=(6.4, 4.8), suptitle='')
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from scipy.stats import norm

def get_plot_grid(n_axes):
    """
    Gets the number of rows and columns with the desired number of axes

    It searches two number n_row and n_col such that n_row*n_col=n_axes
    and minimize the quantity n_row+n_col.

    Parameters
    ----------
        n_axes: int 
            Number of desired axes.
        
    Returns
    -------
        n_rows, n_cols : int 
            Number of rows and columns, respectively.
    """

    min_cols = int(math.floor(math.sqrt(n_axes)))
    for current_cols in np.arange(min_cols, 0, -1):
        quotient = n_axes//current_cols
        if quotient*current_cols == n_axes:
            n_cols = int(current_cols)
            break
    n_rows = int(n_axes/n_cols)
    return n_rows, n_cols

def get_figsize(ax=None, scale=(6.4, 4.8)):
    """
    Gets the figsize that keeps the scale constant for the given number of axes,
    that is, the size of each axis is equal to scale.

    This function assumes that all the axes have the same size when doing this
    task.

    Parameters
    ----------
        ax 
            Figure axes from which we obtain the margins.
        scale : tuple[float, float] 
            Scale of the figure that we want to keep constant.
        
    Returns
    -------
        figw, figh : floats
            Width and heigh of the figure, respectively.
    """

    if ax is None: 
        ax = plt.gca()

    l_margin = ax.figure.subplotpars.left
    r_margin = ax.figure.subplotpars.right
    t_margin = ax.figure.subplotpars.top
    b_margin = ax.figure.subplotpars.bottom
    w = scale[0]
    h = scale[1]
    figw = float(w)/(r_margin-l_margin)
    figh = float(h)/(t_margin-b_margin)

    return figw, figh

def plot_one_vs_each(data, labels, pivot=0, n_timestamps=None, scale=(6.4, 4.8), suptitle='', style='ro', ms=1, sharex=True):
    """
    Plots the variable on column pivot vs the others.

    If n_timestamps is not specified, or if ms is an array of size > 1, then
    the algorithm assumes that each feature have ms.size=n_timestamps columns
    of their own in data, treating it as a concatenated set of time series with
    ms.size=n_timestamps timestamps.

    Parameters
    ----------
        data : 2d array
            Data to plot.
        labels : 1d array or list
            Labels of the data to be plotted.
        pivot : int
            Index of the feature that remains fixed when plotting.
        n_timestamps : int
            Number of timestamps for each feature.
        scale : tuple[float, float]
            Scale to preserve on axes.
        suptitle : str
            suptitle of the plot.
        style : str
            Style of the plot lines or markers.
        ms : float or array
            Size of the markers. If you don't want any markers, set it to None.
        sharex : bool
            Wether or not you want the axes to share the same x axis
        
    Returns
    -------
        fig, ax
            The figure and axes of the plot, respectively.
    """

    # Verify value types and set new values if necessary
    if type(ms) == int or type(ms) == float:
        if n_timestamps is None:
            ms = np.array([ms]*data.shape[1])
        else:
            ms = np.array([ms]*n_timestamps)
    
    if ms is None and n_timestamps is None:
        n_timestamps = 1
    elif n_timestamps is None:
        n_timestamps = ms.size
    
    # Plot data
    n_rows = data.shape[1]/n_timestamps - 1
    fig, ax = plt.subplots(n_rows, 1, sharex=sharex)

    flag = 0  # Indicates if there is only one axis or an array of axes
    if type(ax) != np.ndarray:
        ax = np.array([ax])
        flag = 1

    row_count = 0
    for ind in np.arange(data.shape[1]/n_timestamps):
        if ind != pivot:
            for t in np.arange(n_timestamps):
                if ms is None:
                    ax[row_count].plot(data[:,pivot*n_timestamps+t], data[:,ind*n_timestamps+t], style=style)
                else:
                    ax[row_count].plot(data[:,pivot*n_timestamps+t], data[:,ind*n_timestamps+t], style=style, ms=ms[t])
            ax[row_count].set_xlabel(labels[pivot], fontsize=14)
            ax[row_count].set_ylabel(labels[ind], fontsize=14)
            ax[row_count].grid()
            row_count += 1

    fig.suptitle(suptitle, fontsize=20)
    figw, figh = get_figsize(ax[0], scale=scale)
    fig.set_figheight(figh)
    fig.set_figwidth(figw)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if flag:
        return fig, ax[0]
    return fig, ax

def plot_each_vs_each(data, labels, n_timestamps=None, scale=(6.4, 4.8), suptitle='', style='ro', ms=1):
    """
    Plots each variable vs the others.

    If n_timestamps is not specified, or if ms is an array of size > 1, then
    the algorithm assumes that each feature have ms.size=n_timestamps columns
    of their own in data, treating it as a concatenated set of time series with
    ms.size=n_timestamps timestamps.

    Parameters
    ----------
        data : 2d array
            Data to plot.
        labels : 1d array
            Labels of the data to be plotted.
        n_timestamps : int
            Number of time stamps per feature.
        scale : tuple[float, float]
            Scale to preserve on axes.
        suptitle : str
            Suptitle of the plot.
        style : str
            Style of the plot lines or markers.
        ms : float or array
            Size of the markers. If you don't want any markers, set it to None.
        
    Returns
    -------
        fig, ax
            The figure and axes of the plot, respectively.
    """

    if type(ms) == int or type(ms) == float:
        if n_timestamps is None:
            ms = np.array([ms]*data.shape[1])
        else:
            ms = np.array([ms]*n_timestamps)
    
    if ms is None and n_timestamps is None:
        n_timestamps = 1
    elif n_timestamps is None:
        n_timestamps = ms.size
    
    n_axes = int((data.shape[1]/n_timestamps-1)*(data.shape[1]/n_timestamps)/2)
    n_rows, n_cols = get_plot_grid(n_axes)
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)
    flag = 0  # Indicates if there is only one axis or an array of axes
    if type(ax) != np.ndarray:
        ax = np.array([[ax]])
        flag = 1
    elif ax.size == ax.shape[0]:
        ax = np.array([ax]).T
        flag = 2
    row_count = 0
    col_count = 0
    for row in np.arange(int(data.shape[1]/n_timestamps)):
        for col in np.arange(row + 1, int(data.shape[1]/n_timestamps), 1):
            for t in np.arange(n_timestamps):
                if ms is None:
                    ax[row_count, col_count].plot(data[:,row*n_timestamps+t], data[:,col*n_timestamps+t], style)
                else:
                    ax[row_count, col_count].plot(data[:,row*n_timestamps+t], data[:,col*n_timestamps+t], style, ms=ms[t])
            ax[row_count, col_count].set_xlabel(labels[row], fontsize=14)
            ax[row_count, col_count].set_ylabel(labels[col], fontsize=14)
            ax[row_count, col_count].grid()
            
            row_count += 1
            if row_count >= n_rows:
                row_count = 0
                col_count += 1

    fig.suptitle(suptitle, fontsize=20)
    figw, figh = get_figsize(ax=ax[0, 0], scale=scale)
    fig.set_figheight(figh)
    fig.set_figwidth(figw)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if flag == 1:
        return fig, ax[0, 0]
    elif flag == 2:
        return fig, ax[:, 0]
    return fig, ax

def plot_correlation_matrix(corr_coef, labels, scale=(6.4, 4.8), suptitle='', triangular=True, with_colorbar=False, cmap_style='jet', annotation_color=None, set_bad='w'):
    """
    Creates a color map with the given data and settings.

    Parameters
    ----------
        corr_coef : 2d array
            correlation matrix to plot.
        labels : list or 1d array
            Names of the features.
        scale : tuple[float, float]
            Scale to preserve on axes.
        suptitle : str
            Suptitle of the plot.
        triangular : bool
            If True, only plots the cells below the diagonal.
        with_colorbar : bool
            If True, appends a colorbar to the plot.
        cmap_style : str
            Style to apply to the color map.
        annotation_color : str or None
            color of the annotations to write in each cell. If not defined,
            the algorithm chooses the more contrastant color for each cell.
        set_bad : str
            If triangular is set to True, it sets the color of the annotations
            in the cells not considered for plotting.
    
    Returns
    -------
        fig, ax
            The figure and axes of the plot, respectively.
    """
    if triangular:
        mask =  np.triu(corr_coef, k=0)
    else:
        mask =  np.zeros(corr_coef.shape)
    corr_coef = np.ma.array(corr_coef, mask=mask)

    fig, ax = plt.subplots()
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    cmap = cm.get_cmap(cmap_style)
    cmap.set_bad(set_bad)
    im = ax.imshow(corr_coef, cmap=cmap)
    if with_colorbar:
        fig.colorbar(im, ax=ax)
    color_flag = 0
    if annotation_color is None:
        color_matrix = im.cmap(im.norm(im.get_array()))
        contrast_color_matrix = color_matrix[:,:,:3]
        ind_set_to_0 = contrast_color_matrix > 0.5
        ind_set_to_255 = contrast_color_matrix <= 0.5
        contrast_color_matrix[ind_set_to_0] = 0
        contrast_color_matrix[ind_set_to_255] = 255
        contrast_color_matrix = contrast_color_matrix.astype(int)
        color_flag = 1

    ax.set_xticks(np.arange(corr_coef.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(corr_coef.shape[1]))
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(corr_coef.shape[0]):
        for j in range(corr_coef.shape[1]):
            if color_flag:
                annotation_color = tuple(contrast_color_matrix[i, j, :].tolist())
                annotation_color = '#%02x%02x%02x' % annotation_color
            if not mask[i, j]:
                ax.text(j, i, np.round(corr_coef[i, j],2), ha="center", va="center", color=annotation_color)
            else:
                ax.text(j, i, np.round(corr_coef[i, j],2), ha="center", va="center", color=set_bad)
    
    fig.suptitle(suptitle, fontsize=20)
    figw, figh = get_figsize(ax=ax, scale=scale)
    fig.set_figheight(figh)
    fig.set_figwidth(figw)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, ax

def qq_plot(data, x_label='Randomly generated data', y_label='Errors', data_style='bo', random_style='r-', scale=(6.4, 4.8), suptitle='', standar=False):
    """
    Generates a Q-Q plot for the given data, comparing it with random
    generated from a normal distribution.

    Parameters
    ----------
        data : 1d array
            data to plot.
        x_label : str
            Label of the x axis.
        y_label : str
            Label of the y axis.
        data_style : str
            Style of the markers in the data plot.
        random_style : str
            Style of the line of the random plot.
        scale : tuple[float, float]
            Scale to preserve on axes.
        suptitle : str
            Suptitle of the plot.
        standar : bool
            If set to True, standarize the so it has mean 0 and variance 1.
    
    Returns
    -------
        fig, ax
            The figure and axes of the plot, respectively.
    """
    mean = np.mean(data)
    std = np.std(data)
    if standar:
        data = (data - mean)/std
        mean = 0
        std = 1
    random_data = np.random.normal(loc=mean, scale=std, size=data.size)
    data = np.sort(data)
    random_data = np.sort(random_data)
    fig, ax = plt.subplots()
    ax.plot(random_data, random_data, random_style, label='Normal Distribution')
    ax.plot(random_data, data, data_style)
    ax.legend()
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    fig.suptitle(suptitle, fontsize=20)
    figw, figh = get_figsize(ax=ax, scale=scale)
    fig.set_figheight(figh)
    fig.set_figwidth(figw)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, ax

def plot_histogram(data, x_label='Errors', y_label='Density', scale=(6.4, 4.8), suptitle='', normal=False, standar=False, plot_random_data=True):
    """
    Creates an histogram using the given data. If plot_random_data is set to
    True, it also plots a line over the histogram which contains random
    points generated from a normal distribution.

    Parameters
    ----------
        data : 1d array
            Data to plot.
        x_label : str
            Label for the x axis.
        y_label : str
            Label for the y axis.
        scale : tuple[float, float]
            Scale to preserve on axes.
        suptitle : str
            Suptitle of the plot.
        normal : bool
            If True, the algorithm generates the histogram bins based on
            the Sturgeon criterion. Else, it uses the Doane criterion.
        standar : bool
            If set to True, standarize the so it has mean 0 and variance 1.
        plot_random_data : bool
            If True, it plots a line over the histogram which contains data
            generated from a normal distribution.
        
    Returns
    -------
        fig, ax
            The figure and axes of the plot, respectively.
    """
    mean = np.mean(data)
    std = np.std(data)
    if standar:
        data = (data - mean)/std
        mean = 0
        std = 1
    if normal:
        edges = np.histogram_bin_edges(data, bins='sturges')
    else:
        edges = np.histogram_bin_edges(data, bins='doane')
    fig, ax = plt.subplots()
    ax.hist(data, density = True, bins=edges)
    if plot_random_data:
        data_bottom_percentile = np.sum(data <= edges[0])/data.size
        data_top_percentile = 1 - np.sum(data >= edges[-1])/data.size
        random_bottom_percentile =norm.ppf(data_bottom_percentile, loc=mean, scale=std)
        random_top_percentile =norm.ppf(data_top_percentile, loc=mean, scale=std)
        random_data = np.linspace(random_bottom_percentile, random_top_percentile, data.size)
        ax.plot(random_data, norm.pdf(random_data), 'r-', label='Normal Distribution')
        ax.legend()
    ax.set_xticks(edges)
    ax.set_xticklabels(np.round(edges, 2))
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    fig.suptitle(suptitle, fontsize=20)

    figw, figh = get_figsize(ax=ax, scale=scale)
    fig.set_figheight(figh)
    fig.set_figwidth(figw)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, ax

def plot_scores(train_score, test_score=None, cross_val_score=None, x_array=None, labels=None, x_label='Degrees', y_label='R2', scale=(6.4, 4.8), suptitle='', styles=None, plot_full=False):
    """
    Plot the scores of a model on the train data.

    If test_score is given, it plots additionaly this scores.

    If cross_val_score is given, it plots the means of the given scores.
    If additionaly plot_full is set to True, each cross_val_score is plotted
    individually.

    Parameters
    ----------
        train_scores : 1d array
            Scores of the training data on the models to analyze.
        test_score : 1d array
            Scores of the test data on the models to analyze.
        cross_val_score : 2d array
            Scores of the cross-validation on the models to analyze.
        x_array : 1d array
            Array with the parameters that define the models.
        labels : list
            Labels for the data.
        x_label : str
            Label for the x axis.
        y_label : str
            Label for the y axis.
        scale : tuple[float, float]
            Scale to preserve on axes.
        suptitle : str
            Suptitle of the plot.
        styles : 
            Styles for the lines/markers.
        plot_full : bool
            If True, the function also plots each cross-validation score
            individually. If cross_val_score is not defined, this
            parameter is ignored.
    
    Returns
    -------
        fig, ax
            The figure and axes of the plot, respectively.
    """
    if labels is None:
        labels = ['R2 train scores', 'R2 test scores', 'cross-validation scores', 'cross-validation mean scores']
    if styles is None:
        styles = ['b-', 'r-', 'g-', 'go']
    if x_array is None:
        x_array = np.arange(train_score.size)+1
    if not (cross_val_score is None):
        if cross_val_score.size == cross_val_score.shape[0]:
            cross_val_score = cross_val_score.reshape(-1, 1)
    fig, ax = plt.subplots()
    ax.plot(x_array, train_score, styles[0], label=labels[0])
    if not (test_score is None):
        ax.plot(x_array, test_score, styles[1], label=labels[1])
    if not (cross_val_score is None):
        mean_cross_val_score = np.mean(cross_val_score, axis=0)
        ax.plot(x_array, mean_cross_val_score, styles[2], label=labels[2])
        if plot_full:
            for i in np.arange(cross_val_score.shape[1]):
                if i == 0:
                    ax.plot(x_array, cross_val_score[:,i], styles[3], label=labels[3])
                else:
                    ax.plot(x_array, cross_val_score[:,i], styles[3])
    if not (test_score is None and cross_val_score is None):
        ax.legend()
    ax.grid()
    ax.set_xticks(x_array)
    ax.set_xticklabels(x_array)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    fig.suptitle(suptitle, fontsize=20)

    figw, figh = get_figsize(ax=ax, scale=scale)
    fig.set_figheight(figh)
    fig.set_figwidth(figw)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, ax

def plot_cross_val_scores(scores, x_array=None, labels=None, x_label='Degrees', y_label='R2', scale=(6.4, 4.8), suptitle='', styles=None, with_means=True):
    """
    Plots each cross-validation score individually. If with_means is set to
    True, it also plots the means of cross-validation scores for each
    iteration.

    Parameters
    ----------
        scores : 2d array
            scores to plot.
        x_array : 1d array
            Array with the parameters that define the models.
        labels : list
            Labels for the data.
        x_label : str
            Label for the x axis.
        y_label : str
            Label for the y axis.
        scale : tuple[float, float]
            Scale to preserve on axes.
        suptitle : str
            Suptitle of the plot.
        styles : 
            Styles for the lines/markers.
        with_means : bool
            If True, the function also plots the means of cross-validation
            scores for each iterarion.
    
    Returns
    -------
        fig, ax
            The figure and axes of the plot, respectively.
    """
    if labels is None:
        labels = ['Cross-validation R2 scores', 'cross-validation R2 mean scores']
    if styles is None:
        styles = ['bo', 'b-']
    if scores.size == scores.shape[0]:
        scores = scores.reshape(-1, 1)
    if x_array is None:
        x_array = np.arange(scores.shape[1])+1
    fig, ax = plt.subplots()
    for i in np.arange(scores.shape[0]):
        if i == 0:
            ax.plot(x_array, scores[i,:], styles[0], label=labels[0])
        else:
            ax.plot(x_array, scores[i,:], styles[0])
    if with_means:
        mean_scores = np.mean(scores, axis=0)
        ax.plot(x_array, mean_scores, styles[1], label=labels[1])
        ax.legend()
    ax.grid()
    ax.set_xticks(x_array)
    ax.set_xticklabels(x_array)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    fig.suptitle(suptitle, fontsize=20)

    figw, figh = get_figsize(ax=ax, scale=scale)
    fig.set_figheight(figh)
    fig.set_figwidth(figw)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, ax

def plot_silhouette_samples(samples, cluster_labels, ks=None, x_labels=None, y_label=None, scale=(6.4, 4.8), suptitle='', with_means=True, share_x=True):
    """
    Makes a knife plot with the silhouette samples. If with_means is
    set to True, it also plots a vertical line where the mean is.

    Parameters
    ----------
        samples : list or 1d or 2d array
            samples to plot.
        cluster_labels : list[numpy array]
            list of arrays which contains the labels of each instance.
        ks : 1d array
            array with the number of clusters of each KMeans model analized.
        x_label : str
            Label for the x axis.
        y_label : str
            Label for the y axis.
        scale : tuple[float, float]
            Scale to preserve on axes.
        suptitle : str
            Suptitle of the plot.
        with_means : bool
            If True, the function also plots the means of silhouette samples
            for each iterarion.
        share_x : bool
            Defines if the plots share the same x axis.
        
    Returns
    -------
        fig, ax
            The figure and axes of the plot, respectively.
    """
    if type(samples) == list:
        samples = np.array(samples).T
    if samples.size == samples.shape[0]:
        samples = samples.reshape(-1, 1)
    n_iterations = samples.shape[1]
    if ks is None:
        ks = np.arange(n_iterations) + 2
    if x_labels is None:
        x_labels = []
        for i in np.arange(n_iterations):
            x_labels.append('Silhouette samples, k = ' + str(ks[i]))
    if y_label is None:
        y_label = 'Number of instances'
    n_rows, n_cols = get_plot_grid(n_iterations)
    if share_x:
        fig, ax = plt.subplots(n_rows, n_cols, sharex=share_x)
    else:
        fig, ax = plt.subplots(n_rows, n_cols)
    flag = 0
    if type(ax) != np.ndarray:
        ax = np.array([[ax]])
        flag = 1
    elif ax.size == ax.shape[0]:
        ax = np.array([ax]).T
        flag = 2
    row = 0
    col = 0
    for i in np.arange(n_iterations):
        y_low = 0
        y_high = 0
        iteration_samples = samples[:, i]
        iteration_labels = cluster_labels[:, i]
        unique_labels = np.unique(iteration_labels)
        y_ticks = [y_low]
        for label in unique_labels:
            cluster_samples = iteration_samples[iteration_labels == label]
            cluster_samples = np.sort(cluster_samples)
            y_high += cluster_samples.size
            y_ticks.append(y_high)
            ax[row, col].barh(np.arange(y_low, y_high), cluster_samples, edgecolor='none', height=1)
            y_low += cluster_samples.size
        if with_means:
            samples_mean = np.mean(iteration_samples)
            ax[row, col].axvline(samples_mean, linestyle='--', linewidth=2, color='green')
        
        ax[row, col].set_yticks(y_ticks)
        ax[row, col].set_yticklabels(y_ticks)
        ax[row, col].set_xlabel(x_labels[i], fontsize=14)
        ax[row, col].set_ylabel(y_label, fontsize=14)
        
        row += 1
        if row >= n_rows:
            col += 1
            row = 0
    fig.suptitle(suptitle, fontsize=20)

    figw, figh = get_figsize(ax=ax[0, 0], scale=scale)
    fig.set_figheight(figh)
    fig.set_figwidth(figw)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if flag == 1:
        return fig, ax[0, 0]
    elif flag == 2:
        return fig, ax[:, 0]
    return fig, ax

def plot_clusters(data, cluster_labels, centroids=None, centroid_color='r', centroid_marker='*', centroid_size=250, features_to_plot=None, ks=None, x_labels=None, y_label=None, scale=(6.4, 4.8), suptitle=''):
    """
    Plot the clusters generated by 1 or many KMeans models. If data has
    more than 2 features, the function plots each feature vs the others,
    unless features_to_plot is specified, in which case it only plots
    the specified features.

    Parameters
    ----------
        data : 1d or 2d array
            Data to plot.
        cluster_labels : 1d or 2d array
            Labels generated by each KMeans model.
        centroids : list[numpy array]
            list of the centroids generated by each KMeans model.
        centroid_color : str
            Color to plot the centroids with. If centroids is not specified,
            this parameter is ignored.
        centroid_marker : str
            Marker to use for the centroids. If centroids is not specified,
            this parameter is ignored.
        centroid_size : str
            Size of the centroids marker. If centroids is not specified,
            this parameter is ignored.
        features_to_plot : 1d array
            features for which the plot is made. If not specified, the function
            plots each feature vs each other.
        ks : 1d array
            array with the number of clusters of each KMeans model analized.
        x_labels : str
            Labels for the x axes.
        y_label : str
            Label for the y axis.
        scale : tuple[float, float]
            Scale to preserve on axes.
        suptitle : str
            Suptitle of the plot.
        
    Returns
    -------
        fig, ax
            The figure and axes of the plot, respectively.
    """
    if data.size == data.shape[0]:
        data = data.reshape(-1, 1)
    if cluster_labels.size == cluster_labels.shape[0]:
        cluster_labels = cluster_labels.reshape(-1, 1)
    n_features = data.shape[1]
    n_iterations = cluster_labels.shape[1]
    if features_to_plot is None:
        features_to_plot = np.arange(n_features)
    n_feat_to_plot = features_to_plot.size
    if ks is None:
        ks = np.arange(n_iterations) + 1
    if x_labels is None:
        x_labels = np.array(['k = ' + str(k) for k in ks])
    if n_feat_to_plot == 1:
        n_axes = 1
    else:
        n_axes = n_feat_to_plot*(n_feat_to_plot - 1)/2
    n_axes *= n_iterations
    n_rows, n_cols = get_plot_grid(n_axes)
    fig, ax = plt.subplots(n_rows, n_cols)

    flag = 0
    if type(ax) != np.ndarray:
        ax = np.array([[ax]])
        flag = 1
    elif ax.size == ax.shape[0]:
        ax = np.array([ax]).T
        flag = 2

    row = 0
    col = 0
    for i in np.arange(n_iterations):
        iteration_labels = cluster_labels[:, i]
        unique_labels = np.unique(iteration_labels)
        if n_iterations > 1:
            iteration_centroids = centroids[i]
        else:
            iteration_centroids = centroids
        if iteration_centroids.size == iteration_centroids.shape[0]:
            iteration_centroids = iteration_centroids.reshape(-1, 1)
        x_end = n_feat_to_plot
        if n_feat_to_plot > 1:
            x_end -= 1
        for feat1 in np.arange(x_end):
            feat_x = features_to_plot[feat1]
            x = data[:, feat_x]
            x_centroids = iteration_centroids[:, feat_x]
            y_init = feat1
            if n_feat_to_plot > 1:
                y_init += 1
            for feat2 in np.arange(y_init, n_feat_to_plot):
                feat_y = features_to_plot[feat2]
                y = data[:, feat_y]
                y_centroids = iteration_centroids[:, feat_y]
                for label in unique_labels:
                    cluster_index = iteration_labels == label
                    x_cluster = x[cluster_index]
                    y_cluster = y[cluster_index]
                    ax[row, col].scatter(x_cluster, y_cluster)
                if not (centroids is None):
                    ax[row, col].scatter(x_centroids, y_centroids, c=centroid_color, marker=centroid_marker, s=centroid_size)

            ax[row, col].set_xlabel(x_labels[i], fontsize=14)
            ax[row, col].set_ylabel(y_label, fontsize=14)
            ax[row, col].grid()
            row += 1
            if row >= n_rows:
                col += 1
                row = 0
    fig.suptitle(suptitle, fontsize=20)

    figw, figh = get_figsize(ax=ax[0, 0], scale=scale)
    fig.set_figheight(figh)
    fig.set_figwidth(figw)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if flag == 1:
        return fig, ax[0, 0]
    elif flag == 2:
        return fig, ax[:, 0]
    return fig, ax