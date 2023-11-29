import numpy as np
from scipy.stats import ks_2samp, kstest, norm

def log_hist(data, nbins=30):
    '''
    Function to calculate the log-histogram of a dataset. In particular, it calculates log-spaced bins, and then 
    calculates the counts, probability, and probability density of the data in those bins.
    
    Parameters
    ----------
    data : array-like
        Dataset to calculate the log-histogram of
    nbins : int
        Number of bins to use in the histogram
    
    Returns
    -------
    bins : array-like
        Log-spaced bins
    counts : array-like
        Counts in each bin (sum to total number of data points)
    prob : array-like  
        Probability of each bin (sum to 1)
    pdf : array-like   
        Probability density of each bin (the area sums to 1)
    '''
    
    bins = np.logspace(np.log10(data.min()), np.log10(data.max()), nbins)
    counts = np.histogram(data, bins=bins)[0]
    prob = counts / counts.sum()
    pdf = prob / np.diff(bins)
    
    return bins, counts, prob, pdf

def return_geo(x):
    '''
    Function to return the geographic region of a station based on its latitude.
    
    Parameters
    ----------
    x : float
        Latitude of the station
        
    Returns
    ----------
    region : str
        Geographic region of the station
    '''
    
    if np.abs(x)>60:
        return 'polar'
    elif np.abs(x)>30:
        return 'tropical'
    else:
        return 'equatorial'
    

def maximum_likelihood_exponent(data):
    '''
    This function returns the maximum likelihood exponent of a power law distribution.
    The power law distribution is defined as:
    y = x**(-rho-1)
    ----------
    Parameters
    data: array-like
        The data to fit. If some values are zero, they are removed.
    ----------
    Returns
    rho: float
        The maximum likelihood exponent of the power law distribution
    pvalue: float
        The p-value of the Kolmogorov-Smirnov test
    '''
    rho = 1/np.mean(np.log(data[data>0]))
    # calculate the p-value of the Kolmogorov-Smirnov test
    KS = kstest(data, 'powerlaw', args=(rho,))
    pvalue = KS.pvalue
    
    return rho, pvalue



def mode_with_binning(array, bins=10):
    '''
    Function to calculate the mode of a dataset using binning. In particular, it calculates the histogram of the data
    
    Parameters
    ----------
    array : array-like
        Dataset to calculate the mode of
    bins : array-like
        Bins to use in the histogram
        
    Returns
    ----------
    mode : float
        Mode of the dataset
    '''
    
    # Data binning
    histo, bin_edges = np.histogram(array, bins=bins)

    # Index of the maximum value
    index_max = np.argmax(histo)

    return (bin_edges[index_max]+bin_edges[index_max+1])/2


def two_tailed_pvalue(data, mu=0, sigma=1):
    '''
    Function to calculate the two-tailed p-value of a dataset with respect to a normal distribution.
    
    Parameters
    ----------
    data : array-like
        Dataset to calculate the p-value of
    mu : float
        Mean of the normal distribution
    sigma : float
        Standard deviation of the normal distribution
        
    Returns
    ----------
    pvalue : float
        Two-tailed p-value of the dataset
    '''
    
    # Calculate the p-value
    pvalue = 2*norm.cdf(-np.abs(data-mu)/sigma)
    
    return pvalue