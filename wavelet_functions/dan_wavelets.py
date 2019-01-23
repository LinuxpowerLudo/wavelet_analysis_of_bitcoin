import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (16, 3)

def dwt(input_series, filter_length=2, 
                     scaling_filter_coefficients=[1/(np.power(2, 1/2)), 1/(np.power(2, 1/2))],
                    wavelet_filter_coefficients=[-1/(np.power(2, 1/2)), 1/(np.power(2, 1/2))]):
    """
    Function that computes the dwt of the input series
    Default wavelet is the haar wavelet
    returns:
    (cD, cA)
    """
    series_length = input_series.shape[0]
    
    w_series = np.zeros(int(series_length/2), float)
    v_series = np.zeros(int(series_length/2), float)
    for t in range(int(series_length/2)):
        lw_sum = 0
        lv_sum = 0
        for l in range(filter_length):
            lw_sum+= wavelet_filter_coefficients[l]*input_series[(2*t + 1 - l)%series_length]
            lv_sum+= scaling_filter_coefficients[l]*input_series[(2*t + 1 - l)%series_length]
        
        w_series[t] = lw_sum
        v_series[t] = lv_sum
    #w_series[int(series_length/2)] = 0
    
    w_series = w_series[~np.isnan(w_series)]    
    return w_series, v_series

def idwt(cD, cA, level=1, scaling_filter_coefficients=[1/(np.power(2, 1/2)), 1/(np.power(2, 1/2))],
                    wavelet_filter_coefficients=[-1/(np.power(2, 1/2)), 1/(np.power(2, 1/2))]):
    """
    Function that takes in the detail and approximation coefficients and returns the parent series
    using an inverse discrete wavelet transform
    Default wavelet is the haar wavelet
    """
    
    
    
    filter_length = len(scaling_filter_coefficients)
    new_length = cD.shape[0]*2
    cD_resized = np.zeros((new_length))
    cA_resized = np.zeros((new_length))
    return_series = np.zeros((new_length))
    
    for i in range(new_length):
        if i%2 ==1:
            cD_resized[i] = cD[int(i/2)]
            cA_resized[i] = cA[int(i/2)]
            
    return_series = np.zeros((new_length))
    for t in range(new_length):
        cD_sum = 0
        cA_sum = 0
        for l in range(filter_length):
            cD_sum+= wavelet_filter_coefficients[l]*cD_resized[(t+l)%(new_length)]
            cA_sum+= scaling_filter_coefficients[l]*cA_resized[(t+l)%(new_length)]

        return_series[t] = cD_sum+cA_sum
       
    
    return return_series

def modwt(input_series, filter_length=2, j=1,
        g=[1/np.sqrt(2), 1/np.sqrt(2)],
        h=[1/(np.sqrt(2)), -1/(np.sqrt(2))]):
    """
    Function that computes the dwt of the input series
    Default wavelet is the haar wavelet
    returns:
    (cD, cA)
    """
    gtilda = np.array(g)/np.sqrt(2)
    htilda = np.array(h)/np.sqrt(2)

    vj = np.zeros(input_series.shape)
    wj = np.zeros(input_series.shape)

    N = input_series.shape[0]

    for t in range(N):
        k = t
        wj[t] = htilda[0]*input_series[k]
        vj[t] = gtilda[0]*input_series[k]
        for n in range(1, filter_length):
            k = k - 2**(j-1)
            if k < 0:
                k = k%N
            wj[t] += htilda[n]*input_series[k]
            vj[t] += gtilda[n]*input_series[k]
    return wj, vj

def imodwt(W, V, filter_length=2, j=1,
        g=[1/np.sqrt(2), 1/np.sqrt(2)],
        h=[1/(np.sqrt(2)), -1/(np.sqrt(2))]):
    '''
    Function that takes in the detail and approximation coefficients and returns the parent series
    using an imodwt
    Default wavelet is the haar wavelet
    '''
    if len(W) == 0:
        W = np.zeros(V.shape)
    if len(V) == 0:
        V = np.zeros(W.shape)
    v_output = np.zeros(W.shape)
    N = W.shape[0]
    htilda = np.array(h)/np.sqrt(2)
    gtilda = np.array(g)/np.sqrt(2)
    
    for t in range(N):
        k=t
        v_output[t] = W[k]*htilda[0] + V[k]*gtilda[0]
        for n in range(1, filter_length):
            k += 2**(j-1)
            if k >= N:
                k = k%N
            v_output[t] += W[k]*htilda[n] + V[k]*gtilda[n]
    return v_output

def mra(input_series, num_decompositions, h, g, L):
    """
    Perform mra using modwt
    Args:
        input_series: the series to be decomposed
        num_decompositions: The number of times the modwt will be performed
        h: The coeffs of the high pass filter of the wavelet
        g: The coeffs of the low pass filter of the wavlelet
        L: The length of the wavelet
    
    Returns:
        Void
    """
    plt.plot(input_series)
    plt.title("Original Series")
    plt.show()

    cA = input_series
    for i in range(num_decompositions):
        cD, cA = modwt(cA, j=i+1, filter_length=L, h=h, g=g)
        cD = imodwt(cD, [], j = i+1, filter_length=L, h=h, g=g)
        for j in range(i, 0, -1):
            cD = imodwt([], cD, j = j, filter_length=L, h=h, g=g)
        plt.title("D"+str(i+1))
        plt.plot(cD)
        plt.show()
    for j in range(num_decompositions, 0, -1):
        cA = imodwt([], cA, j=j, g=g, h=h, filter_length = L)
    plt.title("S"+str(num_decompositions))
    plt.plot(cA)
    plt.show()