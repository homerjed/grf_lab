from scipy import fftpack
import numpy as np


def _get_k(input_array, box_dims):
    '''
        Get the k values for input array with given dimensions.
        Return k components and magnitudes.
        For internal use.
    '''
    # Assuming 2d
    x, y = np.indices(input_array.shape, dtype='int32')
    # Centred k-space frequencies (k=0 is at the centre of the map)
    center = np.array([(x.max() - x.min()) / 2, (y.max() - y.min()) / 2])
    # Scale by box length
    kx = 2. * np.pi * (x - center[0]) / box_dims[0]
    ky = 2. * np.pi * (y - center[1]) / box_dims[1]
    # Magnitudes of k vectors
    k = np.sqrt(kx ** 2 + ky ** 2)
    return [kx, ky], k

def _get_kbins(kbins, box_dims, k):
    '''
        Make a list of bin edges if kbins is an integer,
        otherwise return it as it is.
    '''
    kmin = 2. * np.pi / min(box_dims) # Minimum freq is that which fits in the box, scale of box
    kbins = np.linspace(kmin, k.max(), kbins + 1)
    return kbins
def power_spectrum_nd(input_array, box_dims=None):
    ''' 
        Calculate the power spectrum of input_array and return it as an n-dimensional array,
        where n is the number of dimensions in input_array
        box_side is the size of the box in comoving Mpc. If this is set to None (default),
        the internal box size is used
        
        Parameters:
            * input_array (numpy array): the array to calculate the 
                power spectrum of. Can be of any dimensions.
            * box_dims = None (float or array-like): the dimensions of the 
                box. If this is None, the current box volume is used along all
                dimensions. If it is a float, this is taken as the box length
                along all dimensions. If it is an array-like, the elements are
                taken as the box length along each axis.
        
        Returns:
            The power spectrum in the same dimensions as the input array.		
    '''
    box_dims = [box_dims[0]] * len(input_array.shape)

    # Transform to Fourier space
    ft = fftpack.fftshift(fftpack.fftn(input_array.astype('float64')))

    # Calculate power
    power_spectrum = np.abs(ft) ** 2.

	  # Scale by box volume
    boxvol = np.prod(box_dims)
    pixelsize = boxvol / np.prod(input_array.shape)
    power_spectrum *= pixelsize ** 2. / boxvol

    return power_spectrum



def radial_average(input_array, box_dims, kbins):
	'''
	Radially average data.
	
	Parameters: 
		* input_array (numpy array): the data array
		* box_dims = None (float or array-like): the dimensions of the 
			box. If this is None, the current box volume is used along all
			dimensions. If it is a float, this is taken as the box length
			along all dimensions. If it is an array-like, the elements are
			taken as the box length along each axis.
		* kbins = 10 (integer or array-like): The number of bins,
			or a list containing the bin edges. If an integer is given, the bins
			are logarithmically spaced.
			
	Returns:
		A tuple with (data, bins, n_modes), where data is an array with the 
		averaged data, bins is an array with the bin centers and n_modes is the 
		number of modes in each bin

	'''
	k_comp, k = _get_k(input_array, box_dims)

	kbins = _get_kbins(kbins, box_dims, k)
	
	# Bin the data
	dk = (kbins[1:] - kbins[:-1]) / 2.
    
	# Total power in each bin (weights are k-space cell densities from input array)
	outdata, _  = np.histogram(k.flatten(), bins=kbins, weights=input_array.flatten())
    
	# Number of modes in each bin
	n_modes = np.histogram(k.flatten(), bins=kbins)[0].astype('float')
	outdata /= n_modes
	
	return outdata, kbins[:-1] + dk, n_modes


def power_spectrum_1d(input_array_nd, kbins = 50, box_dim = 1.0):
    ''' Calculate the spherically averaged power spectrum of an array 
	and return it as a one-dimensional array.
	
	Parameters: 
		* input_array_nd (numpy array): the data array
		* kbins = 100 (integer or array-like): The number of bins,
			or a list containing the bin edges. If an integer is given, the bins
			are logarithmically spaced.
		* box_dims = None (float or array-like): the dimensions of the 
			box. If this is None, the current box volume is used along all
			dimensions. If it is a float, this is taken as the box length
			along all dimensions. If it is an array-like, the elements are
			taken as the box length along each axis.
		* return_n_modes = False (bool): if true, also return the
			number of modes in each bin
			
	Returns: 
		A tuple with (Pk, bins), where Pk is an array with the 
		power spectrum and bins is an array with the k bin centers.
    '''
    box_dims = [box_dim] * len(input_array_nd.shape)

    input_array = power_spectrum_nd(input_array_nd, box_dims=box_dims)	

    power_spectrum, bins, n_modes = radial_average(
		input_array, 
		kbins=kbins, 
		box_dims=box_dims
    )
    return power_spectrum, bins, n_modes
