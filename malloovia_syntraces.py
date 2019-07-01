"""Basic classes to generate synthetic traces based on periodic
noise and spike components, and to apply global envelopes"""

from enum import Enum
import numpy as np
from numpy.fft import rfft, irfft


class ADSR:
    """ADSR envelope. It is a sequence of floats between 0 and 1
    which is composed of four regions:

        * Attack: it is an inverted exponential which starts at 0
          and reaches 1. The length (in percentage) and the power of the
          exponential can be chosen.
        * Decay: it is an inverted exponential which starts at 1
          and gradually decreases asymptotically towards the "sustain" value.
          The lenght of the decay+sustain (in percentage) and the power of the decay can
          be chosen.
        * Sustain: it is a plateau at "sustain" value. The value of the "sustain"
          is specified as a number beteen 1 and 0 (usually close to 1). The
          length of the plateau cannot be specified. What is specified instead
          is the length of the decay+plateau part. In fact, the plateau can
          be absent if the decay is slow and never reaches the "sustain" value.
        * Release: it is the final part in which the values decay from the
          last value reached during the decay phase until reaching zero. The
          lenght (in percentage) of this phase and the shape of the exponential
          can be chosen.
    """

    def __init__(self, attack, release, sustain, k_a, k_d, k_r):
        """Stores the parameters of the envelope.

        Arguments:

            ``attack``: fraction of the total length to be used by the attack phase.

            ``release``: fraction of the total lencth to be used by the release phase.

            ``sustain``: positive value (<=1.0) of the sustain plateau. This value is reached
               through a decay phase. The length of the decay + plateau phase is computed
               as (1 - `attack` - `release`).

            ``k_a``: power of the exponent for the attack phase.

            ``k_d``: power of the exponent for the decay phase.

            ``k_r``: power of the exponent for the release phase.
        """
        assert attack + release <= 1
        assert sustain <= 1.0
        self.attack = attack
        self.release = release
        self.sustain = sustain
        self.k_a = k_a
        self.k_r = k_r
        self.k_d = k_d

    @staticmethod
    def weighted_exp(nsamples, weight):
        """Exponential between 0 and 1"""
        time = np.linspace(0, 1, nsamples)
        expo = np.exp(weight * time) - 1
        expo /= max(expo)
        return expo

    def get_envelope(self, nsamples):
        """Computes the ADSR envelope for a given number of samples.

        Arguments:

            ``nsamples``: the lenght (in samples) of the ADSR to be computed

        Returns:

            A ``numpy.array`` with the values (between 0 and 1) of the envelope
        """
        # Length of each stage
        l_attack = int(nsamples * self.attack)
        l_release = int(nsamples * self.release)
        l_sustain = nsamples - l_attack - l_release

        attack = ADSR.weighted_exp(l_attack, self.k_a)
        decay = ADSR.weighted_exp(l_sustain, self.k_d)
        release = ADSR.weighted_exp(l_release, self.k_r)

        attack = 1.0 - attack[::-1]
        decay = decay[::-1] * (1.0 - self.sustain) + self.sustain
        release = release[::-1] * self.sustain

        return np.concatenate([attack, decay, release])

class Period(Enum):
    """An enumeration with some units of time and their conversion to minutes and seconds.

    Example usage: ``Period.month.in_minutes()``, ``Period.week.in_seconds()``

    Also: ``Period.month.value`` gives the minutes.
    """

    minute = 1
    hour = 60
    day = 60*24
    week = 60*24*7
    month = 60*24*30
    quarter = 60*24*365//4
    semester = 60*24*365//2
    year = 60*24*365

    def in_minutes(self):
        """Returns the length in minutes"""
        return self.value
    
    def in_seconds(self):
        """Returns the length in seconds"""
        return self.value*60
    
    def in_hours(self):
        """Returns the length in hours"""
        return self.value/60


def periodic_workload(components, nsamples):
    """Generates a periodic workload as the sum of several sinusoids.

    Args:
      ``components``: an iterable of tuples, each one containing the period
          (in timeslots), phase (in timeslots) and amplitude (in arbitrary
          units to be taken as relative weights) of each sinusoid
      ``nsamples``: the number of timeslots to generate

    Returns:
      The periodic workload, normalized between -1 and 1
    """
    y__ = np.zeros(nsamples)
    x__ = np.arange(nsamples)
    for period, phase, amplitude in components:
        y__ += np.sin((x__+phase)*2*np.pi/period) * amplitude
    max_ = np.max(y__)
    min_ = np.min(y__)
    # Return normalized version
    return (y__ - min_)/(max_ - min_)*2 - 1

def rescale_between(workload, minimum=0, maximum=200):
    """Re-scales the given workload, which is expected to be between -1 and 1,
    so that it becomes between the given minimum and maximum values.

    Arguments:
        ``workload``: the normalized workload to rescale (it is not modified)
        ``minimum``: the new minimum
        ``maximum``: the new maximum

    Returns:
        The rescaled workload
    """
    amplitude = (maximum - minimum)/2
    return workload * amplitude + (minimum + amplitude)

def gaussian_noise(workload, fraction=0.1, noisiness=0.2, seed=None):
    """Generates random (gaussian) noise, but scales it proportional to the given
    workload at each timeslot.

    Args:
        ``workload``: input workload (it is not modified)
        ``fraction``: fraction of the workload at each timeslot which is randomized
        ``noisiness``: standard deviation of the normal noise
    Returns:
        ``np.array`` of noise, whose length equals ``workload`` length
    """
    np.random.seed(seed)
    noise = np.random.normal(0, noisiness, len(workload))
    return noise * workload * fraction

def white_noise(nsamples, seed=None):
    np.random.seed(seed)
    return np.random.normal(0,1,nsamples)


# The following functions were extracted from https://github.com/python-acoustics/python-acoustics code
def _ms(x):
    """Mean value of signal `x` squared.
    :param x: Dynamic quantity.
    :returns: Mean squared of `x`.
    """
    return (np.abs(x)**2.0).mean()

def _normalize(y, x=None):
    """normalize power in y to a (standard normal) white noise signal.
    Optionally normalize to power in signal `x`.
    #The mean power of a Gaussian with :math:`\\mu=0` and :math:`\\sigma=1` is 1.
    """
    if x is not None:
        x = _ms(x)
    else:
        x = 1.0
    return y * np.sqrt( x / _ms(y) )

def pink_noise(nsamples, seed=None):
    """
    Pink noise.

    Pink noise has equal power in bands that are proportionally wide.
    Power density decreases with 3 dB per octave.

    """
    # This method uses the filter with the following coefficients.
    #b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    #a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
    #return lfilter(B, A, np.random.randn(N))
    # Another way would be using the FFT
    #x = np.random.randn(N)
    #X = rfft(x) / N
    state = np.random.RandomState(seed)
    uneven = nsamples%2
    X = state.randn(nsamples//2+1+uneven) + 1j * state.randn(nsamples//2+1+uneven)
    S = np.sqrt(np.arange(len(X))+1.) # +1 to avoid divide by zero
    y = (irfft(X/S)).real
    if uneven:
        y = y[:-1]
    return _normalize(y)

def brown_noise(nsamples, seed=None):
    """
    Brown noise.

    Power decreases with -3 dB per octave.
    Power density decreases with 6 dB per octave.
    """
    state = np.random.RandomState(seed)
    uneven = nsamples%2
    X = state.randn(nsamples//2+1+uneven) + 1j * state.randn(nsamples//2+1+uneven)
    S = (np.arange(len(X))+1)# Filter
    y = (irfft(X/S)).real
    if uneven:
        y = y[:-1]
    return _normalize(y)

def adsr_spikes(workload, spikes):
    """Generates spikes around some timeslots, with attack, decay, sustain and release
    times specified by adsr parameters.

    Args:
        workload: input workload (it is not modified)
        spikes: iterable of tuples, each one containing:
           - The timeslot in which the spike starts
           - The width (number of timeslots) of the spike envelope
           - The strength (height of the spike) If ``strength`` integer,
             it is considered an absolute value, in whatever workload units. If not, it is
             a multiplicative factor of the workload at the given timeslot.
           - A :class:`ADSR` instance with the envelope of the spike
    Returns:
        An ``np.array`` containing the generated spikes. This array can be added to the
        original workload.
    """
    result = np.zeros(len(workload))
    for spike in spikes:
        place, width, strength, adsr = spike
        if type(strength) == int:
            amplitude = strength
        else:
            amplitude = workload[place] * strength
        spike = amplitude * adsr.get_envelope(width)
        result[place:place+len(spike)] = spike
    return result