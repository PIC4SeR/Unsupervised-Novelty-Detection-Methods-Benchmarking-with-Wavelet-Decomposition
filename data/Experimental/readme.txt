first 5 sets with the following wave, various amplitudes:
src.utils.generate_wav_file([50,100,150,230,300,440,460,530,600], [.5,.5,.5,1,1,1,.5,.5,.5], 600)

last 3 sets with the following wave, various amplitudes:
src.utils.generate_wav_file([50,100,150,230,300,440,460,530,600], [.5,.5,.5,1,.2,1,.5,.5,2], 600)

def generate_wav_file(freqs: list,mags: list,duration: float,filepath='wavfiles',phis=None,fs=44100):
    """
    Generate a wave file from a list of frequencies and magnitudes
    y = mags[0]+sin(2*pi*freqs[0]*t+phis[0])+mags[1]+sin(2*pi*freqs[1]*t+phis[1])+...
    for a time of "duration" seconds

    Parameters
    ----------
    freqs : list
        List of frequencies
    mags : list
        List of magnitudes
    duration : float
        Duration of the signal in seconds
    filepath : str, optional
        Path to save the wave file, by default ''
    phis : list, optional
        List of phases, by default None
    fs : int, optional
        Sampling frequency, by default 44100
    """
    if len(freqs) != len(mags):
        raise Exception('The length of the input arrays should be the same')
    if phis is None:
        phis = np.zeros(len(freqs))

    # create time values
    t = np.linspace(0, duration, duration * fs, dtype=np.float32)

    # generate y values for signal
    y = np.zeros(len(t))

    for i in range(len(freqs)):
        y += mags[i]*np.sin(2*np.pi*freqs[i]*t+phis[i])

    # create the directory if it does not exist
    pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)

    # concatenate the path
    filepath = pathlib.Path(filepath, f"{'_'.join([str(f) for f in freqs])}.wav")

    # save to wave file
    write_wav(filepath, fs, y)