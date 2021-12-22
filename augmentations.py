import random
import pandas as pd
import sox
import random
import sys, os
import numpy as np
import cv2
import librosa
import pyroomacoustics
import scipy
from scipy.ndimage import filters
from scipy.signal import gaussian
import scipy.fftpack
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.io import wavfile
import scipy as sp
from scipy import signal
import spec_augment_tensorflow
import tensorflow as tf
#pip install tensorflow-addons==0.9.1

#Current file lists many useful audio data augmentations suitable to create spoof artefacts in audio files.
#All below augmentations are the fruit of a deep and long analysis of the original spoofed files provided 
#by the ASVSpoof19 organizers. We try to represent all possible anomalies/artifacts that could be introduced by models (GANs, audio mathematical models, etc.) 
#used to create fake audios. Call to methods uses randomized parameters whenever is possible to generate different variants/intensities of artefacts.
#Methods return a waveform or spectrogram depending on parameter return_audio.

def specAugment(clip, sample_rate, type_augment=4, return_audio=False):
    # Step 0 : extract mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=clip,
                                                     sr=sample_rate,
                                                     n_mels=256,
                                                     hop_length=128,
                                                     fmax=8000)
    mel_spectrogram = tf.cast(mel_spectrogram, dtype=tf.float32)

    # reshape spectrogram shape to [batch_size, time, frequency, 1]
    shape = mel_spectrogram.shape
    mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1], 1))

    # Show Raw mel-spectrogram
    #spec_augment_tensorflow.visualization_spectrogram(mel_spectrogram=mel_spectrogram,
    #                                                  title="Raw Mel Spectrogram")
    if type_augment == 1:
        tau = mel_spectrogram.shape[1]
        warped_masked_spectrogram = spec_augment_tensorflow.time_masking(mel_spectrogram, tau)
    elif type_augment == 2:
        v = mel_spectrogram.shape[0]
        warped_masked_spectrogram = spec_augment_tensorflow.frequency_masking(mel_spectrogram, v)
    elif type_augment == 3:
        warped_masked_spectrogram = spec_augment_tensorflow.sparse_warp(mel_spectrogram)
    else:
        warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram)
    # Show time warped & masked spectrogram
    #spec_augment_tensorflow.visualization_tensor_spectrogram(mel_spectrogram=warped_masked_spectrogram,
    #                                                  title="tensorflow Warped & Masked Mel Spectrogram")
    
    #convert to audio
    shape = warped_masked_spectrogram.shape
    reshaped_warped_masked_spectrogram = np.reshape(warped_masked_spectrogram, (shape[1], shape[2]))
    recovered_audio = librosa.feature.inverse.mel_to_audio (M=reshaped_warped_masked_spectrogram, 
        hop_length=128, sr=sample_rate)
    if return_audio:
        return recovered_audio
    else:
        return to_sp(recovered_audio)

#remix
def remix_audio(clip, sample_rate, return_audio=False):
    #compute beats
    _, beat_frames = librosa.beat.beat_track(y=clip, sr=sample_rate,hop_length=512)
    #Convert from frames to sample indices
    beat_samples = librosa.frames_to_samples(beat_frames)
    #Generate intervals from consecutive events
    intervals = librosa.util.frame(beat_samples, frame_length=2, hop_length=1).T
    #Reverse the beat intervals
    clip_remix = librosa.effects.remix(clip, intervals[::-1])
    if return_audio:
        return clip_remix
    else:
        return to_sp(clip_remix)
    
#autocorrelate #returns covariance matrix
"""def autocorrelate_audio(clip):
    clip_autocorrelate = librosa.autocorrelate(clip)
    return to_sp(clip_autocorrelate)"""

#pitch shift
def pitch_shift_audio(clip, sample_rate, return_audio=False):
    num_steps = random.randint(-15,15)
    if num_steps == 0:
        num_steps = random.randint(1,10)
    num_bins = random.randint(5,24)
    clip_pitch_shifted = librosa.effects.pitch_shift(clip, sample_rate, n_steps=num_steps, bins_per_octave=num_bins) # shifted by 4 half steps # maybe [-15,15] avoid 0, bins in [5,24]
    if return_audio:
        return clip_pitch_shifted
    else:
        return to_sp(clip_pitch_shifted)

#time stretching: https://en.wikipedia.org/wiki/Audio_time_stretching_and_pitch_scaling
def time_stretch_audio(clip, return_audio=False):
    speed_factor = random.uniform(0.7, 1.3) #make sure it is not 1. If < 1, makes the sound deeper, if > 1 creates higher frequencies
    clip_time_stretched = librosa.effects.time_stretch(clip, speed_factor)
    if return_audio:
        return clip_time_stretched
    else:
        return to_sp(clip_time_stretched)

#Spectral Distorsion Model
def spectral_distorsion_model_audio(clip, return_audio=False):
    #Create a random phase distortion model with sigma equal to 0.4
    sigma = random.uniform(0.4, 0.9)
    PDM = np.array([np.complex(np.cos(p), np.sin(p)) for p in np.random.normal(0, sigma, size=513)])
    # Transform the original signal and apply augmentation for each frame
    f = librosa.stft(clip, window='hanning', hop_length=512, n_fft=1024)
    f *= PDM.reshape((-1, 1))
    #Reconstruct transformed signal
    clip_SDM = librosa.istft(f, window='hanning', hop_length=512)
    if return_audio:
        return clip_SDM
    else:
        return to_sp(clip_SDM)

#LPC synthesis: idea from https://stackoverflow.com/questions/61519826/how-to-decide-filter-order-in-linear-prediction-coefficients-lpc-while-calcu
def LPC_synthesis_audio1(clip, sample_rate, return_audio=False):
    order = random.randint(2,16)
    A = librosa.core.lpc(clip, order)
    fs = sample_rate
    rts = np.roots(A)
    rts = rts[np.imag(rts) >= 0]
    angz = np.arctan2(np.imag(rts), np.real(rts))
    frqs = angz * fs / (2 *  np.pi)
    frqs.sort()
    #
    b = np.hstack([[0], -1 * frqs[1:]])
    clip_LPC = scipy.signal.lfilter(b, [1], clip)
    if return_audio:
        return clip_LPC
    else:
        return to_sp(clip_LPC)

## Idea from https://librosa.org/doc/latest/generated/librosa.lpc.html
def LPC_synthesis_audio2(clip, return_audio=False):
    order = random.randint(2,16) # advice on how to choose order: https://stackoverflow.com/questions/61519826/how-to-decide-filter-order-in-linear-prediction-coefficients-lpc-while-calcu
    a = librosa.lpc(clip, order) # order in [2,16]
    b = np.hstack([[0], -1 * a[1:]])
    clip_LPC = scipy.signal.lfilter(b, [1], clip)
    if return_audio:
        return clip_LPC
    else:
        return to_sp(clip_LPC)


#util function
def apply_transfer(signal, transfer, interpolation='linear'):
    constant = np.linspace(-1, 1, len(transfer))
    interpolator = interp1d(constant, transfer, interpolation)
    return interpolator(signal)

# hard limiting
def limiter_audio(clip, sample_rate, return_audio=False): # in [0.3,0.8]
    transfer_len = 1000
    treshold=random.uniform(0.3, 0.8)
    transfer = np.concatenate([ np.repeat(-1, int(((1-treshold)/2)*transfer_len)),
                                np.linspace(-1, 1, int(treshold*transfer_len)),
                                np.repeat(1, int(((1-treshold)/2)*transfer_len)) ])
    clip_limiter = apply_transfer(clip, transfer)
    clip_limiter = clip_limiter * sample_rate
    if return_audio:
        return clip_limiter
    else:
        return to_sp(clip_limiter)

# smooth compression: if factor is small, its near linear, the bigger it is the
# stronger the compression
def arctan_compressor_audio(clip, sample_rate, return_audio=False): # factor in [2,10]
    factor=random.randint(2,10)
    constant = np.linspace(-1, 1, 1000)
    transfer = np.arctan(factor * constant)
    transfer /= np.abs(transfer).max()
    clip_arctan_compressor = apply_transfer(clip, transfer)
    clip_arctan_compressor = clip_arctan_compressor * sample_rate
    if return_audio:
        return clip_arctan_compressor
    else:
        return to_sp(clip_arctan_compressor)

#Smooth spectrum
# from https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
def smooth_spectrum_audio(clip, return_audio=False):
    modes = ['mirror', 'constant', 'nearest', 'wrap', 'interp']
    mode = modes[random.randint(0,4)]
    window_size = 2 * random.randint(7,25) + 1 #must be odd if interp mode
    poly_order = random.randint(2,3)
    clip_smoothed = savgol_filter(clip, window_size, poly_order, mode=mode)
    if return_audio:
        return clip_smoothed
    else:
        return to_sp(clip_smoothed)

#Another smoothing with Fourrier Transform
# Less obvious if it is really useful ??
def smooth_fft_audio(clip, return_audio=False):
    w = scipy.fftpack.rfft(clip)
    spectrum = w**2
    cutoff_idx = spectrum < (spectrum.max()/5)
    w2 = w.copy()
    w2[cutoff_idx] = 0
    clip_fft_smoothed = scipy.fftpack.irfft(w2)
    if return_audio:
        return clip_fft_smoothed
    else:
        return to_sp(clip_fft_smoothed)

#Another smoothing with convolution (moving average)
def smooth_convolve_audio(clip, return_audio=False):
    window_size = len(clip)
    modes = ['mirror', 'constant', 'nearest', 'wrap', 'reflect']
    mode = modes[random.randint(0,4)]
    if random.randint(0,2) == 0:
        kernel = gaussian(window_size, 1) # not really apparent since very small, use with wrap and reflect to amplify maybe
    else: # 2 times more frequent
        kernel = np.ones(window_size) # mean
    kernel = kernel / np.asarray(kernel).sum()
    clip_convolved = filters.convolve1d(clip, kernel, mode=mode)
    if return_audio:
        return clip_convolved
    else:
        return to_sp(clip_convolved)

#Phase vocoder. Phase vocoder. Given an STFT matrix D, speed up by a factor of rate
def phase_vocoder_audio(clip, return_audio=False):
    D = librosa.stft(clip, n_fft=2048, hop_length=512)
    #speed_rate = random.uniform(0.3, 2.) #Maybe remove 1.
    speed_rate = random.uniform(0.85, 1.15) #Adapted to ASC
    clip_speed = librosa.phase_vocoder(D, rate = speed_rate, hop_length=512) # rate in [0.3, 2.]
    clip_speed  = librosa.istft(clip_speed, hop_length=512)
    if return_audio:
        return clip_speed
    else:
        return to_sp(clip_speed)
    
def reverberance_audio(clip, sample_rate, return_audio=False):
    # create transformer
    tfm = sox.Transformer()
    reverberance = random.randint(20,60)
    high_freq_damping = random.randint(30,70)
    room_scale = random.randint(20,100)
    stereo_depth = random.randint(20,100)
    wet_gain = random.randint(-10,10)
    tfm.reverb(reverberance=reverberance, high_freq_damping=high_freq_damping, room_scale=room_scale, stereo_depth=stereo_depth, wet_gain=wet_gain)
    # transform an in-memory array and return an array
    y = tfm.build_array(input_array=clip, sample_rate_in=sample_rate)
    if return_audio:
        return y
    else:
        return to_sp(y)

#Speed tuning
#slightly change the speed of the audio, then pad or slice it.
def speed_tune_audio(clip, sample_rate, return_audio=False):
    speed_rate = np.random.uniform(0.7,1.3)
    wav_speed_tune = cv2.resize(clip, (1, int(len(clip) * speed_rate))).squeeze()
    #print('speed rate: %.3f' % speed_rate, '(lower is faster)')
    if len(wav_speed_tune) < sample_rate: #sample_rate=16000
        pad_len = sample_rate - len(wav_speed_tune)
        wav_speed_tune = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
                               wav_speed_tune,
                               np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
    else: 
        cut_len = len(wav_speed_tune) - sample_rate
        wav_speed_tune = wav_speed_tune[int(cut_len/2):int(cut_len/2)+sample_rate]
    if return_audio:
        return wav_speed_tune
    else:
        return to_sp(wav_speed_tune)

#Add background noise / mix 2 audios
def mix_audios(clip1, clip2, return_audio=False):
    min_length = min(clip1.shape[0], clip2.shape[0])
    if clip2.shape[0]-min_length == 0: #if clip2(=background) is shorter, choose start_=0 to avoid error with np.random.randint
        start_ = 0
    else:
        start_ = np.random.randint(clip2.shape[0]-min_length)
    clip2_slice = clip2[start_ : start_+min_length]
    clip_with_bg = clip1[:min_length] * np.random.uniform(0.8, 1.2) + \
                  clip2_slice[:min_length] * np.random.uniform(0.1, 0.8) # vary values
    if return_audio:
        return clip_with_bg
    else:
        return to_sp(clip_with_bg)


#Speech Enhancement: Collection of single channel noise reduction (SCNR) algorithms for speech
#https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.denoise.html
#https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.denoise.spectral_subtraction.html
def spectral_subtraction_audio(clip, return_audio=False):
    denoised_signal = pyroomacoustics.denoise.spectral_subtraction.apply_spectral_sub(noisy_signal=clip, nfft=512,
                                     db_reduc=10, lookback=5,
                                     beta=20, alpha=3)
    if return_audio:
        return denoised_signal
    else:
        return to_sp(denoised_signal)

#iterative_wiener
#https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.denoise.iterative_wiener.html
def iterative_wiener_audio(clip, return_audio=False):
    denoised_signal = pyroomacoustics.denoise.iterative_wiener.apply_iterative_wiener(noisy_signal=clip, frame_len=512,
                                         lpc_order=20, iterations=2,
                                         alpha=0.8, thresh=0.01)
    if return_audio:
        return denoised_signal
    else:
        return to_sp(denoised_signal)

#apply_subspace
#https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.denoise.subspace.html
def subspace_audio(clip, return_audio=False):
    denoised_signal = pyroomacoustics.denoise.subspace.apply_subspace(noisy_signal=clip, frame_len=256, mu=10,
                                 lookback=10, skip=2, thresh=0.01)
    if return_audio:
        return denoised_signal
    else:
        return to_sp(denoised_signal)

#Griffin Lim vocoder
def Griffin_Lim_vocoder(clip, return_audio=False):
    # These are the parameters of the STFT
    fft_size = 512
    hop = fft_size // 4
    win_a = np.hamming(fft_size)
    win_s = pyroomacoustics.transform.compute_synthesis_window(win_a, hop)
    n_iter = 200

    engine = pyroomacoustics.transform.STFT(
        fft_size, hop=hop, analysis_window=win_a, synthesis_window=win_s
    )
    X = engine.analysis(clip)
    X_mag = np.abs(X)

    griffin_lim_clip = pyroomacoustics.phase.griffin_lim(X_mag, hop, win_a, n_iter=n_iter)
    if return_audio:
        return griffin_lim_clip
    else:
        return to_sp(griffin_lim_clip)

#Add between 2 and 6 random transformations (among 20 transformations) to an audio waveform.
#Each call uses params that are selected randomly within the permitted range of each one.
#def sequencial_transforms(clip, sample_rate, min_transforms=2 , max_transforms=6, include_sinc=True): #20 Transforms in total
def sequencial_transforms(clip, sample_rate, min_transforms=1 , max_transforms=3, include_sinc=True): #20 Transforms in total
    transforms_list = ['compand', 'fade', 'pitch', 'bass', 'bend', 'chorus', 
                               'contrast', 'echo', 'flanger', 'gain', 'hilbert', 'tremolo',
                              'treble', 'tempo', 'speed', 'overdrive', 'phaser', 'reverse',
                              'reverb', 'sinc']
    num_transforms = random.randint(min_transforms,max_transforms)
    
    # create transformer
    tfm = sox.Transformer()
    # trim the audio between 5 and 10.5 seconds.
    ###tfm.trim(5, 10.5) #creates probelms NOOO!
    while num_transforms > 0:
        transform = random.randint(1,20)
        #print(transform)
        if (transform==1):
            # apply compression
            if 'compand' in transforms_list:
                tfm.compand()
                transforms_list.remove('compand')
                num_transforms -= 1
        elif (transform==2):
            # apply a fade in and fade out
            if 'fade' in transforms_list: 
                tfm.fade(fade_in_len=1.0, fade_out_len=0.5)
                transforms_list.remove('fade')
                num_transforms -= 1
        elif (transform==3):
            # shift the pitch up by 2 semitones
            if 'pitch' in transforms_list: 
                n_semitones = random.randint(-10,20)
                quick = [True, False]
                quick = quick[random.randint(0,1)]
                tfm.pitch(n_semitones=n_semitones, quick=quick) #[-10,20]
                transforms_list.remove('pitch')
                num_transforms -= 1
        elif (transform==4):
            #
            if 'bass' in transforms_list: 
                gain_db = random.randint(-20,20)
                slope = random.uniform(0.3, 1.)
                tfm.bass(gain_db=gain_db, frequency=100.0, slope=slope)# slope in [0.3, 1.], gain_db in [-20,20]
                transforms_list.remove('bass')
                num_transforms -= 1
        elif (transform==5):
            #
            if 'bend' in transforms_list: 
                audio_duration = librosa.get_duration(clip)
                num_bends = random.randint(2,5)
                val_cents = np.random.randint(-500,800,num_bends).tolist()
                time_points = np.random.uniform(low=0.1, high=audio_duration, size=num_bends*2)
                time_points = np.sort(time_points)
                start_times = []
                end_times = []
                i = 0
                while i < num_bends * 2:
                    start_times.append(time_points[i])
                    end_times.append(time_points[i+1])
                    i += 2
                #tfm.bend(n_bends=3, start_times=[0.5, 1.2, 2.], end_times=[1.15,1.9, 2.4], cents=[800, 160, -500], frame_rate=25, oversample_rate=16)
                tfm.bend(n_bends=num_bends, start_times=start_times, end_times=end_times, cents=val_cents, frame_rate=25, oversample_rate=16)
                transforms_list.remove('bend')
                num_transforms -= 1
        elif (transform==6):
            #
            if 'chorus' in transforms_list: 
                n_voices = random.randint(2,5)
                tfm.chorus(gain_in=0.5, gain_out=0.9, n_voices=n_voices) # n_voices in [2,5]
                transforms_list.remove('chorus')
                num_transforms -= 1
        elif (transform==7):
            #
            if 'contrast' in transforms_list: 
                amount = random.randint(5,100)
                tfm.contrast(amount=amount) # in [5,100]
                ##tfm.deemph() #NOT TO USE
                ###tfm.earwax() #NOT TO USE
                transforms_list.remove('contrast')
                num_transforms -= 1
        elif (transform==8):
            if 'echo' in transforms_list: 
                tfm.echo()
                transforms_list.remove('echo')
                num_transforms -= 1
        elif (transform==9):
            #
            if 'flanger' in transforms_list: 
                tfm.flanger()
                transforms_list.remove('flanger')
                num_transforms -= 1
        elif (transform==10):
            if 'gain' in transforms_list: 
                gain_db = random.randint(-20,20)
                tfm.gain(gain_db)
                transforms_list.remove('gain')
                num_transforms -= 1
        elif (transform==11):
            #
            if 'hilbert' in transforms_list: 
                tfm.hilbert(num_taps=None)
                #tfm.vol(5) # For bonafide. positive value
                transforms_list.remove('hilbert')
                num_transforms -= 1
        elif (transform==12):
            #
            if 'tremolo' in transforms_list: 
                depth = random.randint(10,100)
                tfm.tremolo(speed=6.0, depth=depth) # depth in [10., 100.]
                transforms_list.remove('tremolo')
                num_transforms -= 1
        elif (transform==13):
            #
            if 'treble' in transforms_list: 
                frequency = random.randint(100,3000)
                slope = random.uniform(0.3, 1.)
                gain_db = random.randint(-20,20)
                tfm.treble(gain_db=gain_db, frequency=frequency, slope=slope) #  slope in [0.3,1.0], frequency in [100, 3000.], db in [-20,20]
                transforms_list.remove('treble')
                num_transforms -= 1
        elif (transform==14):
            #
            if 'tempo' in transforms_list: 
                factor = random.uniform(0.1, 1.9)
                quick = [True, False]
                quick = quick[random.randint(0,1)]
                tfm.tempo(factor=factor, audio_type='s', quick=quick) # factor in [0.1,1.9], quick in [True, False]
                transforms_list.remove('tempo')
                num_transforms -= 1
        elif (transform==15):
            #
            if 'speed' in transforms_list: 
                factor = random.uniform(0.1, 1.9)
                tfm.speed(factor=factor) # factor in [0.1,1.9]
                transforms_list.remove('speed')
                num_transforms -= 1
        elif (transform==16):
            #
            if 'overdrive' in transforms_list: 
                colour = random.uniform(1.,20.)
                gain_db = random.uniform(1.,100.)
                tfm.overdrive(gain_db=gain_db, colour=colour) #both in [-20,20]
                transforms_list.remove('overdrive')
                num_transforms -= 1
            #tfm.pad(start_duration=1.0, end_duration=1.0) # Maybe for bonafide by adding silence to both beginning and end
        elif (transform==17):
            #
            if 'phaser' in transforms_list: 
                gain_in = random.uniform(0.1, 1.)
                modulation_shape = ['triangular', 'sinusoidal']
                modulation_shape = modulation_shape[random.randint(0,1)]
                speed = random.uniform(0.1, 2.)    
                decay = random.uniform(0.1, 0.5)
                delay = random.randint(1,5)
                tfm.phaser(gain_in=gain_in, gain_out=0.74, delay=delay, decay=decay, speed=speed, modulation_shape=modulation_shape) # gain in [0.1, 1.], modulation_shape in ['triangular', 'sinusoidal'], speed in [0.1, 2], decay in [0.1,0.5], delay in [1,5]
                transforms_list.remove('phaser')
                num_transforms -= 1
                #tfm.repeat(count=1) # how many times to repeat audio. Maybe For bonafide
        elif (transform==18):
            #
            if 'reverse' in transforms_list: 
                if (random.randint(0,2)==0):
                    tfm.reverse() # should be scarce
                    transforms_list.remove('reverse')
                    num_transforms -= 1
        elif (transform==19):
            #
            if 'reverb' in transforms_list: 
                reverberance = random.randint(20,60)
                high_freq_damping = random.randint(30,70)
                room_scale = random.randint(20,100)
                stereo_depth = random.randint(20,100)
                wet_gain = random.randint(-10,10)
                tfm.reverb(reverberance=reverberance, high_freq_damping=high_freq_damping, room_scale=room_scale, stereo_depth=stereo_depth, wet_gain=wet_gain)
                transforms_list.remove('reverb')
                num_transforms -= 1
        elif (transform==20):
            if include_sinc:
                if 'sinc' in transforms_list: 
                    # cutoff_freq in [100, 9000], type in ['reject', 'pass', 'low', 'high'], attenuation in [40, 180], phase_response in [10, 90]
                    stop_band_attenuation = random.randint(40,180)
                    phase_response = random.randint(10, 90)
                    filter_types = ['reject', 'pass', 'low', 'high']
                    filter_type = filter_types[random.randint(0,3)]
                    if filter_type in ['low', 'high']:
                        cutoff_freq = random.randint(100, 5000)
                    else:
                        cutoff_freq = np.sort(np.random.randint(100,5000,2)).tolist()
                    tfm.sinc(filter_type=filter_type, cutoff_freq=cutoff_freq, stop_band_attenuation=stop_band_attenuation, phase_response=phase_response)
                    transforms_list.remove('sinc')
                    num_transforms -= 1

    # transform an in-memory array and return an array
    y = tfm.build_array(input_array=clip, sample_rate_in=sample_rate)

    return y

# Call sequencial_transforms and handles eventual exceptions.
#def apply_sequential_transforms(clip, sample_rate, min_transforms=2 , max_transforms=6, return_audio=False):
def apply_sequential_transforms(clip, sample_rate, min_transforms=1 , max_transforms=3, return_audio=False):
    Done = False
    include_sinc = True #sometimes problematic
    while (Done != True):
        try:
            y = sequencial_transforms(clip, sample_rate, min_transforms=min_transforms , max_transforms=max_transforms, include_sinc=include_sinc)
            Done = True
        except:
            include_sinc = False
            continue
    if return_audio:
        return y
    else:
        return to_sp(y)

# add white noise
def add_whitenoise(x, rate=0.002, return_audio=False):
    #return to_sp(x + rate*np.random.randn(len(x)))
    wn_x = x + rate*np.random.randn(len(x))
    if return_audio:
        return wn_x
    else:
        return to_sp(wn_x)

# add pink noise
# https://www.dsprelated.com/showarticle/908.php
def add_pinknoise(x, ncols=11, alpha=0.002, return_audio=False):
    """Generates pink noise using the Voss-McCartney algorithm.
    
    nrows: number of values to generate
    rcols: number of random sources to add
    
    returns: NumPy array
    """
    nrows = len(x)
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)
    
    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)
    pn_x = alpha * total.values + x
    if return_audio:
        return pn_x
    else:
        return to_sp(pn_x)

# add random line
def draw_line(x, length=[5,20], thickness_length=[2,4]): #works with an image as input
    result = np.copy(x)
    width = x.shape[1]
    height = x.shape[0]
    angle = [0, np.pi/2, np.pi, np.pi*3/2]
    np.random.shuffle(angle)

    length = np.random.randint(length[0],length[1])
    x1 = random.randint(length, width-length)
    x2 = x1 + length*np.cos(angle[0])
    y1 = random.randint(length, height-length)
    y2 = y1 + length*np.sin(angle[0])

    thickness = random.randint(thickness_length[0], thickness_length[1])
    color1 = float(np.max(x))

    cv2.line(result, (x1,y1), (int(np.min([width,x2])),int(np.min([height,y2]))), color1, thickness)

    return result

# change Hz to average
def average_Hz(x, length=[2,4]):
    result = np.copy(x)
    height = x.shape[0]

    length = np.random.randint(length[0],length[1])
    begin = np.random.randint(0, height-length)
    for i in range(length):
        result[begin+i] = np.mean(result[begin+i])

    return result

"""def to_img2(x):
    result = cv2.resize(to_sp(x), (224,224))
    return np.array(result)

# change wave data to stft
def to_sp(x, n_fft=512, hop_length=256):
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
    sp = librosa.amplitude_to_db(np.abs(stft))
    return sp"""

#Sound filtering
#from https://github.com/davidpraise45/Audio-Signal-Processing/blob/master/Sound-Filtering.py
def butterworth_sound_filter(clip, sample_rate, return_audio=False):
    rand = random.randint(1,3)
    if rand == 1:
        b,a = signal.butter(5, 1000/(sample_rate/2), btype='highpass') # ButterWorth filter 4350
        filteredSignal = signal.lfilter(b,a,clip)
    elif rand == 2:
        c,d = signal.butter(5, 380/(sample_rate/2), btype='lowpass') # ButterWorth low-filter
        filteredSignal = signal.lfilter(c,d,clip) # Applying the filter to the signal
    else:
        b,a = signal.butter(5, 1000/(sample_rate/2), btype='highpass') # ButterWorth filter 4350
        filteredSignal = signal.lfilter(b,a,clip)
        c,d = signal.butter(5, 380/(sample_rate/2), btype='lowpass') # ButterWorth low-filter
        filteredSignal = signal.lfilter(c,d,filteredSignal) # Applying the filter to the signal
    
    if return_audio:
        return filteredSignal
    else:
        return to_sp(filteredSignal)
    
#checkerboard patterns
def checkerboard_patterns(clip, sample_rate, return_audio=False):
    # Transform the original signal and apply augmentation for each frame
    f = librosa.stft(clip, window='hanning', hop_length=512, n_fft=1024)
    rand = random.randint(1,4)
    if rand == 1:
        x = np.zeros(f.shape)#,dtype=int)
        # fill with 1 the alternate rows and columns
        x[1::2,::2] = 1
        x[::2,1::2] = 1
    elif rand == 2:
        x = np.random.random(f.shape)
    else:
        x = np.random.randint(0,2,f.shape)
        
    f *= x
    result = librosa.istft(f, window='hanning', hop_length=512)
    
    if return_audio:
        return result
    else:
        return to_sp(result)

#mel_to_stft_griffinLim
#The conversion from Mel to STFT spectrogram is not entirely lossless (there may be overlapping frequency ranges, 
#due to the overlapping triangular filters used in the construction of the Mel spectrogram), and the conversion 
#from STFT magnitude spectrogram to the time domain (i.e., to audio) is certainly not perfect, as the STFT magnitude 
#spectrogram is lacking the phase information, which must be approximated using the Griffin Lim algorithm. 
#This approximation is never perfect and introduces phase artifacts (metallic "phasiness").
#Here, the generated Mel spectrogram is used to approximate the STFT magnitude. The STFT spectrogram is 
#then converted back the time domain using the Griffin Lim algorithm.
def mel_to_stft_griffinLim(clip, sample_rate, return_audio=False):
    melspectrogram = librosa.feature.melspectrogram(y=clip, sr=sample_rate, window='hanning')#, n_fft=1024, hop_length=512)
    S = librosa.feature.inverse.mel_to_stft(melspectrogram)
    result = librosa.griffinlim(S)
    if return_audio:
        return result
    else:
        return to_sp(result)
