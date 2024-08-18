"""
Script taken from https://github.com/declare-lab/tango/blob/master/tools/torch_tools.py.
Contains the methods required to compute the mel-spectrogram of an audio.
"""

import torch
import librosa
import torchaudio


def normalize_wav(waveform):
    """
    Normalize the input waveform.

    Params:
        - waveform (torch.FloatTensor): The input waveform tensor.

    Returns:
        - torch.FloatTensor: The normalized waveform tensor.
    """
    waveform = waveform - torch.mean(waveform)
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
    return waveform * 0.5


def pad_wav(waveform, segment_length):
    """
    Pad or truncate the input waveform to the desired segment length.

    Params:
        - waveform (torch.FloatTensor): The input waveform tensor.
        - segment_length (int): The desired segment length. If None, no padding or truncating is performed.

    Returns:
        - torch.FloatTensor: The padded or truncated waveform tensor.
    """
    waveform_length = len(waveform)
    
    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    else:
        pad_wav = torch.zeros(segment_length - waveform_length).to(waveform.device)
        waveform = torch.cat([waveform, pad_wav])
        return waveform
    
    
def _pad_spec(fbank, target_length=1024):
    """
    Pad or truncate the input spectrogram to the desired length and ensure it has an even number of channels.

    Params:
        - fbank (torch.FloatTensor): The input spectrogram tensor of shape (batch, n_frames, channels).
        - target_length (int, optional): The desired number of frames. Defaults to 1024.

    Returns:
        - torch.FloatTensor: The padded or truncated spectrogram tensor.
    """
    batch, n_frames, channels = fbank.shape
    p = target_length - n_frames
    if p > 0:
        pad = torch.zeros(batch, p, channels).to(fbank.device)
        fbank = torch.cat([fbank, pad], 1)
    elif p < 0:
        fbank = fbank[:, :target_length, :]

    if channels % 2 != 0:
        fbank = fbank[:, :, :-1]

    return fbank


def read_wav_file(filename, segment_length, trim_duration=False, trim_start=10, trim_end=20):
    """
    Read, normalize, and optionally trim and pad a WAV file.

    Params:
        - filename (str): The path to the WAV file(s).
        - segment_length (int): The desired segment length.
        - trim_duration (bool, optional): Whether to trim the audio duration. Defaults to False.
        - trim_start (int, optional): The start time in seconds for trimming. Defaults to 10.
        - trim_end (int, optional): The end time in seconds for trimming. Defaults to 20.

    Returns:
        - torch.FloatTensor: The processed waveform tensor.
    """
    waveform, sr = torchaudio.load(filename)  # Faster!!!
    goal_freq = 16000
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=goal_freq)[0]
    try:
        waveform = normalize_wav(waveform)
    except:
        print ("Exception normalizing:", filename)
        waveform = torch.ones(160000)
        
    # Audio Length Trimming
    if trim_duration is not None:
        audio_duration = librosa.get_duration(y=waveform.numpy(), sr=goal_freq)
        trim_start = max(0, min(trim_start, audio_duration))
        trim_end = max(0, min(trim_end, audio_duration))

        # Trimming to specific range
        start_sample = int(goal_freq * trim_start)
        end_sample = int(goal_freq * trim_end)
        waveform = waveform[start_sample:end_sample]
        
        # Calculate and print trimmed audio duration
        trimmed_duration = (end_sample - start_sample) / goal_freq
    
    waveform = pad_wav(waveform, segment_length).unsqueeze(0)
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = 0.5 * waveform
    return waveform


def get_mel_from_wav(audio, _stft):
    """
    Compute the mel spectrogram, log magnitudes, and energy from an audio waveform.

    Params:
        - audio (torch.FloatTensor): The input audio tensor.
        - _stft: The STFT model to compute the spectrogram.

    Returns:
        - melspec: The mel spectrogram.
        - log_magnitudes_stft: The log magnitudes.
        - energy: The energy tensors.
    """
    audio = torch.nan_to_num(torch.clip(audio, -1, 1))
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, log_magnitudes_stft, energy = _stft.mel_spectrogram(audio)
    return melspec, log_magnitudes_stft, energy


def wav_to_fbank(paths, target_length=1024, fn_STFT=None):
    """
    Convert a list of WAV file paths to a batch of mel spectrograms and log magnitudes.

    Params:
        - paths (list of str): List of paths to the WAV files.
        - target_length (int, optional): The target length for the spectrogram. Defaults to 1024.
        - fn_STFT: The STFT model to compute the spectrogram.

    Returns:
        - fbank: The batch of mel spectrograms.
        - log_magnitudes_stft: The log magnitudes.
        - waveform: The waveforms.
    """
    assert fn_STFT is not None

    waveform = torch.cat([read_wav_file(path, target_length * 160, True) for path in paths], 0)  # hop size is 160

    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)
    fbank = fbank.transpose(1, 2)
    log_magnitudes_stft = log_magnitudes_stft.transpose(1, 2)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )

    return fbank, log_magnitudes_stft, waveform