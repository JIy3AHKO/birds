import numpy as np
from scipy import signal


def insert(target, fragment, pos, sr=48000, alpha=0.75, fade=0.1):
    fade_len = int(fade * sr)
    pos = int(pos * sr)

    fade_fn = np.arange(0, alpha, alpha / fade_len)
    fade_len = fade_fn.shape[0]

    target[pos:pos+fade_len] *= 1 - fade_fn
    fragment[:fade_len] *= fade_fn

    target[pos + len(fragment) - fade_len:pos + len(fragment)][::-1] *= 1 - fade_fn
    fragment[-fade_len:][::-1] *= fade_fn

    target[pos + fade_len:pos + len(fragment) - fade_len] *= 1 - alpha
    fragment[fade_len:-fade_len] *= alpha

    target[pos:pos + len(fragment)] += fragment

    return target


def audio_filter(audio, mode, w, wet):
    sos = signal.butter(10, w, mode, output='sos', fs=48000)

    filtered = signal.sosfilt(sos, audio).astype('float32', copy=False)

    return audio * (1 - wet) + filtered * wet


def fade_concat(a1, a2, overlap=0.5, sr=48000):
    overlap = int(sr * overlap)

    assert len(a1) > overlap and len(a2) > overlap

    start = a1[:-overlap]
    end = a2[overlap:]

    blend1 = a1[-overlap:]
    blend2 = a2[:overlap]

    alphas = np.arange(0, 1, 1 / overlap)

    result = np.concatenate([start, blend1 * (1 - alphas) + blend2 * alphas, end])

    return result


def lin_fade(a, mode, duration, sr=48000):
    assert mode in ['in', 'out']
    fade_len = int(duration * sr)
    if mode == 'in':
        a[:fade_len] *= np.arange(0, 1, 1 / fade_len)

    if mode == 'out':
        a[fade_len:] *= np.arange(1, 0, -1 / fade_len)

    return a
