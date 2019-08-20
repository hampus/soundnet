import math
import random
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from numpy.fft import irfft
from scipy.signal.windows import tukey
from scipy.signal import firwin

import coding

SYMBOLS = [-1, 1]


class Modulator:
    def __init__(self, profile):
        self.profile = profile
        self.window = tukey(self.profile.padded_len, self.profile.tukey_r)
        self.pad_block = np.zeros(0)

    def get_frame(self, message):
        # Reset state
        self.pad_block = np.zeros(self.profile.pad_size*2)
        blocks = [np.zeros(math.ceil(self.profile.rate*0.1))]
        # Calculate symbols
        message = self.prepare_message(message)
        # Encode symbols
        for s in range(message.shape[0]):
            blocks.append(self.get_symbol(message[s]))
        # Add final padding block
        blocks.append(self.pad_block)
        data = np.concatenate(blocks)
        # Filter
        if self.profile.highpass > 0:
            hpfilter = firwin(501, self.profile.highpass,
                              fs=self.profile.rate, pass_zero='highpass')
            data = np.convolve(data, hpfilter)
        # Equalize
        data = data / ((np.max(np.absolute(data))) / self.profile.strength)
        # Return data in correct format
        data = data.reshape((len(data), 1))
        #data = np.hstack([data, np.zeros((len(data), 1))])
        return data.astype(np.float32)

    def prepare_message(self, message):
        carriers = self.profile.carriers
        data = self.profile.preamble + message
        print(data)
        data = [x for x in data for _ in range(self.profile.repetition)]
        # print(data)
        padding = (carriers - (len(data) % carriers)) % carriers
        data = ([0] * carriers) + data + ([0] * padding)
        # print(data)
        symbols = len(data) // carriers
        data = np.array(data).reshape((symbols, carriers))
        # print(data)
        # Add differential coding
        for s in range(1, symbols):
            for c in range(carriers):
                data[s, c] = (data[s, c] + data[s-1, c]) % 2
        print(data)
        print('Symbols:', symbols)
        print('Duration:', symbols*self.profile.sym_len/self.profile.rate)
        return data

    def get_symbol(self, values):
        symbol = self.encode_symbol(values)
        pad_len = self.profile.pad_size*2
        if pad_len == 0:
            return symbol
        blocks = [symbol[:pad_len],
                  symbol[pad_len:-pad_len], symbol[-pad_len:]]
        blocks[0] = np.add(blocks[0], self.pad_block)
        self.pad_block = blocks[2]
        return np.concatenate(blocks[:-1])

    def encode_symbol(self, values):
        carriers = self.profile.carriers
        fs = np.zeros(self.profile.fft_len//2 + 1)
        for c in range(carriers):
            f = self.profile.first_carrier + self.profile.carrier_step*c
            fs[f] = SYMBOLS[values[c]]
        block = irfft(fs)

        before = self.profile.guard_len + self.profile.pad_size
        after = self.profile.pad_size
        block = np.pad(block, ((before, after),), 'wrap')

        return np.multiply(block, self.window)


profile = coding.CodingProfile(coding.PROFILE)
profile.print()

modulator = Modulator(profile)
frame = modulator.get_frame([1]*profile.msg_len)
# print(frame)

# plt.plot(frame)
# plt.specgram(frame.reshape(len(frame)), NFFT=256,
#              Fs=modulator.profile.rate, noverlap=0)
# plt.show()

frame = np.concatenate([frame]*10)
while True:
    sd.play(frame, samplerate=modulator.profile.rate, blocking=True)
