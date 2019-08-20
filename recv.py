import math
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from numpy.fft import rfft
from scipy.signal.windows import tukey
import sys

import coding

DURATION = 48000*3
SCAN_STEP = 32


class Demodulator:
    def __init__(self, profile):
        self.profile = profile
        self.used_carriers = (np.arange(self.profile.carriers) *
                              self.profile.carrier_step + self.profile.first_carrier)
        print('Carriers:', self.used_carriers)

    def scan(self, data):
        frame_len = self.profile.frame_len
        print('First scan...')
        errors = []
        for offset in range(0, len(data) - frame_len, SCAN_STEP):
            errors.append(self.scan_error(data[offset:]))
        offset = np.argmin(errors) * SCAN_STEP
        return errors, offset

    def scan_fine(self, data, start_offset):
        frame_len = self.profile.frame_len
        print('Fine scan...')
        errors = []
        start = max(start_offset - SCAN_STEP*2, 0)
        end = min(start_offset + SCAN_STEP*2, len(data) - frame_len)
        for offset in range(start, end, 1):
            errors.append(self.scan_error(data[offset:]))
        offset = np.argmin(errors) + start
        return errors, offset

    def scan_error(self, data):
        preamble = self.profile.preamble
        decoded = self.read_data(data, len(self.profile.preamble))
        error = sum([abs(preamble[i] - decoded[i])
                     for i in range(len(preamble))])
        error /= len(preamble)
        return error

    def read_data(self, data, length):
        carriers = self.profile.carriers
        symbols = math.ceil(length * self.profile.repetition /
                            self.profile.carriers) + 1
        # Read symbols
        frame = self.read_symbols(data, symbols)
        # print(frame)
        # Differential decoding
        weights = np.absolute(frame)**4
        for s in range(symbols-1, 0, -1):
            for c in range(carriers):
                frame[s, c] = np.angle(frame[s, c] / frame[s-1, c])
        frame = np.clip(np.absolute(frame[1:]) / np.pi, 0.0, 1.0)
        frame = np.ravel(frame)[:length * self.profile.repetition]
        weights = np.ravel(weights)[:length * self.profile.repetition]
        # print(frame)
        # Decode repetition
        frame = np.split(frame, length)
        weights = np.split(weights, length)
        for i in range(len(weights)):
            weights[i] = weights[i] / sum(weights[i])
        # print(weights)
        # print(frame)
        frame = np.average(frame, axis=1, weights=weights)
        # print(frame)
        return frame

    def read_symbols(self, data, symbols):
        #print('Reading symbols:', symbols)
        block_step = self.profile.sym_len
        block_len = self.profile.fft_len
        results = []
        for s in range(symbols):
            block = data[block_step*s:block_step*s + block_len]
            if len(block) < block_len:
                print('Missing data')
                sys.exit(1)
            fs = rfft(block)
            fs = np.take(fs, self.used_carriers)
            # print(np.absolute(fs)/np.max(np.absolute(fs)))
            #fs = np.divide(fs, np.clip(np.absolute(fs), 0.0001, None))
            # Add 1 to all (close to) zero elements
            fs = np.add(fs, (np.absolute(fs) < 0.0001).astype(int))
            # print(fs)
            results.append(fs)
        return np.vstack(results)

    def read_frame(self, data):
        length = len(self.profile.preamble) + self.profile.msg_len
        decoded = self.read_data(data, length)
        return decoded[len(self.profile.preamble):]


profile = coding.CodingProfile(coding.PROFILE)
profile.print()

demodulator = Demodulator(profile)

print('Recording...')
data = sd.rec(DURATION, samplerate=demodulator.profile.rate,
              channels=1, dtype=np.float32, blocking=True)
data = data.reshape(len(data))
print('Done')


errors, offset = demodulator.scan(data)
print('Best offset:', offset)
print('Min error:', min(errors))
plt.plot(errors)
plt.show()

errors, offset = demodulator.scan_fine(data, offset)
print('Best offset:', offset)
print('Min error:', min(errors))
data = data[offset:]

plt.specgram(data, NFFT=demodulator.profile.fft_len,
             Fs=demodulator.profile.rate, noverlap=0)
plt.show()


decoded = demodulator.read_frame(data)
print('Lowest value:', min(decoded))
print('Decoded frame:', decoded)
decoded = [int(round(x)) for x in decoded]
print('Decoded frame:', decoded)
ber = 1 - sum(decoded) / len(decoded)
print('BER:', ber)

# plt.plot(errors)
#plt.specgram(data, NFFT=256, Fs=demodulator.profile.rate, noverlap=0)
# plt.show()
