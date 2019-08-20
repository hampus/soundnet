import math

AUDIBLE = {
    'rate': 48000,
    'freq_min': 2000,
    'freq_max': 18000,
    'carrier_step': 20,
    'fft_len': 1024,
    'guard_len': 256,
    'pad_size': 256,
    'preamble': [1, 1, 1, 1, 0, 0, 0, 1]*3,
    'repetition': 3,
    'strength': 0.9,
    'msg_len': 64,
    'highpass': 0,
}

SILENT = {
    'rate': 48000,
    'freq_min': 17000,
    'freq_max': 20000,
    'carrier_step': 3,
    'fft_len': 1024,
    'guard_len': 256,
    'pad_size': 256,
    'preamble': [1, 1, 1, 0, 0, 1]*3,
    'repetition': 3,
    'strength': 0.9,
    'msg_len': 160,
    'highpass': 16500,
}

CABLE = {
    'rate': 48000,
    'freq_min': 100,
    'freq_max': 20000,
    'carrier_step': 1,
    'fft_len': 1024,
    'guard_len': 128,
    'pad_size': 64,
    'preamble': [1, 1, 1, 1, 0, 0, 0, 1]*25,
    'repetition': 1,
    'strength': 0.9,
    'msg_len': 4*1024,
    'highpass': 0,
}

CABLE_SLOW = {
    'rate': 48000,
    'freq_min': 1000,
    'freq_max': 15000,
    'carrier_step': 2,
    'fft_len': 1024,
    'guard_len': 256,
    'pad_size': 64,
    'preamble': [1, 1, 1, 1, 0, 0, 0, 1]*50,
    'repetition': 1,
    'strength': 0.9,
    'msg_len': 4*1024,
    'highpass': 0,
}

PROFILE = SILENT


class CodingProfile:
    def __init__(self, profile):
        self.rate = profile['rate']
        self.freq_min = profile['freq_min']
        self.freq_max = profile['freq_max']
        self.carrier_step = profile['carrier_step']
        self.fft_len = profile['fft_len']
        self.guard_len = profile['guard_len']
        self.pad_size = profile['pad_size']
        self.repetition = profile['repetition']
        self.preamble = profile['preamble']
        self.strength = profile['strength']
        self.msg_len = profile['msg_len']
        self.highpass = profile['highpass']
        self.sym_len = self.fft_len + self.guard_len
        self.freq_per_carrier = self.rate / self.fft_len
        self.first_carrier = math.ceil(
            self.freq_min / (self.rate / self.fft_len))
        self.carriers = math.floor(
            ((self.freq_max / (self.rate / self.fft_len)) - self.first_carrier) / self.carrier_step)
        self.padded_len = self.sym_len + self.pad_size*2
        self.tukey_r = 1.0 - (self.padded_len -
                              self.pad_size*4) / self.padded_len
        self.frame_len = (1 + math.ceil(
            (len(self.preamble) + self.msg_len)*self.repetition / self.carriers)) * self.sym_len

    def print(self):
        print('Modulation profile:')
        print('Rate:', self.rate)
        print('Symbol length:', self.sym_len)
        print('FFT length:', self.fft_len)
        print('Guard interval:', self.guard_len / self.fft_len)
        print('Padding:', self.pad_size)
        print('Padded length:', self.padded_len)
        print('Tukey r:', self.tukey_r)
        print('Code reptition:', self.repetition)
        print('First carrier:', self.first_carrier)
        print('Carriers:', self.carriers)
        print('Carrier step:', self.carrier_step)
        print('First freq:', self.first_carrier * self.freq_per_carrier)
        print('Last freq:', (self.first_carrier + (self.carriers-1)
                             * self.carrier_step) * self.freq_per_carrier)
        print('Carrier separation (Hz):',
              self.carrier_step * self.freq_per_carrier)
        print('Symbols per second:', self.rate / self.sym_len)
        print('Frame length:', self.frame_len)
        print('Frame duration:', self.frame_len / self.rate)
        print('High pass filter:', self.highpass)
