import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz 

"""

references:

- [Abel, Jonathan S., and David P. Berners. "Filter design using second-order peaking and shelving sections." ICMC. 2004.](https://quod.lib.umich.edu/i/icmc/bbp2372.2004.152/1/--filter-design-using-second-order-peaking-and-shelving?page=root;size=150;view=text)
- [Välimäki, Vesa, and Joshua D. Reiss. "All about audio equalization: Solutions and frontiers." Applied Sciences 6.5 (2016): 129.](https://www.mdpi.com/2076-3417/6/5/129)

"""


class GraphicEQ():
    def __init__(self, sr, octave="1/3oct"):
        self.sr = sr
        self.octave = octave
        
        self.set_params()
        self.set_coefs()
    
    def dB2amp(self, dB):
        return 10.0 ** (dB/40.0)

    def f2w(self, f):
        return 2.0 * np.pi * f / self.sr
        
    def set_params(self):
        if self.octave == "1/1oct": # 1オクターブバンドの場合
            self.bandfreqs = np.array([
                [22,       31.5,    44],
                [44,       63,      88],
                [88,      125,     177],
                [177,     250,     355],
                [355,     500,     710],
                [710,    1000,    1420],
                [1420,   2000,    2840],
                [2840,   4000,    5680],
                [5680,   8000,   11360],
                [11360, 16000,   22720]
            ])
        elif self.octave == "1/3oct": # 1/3オクターブバンドの場合
            self.bandfreqs = np.array([ #
                [   22.4,   25.2,  28.2], [  28.2,   31.5,   35.5], [   35.5,   40,    44.7], 
                [   44.7,   50,    56.2], [  56.2,   63,     70.8], [   70.8,   80,    89.1], 
                [   89.1,  100,   112],   [  112,   125,    141],   [  141,    160,   178],  
                [  178,    200,   224],   [  224,   250,    282],   [  282,    315,   355],  
                [  355,    400,   447],   [  447,   500,    562],   [  562,    630,   708], 
                [  708,    800,   891],   [  891,  1000,   1122],   [ 1122,   1250,  1413], 
                [ 1413,   1600,  1778],   [ 1778,  2000,   2239],   [ 2239,   2500,  2818], 
                [ 2818,   3150,  3548],   [ 3548,  4000,   4467],   [ 4467,   5000,  5623], 
                [ 5623,   6300,  7079],   [ 7079,  8000,   8913],   [ 8913,  10000, 11220], 
                [11220,  12500, 14130],   [14130, 16000,  17780],   [17780,  20000, 22390]
            ])
        
        self.n_band   = self.bandfreqs.shape[0]
        
        # バンド幅, ゲインを計算
        self.bandwidth = np.zeros(self.n_band)
        self.gain_db = np.ones(self.n_band) * 3
        self.amp = self.dB2amp(self.gain_db)
        
        for k, (flow, fc, fup) in enumerate(self.bandfreqs):
            self.bandwidth[k] = np.log2(fup) - np.log2(flow)
        
        self.b = np.zeros((self.n_band, 3))
        self.a = np.zeros((self.n_band, 3))
        self.q = np.zeros(self.n_band)
    
    def set_coefs(self, use_opt_amp=False):
        amp = self.opt_amp if use_opt_amp else self.amp
        
        for k in range(self.n_band):
            fc = self.bandfreqs[k, 1]
            w = self.f2w(fc)
            cos_w, sin_w = np.cos(w), np.sin(w)
            #alpha = 0.5 * sinw / self.q
            alpha = sin_w * np.sinh(0.5*np.log(2)*self.bandwidth[k]*w/sin_w)
            self.q[k] = 0.5*sin_w/alpha

            # set coef
            self.b[k, 0] = 1.0 + alpha * amp[k]
            self.b[k, 1] = -2.0 * cos_w
            self.b[k, 2] = 1.0 - alpha * amp[k]
            self.a[k, 0] = 1.0 + alpha / amp[k]
            self.a[k, 1] = -2.0 * cos_w
            self.a[k, 2] = 1.0 - alpha / amp[k]

            self.b[k] /= self.a[k,0]
            self.a[k] /= self.a[k,0]
        
    def freqz(self, worN=4096*10, plot_on=True):
        self.h_list = [None for k in range(self.n_band)]
        for k in range(self.n_band):
            self.h_w, self.h_list[k] = freqz(self.b[k], a=self.a[k], worN=worN)
            
        self.freqs = self.h_w / np.pi * (self.sr//2)
        
        self.h_series = np.ones(worN, dtype=np.complex)
        if plot_on:
            plt.clf()
            plt.subplot(2,1,1)
        for k in range(self.n_band):
            fc = self.bandfreqs[k, 1]
            self.h_series *= self.h_list[k]            
            
            if plot_on:
                plt.plot(self.freqs, 20*np.log10(np.abs(self.h_list[k])))
                            
        if plot_on:
            plt.xscale("log")
            plt.grid()
            plt.xlim([10,self.sr//2])
            plt.ylabel("dB")

            plt.subplot(2,1,2)
            plt.plot(self.freqs, 20*np.log10(np.abs(self.h_series)))
            plt.xscale("log")
            plt.grid()
            plt.xlim([10,self.sr//2])
            plt.ylabel("dB")
            
            plt.tight_layout()
            plt.show()
        
    def optimize_gain(self):
        self.gain_mx = np.zeros((self.n_band, self.n_band))
        self.freqs_fc_indices = np.array([
            np.abs(np.asarray(self.freqs) - self.bandfreqs[k, 1]).argmin()
            for k in range(self.n_band)
        ])
        self.freqs_fc = self.freqs[self.freqs_fc_indices]
        
        for k in range(self.n_band):
            fc_k = self.bandfreqs[k, 1]
            self.gain_mx[k, :] = 20*np.log10(np.abs(self.h_list[k])[self.freqs_fc_indices])
            self.gain_mx[k, :] /= self.gain_mx[k, k] 

        
        self.inv_gain_mx = np.linalg.inv(self.gain_mx)
        self.opt_gain_db = np.dot(self.inv_gain_mx, self.gain_db)
        self.opt_amp = self.dB2amp(self.opt_gain_db)
                
       
       
if __name__ == "__main__":
    eq = GraphicEQ(44100)
    eq.freqz(plot_on=True)
    eq.optimize_gain()
    
    np.set_printoptions(precision=2, suppress=True)
    print(eq.gain_mx, "\n")
    print(eq.opt_amp)
    
    eq.set_coefs(use_opt_amp=True)
    eq.freqz(plot_on=True)
    
    plt.clf()
    plt.title(f"gain_mx**0.25")
    plt.imshow(eq.gain_mx**0.25)
    plt.colorbar()
    plt.show()
    
    import code
    console = code.InteractiveConsole(locals=locals())
    console.interact()
        
        
        