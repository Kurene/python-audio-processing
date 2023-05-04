import os
import numpy as np
import threading
from numba import jit
import sounddevice as sd
import soundfile as sf


class Player(threading.Thread):
    def __init__(self, filepath, effector=None, blocksize=1024, n_buf=10):
        super(Player, self).__init__()
        
        self.blocksize  = blocksize
        self.n_buf = n_buf
        self.bufsize = self.blocksize*self.n_buf

        # エフェクト用の関数
        self.effector = effector 

        # ファイル読込
        self.load(filepath)

        
    def load(self, filepath):
        # 音声ファイルを読み込み
        sig, sr = sf.read(filepath, always_2d=True)
        
        self.sig = sig
        self.sr  = sr
        self.n_samples  = sig.shape[0]
        self.n_channels = sig.shape[1]

        self.x_tmp = np.zeros((self.blocksize, self.n_channels))
        self.y_tmp = np.zeros((self.blocksize, self.n_channels))
        
        # バッファ関連
        self.x_buf = np.zeros((self.bufsize, self.n_channels))
        self.y_buf = np.zeros((self.bufsize, self.n_channels))
        
        self.sig_save = np.zeros((self.n_samples + self.blocksize, self.n_channels))
        self.savefilepath = os.path.splitext(filepath)[0] + "_out.wav"
        

    def save_buf(self):
        slc_dst = slice(0, self.blocksize*(self.n_buf-1))
        slc_src = slice(self.blocksize, self.blocksize*self.n_buf)
        self.x_buf[slc_dst, :] = self.x_buf[slc_src, :]
        self.y_buf[slc_dst, :] = self.y_buf[slc_src, :]
        
        slc = slice(self.blocksize*(self.n_buf-1), self.blocksize*self.n_buf)
        self.x_buf[slc, :] = self.x_tmp[:, :]
        self.y_buf[slc, :] = self.y_tmp[:, :]
        
    def callback(self, indata, outdata, frames, time, status):
        chunksize = min(self.n_samples - self.current_frame, frames)
        
        # チャンネルごとの信号処理
        self.x_tmp[:] *= 0.0
        self.y_tmp[:] *= 0.0
        self.x_tmp[0:chunksize] = self.sig[self.current_frame:self.current_frame + chunksize]
        for k in range(self.n_channels): 
            if self.effector is None:
                outdata[:, k] = self.x_tmp # バイパス処理
            else: # エフェクト処理
                self.effector(
                    self.sr,
                    self.blocksize,
                    self.bufsize,
                    self.x_tmp[:,k],
                    self.y_tmp[:,k],
                    self.x_buf[:,k],
                    self.y_buf[:,k]
                )

            outdata[0:chunksize, k] = self.y_tmp[0:chunksize,k]
            
        self.sig_save[self.current_frame:self.current_frame + chunksize] = outdata[0:chunksize]
        
        # バッファに現在のフレームの信号を保存
        self.save_buf()
        
        if chunksize < frames:
            raise sd.CallbackStop()
        
        self.current_frame += chunksize
    
    def stop(self):
        self.event.set()
    
    def run(self):
        self.current_frame = 0
        self.sig_save[:] *= 0.0
        self.event = threading.Event()
        
        with sd.Stream(
            samplerate=self.sr, 
            blocksize=self.blocksize,
            channels=self.n_channels,
            callback=self.callback, 
            finished_callback=self.event.set
        ):
            self.event.wait()
    
    def save(self):
        sf.write(self.savefilepath, self.sig_save, self.sr)
#=============================================================================


if __name__ == "__main__":
    # ここから実装
    # 単チャネルの信号を処理する関数を実装
    """
    @jit
    def effector(blocksize, x, y, x_buf, y_buf):
        pass
    """

    @jit
    def delay(sr, blocksize, bufsize, x, y, x_buf, y_buf):
        fo    = 10
        tau   = sr // fo
        alpha = 0.6
        for k in range(blocksize):
            if k >= tau:
                y[k] = x[k] - alpha * y[k - tau]
            else:
                y[k] = x[k] - alpha * y_buf[bufsize-1-tau+ k]


    filepath = "./audio.wav"
    player = Player(filepath, effector=delay)
    player.start()
    # player.stop()
    # player.save()

    import code
    console = code.InteractiveConsole(locals=locals())
    console.interact()