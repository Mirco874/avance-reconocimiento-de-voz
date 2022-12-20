import argparse
from platform import platform
import queue
import sys
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
np.seterr(divide='ignore', invalid='ignore') ###

import sounddevice as sd
from pynput.keyboard import *
from keras.models import load_model

import librosa
import librosa.display 
import librosa.feature 

listen=False
trained_model=load_model('my_sr_model.h5')

def extract_features_voice(voiceSignal,sr):
  mfcc = librosa.feature.mfcc(y=voiceSignal, sr=sr, n_mfcc=40)
  mfcc /=np.array(np.absolute(mfcc)) # dividir entre una matriz generada a partir de valores absolutos
  return np.ndarray.flatten(mfcc)[:25000] #eliminar multidimensionalidad


def press_on(key):
    global listen
    if key==Key.space:
        listen=True

def press_off(key):
    global plotdata
    global listen
    if key==Key.space:
        listen=False

    print("=======PLOT DATA========")
    print(plotdata.flatten().shape)
    print("voice signal shape: "+str(plotdata.shape))

    voice_MFCC=extract_features_voice(plotdata.flatten(),22050)
    print( "MFCC shape: " + str(voice_MFCC.shape))

    where_are_NaNs = np.isnan(voice_MFCC)
    voice_MFCC[where_are_NaNs] = 0
    
    voice_MFCC_list=voice_MFCC.tolist()
    print(voice_MFCC_list)
    print("========================")
    
    
    # results =trained_model.predict([voice_MFCC_list])[0]

    # labels=["Aula","Borrar","Cero","Cinco","Cuatro","Dos","Limpiar","Nueve","Ocho","Seis","Siete","Tres","Uno"]
    # index_of_max=results.argmax()
    # print("you say: " + str(labels[index_of_max]) +" ?")

def int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-l', '--list-devices', action='store_true',help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()

if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])

parser.add_argument('channels', type=int, default=[1], nargs='*', metavar='CHANNEL',help='input channels to plot (default: the first)')
parser.add_argument('-d', '--device', type=int_or_str,help='input device (numeric ID or substring)')
parser.add_argument('-w', '--window', type=float, default=200, metavar='DURATION',help='visible time slot (default: %(default)s ms)')
parser.add_argument('-i', '--interval', type=float, default=30,help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument('-b', '--blocksize', type=int, help='block size (in samples)')    ##900
parser.add_argument('-r', '--samplerate', type=float, help='sampling rate of audio device')   ## 41000
parser.add_argument('-n', '--downsample', type=int, default=10, metavar='N',help='display every Nth sample (default: %(default)s)')

args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
q = queue.Queue()


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata[::args.downsample, mapping])


def update_plot(frame):
    global listen
    global plotdata
    
    while listen:
        try:
            data = q.get_nowait()
        except queue.Empty:            
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines

try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    length = 22580 #int(args.window * args.samplerate / (1000 * args.downsample))
    print("------data-----")
    print(args.window)
    print(args.samplerate)
    print(args.downsample)
    print(length)
    print("---------------")

    plotdata = np.zeros((length, len(args.channels)))
 
    fig, ax = plt.subplots()

    lines = ax.plot(plotdata)

    if len(args.channels) > 1:
        ax.legend([f'channel {c}' for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
    ax.axis((0, len(plotdata), -0.10, 0.10))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(bottom=False, top=False, labelbottom=False,
                   right=False, left=False, labelleft=False)
    fig.tight_layout(pad=0)

    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback)

    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)
    
    with stream,Listener(on_press=press_on, on_release= press_off) as listener:
        plt.show()
        listener.join()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))