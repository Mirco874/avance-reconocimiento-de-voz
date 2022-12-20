import sys, threading
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt, QTimer, QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QThread, QObject, pyqtSignal as Signal, pyqtSlot as Slot
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


#se encuentra a la escucha
listen=False
# palabras que puede detectar 
labels=["Aula","Borrar","Cero","Cinco","Cuatro","Dos","Limpiar","Nueve","Ocho","Quebeq","Seis","Siete","Tres","Uno"]
# lista de numeros
numeros=["Cero","Uno","Dos","Tres","Cuatro","Cinco","Seis","Siete","Ocho","Nueve"]
#modelo entrenado
trained_model=load_model('my_sr_model.h5')
#comando actual
command=[]

#funcion para actualizar el comando segun la palabra percibida.
def updateCommand(word):
    global command
    global numeros

    if(len(command)==0 and word=="Quebeq" ):
        command.append(word)

    if(len(command)==1 and word=="Limpiar" ):
        command.append(word)

    if(len(command)==2 and word=="Aula" ):
        command.append(word)

    if(len(command)==3):
        command.append("")

    if(len(command)==4 and word in numeros):
        if(word=="Uno"):
            command[3]=command[3]+"1"
        elif(word=="Dos"):
            command[3]=command[3]+"2"
        elif(word=="Tres"):
            command[3]=command[3]+"3"
        elif(word=="Cuatro"):
            command[3]=command[3]+"4"
        elif(word=="Cinco"):
            command[3]=command[3]+"5"
        elif(word=="Seis"):
            command[3]=command[3]+"6"
        elif(word=="Siete"):
            command[3]=command[3]+"7"
        elif(word=="Ocho"):
            command[3]=command[3]+"8"            
        elif(word=="Nueve"):
            command[3]=command[3]+"9"      

    if(len(command)==4 and word=="Borrar"):
        command[3]=command[3][0:-1] 
    print(command)

# obtener los mfcc de la señal de la voz ingresada por el microfono
def extract_features_voice(voiceSignal,sr):
    mfcc = librosa.feature.mfcc(y=voiceSignal, sr=sr, n_mfcc=40)
    mfcc /=np.array(np.absolute(mfcc))
    return np.ndarray.flatten(mfcc)[:25000] 

# funcion para empezar a escuchar las palabras
def press_on(key):
    global listen
    if key==Key.space:
        listen=True

# funcion para terminar de escuchar y procesar la entrada
def press_off(key):
    global plotdata
    global listen
    if key==Key.space:
        listen=False

    print("=======PLOT DATA========")
    print(plotdata.flatten().shape)
    print("voice signal shape: "+str(plotdata.flatten().shape))

    # extraccion de los mfcc de la voz
    voice_MFCC=extract_features_voice(plotdata.flatten(),22050)
    print( "MFCC shape: " + str(voice_MFCC.shape))
    # remplazar los nan por 0
    where_are_NaNs = np.isnan(voice_MFCC)
    voice_MFCC[where_are_NaNs] = 0
    
    # generar una lista de la señal de voz ingresada
    voice_MFCC_list=voice_MFCC.tolist()
    # predecir la palabra
    results =trained_model.predict([voice_MFCC_list])[0]
    print("did you say: " + str(labels[results.argmax()]) +" ?")

    #actualizar el comando segun la palabra predicha
    updateCommand(labels[results.argmax()])

    print("========================")
    
# funcion para convertir enteros a cadenas
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
mapping = [c - 1 for c in args.channels]  
q = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata[::args.downsample, mapping])

# actualizacion de la grafica, segun la entrada de voz
def update_plot(frame):
    global listen
    global plotdata
# debe dibujar la señal de voz si se encuentra escuchando     
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
    # el ancho de la figura sera igual que el sr
    length = 22580 #int(args.window * args.samplerate / (1000 * args.downsample))
    #c creacion del lienzo
    plotdata = np.zeros((length, len(args.channels)))
    fig, ax = plt.subplots()
    #dibujo de la onda de la señal
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

    # inicio de la animacion
    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)



except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))


class Piso:
	def __init__(self):
		self.aulas = []

	def nuevaAula(self, aula):
		self.aulas.append(aula)

	def getAulas(self):
		return self.aulas

class Aula:
	def __init__(self, nombre, x, y, d):
		self.nombre = nombre
		self.estado = "sucio"
		self.x = x
		self.y = y
		self.d = d

	def getDat(self):
		return self.x, self.y, self.d, self.estado, self.nombre

	def limpiar(self):
		self.estado = "limpio"

class Edificio(QMainWindow):

	work_requested = Signal(str)

	def __init__(self):
		super().__init__()

		self.piso = 0
		self.x = 490
		self.y = 560

		self.pisos = []
		piso1 = Piso()
		piso1.nuevaAula(Aula("0a", 0, 200, 200))
		piso1.nuevaAula(Aula("0b", 200, 0, 200))
		piso1.nuevaAula(Aula("0c", 600, 0, 200))
		piso1.nuevaAula(Aula("0d", 800, 200, 200))
		piso2 = Piso()
		piso2.nuevaAula(Aula("1a", 0, 200, 200))
		piso2.nuevaAula(Aula("1b", 200, 0, 200))
		piso2.nuevaAula(Aula("1c", 400, 0, 200))
		piso2.nuevaAula(Aula("1d", 600, 0, 200))
		piso2.nuevaAula(Aula("1e", 800, 200, 200))
		piso3 = Piso()
		piso3.nuevaAula(Aula("2a", 0, 200, 200))
		piso3.nuevaAula(Aula("2b", 400, 0, 200))
		piso3.nuevaAula(Aula("2c", 800, 200, 200))
		piso4 = Piso()
		piso4.nuevaAula(Aula("3a", 0, 200, 200))
		piso4.nuevaAula(Aula("3b", 200, 0, 200))
		piso4.nuevaAula(Aula("3c", 400, 0, 200))
		piso4.nuevaAula(Aula("3d", 600, 0, 200))
		piso4.nuevaAula(Aula("3e", 800, 200, 200))

		self.pisos.append(piso1)
		self.pisos.append(piso2)
		self.pisos.append(piso3)
		self.pisos.append(piso4)

		self.label = QtWidgets.QLabel()
		canvas = QtGui.QPixmap(1000, 600)
		self.label.setPixmap(canvas)
		self.setCentralWidget(self.label)

		self.dibujarPiso()

		self.worker = Lector()
		self.worker_thread = QThread()

		self.worker.accion.connect(self.actualizar)

		self.work_requested.connect(self.worker.leer)

		self.worker.moveToThread(self.worker_thread)

		self.worker_thread.start()

		self.work_requested.emit("")

	def dibujarPiso(self):
		painter = QtGui.QPainter(self.label.pixmap())
		pen = QtGui.QPen()
		pen.setWidth(40)
		pen.setColor(QtGui.QColor('white'))
		painter.setPen(pen)
		painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
		painter.drawRect(0, 0, 1000, 600)

		aulas = self.pisos[self.piso].getAulas()

		font = QtGui.QFont()
		font.setPointSize(45)
		painter.setFont(font)
		pen.setColor(QtGui.QColor('black'))
		painter.setPen(pen)
		painter.drawText(0, 350, 1000, 250, Qt.AlignHCenter, str(self.piso))

		

		painter.end()

		for aula in aulas:
			self.dibujarAula(aula)

		painter = QtGui.QPainter(self.label.pixmap())
		pen.setWidth(20)
		painter.setPen(pen)
		painter.drawRect(self.x, self.y, 20, 20)

	def dibujarAula(self, aula):
		self.update()
		x, y, d, estado, nombre = aula.getDat()
		painter = QtGui.QPainter(self.label.pixmap())
		pen = QtGui.QPen()
		pen.setColor(QtGui.QColor('black'))
		if estado == "sucio":
			painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
		else:
			painter.setBrush(QBrush(Qt.green, Qt.SolidPattern))
		pen.setWidth(3)
		painter.setPen(pen)
		painter.drawRect(x, y, d, d)
		font = QtGui.QFont()
		font.setPointSize(15)
		painter.setFont(font)
		painter.drawText(x, y+(d//2), d, d, Qt.AlignHCenter, nombre)
		painter.end()

	def existeEnPiso(self, nom):
		res = False
		for aula in self.pisos[self.piso].aulas:
			if aula.nombre == nom:
				res = True
		return res

	def limpiar(self, nom):
		res = None
		for aula in self.pisos[self.piso].aulas:
			if aula.nombre == nom:
				res = aula
		res.limpiar()
		x1, y1, d, estado, nombre = res.getDat()
		self.x = x1 + d//2 - 10
		self.y = y1 + d//2 - 10

	def resetRobot(self):
		self.x = 490
		self.y = 560

	def actualizar(self, action):
		if action == "subir":
			if self.piso < len(self.pisos) -1:
				self.piso = self.piso + 1
				self.resetRobot()
				self.dibujarPiso()
			else:
				print("mas arriba y te vas con diosito")
		elif action == "bajar":
			if self.piso >= 1:
				self.piso = self.piso - 1
				self.resetRobot()
				self.dibujarPiso()
			else:
				print("ya es toda wey, bajale para arriba")
		elif action[0:8] == "limpiar ":
			nom = action[8:-1]+action[-1]
			if self.existeEnPiso(nom):
				self.limpiar(nom)
				self.dibujarPiso()
			else:
				print("echale mas ganas, ese cuarto no existe en este piso")
		else:
			print("no se que dice pero miente")

class Lector(QObject):
	accion = Signal(str)

	@Slot()
	def leer(self):
		while True:
			x = input()
			self.accion.emit(x)

app = QtWidgets.QApplication(sys.argv)

window = Edificio()

# creacion del hilo para mostrar la grafica y escuchar los eventos de los botones
with stream,Listener(on_press=press_on, on_release= press_off) as listener:
    plt.show()  ####
    window.show()
    app.exec_()
    listener.join()
