#Importaciones:
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter.filedialog import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import os.path

#Declaración de la ventana tkinter, tamaño y nombre
root = Tk()
root.geometry("600x580")
root.title("Red Neuronal para predecir Cáncer de seno")
root.config(bg="yellow")

#Variables de tkinter:
filen = StringVar(value="") #Para guardar la ruta del archivo
trainp = IntVar() #Para guardar el porcentaje de datos de entrenamiento
nnodos = StringVar() #Cantidad de nodos por capa
maxit = IntVar() #Numero maximo de iteraciones
activ = StringVar(value="relu") #Para especificar función de activación
solv = StringVar(value="adam") #Para especificar el metodo para la optimización de pesos.
preciB = IntVar() #precisión para Benignas
preciM = IntVar() #precisión para Malignas
#Matriz de confusion:
cantBc = IntVar() 
cantBi = IntVar()
cantMc = IntVar()
cantMi = IntVar()

#Funciones:

#Función para saber si un numero es un entero:
def es_entero(variable):
   try:
      int(variable)
      return True
   except:
      return False

#Función para obtener la ruta de un archivo
def abrir():
   ruta=askopenfilename()
   filen.set(ruta)

#Funcion que se llama cuando se apreta el boton para entrenar la red neuronal y probarla
def entrenar_probar():
   extension = os.path.splitext(filen.get())[1] #Se obtiene la extensión del archivo
   if (extension == ".csv"): #Se comprueba si la extensión es csv
      mamd = pd.read_csv(filen.get())
   else: #Se lanza mensaje de error sino es csv
      messagebox.showwarning(message="La extensión del archivo seleccionado debe ser igual a '.csv'.", title="Extensión incorrecta")
      return
   #Se limpian y corrijen los datos:
   mamd = mamd.drop(mamd[mamd['age']==-100000].index)
   mamd = mamd.drop(mamd[mamd['shape']==-100000].index)
   mamd = mamd.drop(mamd[mamd['margin']==-100000].index)
   mamd = mamd.drop(mamd[mamd['density']==-100000].index)
   mamd = mamd.drop(mamd[mamd['severity']==-100000].index)
   if (trainp.get() < 100 and trainp.get() > 2): #Se comprueba si el porcentaje de entrenamiento es adecuado
      tsize = 1 - (trainp.get()/100)
   else: #Se lanza mensaje de error sino es así
      messagebox.showwarning(message="El porcentaje de datos de entrenamiento debe ser menor a 100% y mayor a 2%.", title="Porcentaje incorrecto")
      return
   training_set, validation_set = train_test_split(mamd, test_size = tsize, random_state = 21) #Se divide la data de entrenamieno y la de prueba
   X_train = training_set.iloc[:,0:-1].values
   Y_train = training_set.iloc[:,-1].values
   X_val = validation_set.iloc[:,0:-1].values
   y_val = validation_set.iloc[:,-1].values
   stnods = nnodos.get()
   hlsnodos = []
   for x in stnods.split(','): #Se comprueba que las cantidades de nodos por capa sean enteros
      if es_entero(x):
         hlsnodos.append(int(x))
      else: #Se lanza mensaje de error sino es así
         messagebox.showwarning(message="La cantidad de nodos por cada capa debe ser un numero entero.", title="Información incorrecta")
         return
   if (maxit.get() < 1000000 and maxit.get() > 10): #Se comprueba si el numero maximo de iteraciones es adecuado
      #Se crea y establece la red neuronal (MLP):
      classifier = MLPClassifier(hidden_layer_sizes=hlsnodos, max_iter=maxit.get(),activation = activ.get(),solver=solv.get(),random_state=1)
   else: #Se lanza mensaje de error sino es así
      messagebox.showwarning(message="El numero maximo de iteraciones debe ser menor a 1000000 y mayor a 10", title="Numero incorrecto")
      return
   classifier.fit(X_train, Y_train) #Se entrena el MLP
   y_pred = classifier.predict(X_val) #Se prueba el MLP
   cm = confusion_matrix(y_pred, y_val) #Se crea la matriz de confusión
   preciB.set((cm[0][0]/(cm[0][0]+cm[1][0]))*100) #Se actualizan datos para ser mostrados en pantalla
   preciM.set((cm[1][1]/(cm[0][1]+cm[1][1]))*100)
   cantBc.set(cm[0][0])
   cantBi.set(cm[1][0])
   cantMc.set(cm[1][1])
   cantMi.set(cm[0][1])

#Mensaje en Tkinter de selección de archivos, boton y caja de texto respectiva
mensArchivo = Label(root, text="Selecciona el archivo con los datos", background="orange")
mensArchivo.place(x=30, y=20)
entryArchivo = Entry(root, textvariable=filen, width=70)
entryArchivo.place(x=30, y=50)
botonAbrirArchivo =Button(root,text="Seleccionar archivo", command=abrir)
botonAbrirArchivo.place(x=470, y=48)

#Mensajes en Tkinter para ingreso de datos afines al entrenamiento y prueba de la red neuronal, y cajas de texto respectivas
mensTrain = Label(root, text="Ingrese el porcentaje de datos de entrenamiento (%): ", background="pink")
mensTrain.place(x=30, y=100)
entryTrain = Entry(root, textvariable=trainp, width=10)
entryTrain.place(x=500, y=100)
mensNnodos = Label(root, text="Ingrese el numero de nodos por capa ordenadamente separados por comas: ", background="orange")
mensNnodos.place(x=30, y=130)
entryNnodos = Entry(root, textvariable=nnodos, width=60)
entryNnodos.place(x=33, y=155)
mensMaxit = Label(root, text="Ingrese el numero maximo de iteraciones: ", background="pink")
mensMaxit.place(x=30, y=180)
entryMaxit = Entry(root, textvariable=maxit, width=10)
entryMaxit.place(x=500, y=180)

#Cajas de selección y Labels, para seleccionar función de activación y metodo de optimización
mensActive = Label(root, text="Función de activación: ", background="orange")
mensActive.place(x=30, y=220)
combActive = ttk.Combobox(root, values=["relu", "logistic", "tanh", "identity"], state='readonly', textvariable=activ)
combActive.place(x=30, y=250)
mensSolver = Label(root, text="Optimización de pesos: ", background="pink")
mensSolver.place(x=200, y=220)
combSolver = ttk.Combobox(root, values=["adam", "lbfgs", "sgd"], state='readonly', textvariable=solv)
combSolver.place(x=200, y=250)

#Boton que llama a la función entrenar_probar
botonEntrenar =Button(root,text="Entrenar y Probar", command=entrenar_probar)
botonEntrenar.place(x=30, y=300)

#Labels y cajas de texto que dan información sobre la precisión de diagnostico de la red neuronal
mporceBeni = Label(root, text="Porcentaje de precisión para diagnosticar un tumor benigno (%): ", background="orange")
mporceBeni.place(x=30, y=360)
entrympBeni = Entry(root, textvariable=preciB, width=10)
entrympBeni.place(x=400, y=360)
mporceMali = Label(root, text="Porcentaje de precisión para diagnosticar un tumor maligno (%): ", background="pink")
mporceMali.place(x=30, y=390)
entrympMali = Entry(root, textvariable=preciM, width=10)
entrympMali.place(x=400, y=390)

#Labels y cajas de texto que dan información sobre la cantidad de tumores benignos diagnosticados correcta e incorrectamente
mcBenic = Label(root, text="Cantidad de tumores benignos diagnosticados correctamente: ", background="orange")
mcBenic.place(x=30, y=430)
entryBenic = Entry(root, textvariable=cantBc, width=10)
entryBenic.place(x=400, y=430)
mcBenii = Label(root, text="Cantidad de tumores benignos diagnosticados como malignos: ", background="pink")
mcBenii.place(x=30, y=460)
entryBenii = Entry(root, textvariable=cantBi, width=10)
entryBenii.place(x=400, y=460)

#Labels y cajas de texto que dan información sobre la cantidad de tumores malignos diagnosticados correcta e incorrectamente
mcMalic = Label(root, text="Cantidad de tumores malignos diagnosticados correctamente: ", background="orange")
mcMalic.place(x=30, y=500)
entryMalic = Entry(root, textvariable=cantMc, width=10)
entryMalic.place(x=400, y=500)
mcMalii = Label(root, text="Cantidad de tumores malignos diagnosticados como benignos: ", background="pink")
mcMalii.place(x=30, y=530)
entryMalii = Entry(root, textvariable=cantMi, width=10)
entryMalii.place(x=400, y=530)

#Loop de Tkinter
root.mainloop()
