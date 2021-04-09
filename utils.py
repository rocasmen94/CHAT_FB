import numpy as np
import copy
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import normalize

def read(file):
    data = np.genfromtxt(file, delimiter=',')
    # n = int(data[0,0])

    X = data[1:, :-1]
    p, n = X.shape
    y = data[1:, n:]

    num_classes = len(np.unique(y))
    one_hot = np.identity(num_classes)
    y_one_hot = np.zeros((p, num_classes))

    for i, c in enumerate(y):
        y_one_hot[i] = one_hot[int(c)]

    return X, y_one_hot


def best_point_fuerza_bruta(conjuntoX, conjuntoY, initial_step, variation_step, search_depth, explore_outside, CHAT):
    point_to_test = copy.copy(conjuntoX)
    explorar = False
    for i in range(search_depth):
        combinations = CHAT.get_meshgrid(point_to_test, initial_step, CHAT, explore_outside, explorar)
        # best_point es un arreglo de 1x2 donde el primer valor es el resb error
        # y el segundo es el punto para dicho valor
        reb_error, best_point = CHAT.ECT_resub_error(conjuntoX, conjuntoY, combinations, CHAT)
        point_to_test = best_point[1]
        initial_step = initial_step / variation_step
        explorar = True
    return best_point, combinations, reb_error


def LOOCV_CHAT(conjuntoX, conjuntoY, best_point, CHAT):
    # Método de validación LOOCV
    LOOCV = LeaveOneOut()
    LOOCV.get_n_splits(conjuntoX)

    # Se coloca el mejor de los puntos evaluados
    translation_point = best_point  # el mejor punto encontrado con fuerza bruta

    # Apartir de este punto se realiza el algoritmo CHAT para cada una de las combinaciones
    aciertos = 0
    incorrectos = 0
    lista_recuperados = list()
    for train_index, test_index in LOOCV.split(conjuntoX):
        X_train, X_test = conjuntoX[train_index], conjuntoX[test_index]
        y_train, y_test = conjuntoY[train_index], conjuntoY[test_index]

        X_train_norm, _ = normalize(X_train, axis=0, norm='max', return_norm=True)
        X_test_norm, _ = normalize(X_test, axis=0, norm='max', return_norm=True)
        # FASE DE ENTRENAMIENTO
        conjuntoX_tras = X_train_norm - translation_point
        # print(conjuntoX_tras)
        Memoria = CHAT.aprendizaje(conjuntoX_tras, y_train)

        # FASE DE PRUEBA
        conjuntoX_tras_test = X_test_norm - translation_point
        patron_recup = CHAT.recuperar(Memoria, conjuntoX_tras_test)
        lista_recuperados.append(patron_recup)

        if (y_test == patron_recup).all():
            aciertos += 1
        else:
            incorrectos += 1

    accuracy = aciertos / (aciertos + incorrectos)
    print(f"Patrones correctamente clasificados: {aciertos} \nPatrones mal clasificados: {incorrectos} ")
    print(f"Accuracy general del algoritmo: {round(accuracy * 100, 4)}%")
    return lista_recuperados
