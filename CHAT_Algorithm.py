import numpy as np


class CHAT:

    def vector_medio(self, conjuntoX):
        p, n = conjuntoX.shape
        conjuntoX_media = np.zeros((1, n))

        for p in range(0, len(conjuntoX)):
            conjuntoX_media += conjuntoX[p]
        conjuntoX_media = conjuntoX_media / len(conjuntoX)

        # conjuntoX_tras = conjuntoX - conjuntoX_media
        return conjuntoX_media

    def aprendizaje(self, conjuntoX, conjuntoY):
        m = len(conjuntoX[0])
        n = len(conjuntoY[0])
        MA = np.zeros((n, m))

        for p in range(0, len(conjuntoY)):
            patronX = np.transpose(conjuntoX[p])
            patronY = conjuntoY[p]
            for i in range(0, n):
                MA[i] += np.dot(patronY[i], patronX)
        return MA

    def recuperar(self, memoria, patron):
        patron_recuperado = np.dot(memoria, patron.T)
        # print(patron_recuperado)
        valorMax = np.max(patron_recuperado)
        vectorBool = [r == valorMax for r in patron_recuperado]
        vectorBool = [int(p) for p in vectorBool]
        return vectorBool

    # Se realiza la transición buscando entre los rasgos el valor min y max
    def Do_VectorTras(self, conjuntoX, step, explore_outside):
        _, n = conjuntoX.shape
        conjuntoX_media = np.array([[]] * n + [[1]])[:-1]

        #   Se obtiene el rango min y max para cada valor
        minXrasgo = np.min(conjuntoX, axis=0)
        maxXrasgo = np.max(conjuntoX, axis=0)

        for rasgo in range(0, n):
            if minXrasgo[rasgo] == maxXrasgo[rasgo]:
                conjuntoX_media[rasgo] = [minXrasgo[rasgo]]
            else:
                #       Se generan los datos entre el valor min y max con el step establecido
                conjuntoX_media[rasgo] = np.arange(start=minXrasgo[rasgo]-explore_outside,
                                                   stop=maxXrasgo[rasgo]+explore_outside,
                                                   step=step)
                # conjuntoX_media[rasgo] = np.linspace(start=minXrasgo[rasgo],stop=maxXrasgo[rasgo],num=10)

        return conjuntoX_media

    # Similar a la función Do_VectorTras pero se pasa un punto y una desviación estandar
    # para que apartir de dicho punto se genere un nuevo campo de busqueda

    def Do_VectorTras_Explora(self, bestPoint, step):
        n = len(bestPoint)
        conjuntoX_media = np.array([[]] * n + [[1]])[:-1]

        for rasgo in range(0, n):
            conjuntoX_media[rasgo] = np.arange(start=bestPoint[rasgo] - step,
                                               stop=bestPoint[rasgo] + step + 0.1,
                                               step=step)
            # conjuntoX_media[rasgo] = np.linspace(start=minXrasgo[rasgo],stop=maxXrasgo[rasgo],num=10)

        return conjuntoX_media

    def get_meshgrid(self, points, step, CHAT, explore_outside, explorar):
        # Se obtiene el vector de puntos entre el min y max
        if not explorar:
            puntos_a_evaluar = CHAT.Do_VectorTras(points, step, explore_outside)
        else:
            puntos_a_evaluar = CHAT.Do_VectorTras_Explora(points, step)
        # Se realiza la combinatoria para todos los puntos encontrados de acuerdo con la cantidad de rasgos
        mesh = np.array(np.meshgrid(*puntos_a_evaluar))
        # combinations es un vector con el conjunto de puntos por rasgos a evaluar
        combinations = mesh.T.reshape(-1, len(puntos_a_evaluar))
        return combinations

    def ECT_resub_error(self, conjuntoX, conjuntoY, combinations, CHAT):

        # Apartir de este punto se realiza el algoritmo CHAT para cada una de las combinaciones
        best_resubstitution_error = 0
        # best_results = np.array(([], []))
        reb_error = []
        for i in range(0, len(combinations)):
            conjuntoX_tras = conjuntoX - combinations[i]
            Memoria = CHAT.aprendizaje(conjuntoX_tras, conjuntoY)

            # resubstitution_error = 0
            aciertos = 0
            # Se realiza resubstitution error para comprobar en cada punto el desempeño obtenido
            for r, p in enumerate(conjuntoX_tras):
                patron_recup = CHAT.recuperar(Memoria, p)
                if (conjuntoY[r] == patron_recup).all():
                    aciertos += 1
            resubstitution_error = aciertos / len(conjuntoX_tras)
            reb_error.append(resubstitution_error)
            # Se evalua si el punto es mejor que otro punto probado. Si lo es, se guarda y se imprime el punto
            if resubstitution_error > best_resubstitution_error:
                best_resubstitution_error = resubstitution_error
                best_results = np.array((best_resubstitution_error, combinations[i]))
                print("Resubstitution Error with the point: ", best_results)
        return reb_error, best_results





