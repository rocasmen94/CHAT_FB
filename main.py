from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

from CHAT_Algorithm import CHAT
from utils import *
from Visualizacion_Datos import visualizar_datos

# reb_error = es un array con todos los resubstitution errors de cada punto probado
# combinations = son las combinaciones de todos lo puntos

# PARAMETROS INICIALES
ruta = r'C:\Users\CASTA\OneDrive\3° Semestre\Memorias Asociativas\Modificaciones CHAT\Datasets\ecoli-0-6-7_vs_5.csv'
initial_step = 0.25
explore_outside = 0
variation_step = 2
search_depth = 3

CHAT = CHAT()
conjuntoX, conjuntoY = read(ruta)
conjuntoX_norm, norm = normalize(conjuntoX, axis=0, norm='max', return_norm=True)
best_point, combinations, reb_error = best_point_fuerza_bruta(conjuntoX_norm, conjuntoY, initial_step, variation_step,
                                                              search_depth, explore_outside, CHAT)
# visualizar_datos(combinations, reb_error)
# plt.show()

# para CHAT normal
# predict_list = LOOCV_CHAT(conjuntoX, conjuntoY, CHAT.vector_medio(conjuntoX), CHAT)
# best_point[1] = best_point[1]*norm #se regresa el mejor punto sin normalizar
predict_list = LOOCV_CHAT(conjuntoX, conjuntoY, best_point[1], CHAT)
y_true = np.argmax(conjuntoY, 1)
y_pred = np.argmax(np.array(predict_list), 1)

print(balanced_accuracy_score(y_true, y_pred))
print(f1_score(y_true, y_pred, average='micro'))
print(confusion_matrix(y_true, y_pred))

