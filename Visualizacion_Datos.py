import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import copy
import seaborn as sns
# %matplotlib inline

def visualizar_datos(combinations, reb_error):

    # Se realiza un diccionario con cada uno de los puntos probados
    # para poder manipular los datos con pandas

    # data = {'Sepal Length': combinations[:][:,0], 'Sepal Width': combinations[:][:,1],
    #         'Petal Length': combinations[:][:,2], 'Petal Width': combinations[:][:,3],
    #         'Reb_Error': reb_error[:]}

    data = {'Sepal Length': combinations[:][:,0], 'Sepal Width': combinations[:][:,1],
            'Reb_Error': reb_error[:]}
    dataframe = pd.DataFrame(data)
    stdeviation = dataframe.std(axis=0)
    # dataframe.head()

    # LA SIGUIENTE LÍNEA DE CÓDIGO ES PARA GUARDAR EL ARCHIVO EN EXCEL
    # dataframe.to_excel("CHAT_FB_bezdekIris_2.xlsx",
    #              sheet_name='Chat_Fuerza_bruta')

    # Se realiza un mapheat para cada una de las combinaciones de rasgos posibles

    heatmap1_data = pd.pivot_table(round(dataframe,2), values='Reb_Error',
                         index=['Sepal Width'],
                         columns='Sepal Length')


    # Imprime los primeros tres pares de rasgos
    f, ax = plt.subplots(1,2,figsize=(25, 6))
    hm1 = sns.heatmap(heatmap1_data, linewidths=.01, ax=ax[0])
    return f
    # t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=8)