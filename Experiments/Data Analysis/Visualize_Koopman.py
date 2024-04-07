import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from copy import deepcopy


DF = pd.read_csv("Koopman Data\\Koopman_Testing_06_04_2024_10_04_48.csv")
corrected_time = DF.loc[:,["time_delta_s","sequence_num"]].groupby(['sequence_num']).transform(lambda x: (x-x.iloc[0]))
DF["corrected_time_s"] = corrected_time



## ---- Plot the states (jaw pressures) and controls (x,y and the commanded grasper pressure) as functions of time, colored by the sequence number, time not corrected
DF_melt0 = DF.melt(id_vars=['time_delta_s','sequence_num'],
                        value_vars=['P_jaw1_psi', 'P_jaw2_psi','P_jaw3_psi','x_mm','y_mm','P_closure_psi',
                                    'commanded_x_mm',' commanded_y_mm','commanded_closure_pressure_psi'],var_name = 'Variable',value_name ='Values')

fig0 = px.line(DF_melt0,x = 'time_delta_s',y = 'Values', color = "sequence_num", symbol = "sequence_num",facet_row = 'Variable')
fig0.update_yaxes(matches = None, showticklabels=True)
fig0.show()



## ---- Plot the jaw pressures in jaw 1 stacked ---- ##
fig1 = px.line(DF,x = 'corrected_time_s',y = 'P_jaw1_psi', color = "sequence_num")
fig1.show()


## ---- Plot the jaw pressures as functions of time, colored by the sequence number, time not corrected
DF_melt2 = DF.melt(id_vars=['time_delta_s','sequence_num'],
                        value_vars=['P_jaw1_psi', 'P_jaw2_psi','P_jaw3_psi'],var_name = 'Variable',value_name ='Values')

fig2 = px.line(DF_melt2,x = 'time_delta_s',y = 'Values', color = "sequence_num", symbol = "sequence_num",facet_row = 'Variable')
fig2.show()

## ---- Look at max jaw pressure as a function of the commanded pressure for each jaw ---- ##
#Note: results will be confusing because if it is a step down (i.e. prev pressure is higher than current commanded pressure), there will be a negative correlation. Need to add a secondary variable to color by that encodes step up or step down
DF[["Max_jaw1_Pressure","Max_jaw2_Pressure","Max_jaw3_Pressure"]] = DF.loc[:,["P_jaw1_psi","P_jaw2_psi","P_jaw3_psi","sequence_num"]].groupby(['sequence_num']).transform(lambda x: max(x))
DF_melt3 = DF.melt(id_vars=['time_delta_s','sequence_num','commanded_closure_pressure_psi'],
                        value_vars=["Max_jaw1_Pressure","Max_jaw2_Pressure","Max_jaw3_Pressure"],var_name = 'Variable',value_name ='Values')

fig3 = px.line(DF_melt3,x = 'commanded_closure_pressure_psi',y = 'Values', symbol = "sequence_num",facet_row = 'Variable')
fig3.show()

## ---- Look at average jaw pressure as a function of the commanded pressure for each jaw in the last second, i.e ~20 samples ---- ##
DF[["Mean_jaw1_Pressure","Mean_jaw2_Pressure","Mean_jaw3_Pressure"]] = DF.loc[:,["P_jaw1_psi","P_jaw2_psi","P_jaw3_psi","sequence_num"]].groupby(['sequence_num']).transform(lambda x: np.mean(x.iloc[-20:]))
DF_melt4 = DF.melt(id_vars=['time_delta_s','sequence_num','commanded_closure_pressure_psi'],
                        value_vars=["Mean_jaw1_Pressure","Mean_jaw2_Pressure","Mean_jaw3_Pressure"],var_name = 'Variable',value_name ='Values')

fig4 = px.line(DF_melt4,x = 'commanded_closure_pressure_psi',y = 'Values', symbol = "sequence_num",facet_row = 'Variable')
fig4.show()


## Look at all the commanded grasper pressure and the measured grasper pressure
fig5 = px.scatter(DF,x = 'commanded_closure_pressure_psi',y = 'P_closure_psi', symbol = "sequence_num")
fig5.show()

## Look at all the commanded grasper pressure and the measured grasper pressure in a box plot
fig6 = px.box(DF,x = 'commanded_closure_pressure_psi',y = 'P_closure_psi', points = "all")
fig6.show()
