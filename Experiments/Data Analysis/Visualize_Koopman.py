import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from copy import deepcopy

import pykoopman as pk
from pydmd import DMD
from pydmd.plotter import plot_eigs, plot_summary


DF = pd.read_csv("Koopman Data/Koopman_Testing_06_04_2024_10_04_48.csv")
corrected_time = DF.loc[:,["time_delta_s","sequence_num"]].groupby(['sequence_num']).transform(lambda x: (x-x.iloc[0]))
DF["corrected_time_s"] = corrected_time



# ## ---- Plot the states (jaw pressures) and controls (x,y and the commanded grasper pressure) as functions of time, colored by the sequence number, time not corrected
# DF_melt0 = DF.melt(id_vars=['time_delta_s','sequence_num'],
#                         value_vars=['P_jaw1_psi', 'P_jaw2_psi','P_jaw3_psi','x_mm','y_mm','P_closure_psi',
#                                     'commanded_x_mm',' commanded_y_mm','commanded_closure_pressure_psi'],var_name = 'Variable',value_name ='Values')
#
# fig0 = px.line(DF_melt0,x = 'time_delta_s',y = 'Values', color = "sequence_num", symbol = "sequence_num",facet_row = 'Variable')
# fig0.update_yaxes(matches = None, showticklabels=True)
# fig0.show()
#
#
# ## ---- Plot the jaw pressures in jaw 1 stacked ---- ##
# fig1 = px.line(DF,x = 'corrected_time_s',y = 'P_jaw1_psi', color = "sequence_num")
# fig1.show()
#
#
# ## ---- Plot the jaw pressures as functions of time, colored by the sequence number, time not corrected
# DF_melt2 = DF.melt(id_vars=['time_delta_s','sequence_num'],
#                         value_vars=['P_jaw1_psi', 'P_jaw2_psi','P_jaw3_psi'],var_name = 'Variable',value_name ='Values')
#
# fig2 = px.line(DF_melt2,x = 'time_delta_s',y = 'Values', color = "sequence_num", symbol = "sequence_num",facet_row = 'Variable')
# fig2.show()
#
# ## ---- Look at max jaw pressure as a function of the commanded pressure for each jaw ---- ##
# #Note: results will be confusing because if it is a step down (i.e. prev pressure is higher than current commanded pressure), there will be a negative correlation. Need to add a secondary variable to color by that encodes step up or step down
# DF[["Max_jaw1_Pressure","Max_jaw2_Pressure","Max_jaw3_Pressure"]] = DF.loc[:,["P_jaw1_psi","P_jaw2_psi","P_jaw3_psi","sequence_num"]].groupby(['sequence_num']).transform(lambda x: max(x))
# DF_melt3 = DF.melt(id_vars=['time_delta_s','sequence_num','commanded_closure_pressure_psi'],
#                         value_vars=["Max_jaw1_Pressure","Max_jaw2_Pressure","Max_jaw3_Pressure"],var_name = 'Variable',value_name ='Values')
#
# fig3 = px.line(DF_melt3,x = 'commanded_closure_pressure_psi',y = 'Values', symbol = "sequence_num",facet_row = 'Variable')
# fig3.show()
#
# ## ---- Look at average jaw pressure as a function of the commanded pressure for each jaw in the last second, i.e ~20 samples ---- ##
# DF[["Mean_jaw1_Pressure","Mean_jaw2_Pressure","Mean_jaw3_Pressure"]] = DF.loc[:,["P_jaw1_psi","P_jaw2_psi","P_jaw3_psi","sequence_num"]].groupby(['sequence_num']).transform(lambda x: np.mean(x.iloc[-20:]))
# DF_melt4 = DF.melt(id_vars=['time_delta_s','sequence_num','commanded_closure_pressure_psi'],
#                         value_vars=["Mean_jaw1_Pressure","Mean_jaw2_Pressure","Mean_jaw3_Pressure"],var_name = 'Variable',value_name ='Values')
#
# fig4 = px.line(DF_melt4,x = 'commanded_closure_pressure_psi',y = 'Values', symbol = "sequence_num",facet_row = 'Variable')
# fig4.show()
#
#
# ## Look at all the commanded grasper pressure and the measured grasper pressure
# fig5 = px.scatter(DF,x = 'commanded_closure_pressure_psi',y = 'P_closure_psi', symbol = "sequence_num")
# fig5.show()
#
# ## Look at all the commanded grasper pressure and the measured grasper pressure in a box plot
# fig6 = px.box(DF,x = 'commanded_closure_pressure_psi',y = 'P_closure_psi', points = "all")
# fig6.show()


#---- extended DMD w/ control ----# -> see this example: https://pykoopman.readthedocs.io/en/master/tutorial_koopman_edmdc_for_vdp_system.html

uniqSeq = np.unique(DF["sequence_num"])
train_prop = 0.7 #proportion of test prop
ran = range(0,int(np.floor(train_prop*len(uniqSeq))))

train_seq = [uniqSeq[np.random.randint(np.min(0),np.max(len(uniqSeq)))] for x in ran]
train_idx = [x in train_seq for x in DF["sequence_num"]]
test_seq = [x for x in uniqSeq if x not in train_seq]
test_idx = [x in test_seq for x in DF["sequence_num"]]

#get the states

states =["P_jaw1_psi","P_jaw2_psi","P_jaw3_psi","x_mm","y_mm","z_mm"]
state_data = np.array(DF.loc[train_idx,states]).T #will be an ns x n_t matrix, where ns is the number of states, n_t is the number of datapoints

#get the controls
controls = ["commanded_closure_pressure_psi","commanded_x_mm"," commanded_y_mm"]
control_data = np.array(DF.loc[train_idx,controls]).T #need one less control than than state

#from the states, create the observables
n_delay = 2
ob1 = pk.observables.Identity()
ob2 = pk.observables.Polynomial(degree=3)
ob3 = pk.observables.TimeDelay(delay = 1, n_delays = n_delay)
obs = ob1 + ob2 + ob3



#---- Koopman ---#
#Fit koopman
EDMDc = pk.regression.EDMDc()
#model = pk.Koopman(observables = obs, regressor = EDMDc)
model = pk.Koopman(observables = ob1, regressor = EDMDc)

#model.fit(x= state_data[:,n_delay:-1].T, y = state_data[:,n_delay+1:].T,u = control_data[:,n_delay:-1].T)
model.fit(x= state_data[:,0:-1].T, u = control_data[:,0:-1].T)
#Xkoop = model.simulate(x, u[:, np.newaxis], n_steps=n_int-1)


#---- Validate ---#
# add column for prediction, whether is train or test
DF["Prediction"] = np.nan
DF["Train_or_Test"] = "Train"
for k in uniqSeq:
    idx = np.array([x==k for x in DF["sequence_num"]])
    nidx = np.where(idx == True)[0]
    num_steps = np.size(nidx)
    Xkoop = model.simulate(DF.loc[nidx[0],states].to_numpy(), DF.loc[nidx,controls].to_numpy(), n_steps=(num_steps))
    DF.loc[nidx,"Prediction"] = pd.Series(list(Xkoop),index = nidx)
    DF.loc[nidx,"Train_or_Test"] = "Train" if k in train_seq else "Test"

#expand column with list of koopman prediction to prediction of state variables
predict_state_names=[x+"_prediction" for x in states]
newDF = DF[["Prediction"]].apply(lambda x:  pd.Series([v for lst in x for v in lst],index = predict_state_names), axis=1, result_type="expand")
DF = pd.DataFrame.merge(DF,newDF,left_index=True,right_index=True)

DF_melt7 = DF.melt(id_vars=['corrected_time_s','sequence_num',"Train_or_Test"],
                        value_vars=['P_jaw1_psi', 'P_jaw2_psi','P_jaw3_psi', 'P_jaw1_psi_prediction','P_jaw2_psi_prediction','P_jaw3_psi_prediction'],var_name = 'Variable',value_name ='Values')

fig7 = px.line(DF_melt7,x = 'corrected_time_s',y = 'Values', color = "sequence_num", symbol = "sequence_num",facet_row = 'Variable',facet_col = "Train_or_Test")
fig7.update_yaxes(range=[0, 1])
fig7.show()

