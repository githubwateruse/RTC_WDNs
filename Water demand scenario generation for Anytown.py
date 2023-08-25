# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 18:36:58 2021

@author: sp825
"""


import numpy as np
import epanet

from epanet import toolkit as tk
import math
import random
import matplotlib.pyplot as plt
from scipy.stats.distributions import truncnorm
import pyDOE as pydoe



np.random.seed(1)
lhsamplesize = 1500 # the number of training scnarios (i.e. 1000) + the number of testing scenarios (i.e. 500)
Cv = 0.5


# Basic information
#-------------------------------------------------------------
ENpro = tk.createproject()
tk.open(ENpro, "Anytown_revised_1h.inp", "Anytown_revised_1h.rpt", "")




nnodes = tk.getcount(ENpro, tk.NODECOUNT)
nlinks = tk.getcount(ENpro, tk.LINKCOUNT)

pump_group = []
for i in range (nlinks):
    link_type = tk.getlinktype(ENpro, i+1)
    link_id = tk.getlinkid(ENpro, i+1)
    # print(i+1, link_id, link_type)
    if link_type == 2:
        pump_group.append(i+1)
  
"""  """     
demand_group = []
tank_group = []
MeanDemand_group = []
for i in range(nnodes): 
    node_type = tk.getnodetype(ENpro, i+1)
    node_id = tk.getnodeid(ENpro, i+1)
    # print(i+1, node_id, node_type)
    
    
    bd = tk.getnodevalue(ENpro, i+1, tk.BASEDEMAND)
    # print(i+1, node_id, node_type, bd)
    
    if node_type == 0 and bd != 0:
        demand_group.append(i+1)
        MeanDemand_group.append(bd)
    elif node_type == 2:
        tank_group.append(i+1)
tk.close(ENpro)
tk.deleteproject(ENpro)


#For every generation, Latin Hypervolume Sampling on the demand multiplier
LhsRandom = pydoe.lhs(len(MeanDemand_group), lhsamplesize)





VarianceDemand_group = [Cv * value for value in MeanDemand_group]

# print(VarianceDemand_group)



for i in range(len(MeanDemand_group)):
    left_clip = (0.3*MeanDemand_group[i] - MeanDemand_group[i]) / VarianceDemand_group[i]
    right_clip = (1.3*MeanDemand_group[i] - MeanDemand_group[i]) / VarianceDemand_group[i]
    LhsRandom[:, i] = truncnorm(left_clip, right_clip, \
                                loc=MeanDemand_group[i], scale=VarianceDemand_group[i]).ppf(LhsRandom[:, i])

len_for_train = 1000

import xlsxwriter 
workbook_training = xlsxwriter.Workbook('LhsRandom_results_for_training.xlsx')
worksheet_training = workbook_training.add_worksheet('sheet1')
for i in range (len_for_train):
    worksheet_training.write_column(0, i, LhsRandom[i])
workbook_training.close()     

workbook_testing = xlsxwriter.Workbook('LhsRandom_results_for_testing.xlsx')
worksheet_testing = workbook_testing.add_worksheet('sheet1')
for i in range (len_for_train, len(LhsRandom)):
    worksheet_testing.write_column(0, i-len_for_train, LhsRandom[i])
workbook_testing.close()     


