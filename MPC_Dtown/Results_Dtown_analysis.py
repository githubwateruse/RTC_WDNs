# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 12:20:08 2022

@author: sp825
"""





import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
np.random.seed(1)
import math


import copy
import random
import pyDOE as pydoe
import sys
import xlrd 

import epanet

from epanet import toolkit as tk

import xlsxwriter 

EPSILON = sys.float_info.epsilon






 
RequiredPressure = 25   

Maxlevels = [6.75, 6.5, 5, 5.5, 4.5, 5.9, 4.7]
# some global variables of genetic algorithm (GA)
EP_LEN = 24 # The length od decision time periods
nvars = 25

##-----------------------------------------------------------------------------------------------------------------
 

# some global variables of Anytown_1h
PenaltyFactor = 360
# price = 0.189
price = 0.189
ErrorpenaltyFactor = 0
ActionPenalty = 1




 
#-------------------------------------------------------------
ENpro = tk.createproject()
tk.open(ENpro, "D_town_ori_mod.inp", "D_town_ori_mod.rpt", "")




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
NODEID_group = []
for i in range(nnodes): 
    node_type = tk.getnodetype(ENpro, i+1)
    node_id = tk.getnodeid(ENpro, i+1)
    # print(i+1, node_id, node_type)
    NODEID_group.append(node_id)
    
    bd = tk.getnodevalue(ENpro, i+1, tk.BASEDEMAND)
    # print(i+1, node_id, node_type, bd)
    
    if node_type == 0 and bd != 0:
        demand_group.append(i+1)
        MeanDemand_group.append(bd)
    elif node_type == 2:
        tank_group.append(i+1)


 


for i in pump_group:
    id_ = tk.getlinkid(ENpro, i)
    print("Pump_group", id_)


 
for i in range (5):
    pattern_id = tk.getpatternid(ENpro, i+1)
    pattern_len = tk.getpatternlen(ENpro, i+1)
    # print(i+1, pattern_id, pattern_len)

Pattern_DMA1 = []
for i in range (168):
    pattern_value = tk.getpatternvalue(ENpro, 1, i+1)
    # print(pattern_value)
    Pattern_DMA1.append(pattern_value)

Pattern_DMA2 = []
for i in range (168):
    pattern_value = tk.getpatternvalue(ENpro, 2, i+1)
    # print(pattern_value)
    Pattern_DMA2.append(pattern_value)

Pattern_DMA3 = []
for i in range (168):
    pattern_value = tk.getpatternvalue(ENpro, 3, i+1)
    # print(pattern_value)
    Pattern_DMA3.append(pattern_value)
    
Pattern_DMA4 = []
for i in range (168):
    pattern_value = tk.getpatternvalue(ENpro, 4, i+1)
    # print(pattern_value)
    Pattern_DMA4.append(pattern_value)

Pattern_DMA5 = []
for i in range (168):
    pattern_value = tk.getpatternvalue(ENpro, 5, i+1)
    # print(pattern_value)
    Pattern_DMA5.append(pattern_value)


 
Node_Pattern = []
DMA_1_node_group = []
DMA_2_node_group = []
DMA_3_node_group = []
DMA_4_node_group = []
DMA_5_node_group = []

for i in demand_group:
    id_ = tk.getnodeid(ENpro, i)
    pn = tk.getnodevalue(ENpro, i, tk.PATTERN)
    # print(id_, pn)
    Node_Pattern.append(pn)
    
    

tk.close(ENpro)
tk.deleteproject(ENpro)

Multipliers_DMA_1 = Pattern_DMA1
Multipliers_DMA_2 = Pattern_DMA2
Multipliers_DMA_3 = Pattern_DMA3
Multipliers_DMA_4 = Pattern_DMA4
Multipliers_DMA_5 = Pattern_DMA5



 
#-------------------------------------------------------------
 
def DEMAND_Observation(basedemand_scenario, period):
    Demand_observation_group = np.zeros(len(demand_group)) 
    
    for j in range (len(basedemand_scenario)):
        demand_pattern_index = Node_Pattern[j]
        if demand_pattern_index == 1:
            multiplier = Multipliers_DMA_1[period]
        if demand_pattern_index == 2:
            multiplier = Multipliers_DMA_2[period]
        if demand_pattern_index == 3:
            multiplier = Multipliers_DMA_3[period]
        if demand_pattern_index == 4:
            multiplier = Multipliers_DMA_4[period]
        if demand_pattern_index == 5:
            multiplier = Multipliers_DMA_5[period]
        
        Demand_observation_group[j] = multiplier*basedemand_scenario[j]/2
        
    # return Demand_obser_group_1, Demand_obser_group_2, Demand_obser_group_3, Demand_obser_group_4, Demand_obser_group_5
    return Demand_observation_group



 
def REWARD_CALCULATION(EnergyCost, critical_nodes_pressure,  action_pre, action_current):
    
    # 1) energy calculation
    Energy_spend = price*EnergyCost
        
    # 2) Required pressure judgement (soft constraint)
    if critical_nodes_pressure >= RequiredPressure:
        Pressure_spend = 0
    else:
        Pressure_spend = RequiredPressure - critical_nodes_pressure
    
    Action_difference = 0 #abs(action_pre - action_current)
    for k in range (len(action_pre)):
        action_diff = abs(action_pre[k] - action_current[k])
        Action_difference = Action_difference + action_diff
    hourly_reward = (Energy_spend  + PenaltyFactor*Pressure_spend) + ActionPenalty * Action_difference 
    
    hourly_reward_2 = Energy_spend + ActionPenalty * Action_difference
    
    return hourly_reward, hourly_reward_2


 
def Objfunction(pump_solution, period_number, Basedemand_scenario, Tanklevel_Observation):
    # some sub-globalvariables
    EnergyCost = 0
    ErrorConstr = 0
    
    Ph = tk.createproject()
    
    tk.open(Ph, "D_town_mod_1h.inp", "D_town_mod_1h.rpt", "")
    
    
    # set the corresponding water demand scenario 
    # which is based on the period_number and Header
    period_demand_observation = DEMAND_Observation(Basedemand_scenario, period_number)
    for i in range (len(demand_group)):
        tk.setnodevalue(Ph, demand_group[i], tk.BASEDEMAND, period_demand_observation[i])
     
    Timber1 = np.round(float(pump_solution[0]), decimals = 2)
    Timber2 = np.round(float(pump_solution[1]), decimals = 2)
    Timber3 = np.round(float(pump_solution[2]), decimals = 2)
    Timber4 = np.round(float(pump_solution[3]), decimals = 2)
    Timber5 = np.round(float(pump_solution[4]), decimals = 2)
    
    if Timber1 < 0.5:
        # Reference: Gergely Hajgato (2020). Deep Reinforcement Learning for Real-Time Optimization of Pumps in Water Distribution Systems.
        Timber1 = 0
    if Timber2 < 0.5:
        Timber2 = 0
    if Timber3 < 0.5:
        Timber3 = 0
    if Timber4 < 0.5:
        Timber4 = 0
    if Timber5 < 0.5:
        Timber5 = 0
    
    tk.setpatternvalue(Ph, 6, 1, Timber1)
    tk.setpatternvalue(Ph, 7, 1, Timber2)
    tk.setpatternvalue(Ph, 8, 1, Timber3)
    tk.setpatternvalue(Ph, 9, 1, Timber4)
    tk.setpatternvalue(Ph, 10, 1, Timber5)
    
    
    # settings of tanks
    for i in range (len(tank_group)):
        # print("Tanklevel_Observation", Tanklevel_Observation)
        tk.setnodevalue(Ph, tank_group[i], tk.TANKLEVEL, Tanklevel_Observation[i])
    
     
    tk.setdemandmodel(Ph, tk.PDA, 0, 25, 0.5)    
    # hydraulic simulation
    
    tk.openH(Ph)
    tk.initH(Ph, tk.NOSAVE)
    while True:
        t = tk.runH(Ph)
        
        if t == 0:
            # Tanklevel_Start, which is supposed to be equivalent to Tanklevel_Observation
            Tanklevel_Start = []
            for i in tank_group:
                d = tk.getnodevalue(Ph, i, tk.HEAD)
                e = tk.getnodevalue(Ph, i, tk.ELEVATION)
                Tanklevel_Start.append(d-e)
        
        if t % 3600 == 0:
            if t > 0:
                # Read the critical node pressure
                PRESSURE_ = []
                for i in demand_group:
                    p = tk.getnodevalue(Ph, i, tk.PRESSURE)
                    #print(p)
                    #HydraulicConstr += max(0, RequiredPressure - p)
                    PRESSURE_.append(p)
                critical_nodes_pressure = np.min(PRESSURE_)
        
        tstep = tk.nextH(Ph)
        # Energy consumption calculation
        for i in pump_group:
            EnergyCost += tk.getlinkvalue(Ph, i, tk.ENERGY) * tstep / 3600
        
        
                
        
        # final states of tanks after the 1h-hydraulic simulation
        if tstep <= 0:
            final_tank_level = []
            for i in range (len(tank_group)):
                d = tk.getnodevalue(Ph, tank_group[i], tk.HEAD)
                e = tk.getnodevalue(Ph, tank_group[i], tk.ELEVATION)
                if d - e <= 0: 
                    final_tank_level.append(0)
                elif d - e >= Maxlevels[i]: 
                    final_tank_level.append(Maxlevels[i]) 
                else:
                    final_tank_level.append(d - e)
                    
            break
    
    tk.closeH(Ph)
    tk.close(Ph)
    tk.deleteproject(Ph)
    
    return EnergyCost, critical_nodes_pressure, final_tank_level, ErrorConstr







 
def Single_period_Solution_Evaluation(pump_solution, Demand_scenario, period_number, tanklevel_observation):
    
    EnergyCost, critical_nodes_pressure, Final_Tank_Level, ErrorConstr = Objfunction(pump_solution, period_number, Demand_scenario, tanklevel_observation)

    return Final_Tank_Level, critical_nodes_pressure, EnergyCost

    
    



 
#-------------------------------------------------------------
def MPC_GAOptimize(Demand_scenario, MPC_solutions):
    Total_costs = 0
    MIN_node_pressure = []
 
    
    action_pre = [1, 1, 1, 1, 1]
    Tanklevel_Observation = [3, 3, 2.5, 5.2, 1, 0.5, 2.5]
    for time_period in range (EP_LEN):
        Final_Tank_Level, min_node_pressure, EnergyCost = Single_period_Solution_Evaluation(MPC_solutions[time_period], Demand_scenario, time_period, Tanklevel_Observation)
        Tanklevel_Observation = Final_Tank_Level
        
        MIN_node_pressure.append(min_node_pressure)
        
         
        action_current = []
        for x in range (len(MPC_solutions[time_period])):
            if MPC_solutions[time_period][x] < 0.5:
                ps = 0
            else:
                ps = 1
            action_current.append(ps)
        hourly_reward, hourly_reward_2 = REWARD_CALCULATION(EnergyCost, min_node_pressure, action_pre, action_current)
        
        
        Total_costs = Total_costs + hourly_reward_2 
        action_pre = action_current
        
    return [Total_costs], MIN_node_pressure            



#-------------------------------------------------------------
def main_MPC(header_base, header_type, header_range):
    
    
    path = header_base + "LHS sample results_Dtown/LhsRandom_results_for_" + header_type + ".xlsx"   # 
    workbook = xlrd.open_workbook(path)
    data_sheet_scenario = workbook.sheet_by_index(0) # 
    rowNum = data_sheet_scenario.nrows
    colNum = data_sheet_scenario.ncols
    
    print("header_base", header_base, "header_type", header_type, "header_range", header_range)
    #print(rowNum,colNum)    
    cols = np.zeros((colNum,rowNum))
    
    Overview_reward = []
    Overview_pressure_violation = 0
    for i in range (colNum):
        print("Running", "scenario ", i)
        cols = data_sheet_scenario.col_values(i)
        demand_scenario = cols
        
        # F:\Dtown_perfect_uncertainty\MPC_results\MPC_testing_0/Deamnd Scenario_0_pump solution.xlsx
        
        
        path_read = header_base + "MPC_results/MPC_" + header_type + "_" + header_range + "/"
        path_solution = path_read + "Deamnd Scenario_" + str(i) + "_pump solution.xlsx"
        workbook_solution = xlrd.open_workbook(path_solution)
        data_sheet_solution = workbook_solution.sheet_by_index(0) # 
        
        MPC_solutions = []
        for m in range (EP_LEN):
            mpc_solution = data_sheet_solution.row_values(m)
            MPC_solutions.append(mpc_solution)
            
        # print(MPC_solutions)
        
        Total_costs, MIN_node_pressure = MPC_GAOptimize(demand_scenario, MPC_solutions)
        Overview_reward.append(Total_costs)
        
        for mmm in range (len(MIN_node_pressure)):
            aaa = MIN_node_pressure[mmm]
            if aaa < RequiredPressure:
                Overview_pressure_violation = Overview_pressure_violation + 1
                 
        workbook_wr = xlsxwriter.Workbook(path_read + 'Results_'+ 'scenario_'+str(i) +'.xlsx')
        worksheet_wr = workbook_wr.add_worksheet('sheet1')
        
        worksheet_wr.write_column(0, 0, Total_costs)
        worksheet_wr.write_column(0, 1, MIN_node_pressure)
        workbook_wr.close()  
    print("mean cost", np.mean(Overview_reward), "The number of constraint violations", Overview_pressure_violation)
        
    

    
if __name__ == "__main__":
    import time
    starttime = time.time()
    
    
    
    Header_base = "F:/Dtown_perfect_uncertainty/"
    Header_type = ["training", "testing"]
    Header_range = ["0","5"]
    
    # INPUT_TYPE= [0, 1, 2]
    INPUT_TYPE = [0]
    for x in INPUT_TYPE:
        Input_type = x
    
        if Input_type == 0:
            header_type = "training"
            header_range = "0"
        
        if Input_type == 1:
            header_type = "testing"
            header_range = "0"
        
        
            
        main_MPC(Header_base, header_type, header_range)
    

    endtime = time.time()
    print ('The Running Time:' , endtime - starttime)
    print ("Run over!")
