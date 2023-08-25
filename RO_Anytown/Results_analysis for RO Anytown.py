# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 03:59:12 2022

@author: sp825
"""


import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
np.random.seed(1)
import math
np.random.seed(1) # random seeds


import copy
import random
import pyDOE as pydoe
import sys
import xlrd 

import epanet

from epanet import toolkit as tk

import xlsxwriter 

EPSILON = sys.float_info.epsilon

Header_base = 'F:/Anytown_PPO_two_tariffs/Result_Anytown_GA_robust_economy7/'

# some global variables of genetic algorithm (GA)

EP_LEN = 24 # The length od decision time periods
nvars = 24



##-----------------------------------------------------------------------------------------------------------------
# some global variables of Anytown_1h
PenaltyFactor = 30
Peakprice = 0.22
Offpeakprice = 0.111
RequiredPressure = 40
ErrorpenaltyFactor = 0
ActionPenalty = 1



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

print(pump_group)
print(tank_group)
print(demand_group)


Multipliers = [ 1.0, 1.0, 1.0,0.9, 0.9, 0.9, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 1.2, 1.2, 1.2, 1.3, 1.3, 1.3, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1]

print(len(Multipliers))



#-------------------------------------------------------------


def DEMAND_Observation(basedemand_scenario, period):
    
    Demand_observation_group = np.zeros(19)
    
    
    multiplier = Multipliers[period] # 
    for j in range (len(basedemand_scenario)):
        Demand_observation_group[j] = multiplier*basedemand_scenario[j]
    return Demand_observation_group



def REWARD_CALCULATION(EnergyCost, critical_nodes_pressure, period_number, ErrorConstr, action_pre, action_current):
    
    # 1) energy calculation
    # judging peaktime or offpeaktime by the parameter 'period'
    period_order = period_number % 24
    # For each day, the front 12 hours belong to offpeaktime, and the rest 12 hours belong to peak time
    if period_order < 12 and period_order > 4: 
        # the corresponding period belongs to offpeaktime periods
        Energy_spend = Offpeakprice*EnergyCost
    else:
        # the corresponding period belongs to peaktime periods
        Energy_spend = Peakprice*EnergyCost
        
    # 2) Required pressure judgement (soft constraint)
    if critical_nodes_pressure >= RequiredPressure:
        Pressure_spend = 0
    else:
        Pressure_spend = RequiredPressure - critical_nodes_pressure
    
    hourly_reward_1 = (Energy_spend  + PenaltyFactor*Pressure_spend) + ActionPenalty * abs(action_pre - action_current)
    hourly_reward_2 = Energy_spend + ActionPenalty * abs(action_pre - action_current)
    
    return hourly_reward_1, hourly_reward_2 




def Objfunction(pump_solution, period_number, Basedemand_scenario, Tanklevel_Observation):
    # some sub-globalvariables
    EnergyCost = 0
    ErrorConstr = 0
    
    Ph = tk.createproject()
    tk.open(Ph, "Anytown_revised_1h.inp", "Anytown_revised_1h.rpt", "")
    
    
    # set the corresponding water demand scenario 
    # which is based on the period_number and Header
    period_demand_observation = DEMAND_Observation(Basedemand_scenario, period_number)
    for i in range (len(demand_group)):
        tk.setnodevalue(Ph, demand_group[i], tk.BASEDEMAND, period_demand_observation[i])
    
    
    
    # settings of variable speed pumps
    Timber = float(pump_solution) 
    
    
    Timber = np.round(Timber, decimals = 2)
    
    if Timber < 0.5:
        # Reference: Gergely Hajgato (2020). Deep Reinforcement Learning for Real-Time Optimization of Pumps in Water Distribution Systems.
        Timber = 0
    
    
    tk.setpatternvalue(Ph, 2, 1, Timber)
    tk.setpatternvalue(Ph, 3, 1, Timber)
    tk.setpatternvalue(Ph, 4, 1, Timber)
    
    
    # settings of tanks
    for i in range (len(tank_group)):
        # print("Tanklevel_Observation", Tanklevel_Observation)
        tk.setnodevalue(Ph, tank_group[i], tk.TANKLEVEL, Tanklevel_Observation[i])
        
    tk.setdemandmodel(Ph, tk.PDA, 0, 40, 0.5)    
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
            for i in tank_group:
                d = tk.getnodevalue(Ph, i, tk.HEAD)
                e = tk.getnodevalue(Ph, i, tk.ELEVATION)
                if d - e <= 10:
                    final_tank_level.append(10)
                    # Check the violation during hydraulic simulation process
                    # ErrorConstr = ErrorConstr 
                elif d - e >= 35:
                    final_tank_level.append(35)
                else:
                    final_tank_level.append(d - e)
                
            break
    
    tk.closeH(Ph)
    tk.close(Ph)
    tk.deleteproject(Ph)
    
    return EnergyCost, critical_nodes_pressure, final_tank_level, ErrorConstr





#-------------------------------------------------------------
    


def GA_Evaluation(ga_solution, DEMAND_Scenario):
    Total_costs = 0
    Pump_solutions = []
    Energy_power = []
    MIN_node_pressure = []
    Tank1_level = []
    Tank2_level = []
    Tank3_level = []
    Tanklevel_Observation = [10, 10, 10] # initial tank levels
    Tank1_level.append(Tanklevel_Observation[0])
    Tank2_level.append(Tanklevel_Observation[1])
    Tank3_level.append(Tanklevel_Observation[2])

    Basedemand_scenario = DEMAND_Scenario
    action_pre = 1
    for period in range(EP_LEN):
        #Demand_Observation = DEMAND_Observation(Basedemand_scenario, period)
        #print('ga_solution', ga_solution)
        pump_solution = ga_solution[period] # the pump solutions for the corresponding decision time periods
        
        if pump_solution < 0.5:
            action_current = 0
            pump_current = 0
        else:
            action_current = 1
            pump_current = np.round(pump_solution, decimals = 2)
        Pump_solutions.append(pump_current)
        EnergyCost, critical_nodes_pressure, final_tank_level, ERRORconstr = Objfunction(pump_solution, period, Basedemand_scenario, Tanklevel_Observation)
        Energy_power.append(EnergyCost)
        MIN_node_pressure.append(critical_nodes_pressure)
        
        Tank1_level.append(final_tank_level[0])
        Tank2_level.append(final_tank_level[1])
        Tank3_level.append(final_tank_level[2])
        
        hourly_reward_1, hourly_reward_2 = REWARD_CALCULATION(EnergyCost, critical_nodes_pressure, period, ERRORconstr, action_pre, action_current)
        action_pre = action_current
        Tanklevel_Observation = final_tank_level 
        
        
        Total_costs += hourly_reward_2
        

        
    return [Total_costs], Pump_solutions, Energy_power, MIN_node_pressure, Tank1_level, Tank2_level, Tank3_level 





    
    
def GA_Evaluation_multipool(ga_solution, header_write):

    
    path = Header_base + 'LHS sample results/LhsRandom_results_for_'+ header_write +'_0.xlsx'    
    workbook_scenario = xlrd.open_workbook(path)
    data_sheet_scenario = workbook_scenario.sheet_by_index(0) 
    rowNum_S = data_sheet_scenario.nrows
    colNum_S = data_sheet_scenario.ncols
    
    for i in range (colNum_S):
        print("Running", header_write, "scenario ", i)
        workbook_wr = xlsxwriter.Workbook(Header_base + 'Results_'+ header_write +'_scenario_'+str(i) +'.xlsx')
        worksheet_wr = workbook_wr.add_worksheet('sheet1')
        
        cols = data_sheet_scenario.col_values(i)
        demand_scenario = cols
        Total_costs, Pump_solutions, Energy_power, MIN_node_pressure, Tank1_level, Tank2_level, Tank3_level = GA_Evaluation(ga_solution, demand_scenario)
    


        worksheet_wr.write_row(0, 0, Total_costs)
        worksheet_wr.write_row(1, 0, Pump_solutions)
        worksheet_wr.write_row(2, 0, Energy_power)
        worksheet_wr.write_row(3, 0, MIN_node_pressure)
        worksheet_wr.write_row(4, 0, Tank1_level)
        worksheet_wr.write_row(5, 0, Tank2_level)
        worksheet_wr.write_row(6, 0, Tank3_level)
        workbook_wr.close()    
    


    
if __name__ == "__main__":
    import time
    starttime = time.time()
    
    path_solution = '.../Robust GA_pump solution299.xlsx'
    workbook_solution = xlrd.open_workbook(path_solution)
    data_sheet_solution = workbook_solution.sheet_by_index(0)  
    Robust_GA_solution = data_sheet_solution.row_values(299)


    print("Robust_GA_solution", Robust_GA_solution)
    
    
    Header_write = ["training", "testing"]
    for x in Header_write:
        header_write = x
    
    
        GA_Evaluation_multipool(Robust_GA_solution, header_write)
    
    endtime = time.time()
    print ('The Running Time:' , endtime - starttime)
    print ("Run over!")
