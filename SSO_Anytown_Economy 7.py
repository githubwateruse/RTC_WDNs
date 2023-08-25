# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 07:35:06 2022

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



# some global variables of genetic algorithm (GA)

EP_LEN = 24 # The length od decision time periods
nvars = 24
popsize = 200
nogen = 300

tournamentSize = 4 
pCross = 0.7 
pMutation = 0.05
sbx_index = 1 
pm_index = 1 

lowbound = np.zeros(nvars)
highbound = np.ones(nvars) 



##-----------------------------------------------------------------------------------------------------------------
# some global variables of Anytown_1h
PenaltyFactor = 30
Peakprice = 0.22
Offpeakprice = 0.111
RequiredPressure = 40
ErrorpenaltyFactor = 0
ActionPenalty = 1

# Basic information of the Anytown network
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

# Multipliers of the scheduling period
Multipliers = [ 1.0, 1.0, 1.0,0.9, 0.9, 0.9, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 1.2, 1.2, 1.2, 1.3, 1.3, 1.3, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1]

print(len(Multipliers))


#-------------------------------------------------------------

# Read the water demand scenario
def DEMAND_Observation(basedemand_scenario, period):
    
    Demand_observation_group = np.zeros(19)
    
    # 确定相应时段的乘子
    multiplier = Multipliers[period] # 
    for j in range (len(basedemand_scenario)):
        Demand_observation_group[j] = multiplier*basedemand_scenario[j]
    return Demand_observation_group




# Reward function calculation
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
    
    hourly_reward = (Energy_spend  + PenaltyFactor*Pressure_spend) + ActionPenalty * abs(action_pre - action_current) + ErrorConstr
    # reward calculation, including the ErrorConstr
#    if ErrorConstr == 0:
#        hourly_reward = (Energy_spend  + PenaltyFactor*Pressure_spend)
#        
#    else:
#        hourly_reward = ErrorpenaltyFactor * ErrorConstr
    #return -1*np.sqrt(hourly_reward)
#    
#    if ErrorConstr == 0 and pump_solution[0] < 0.5:
#        hourly_reward = hourly_reward - 1000
     
    # hourly_reward = -1*hourly_reward
    
    return hourly_reward 



# Hydrulic simulation
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
    Timber = float(pump_solution) # 
    
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
                else:
                    final_tank_level.append(d - e)
                
            break
    
    tk.closeH(Ph)
    tk.close(Ph)
    tk.deleteproject(Ph)
    
    return EnergyCost, critical_nodes_pressure, final_tank_level, ErrorConstr





# Genetic algorithm optimizer
#-------------------------------------------------------------
def GAOptimize(Demand_scenario, number, header_base, header_type, header_range):
    workbook_ps = xlsxwriter.Workbook(header_base + 'GA_' + header_type + '_'+header_range +'/' + 'Deamnd Scenario_' + '' + str(number) + '' + '_pump solution' +'.xlsx')
    worksheet_ps = workbook_ps.add_worksheet('sheet1')
    
    workbook_curve = xlsxwriter.Workbook(header_base + 'GA_' + header_type + '_'+header_range +'/' + 'Deamnd Scenario_' + '' + str(number) + '' + '_rewards curve' +'.xlsx')
    worksheet_curve = workbook_curve.add_worksheet('sheet1')
    # Initialize the population
    # for continuous decision variables
    population = lowbound + pydoe.lhs(nvars, popsize) * (highbound - lowbound)
    
    # for discrete decision variables
    #population = lowbound + pydoe.lhs(nvars, popsize) * (highbound - lowbound)

                 
    # Evaluate all individuals
    functionValues = []
    for row in range(popsize):
        """ GA solution evaluation: the corresponding function (GA_Evaluation) """
        functionValues.append(GA_Evaluation(population[row, :], Demand_scenario))
    functionValues = np.array(functionValues)

    # Save the best individual
    indmin = np.argmin(functionValues)
    bestValue = functionValues[indmin]
    bestIndividual = copy.deepcopy(population[indmin, :])
              
    
   
    # Main loop
    for ngen in range(nogen):

        # Do tournament selection to select the parents
        competitors = np.random.randint(0, popsize, (popsize, tournamentSize))
        ind = np.argmin(functionValues[competitors], axis=1)
        winnerIndices = np.zeros(popsize, dtype=int)
        for i in range(tournamentSize):  # This loop is short
            winnerIndices[np.where(ind == i)] = competitors[np.where(ind == i), i]
        parent1 = population[winnerIndices[0:popsize // 2], :]
        parent2 = population[winnerIndices[popsize // 2:popsize], :]

        #SBX
        cross = np.where(np.random.rand(popsize // 2) < pCross)[0]
        for i in cross:
            child1 = copy.deepcopy(parent1[i, :])
            child2 = copy.deepcopy(parent2[i, :])
            for j in range(nvars):
                if random.uniform(0.0, 1.0) <= 0.5:
                    x1 = float(child1[j])
                    x2 = float(child2[j])
                    x1, x2 = sbx_crossover(x1, x2, lowbound[j], highbound[j], sbx_index)
                    child1[j] = x1
                    child2[j] = x2
            parent1[i, :] = copy.deepcopy(child1)
            parent2[i, :] = copy.deepcopy(child2)
        population = np.concatenate((parent1, parent2))

        #PM
        for ind in range(popsize):
            child = copy.deepcopy(population[ind, :])
            for i in range(nvars):
                if random.uniform(0.0, 1.0) <= pMutation:
                    x1 = pm_mutation(float(child[i]), lowbound[i], highbound[i], pm_index)
                    child[i] = x1
            population[ind, :] = copy.deepcopy(child)

        #  Evaluate all individuals
        functionValues = []
        for row in range(popsize):
            functionValues.append(GA_Evaluation(population[row, :], Demand_scenario))
        functionValues = np.array(functionValues)

        # Save the best individual
        indmin = np.argmin(functionValues)
        indmax = np.argmax(functionValues)
        bestValue_Candidate = functionValues[indmin]

        # Generational Elitist Keep the best individual
        if bestValue_Candidate > bestValue:
            population[indmax, :] = copy.deepcopy(bestIndividual)
            functionValues[indmax] = bestValue
        else:
            bestIndividual = copy.deepcopy(population[indmin, :])
            bestValue = bestValue_Candidate
        
        print (ngen, bestValue)
        
        #print(bestValue)
        #print(bestIndividual)
        
        Optimal_pump_solution_per_generation = []
        # Recording the results (i.e., pump solutions, total rewards per generation)
        for i in range (len(bestIndividual)):
            Optimal_pump_solution_per_generation.append(bestIndividual[i])

        worksheet_ps.write_row(ngen, 0, Optimal_pump_solution_per_generation)
        
        Total_reward_per_generation = []
        Total_reward_per_generation.append(bestValue)
        worksheet_curve.write_column(ngen, 0, Total_reward_per_generation) 
    
    
    
    workbook_ps.close() 
    workbook_curve.close()


    
def pm_mutation(x, lb, ub, distribution_index):
    #
    u = random.uniform(0, 1)
    dx = ub - lb

    if u < 0.5:
        bl = (x - lb) / dx
        b = 2.0 * u + (1.0 - 2.0 * u) * pow(1.0 - bl, distribution_index + 1.0)
        delta = pow(b, 1.0 / (distribution_index + 1.0)) - 1.0
    else:
        bu = (ub - x) / dx
        b = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * pow(1.0 - bu, distribution_index + 1.0)
        delta = 1.0 - pow(b, 1.0 / (distribution_index + 1.0))

    x = x + delta * dx
    x = clip(x, lb, ub)

    return x

def sbx_crossover(x1, x2, lb, ub, distribution_index):

    dx = x2 - x1

    if dx > EPSILON:
        if x2 > x1:
            y2 = x2
            y1 = x1
        else:
            y2 = x1
            y1 = x2

        beta = 1.0 / (1.0 + (2.0 * (y1 - lb) / (y2 - y1)))
        alpha = 2.0 - pow(beta, distribution_index + 1.0)
        rand = random.uniform(0.0, 1.0)

        if rand <= 1.0 / alpha:
            alpha = alpha * rand
            betaq = pow(alpha, 1.0 / (distribution_index + 1.0))
        else:
            alpha = alpha * rand
            alpha = 1.0 / (2.0 - alpha)
            betaq = pow(alpha, 1.0 / (distribution_index + 1.0))

        x1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
        beta = 1.0 / (1.0 + (2.0 * (ub - y2) / (y2 - y1)))
        alpha = 2.0 - pow(beta, distribution_index + 1.0)

        if rand <= 1.0 / alpha:
            alpha = alpha * rand
            betaq = pow(alpha, 1.0 / (distribution_index + 1.0))
        else:
            alpha = alpha * rand
            alpha = 1.0 / (2.0 - alpha)
            betaq = pow(alpha, 1.0 / (distribution_index + 1.0))

        x2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

        # randomly swap the values
        if bool(random.getrandbits(1)):
            x1, x2 = x2, x1

        x1 = clip(x1, lb, ub)
        x2 = clip(x2, lb, ub)

    return x1, x2

def clip(value, min_value, max_value):

    return max(min_value, min(value, max_value))


#-------------------------------------------------------------
def GA_Evaluation(ga_solution, DEMAND_Scenario):
    fitness_value = 0
    Tanklevel_Observation = [10, 10, 10] # initial tank levels
    
    Basedemand_scenario = DEMAND_Scenario
    action_pre = 1
    
    for period in range(EP_LEN):
        #Demand_Observation = DEMAND_Observation(Basedemand_scenario, period)
        #print('ga_solution', ga_solution)
        pump_solution = ga_solution[period] # the pump solutions for the corresponding decision time periods
        
        if pump_solution < 0.5:
            action_current = 0
        else:
            action_current = 1
        
        EnergyCost, critical_nodes_pressure, final_tank_level, ERRORconstr = Objfunction(pump_solution, period, Basedemand_scenario, Tanklevel_Observation)
        
        
        
        hourly_reward = REWARD_CALCULATION(EnergyCost, critical_nodes_pressure, period, ERRORconstr, action_pre, action_current)
        
        action_pre = action_current
        Tanklevel_Observation = final_tank_level 
        fitness_value += hourly_reward
    
    return fitness_value


def main(header_base, header_type, header_range):
    
    
    
    path = header_base + "LHS sample results/LhsRandom_results_for_" + header_type + "_" + header_range + ".xlsx"   #设置路径
    workbook = xlrd.open_workbook(path)
    data_sheet = workbook.sheet_by_index(0) 
    rowNum = data_sheet.nrows
    colNum = data_sheet.ncols
    
    #print(rowNum,colNum)    
    cols = np.zeros((colNum,rowNum))
    
    # each column is the base nodal demand combination for a demand scenario
    
    pool = multiprocessing.Pool(10) # parallel computing
    
    for i in range (colNum):
        cols[i] = data_sheet.col_values(i)
        demand_scenario = cols[i]
        # GAOptimize(demand_scenario, i, header_base, header_type, header_range)
        pool.apply_async(func=GAOptimize,args=(demand_scenario, i, header_base, header_type, header_range))
    
    pool.close()
    pool.join()
    
    
    
    
    

        

    
if __name__ == "__main__":
    import time
    starttime = time.time()
    
    
    Header_base = " " # Header_base is the path to store the calculation results
    Header_type = ["training", "testing"]
    Header_range = ["0"]
    
    
    for i in (Header_type):
        for j in (Header_range):
            header_base = Header_base
            header_type = i
            header_range = j
            
            main(header_base, header_type, header_range)

    endtime = time.time()
    print ('The Running Time:' , endtime - starttime)
    print ("Run over!")
