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

Low = 0.95
HIGH = 1.05


RequiredPressure = 25   

Maxlevels = [6.75, 6.5, 5, 5.5, 4.5, 5.9, 4.7]
Minlevels = [0, 0, 0, 0, 0, 0, 0]
# some global variables of genetic algorithm (GA)
EP_LEN = 24 # The length od decision time periods
nvars = 10
popsize = 50
nogen = 50

tournamentSize = 4 
pCross = 0.7 
pMutation = 0.05
sbx_index = 1 
pm_index = 1 

lowbound = np.zeros(nvars)
highbound = np.ones(nvars) 

##-----------------------------------------------------------------------------------------------------------------
# some global variables of Anytown_1h
PenaltyFactor = 360 
# price = 0.189
price = 0.189
ErrorpenaltyFactor = 0
ActionPenalty = 1


#-------------------------------------------------------------
ENpro = tk.createproject()
tk.open(ENpro, "D_town_mod.inp", "D_town_mod.rpt", "")




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


def REWARD_CALCULATION(EnergyCost, critical_nodes_pressure, period_number, ErrorConstr, action_pre, action_current):
    
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
    hourly_reward = (Energy_spend  + PenaltyFactor*Pressure_spend) + ActionPenalty * Action_difference + ErrorConstr

    
    return hourly_reward 




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
    
    # The pressure constraint is 25m
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




def Control_Horizon_Optimize(Demand_scenario, time_period, Tanklevel_Observation, action_pre):
    # Initialize the population
    # for continuous decision variables
    # multi_period_solution
    population = lowbound + pydoe.lhs(nvars, popsize) * (highbound - lowbound)
    
    
    # Evaluate all individuals
    functionValues = []
    for row in range(popsize):
        functionValues.append(Multi_period_Solution_Evaluation(population[row, :], Demand_scenario, time_period, Tanklevel_Observation, action_pre))
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
            functionValues.append(Multi_period_Solution_Evaluation(population[row, :], Demand_scenario, time_period, Tanklevel_Observation, action_pre))
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
        
        # print (ngen, bestValue)
        
        #print(bestValue)
        #print(bestIndividual)
        
        # Optimal_pump_solution_per_generation = []
        # # Recording the results (i.e., pump solutions, total rewards per generation)
        # for i in range (len(bestIndividual)):
        #     Optimal_pump_solution_per_generation.append(bestIndividual[i])

    Best_Individual = bestIndividual
    solution_in_control_horizon = [Best_Individual[0:5], Best_Individual[5:10]]

    return solution_in_control_horizon
    



def Multi_period_Solution_Evaluation(multi_period_solution, Demand_scenario, time_period, Tanklevel_Observation, Action_Pre):
    Fitness_Value = 0
    tanklevel_observation = Tanklevel_Observation
    action_pre = Action_Pre
    for i in range (2):
        period_number = time_period + i
        pump_solution = multi_period_solution[5*i: 5*i + 5]
        
        action_current = []
        for x in range (len(pump_solution)):
            if pump_solution[x] < 0.5:
                ps = 0
            else:
                ps = 1
            action_current.append(ps)
        EnergyCost, critical_nodes_pressure, final_tank_level, ErrorConstr = Objfunction(pump_solution, period_number, Demand_scenario, tanklevel_observation)
        
        
        fitness_value = REWARD_CALCULATION(EnergyCost, critical_nodes_pressure, period_number, ErrorConstr, action_pre, action_current)
        Fitness_Value = Fitness_Value + fitness_value
        
        tanklevel_observation = final_tank_level
        action_pre = action_current
    return Fitness_Value



 
def Single_period_Solution_Evaluation(pump_solution, Demand_scenario, period_number, tanklevel_observation):
    
    EnergyCost, critical_nodes_pressure, Final_Tank_Level, ErrorConstr = Objfunction(pump_solution, period_number, Demand_scenario, tanklevel_observation)
    print("period_number", period_number, "critical node pressure", critical_nodes_pressure, "Energy", EnergyCost)
    return Final_Tank_Level, period_number, critical_nodes_pressure, EnergyCost

    
def Tanklevel_inaccurate(Tanklevel_Observation_real):
    Tanklevel_Observation_input = []
    
    for i in range (len(Tanklevel_Observation_real)):
        tl = Tanklevel_Observation_real[i] * np.random.uniform(low = Low, high = HIGH)
        
        if tl >= Maxlevels[i]:
            tl = Maxlevels[i]
        if tl <= Minlevels[i]:
            tl = Minlevels[i]
            
        Tanklevel_Observation_input.append(tl)
        
    
    
    return Tanklevel_Observation_input





#-------------------------------------------------------------
def MPC_GAOptimize(Demand_scenario, number, header_base, header_type, header_range):
    workbook_ps = xlsxwriter.Workbook(header_base + 'MPC_results/' + 'MPC_' + header_type + '_'+header_range +'/' + 'Deamnd Scenario_' + '' + str(number) + '' + '_pump solution' +'.xlsx')
    worksheet_ps = workbook_ps.add_worksheet('sheet1')

    

    action_pre = [1, 1, 1, 1, 1]
    Tanklevel_Observation_real = [3, 3, 2.5, 5.2, 1, 0.5, 2.5]
    
    
    
    for time_period in range (EP_LEN - 1):
        
        if header_range == "0":
            Tanklevel_Observation_input = Tanklevel_Observation_real 
        else:
            # header_range == "5"
            Tanklevel_Observation_input = Tanklevel_inaccurate(Tanklevel_Observation_real)
            
        if time_period < EP_LEN - 2:
            solution_in_control_horizon = Control_Horizon_Optimize(Demand_scenario, time_period, Tanklevel_Observation_input, action_pre)
            Final_Tank_Level, period_number, critical_nodes_pressure, EnergyCost = Single_period_Solution_Evaluation(solution_in_control_horizon[0], Demand_scenario, time_period, Tanklevel_Observation_real)
            
            Tanklevel_Observation_real = Final_Tank_Level
            worksheet_ps.write_row(time_period, 0, solution_in_control_horizon[0])
        else:
            solution_in_control_horizon = Control_Horizon_Optimize(Demand_scenario, time_period, Tanklevel_Observation_input, action_pre)
            # Final_Tank_Level = Single_period_Solution_Evaluation(solution_in_control_horizon[0], Demand_scenario, time_period, Tanklevel_Observation_real)
            # Tanklevel_Observation_real = Final_Tank_Level
            # Final_Tank_Level = Single_period_Solution_Evaluation(solution_in_control_horizon[1], Demand_scenario, time_period + 1, Tanklevel_Observation_real)
            # Tanklevel_Observation_real = Final_Tank_Level
            
            worksheet_ps.write_row(time_period, 0, solution_in_control_horizon[0])
            worksheet_ps.write_row(time_period + 1, 0, solution_in_control_horizon[1])
            # worksheet_ps.write_row(time_period + 2, 0, solution_in_control_horizon[2])
            # worksheet_ps.write_row(time_period + 3, 0, solution_in_control_horizon[3])
            # worksheet_ps.write_row(time_period + 4, 0, solution_in_control_horizon[4])
    workbook_ps.close() 



    
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
def main_MPC(header_base, header_type, header_range):
    
    
    path = header_base + "LHS sample results_Dtown/LhsRandom_results_for_" + header_type + ".xlsx"   #设置路径
    workbook = xlrd.open_workbook(path)
    data_sheet = workbook.sheet_by_index(0) # 
    rowNum = data_sheet.nrows
    colNum = data_sheet.ncols
    
    #print(rowNum,colNum)    
    cols = np.zeros((colNum,rowNum))
     
    
    print("header_type", header_type, "header_range", header_range, "num_scenarios", colNum)
    pool = multiprocessing.Pool(10) #  
    
    for i in range (colNum):
        cols[i] = data_sheet.col_values(i)
        demand_scenario = cols[i]
        
        scenario_num = i        
        # MPC_GAOptimize(demand_scenario, i, header_base, header_type, header_range)
        pool.apply_async(func=MPC_GAOptimize,args=(demand_scenario, scenario_num, header_base, header_type, header_range))
    
    pool.close()
    pool.join()
    
    
    
    
    

        

    
if __name__ == "__main__":
    import time
    starttime = time.time()
     
    Input_type = 1
    Header_base = " "
    Header_type = ["training", "testing"]
    Header_range = ["0","5"]
    
    
    
    
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
