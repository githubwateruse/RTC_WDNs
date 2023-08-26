# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 09:03:51 2023

@author: sp825
"""


import torch
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous

import random
import math
import matplotlib.pyplot as plt

import xlrd
import xlsxwriter

import epanet

from epanet import toolkit as tk
import xlsxwriter


Header_base ="E:/Dtown_perfect_uncertainty/"
Header_for_train = "LHS sample results_Dtown/LhsRandom_results_for_training.xlsx"

Maxlevels = [6.75, 6.5, 5, 5.5, 4.5, 5.9, 4.7]
Minlevels = [0, 0, 0, 0, 0, 0, 0]

##-----------------------------------------------------------------------------------------------------------------

 
PenaltyFactor = 360
# price = 0.189
price = 0.189
RequiredPressure = 25  
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


# print(len(pump_group))
print(len(tank_group))
# print(len(demand_group))



for i in pump_group:
    id_ = tk.getlinkid(ENpro, i)
    print("Pump_group", id_)



 
for i in range (5):
    pattern_id = tk.getpatternid(ENpro, i+1)
    pattern_len = tk.getpatternlen(ENpro, i+1)
    # print(i+1, pattern_id, pattern_len)

Pattern_DMA1 = []
for i in range (24):
    pattern_value = tk.getpatternvalue(ENpro, 1, i+1)
    # print(pattern_value)
    Pattern_DMA1.append(pattern_value)

Pattern_DMA2 = []
for i in range (24):
    pattern_value = tk.getpatternvalue(ENpro, 2, i+1)
    # print(pattern_value)
    Pattern_DMA2.append(pattern_value)

Pattern_DMA3 = []
for i in range (24):
    pattern_value = tk.getpatternvalue(ENpro, 3, i+1)
    # print(pattern_value)
    Pattern_DMA3.append(pattern_value)
    
Pattern_DMA4 = []
for i in range (24):
    pattern_value = tk.getpatternvalue(ENpro, 4, i+1)
    # print(pattern_value)
    Pattern_DMA4.append(pattern_value)

Pattern_DMA5 = []
for i in range (24):
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
    
    # if pn == 1:
    #     DMA_1_node_group.append(i)
    # if pn == 2:
    #     DMA_2_node_group.append(i)
    # if pn == 3:
    #     DMA_3_node_group.append(i)
    # if pn == 4:
    #     DMA_4_node_group.append(i)
    # if pn == 5:
    #     DMA_5_node_group.append(i)

# print("DMA_1", len(DMA_1_node_group))
# print("DMA_2", len(DMA_2_node_group))
# print("DMA_3", len(DMA_3_node_group))
# print("DMA_4", len(DMA_4_node_group))
# print("DMA_5", len(DMA_5_node_group))

tk.close(ENpro)
tk.deleteproject(ENpro)

 
# Multipliers_DMA_1 = Pattern_DMA1
# Multipliers_DMA_2 = Pattern_DMA2
# Multipliers_DMA_3 = Pattern_DMA3
# Multipliers_DMA_4 = Pattern_DMA4
# Multipliers_DMA_5 = Pattern_DMA5

 
Multipliers_DMA_1 = []
Multipliers_DMA_2 = []
Multipliers_DMA_3 = []
Multipliers_DMA_4 = []
Multipliers_DMA_5 = []
for i in range (len(Pattern_DMA1)):
    # print(len(Pattern_DMA1))
    for j in range (12):
        Multipliers_DMA_1.append(Pattern_DMA1[i])
        Multipliers_DMA_2.append(Pattern_DMA2[i])
        Multipliers_DMA_3.append(Pattern_DMA3[i])
        Multipliers_DMA_4.append(Pattern_DMA4[i])
        Multipliers_DMA_5.append(Pattern_DMA5[i])



 
#-------------------------------------------------------------
 
def DEMAND_Observation(basedemand_scenario, period, ep_len):
    # print(len(basedemand_scenario))
    # print(len(basedemand_scenario[0]))
    Demand_obs_current = np.zeros(len(demand_group))
    if period < ep_len: 
        # 24 * 12 = 2016
        hour_pattern_current = period % 24
        week_pattern_current = math.floor(period/24) 
        for j in range (len(demand_group)):
            demand_pattern_index = Node_Pattern[j]
            if demand_pattern_index == 1:
                multiplier = Multipliers_DMA_1[hour_pattern_current]
            if demand_pattern_index == 2:
                multiplier = Multipliers_DMA_2[hour_pattern_current]
            if demand_pattern_index == 3:
                multiplier = Multipliers_DMA_3[hour_pattern_current]
            if demand_pattern_index == 4:
                multiplier = Multipliers_DMA_4[hour_pattern_current]
            if demand_pattern_index == 5:
                multiplier = Multipliers_DMA_5[hour_pattern_current]
            Demand_obs_current[j] = multiplier* basedemand_scenario[week_pattern_current][j]/2
   
    return Demand_obs_current



 
def REWARD_CALCULATION(EnergyCost, critical_nodes_pressure, period_number, ErrorConstr, pump_solution, action_pre, action_current):
    
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

    hourly_reward = -1*hourly_reward
    return hourly_reward 



 
def Objfunction(pump_solution, demand_obs_current, Tanklevel_Observation):
    # some sub-globalvariables
    EnergyCost = 0
    ErrorConstr = 0
    
    Ph = tk.createproject()
    
    tk.open(Ph, "D_town_mod_1h.inp", "D_town_mod_1h.rpt", "")
    
    
    # set the corresponding water demand scenario 
    # which is based on the period_number and Header
    for i in range (len(demand_group)):
        tk.setnodevalue(Ph, demand_group[i], tk.BASEDEMAND, demand_obs_current[i])
     
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
    
    # The pressure constraint is 25 m
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
        
        # control step 为 5min, i.e., 300s
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


##-----------------------------------------------------------------------------------------------------------------
## State vector standardization
Maxlevels = [6.75, 6.5, 5, 5.5, 4.5, 5.9, 4.7]
def tanklevel_standardization(Tanklevel_Observation): 
    standard_tanklevel = []
    for i in range (len(Tanklevel_Observation)):
        sd_level = (Tanklevel_Observation[i] - 0)/Maxlevels[i]
        standard_tanklevel.append(sd_level)
    return standard_tanklevel



##-----------------------------------------------------------------------------------------------------------------
## 
path_train = Header_base + Header_for_train    
workbook_train = xlrd.open_workbook(path_train)
data_sheet_train = workbook_train.sheet_by_index(0)  
rowNum_train = data_sheet_train.nrows
colNum_train = data_sheet_train.ncols

print("training sets", rowNum_train,colNum_train)    
cols_train = np.zeros((colNum_train,rowNum_train))

for i in range (colNum_train):
    cols_train[i] = data_sheet_train.col_values(i)





def training_scenario_generation():
    
    Training_scenario = []
    duration_week = 150

    for j in range (duration_week):
        
        AAA = cols_train[j] 
        AAA_multiply = []
        for m in range (len(AAA)):
            #aaa = AAA[m]*Multiplier_node_random[j*len(AAA)+m]
            aaa = AAA[m]*1
            AAA_multiply.append(aaa)
        Training_scenario.append(AAA_multiply)
    
    return Training_scenario


 

##-----------------------------------------------------------------------------------------------------------------
## Sub-Functions of Main Loop Module
    
# Main_Train part of the Main Loop Module

def Main_Loop_Training(State_vector, number, seed, uncertainty):
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_episodes", type=int, default=int(1000), help=" Maximum number of training episodes")
    # parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    # parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Beta", help="Beta or Gaussian") # Beta distribution is adopted here
    parser.add_argument("--batch_size", type=int, default=30, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=30, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=30, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=0.0001, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=0.0001, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.999, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default = 2, help="PPO parameter")
    
    
    # next, some tricks are adopted whereas others are not used
    parser.add_argument("--use_adv_norm", type=bool, default=False, help="Trick 1:advantage normalization")
    # state norm is not used here
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=False, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=False, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick 10: tanh activation function")

    args = parser.parse_args()
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    args.action_dim = 5
    args.max_action = 1
    args.min_action = 0
    args.max_episode_steps = 24*150
    
    # args.state_dim is identified based on the type of State_vector
    if State_vector == "SV1":
        args.state_dim = 12 # There are 7 tanks in D town and 5 pump stations in D town
    
        
    print("state vector type", State_vector)
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)
    
    # agent.load(f"./" + "PPO_models_SV1_5m_0/371_-12360.219990940825")
    # state_norm is not used here
    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
    
    workbook_curve = xlsxwriter.Workbook(Header_base + '/Convergence_curve_'+ State_vector +'_1h.xlsx')
    worksheet_curve = workbook_curve.add_worksheet('sheet1')
    
    for ep in range (args.max_train_episodes):
        ep_r = 0
        
        # initialize the initial tank levels randomly
        Tanklevel_Observation = []
        random_sampling = np.random.random(len(tank_group)) 
        for i in range  (len(tank_group)):
            init_tanklevel = Maxlevels[i] * random_sampling[i]
            Tanklevel_Observation.append(init_tanklevel)
        
        
        Basedemand_scenario = training_scenario_generation()
        action_pre = [1, 1, 1, 1, 1]
        # print("scenario data reading is finished, ", len(Basedemand_scenario), " next step: training")
      
        EP_pump_solution = []
        
        if args.use_reward_scaling:
            reward_scaling.reset()
        
        done = False
        EP_LEN = 24*150
        batch_number = 0
        
        for period in range(EP_LEN):
            Demand_obs_current = DEMAND_Observation(Basedemand_scenario, period, EP_LEN)

            Standard_tanklevel = tanklevel_standardization(Tanklevel_Observation)
            #print('current_tank level', Standard_tanklevel)

            Observation = np.append(Standard_tanklevel, action_pre)
            
            pump_solution, solution_logprob = agent.choose_action(Observation)
            
    
            # Interaction with EPANET: Hydraulic simulations
            EnergyCost, critical_nodes_pressure, final_tank_level, ERRORconstr = Objfunction(pump_solution, Demand_obs_current, Tanklevel_Observation)
            
            action_current = []
            for i in range (len(pump_solution)):
                if pump_solution[i] < 0.5:
                    action_ = 0
                else:
                    action_ = 1
                action_current.append(action_)

                
            # reward
            hourly_reward = REWARD_CALCULATION(EnergyCost, critical_nodes_pressure, period, ERRORconstr, pump_solution, action_pre, action_current)
            
            # next state vector St+1
            Next_tanklevel_Observation = final_tank_level 
            
            
            Nx_Sd_tanklevel_Observation = tanklevel_standardization(Next_tanklevel_Observation)
                     

            Observation_ = np.append(Nx_Sd_tanklevel_Observation, action_current)
            """ Important! The states of the tank levels are supposed to be updated for next decision periods """
            Tanklevel_Observation = final_tank_level
            action_pre = action_current
            
            
            ep_r += hourly_reward
            if args.use_reward_scaling:
                hourly_reward = reward_scaling(hourly_reward)
            
            if period == EP_LEN - 1:
                dw = False
                done = True
            else:
                dw = False
                done = False
                
            replay_buffer.store(Observation, pump_solution, solution_logprob, hourly_reward, Observation_, dw, done)
            
            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                batch_number = batch_number + 1
                agent.update(replay_buffer, ep)
                replay_buffer.count = 0
    
        agent.save(f"./" + "PPO_models" + "_" + State_vector + "_1h_" + uncertainty + "/" + str(ep) + "_" + str(ep_r/150))
        EP_total_Reward = []
        EP_total_Reward.append(ep_r/150)
        print("Episode", ep, "reward", ep_r/150)
        worksheet_curve.write_row(ep,0, EP_total_Reward)
    workbook_curve.close()


##-----------------------------------------------------------------------------------------------------------------
## General command

if __name__ == "__main__":
    import time
    starttime = time.time()
    
    # state vector
    State_vector = "SV1"  
    
    number = 1
    seed = 1
    
    # Training
    Main_Loop_Training(State_vector, number, seed, uncertainty)
    

    endtime = time.time()
    print ('The Running Time:' , endtime - starttime)
    print ("Run over!")







