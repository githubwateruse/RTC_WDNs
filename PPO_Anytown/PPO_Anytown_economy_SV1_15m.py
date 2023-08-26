# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 20:10:04 2022

@author: sp825
"""

""" control step is set at 30m, the inp file is supposed to be Anytown_15m """
""" the tstep is supposed to be 900 s """


import torch
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
import gym
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

Header_base ="D:/Anytown_PPO_economy7/"
Header_for_train = "LHS sample results/LhsRandom_results_for_training.xlsx"

""" There are two kinds of tarriffs: Standard vs Economy 7

Standard: 18.9 p/ kWh
Economy 7: 22.0 p/kWh (Avg unit Day Price)
           11.1 p/kWh (Avg unit Night Price, 11pm-6am)

"""

##-----------------------------------------------------------------------------------------------------------------
## 

# some global variables of Anytown_1h
PenaltyFactor = 7.5
Peakprice = 0.22
Offpeakprice = 0.111
RequiredPressure = 40
ErrorpenaltyFactor = 0
ActionPenalty = 1

#  
#-------------------------------------------------------------
ENpro = tk.createproject()
tk.open(ENpro, "Anytown_revised_30m.inp", "Anytown_revised_30m.rpt", "")




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

 
multipliers = [ 1.0, 1.0, 1.0,0.9, 0.9, 0.9, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 1.2, 1.2, 1.2, 1.3, 1.3, 1.3, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1]

Multipliers = []
for i in range (len(multipliers)):
    Multipliers.append(multipliers[i])
    Multipliers.append(multipliers[i])
    Multipliers.append(multipliers[i])
    Multipliers.append(multipliers[i])

print(len(Multipliers))
# print(Multipliers)

 
#-------------------------------------------------------------

 
def DEMAND_Observation(basedemand_scenario, period, ep_len): 
    Demand_obs_current = np.zeros(len(demand_group))
    if period < ep_len:
        hour_pattern_current = period % 96
        multiplier_current = Multipliers[hour_pattern_current]
        day_pattern_current = math.floor(period /96)
        
        for j in range (len(demand_group)):
            Demand_obs_current[j] = multiplier_current*basedemand_scenario[day_pattern_current][j]*1
        
    return Demand_obs_current

 
def REWARD_CALCULATION(EnergyCost, critical_nodes_pressure, period_number, ErrorConstr, pump_solution, action_pre, action_current):
    
    # 1) energy calculation
    # judging peaktime or offpeaktime by the parameter 'period'
    period_order = period_number % 96
    # For each day, the front 12 hours belong to offpeaktime, and the rest 12 hours belong to peak time
    if period_order < 48 and period_order > 19: 
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

     
    hourly_reward = -1*hourly_reward
    
    return hourly_reward 

 
def Objfunction(pump_solution, demand_obs_current, Tanklevel_Observation):
    # some sub-globalvariables
    EnergyCost = 0
    ErrorConstr = 0
    
    Ph = tk.createproject()
    tk.open(Ph, "Anytown_revised_15m.inp", "Anytown_revised_15m.rpt", "")
    
    
    # set the corresponding water demand scenario 
    # which is based on the period_number and Header
    
    for i in range (len(demand_group)):
        tk.setnodevalue(Ph, demand_group[i], tk.BASEDEMAND, demand_obs_current[i])
     
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
        
        if t % 900 == 0:
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


##-----------------------------------------------------------------------------------------------------------------
## State vector standardization

def demand_vector_standardization(Demand_Observation):
    # MeanDemand_group is used for demand vector standardization
    standard_demand_observation = []
    node_num = len(MeanDemand_group)
    for i in range (len(Demand_Observation)):
        a = i % node_num
        sd_node_demand = Demand_Observation[i]/MeanDemand_group[a]
        standard_demand_observation.append(sd_node_demand)
    return standard_demand_observation



def tanklevel_standardization(Tanklevel_Observation):
    # Tank levels are supposed to be between 10 and 35m
    # which is limited by the tank size of Anytown
    standard_tanklevel = []
    for i in range (len(Tanklevel_Observation)):
        sd_level = (Tanklevel_Observation[i] - 10)/25
        standard_tanklevel.append(sd_level)
    return standard_tanklevel


##-----------------------------------------------------------------------------------------------------------------
##  
path_train = Header_base + Header_for_train   # 
workbook_train = xlrd.open_workbook(path_train)
data_sheet_train = workbook_train.sheet_by_index(0) # 
rowNum_train = data_sheet_train.nrows
colNum_train = data_sheet_train.ncols

print("training sets", rowNum_train,colNum_train)    
cols_train = np.zeros((colNum_train,rowNum_train))

for i in range (colNum_train):
    cols_train[i] = data_sheet_train.col_values(i)



    

def training_scenario_generation(scenario_number):
    # It is called by the main function "Main_Train" 
    scenario_number = 0
    Training_scenario = []
    duration_day = 1000

    for j in range (duration_day):
        AAA = cols_train[duration_day *scenario_number + j] 
        AAA_multiply = []
        for m in range (len(AAA)):
            #aaa = AAA[m]*Multiplier_node_random[j*len(AAA)+m]
            aaa = AAA[m]*1
            AAA_multiply.append(aaa)
        Training_scenario.append(AAA_multiply)
    
    #for j in range (duration_day):
    #    Training_scenario.append(cols_train[duration_day *scenario_number + j] * Multiplier_day[j])
    return Training_scenario



def ONE_HOT_time(t):
    hr = t%24
    T = np.zeros(24)
    T[hr] = 1
    return T



##-----------------------------------------------------------------------------------------------------------------
## Sub-Functions of Main Loop Module
    
# Main_Train part of the Main Loop Module

def Main_Loop_Training(State_vector, number, seed):
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_episodes", type=int, default=int(500), help=" Maximum number of training episodes")
    # parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    # parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Beta", help="Beta or Gaussian") # Beta distribution is adopted here
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=100, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=30, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.999, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default =10, help="PPO parameter")
    
    
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
    
    args.action_dim = 1
    args.max_action = 1
    args.min_action = 0
    args.max_episode_steps = 96*1000
    
    # args.state_dim is identified based on the type of State_vector
    if State_vector == "SV1":
        args.state_dim = 4 # 3+19+24
    
        
    print("state vector type", State_vector)
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)
    
   
    # state_norm is not used here
    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
    
    workbook_curve = xlsxwriter.Workbook(Header_base + '/Convergence_curve_'+ State_vector +'_15m.xlsx')
    worksheet_curve = workbook_curve.add_worksheet('sheet1')
    
    for ep in range (args.max_train_episodes):
        ep_r = 0
        
        # initialize the initial tank levels randomly
        Tanklevel_Observation = np.random.random(3)*25 + 10 
        
        scenario_number = 0
        Basedemand_scenario = training_scenario_generation(scenario_number)
        action_pre = 1
        # print("scenario data reading is finished, ", len(Basedemand_scenario), " next step: training")
      
        EP_pump_solution = []
        
        if args.use_reward_scaling:
            reward_scaling.reset()
        
        done = False
        EP_LEN = 24000*4 #  
        # print("episode", ep, " started")
        for period in range(EP_LEN):
            Demand_obs_current = DEMAND_Observation(Basedemand_scenario, period, EP_LEN)

            Standard_demand_observation = demand_vector_standardization(Demand_obs_current)
            
            Standard_tanklevel = tanklevel_standardization(Tanklevel_Observation)
            #print('current_tank level', Standard_tanklevel)
            
            Tone_hot = ONE_HOT_time(period)
            
            
            Observation = np.append(Standard_tanklevel, action_pre)
            
            pump_solution, solution_logprob = agent.choose_action(Observation)
            
    
            # Interaction with EPANET: Hydraulic simulations
            EnergyCost, critical_nodes_pressure, final_tank_level, ERRORconstr = Objfunction(pump_solution, Demand_obs_current, Tanklevel_Observation)
            
            
            if pump_solution[0] < 0.5:
                action_current = 0
            else:
                action_current = 1
                
            # reward
            hourly_reward = REWARD_CALCULATION(EnergyCost, critical_nodes_pressure, period, ERRORconstr, pump_solution, action_pre, action_current)
            
            # next state vector St+1
            Next_tanklevel_Observation = final_tank_level 
            
            Demand_obs_next = DEMAND_Observation(Basedemand_scenario, period + 1, EP_LEN)
            Nx_Sd_tanklevel_Observation = tanklevel_standardization(Next_tanklevel_Observation)
            #print('next_tank level', Nx_Sd_tanklevel_Observation)
            Nx_Sd_demand_observation = demand_vector_standardization(Demand_obs_next)
            
            
            T_one_hot = ONE_HOT_time(period + 1)
            

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
                agent.update(replay_buffer, ep)
                replay_buffer.count = 0
    
        agent.save(f"./" + "PPO_models" + "_" + State_vector + "_15m/" + str(ep) + "_" + str(ep_r))
        EP_total_Reward = []
        EP_total_Reward.append(ep_r/1000)
        print("Episode", ep, "reward", ep_r/1000)
        worksheet_curve.write_row(ep,0, EP_total_Reward)
    workbook_curve.close()


##-----------------------------------------------------------------------------------------------------------------
## General command

if __name__ == "__main__":
    import time
    starttime = time.time()
    
    # state vector
    State_vector = "SV1" # 
    
    number = 1
    seed = 1
    
    # Training
    Main_Loop_Training(State_vector, number, seed)
    

    endtime = time.time()
    print ('The Running Time:' , endtime - starttime)
    print ("Run over!")



