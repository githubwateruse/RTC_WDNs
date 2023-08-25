# Introduction
Real-time pump scheduling in water distribution networks using deep reinforcement learning

Pump scheduling in water distribution networks influences energy efficiency and water supply reliability. Conventional optimization methods usually face challenges in intensive computational requirements and water demand uncertainty handling. This study presents a deep reinforcement learning method, Proximal Policy Optimization (PPO), for real-time pump scheduling in water distribution networks. The PPO agents are trained to develop off-line policies in advance, avoiding the online optimization process during the scheduling period. 

The PPO algorithm is provided by https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/5.PPO-continuous

The hydraulic simulation of the water distribution networks is based on the EPANET engine. The Python packages related to the EPANET engine is provided by https://github.com/OpenWaterAnalytics/EPANET


# INP files
The modified Anytown and the modified D-town networks are used as the test cases. The Inp flies of the Anytown and the D-town networks are listed as below.  
Anytown_revised_1h.inp: The Inp file of the Anytown network. The scheduling step size is 1 hour.  
Anytown_revised_15m.inp: The Inp file of the Anytown network. The scheduling step size is 15 min.  
D_town_mod_1h.inp: The Inp file of the D-town network. The scheduling step size is 1 hour.  
D_town_mod_5m.inp: The Inp file of the D-town network. The scheduling step size is 5 min.  

# Dependencies
python==3.8.0  
numpy==1.23.4  
pytorch==1.13.0  
owa-epanet==2.2.4  
matplotlib==3.6.1  
scipy==1.9.3  
pyDOE==0.3.8  
xlrd==1.2.0  
xlsxwriter==3.0.3  



