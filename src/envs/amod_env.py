"""
Autonomous Mobility-on-Demand Environment
-----------------------------------------
This file contains the specifications for the AMoD system simulator. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks (Section III-C in the paper)
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks (Section III-C in the paper)
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""
from collections import defaultdict
import numpy as np
import networkx as nx
from copy import deepcopy
import json

class AMoD:
    # initialization
    def __init__(self, scenario, beta=0.2, reward_mode="revenue"): # updated to take scenario and beta (cost for rebalancing) as input
        self.scenario = deepcopy(scenario) # I changed it to deep copy so that the scenario input is not modified by env
        self.reward_mode = reward_mode # which matching objective to optimize: revenue | served | worst | blend
        # service objectives (served/worst) are cost-blind: they ignore rebalancing cost in
        # the reward so they reposition freely. cost-aware objectives (revenue/blend) pay it.
        # NOTE: rebalancing cost is still MEASURED in info for every mode -- only the reward
        # signal changes -- so we can still report deadhead cost for the cost-blind agents.
        self.charge_reb_cost = reward_mode not in ("served", "worst")
        self.rng = np.random.default_rng(scenario.sd)
        self.G = scenario.G # Road Graph: node - region, edge - connection of regions, node attr: 'accInit', edge attr: 'time'
        self.demandTime = self.scenario.demandTime
        self.rebTime = self.scenario.rebTime
        self.time = 0 # current time
        self.tf = scenario.tf # final time
        self.demand = defaultdict(dict) # demand
        self.depDemand = dict()
        self.arrDemand = dict()
        self.region = list(self.G) # set of regions
        for i in self.region:
            self.depDemand[i] = defaultdict(float)
            self.arrDemand[i] = defaultdict(float)
            
        self.price = defaultdict(dict) # price
        for i,j,t,d,p in scenario.tripAttr: # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i,j][t] = d
            self.price[i,j][t] = p
            self.depDemand[i][t] += d
            self.arrDemand[i][t+self.demandTime[i,j][t]] += d
        self.acc = defaultdict(dict) # number of vehicles within each region, key: i - region, t - time
        self.dacc = defaultdict(dict) # number of vehicles arriving at each region, key: i - region, t - time
        self.rebFlow = defaultdict(dict) # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
        self.paxFlow = defaultdict(dict) # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
        self.edges = [] # set of rebalancing edges
        self.nregion = len(scenario.G) # number of regions
        for i in self.G:
            self.edges.append((i,i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.nedge = [len(self.G.out_edges(n))+1 for n in self.region] # number of edges leaving each region        
        for i,j in self.G.edges:
            self.G.edges[i,j]['time'] = self.rebTime[i,j][self.time]
            self.rebFlow[i,j] = defaultdict(float)
        for i,j in self.demand:
            self.paxFlow[i,j] = defaultdict(float)            
        for n in self.region:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)   
        self.beta = beta * scenario.tstep
        t = self.time
        self.servedDemand = defaultdict(dict)
        for i,j in self.demand:
            self.servedDemand[i,j] = defaultdict(float)
        
        self.N = len(self.region) # total number of cells
        
        # add the initialization of info here
        self.info = dict.fromkeys(['revenue', 'served_demand', 'rebalancing_cost', 'operating_cost'], 0)
        self.reward = 0
        # observation: current vehicle distribution, time, future arrivals, demand        
        self.obs = (self.acc, self.time, self.dacc, self.demand)

    def greedy_matching(self):
        t = self.time
        flow = defaultdict(float)
        available = {i: int(self.acc[i][t]) for i in self.region}

        # build list of individual requests from the demand buckets
        requests = []
        for (i, j) in self.demand:
            if t not in self.demand[i, j]:
                continue
            count = int(round(self.demand[i, j][t]))
            requests.extend([(i, j)] * count)

        # shuffle to simulate random arrival order within the timestep
        self.rng.shuffle(requests)

        # serve in order: each request gets a car if one is available in its origin zone
        for i, j in requests:
            if available[i] > 0:
                flow[i, j] += 1
                available[i] -= 1
            # else: request is dropped (no car available in origin zone)

        paxAction = [flow[i, j] if (i, j) in flow else 0 for i, j in self.edges]
        return paxAction

    # pax step
    def pax_step(self, paxAction):
        t = self.time
        self.reward = 0
        for i in self.region:
            self.acc[i][t+1] = self.acc[i][t]
        self.info['served_demand'] = 0 # initialize served demand
        self.info["operating_cost"] = 0 # initialize operating cost
        self.info['revenue'] = 0
        self.info['rebalancing_cost'] = 0
        self.paxAction = paxAction
        # per-zone bookkeeping for worst-zone / blend rewards (independent of revenue/served accounting)
        self.servedZone = {n: 0.0 for n in self.region} # pax served originating from each zone this step
        self.demandZone = {n: 0.0 for n in self.region} # total demand originating from each zone this step
        for (oi, oj) in self.demand:
            if t in self.demand[oi, oj]:
                self.demandZone[oi] += self.demand[oi, oj][t]
        # serving passengers

        for k in range(len(self.edges)):
            i,j = self.edges[k]
            if (i,j) not in self.demand or t not in self.demand[i,j] or self.paxAction[k]<1e-3:
                continue
            # I moved the min operator above, since we want paxFlow to be consistent with paxAction
            assert paxAction[k] < self.acc[i][t+1] + 1e-3
            self.paxAction[k] = min(self.acc[i][t+1], paxAction[k])            
            self.servedDemand[i,j][t] = self.paxAction[k]
            self.paxFlow[i,j][t+self.demandTime[i,j][t]] = self.paxAction[k]
            self.info["operating_cost"] += self.demandTime[i,j][t]*self.beta*self.paxAction[k]
            self.acc[i][t+1] -= self.paxAction[k]
            self.info['served_demand'] += self.servedDemand[i,j][t]
            self.servedZone[i] += self.paxAction[k]
            self.dacc[j][t+self.demandTime[i,j][t]] += self.paxFlow[i,j][t+self.demandTime[i,j][t]]
            self.reward += self.paxAction[k]*(self.price[i,j][t] - self.demandTime[i,j][t]*self.beta)            
            self.info['revenue'] += self.paxAction[k]*(self.price[i,j][t])

        # select the matching reward according to the configured objective.
        # revenue/served info above is always populated so every mode can be logged;
        # only the reward signal handed to the agent changes here.
        # no reward scaling: served/worst are cost-blind (see charge_reb_cost), so there is no
        # competing rebalancing cost the matching signal has to be balanced against.
        if self.reward_mode == "revenue":
            pass  # self.reward already holds revenue minus per-trip operating cost
        elif self.reward_mode == "served":
            self.reward = self.info['served_demand']  # price-blind count of served pax
        elif self.reward_mode == "worst":
            # service rate of the worst-served zone (min over zones that had demand this step)
            ratios = [self.servedZone[n] / self.demandZone[n]
                      for n in self.region if self.demandZone[n] > 0]
            self.reward = min(ratios) if ratios else 0.0
        else:
            raise ValueError(f"unknown reward_mode: {self.reward_mode!r} (expected revenue|served|worst)")

        self.obs = (self.acc, self.time, self.dacc, self.demand) # for acc, the time index would be t+1, but for demand, the time index would be t
        done = False # if passenger matching is executed first
        return self.obs, max(0,self.reward), done, self.info
    
    # reb step
    def reb_step(self, rebAction):
        t = self.time
        self.reward = 0 # reward is calculated from before this to the next rebalancing, we may also have two rewards, one for pax matching and one for rebalancing
        self.rebAction = rebAction      
        # rebalancing
        for k in range(len(self.edges)):
            i,j = self.edges[k]    
            if (i,j) not in self.G.edges:
                continue
            # TODO: add check for actions respecting constraints? e.g. sum of all action[k] starting in "i" <= self.acc[i][t+1] (in addition to our agent action method)
            # update the number of vehicles
            self.rebAction[k] = min(self.acc[i][t+1], rebAction[k]) 
            self.rebFlow[i,j][t+self.rebTime[i,j][t]] = self.rebAction[k]       
            self.acc[i][t+1] -= self.rebAction[k] 
            self.dacc[j][t+self.rebTime[i,j][t]] += self.rebFlow[i,j][t+self.rebTime[i,j][t]]   
            cost = self.rebTime[i,j][t]*self.beta*self.rebAction[k]
            self.info['rebalancing_cost'] += cost  # always measured, for all modes
            self.info["operating_cost"] += cost
            if self.charge_reb_cost:               # only cost-aware objectives pay it in the reward
                self.reward -= cost
        # arrival for the next time step, executed in the last state of a time step
        # this makes the code slightly different from the previous version, where the following codes are executed between matching and rebalancing        
        for k in range(len(self.edges)):
            i,j = self.edges[k]    
            if (i,j) in self.rebFlow and t in self.rebFlow[i,j]:
                self.acc[j][t+1] += self.rebFlow[i,j][t]
            if (i,j) in self.paxFlow and t in self.paxFlow[i,j]:
                self.acc[j][t+1] += self.paxFlow[i,j][t] # this means that after pax arrived, vehicles can only be rebalanced in the next time step, let me know if you have different opinion
            
        self.time += 1
        self.obs = (self.acc, self.time, self.dacc, self.demand) # use self.time to index the next time step
        for i,j in self.G.edges:
            self.G.edges[i,j]['time'] = self.rebTime[i,j][self.time]
        done = (self.tf == t+1) # if the episode is completed
        return self.obs, self.reward, done, self.info
    
    def reset(self):
        # reset the episode
        self.acc = defaultdict(dict)
        self.dacc = defaultdict(dict)
        self.rebFlow = defaultdict(dict)
        self.paxFlow = defaultdict(dict)
        self.edges = []
        for i in self.G:
            self.edges.append((i,i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.demand = defaultdict(dict) # demand
        self.price = defaultdict(dict) # price
        tripAttr = self.scenario.get_random_demand(reset=True)
        self.regionDemand= defaultdict(dict)
        for i,j,t,d,p in tripAttr: # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i,j][t] = d
            self.price[i,j][t] = p
            if t not in self.regionDemand[i]:
                self.regionDemand[i][t] = 0
            else:
                self.regionDemand[i][t] +=d
            
        self.time = 0
        for i,j in self.G.edges:
            self.rebFlow[i,j] = defaultdict(float)
            self.paxFlow[i,j] = defaultdict(float)            
        for n in self.G:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float) 
        t = self.time
        for i,j in self.demand:
            self.servedDemand[i,j] = defaultdict(float)
         # TODO: define states here
        self.obs = (self.acc, self.time, self.dacc, self.demand)      
        self.reward = 0
        return self.obs
   
    
    
class Scenario:
    def __init__(self, N1=2, N2=4, tf=60, sd=None, ninit=5, tripAttr=None, demand_input=None, demand_ratio = None,
                 trip_length_preference = 0.25, grid_travel_time = 1, fix_price=True, alpha = 0.2, json_file = None, json_hr = 9, json_tstep = 2, varying_time=False, json_regions = None, fleet_scale = 1.0):
        # fleet_scale: multiplier on the initial fleet size (1.0 = as-specified, <1 shrinks the
        # fleet to create a supply-binding regime, >1 loosens supply)
        self.fleet_scale = fleet_scale
        # trip_length_preference: positive - more shorter trips, negative - more longer trips
        # grid_travel_time: travel time between grids
        # demand_input： list - total demand out of each region, 
        #          float/int - total demand out of each region satisfies uniform distribution on [0, demand_input]
        #          dict/defaultdict - total demand between pairs of regions
        # demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        # static_demand will then be sampled according to a Poisson distribution
        # alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        self.sd = sd
        if sd != None:
            np.random.seed(self.sd)
        if json_file == None:    
            self.varying_time = varying_time
            self.is_json = False
            self.alpha = alpha
            self.trip_length_preference = trip_length_preference
            self.grid_travel_time = grid_travel_time
            self.demand_input = demand_input
            self.fix_price = fix_price
            self.N1 = N1
            self.N2 = N2
            self.G = nx.complete_graph(N1*N2)
            self.G = self.G.to_directed()
            self.demandTime = dict()
            self.rebTime = dict()
            self.edges = list(self.G.edges) + [(i,i) for i in self.G.nodes]
            for i,j in self.edges:
                self.demandTime[i,j] = defaultdict(lambda:(abs(i//N1-j//N1) + abs(i%N1-j%N1))*grid_travel_time)
                self.rebTime[i,j] = defaultdict(lambda:(abs(i//N1-j//N1) + abs(i%N1-j%N1))*grid_travel_time)
            
            for n in self.G.nodes:
                self.G.nodes[n]['accInit'] = int(ninit * self.fleet_scale)
            self.tf = tf
            self.demand_ratio = defaultdict(list)
            
            if demand_ratio == None or type(demand_ratio) == list:            
                for i,j in self.edges:
                    if type(demand_ratio) == list:
                        self.demand_ratio[i,j] = list(np.interp(range(0,tf), np.arange(0,tf+1, tf/(len(demand_ratio)-1)), demand_ratio))+[demand_ratio[-1]]*tf
                    else:
                        self.demand_ratio[i,j] = [1]*(tf+tf)
            else:
                for i,j in self.edges:
                    if (i,j) in demand_ratio:
                        self.demand_ratio[i,j] = list(np.interp(range(0,tf), np.arange(0,tf+1, tf/(len(demand_ratio[i,j])-1)), demand_ratio[i,j]))+[1]*tf
                    else:
                        self.demand_ratio[i,j] = list(np.interp(range(0,tf), np.arange(0,tf+1, tf/(len(demand_ratio['default'])-1)), demand_ratio['default']))+[1]*tf
            if self.fix_price: # fix price
                self.p = defaultdict(dict)
                for i,j in self.edges:
                    self.p[i,j] = (np.random.rand()*2+1)*(self.demandTime[i,j][0]+1)
            if tripAttr != None: # given demand as a defaultdict(dict)
                self.tripAttr = deepcopy(tripAttr)
            else:
                self.tripAttr = self.get_random_demand() # randomly generated demand
            
        
        else:
            self.varying_time = varying_time
            self.is_json = True
            with open(json_file,"r") as file:
                data = json.load(file)
            self.tstep = json_tstep
            self.N1 = data["nlat"]
            self.N2 = data["nlon"]
            self.demand_input = defaultdict(dict)
            self.json_regions = json_regions
            
            if json_regions != None:
                self.G = nx.complete_graph(json_regions)
            elif 'region' in data:
                self.G = nx.complete_graph(data['region'])
            else:
                self.G = nx.complete_graph(self.N1*self.N2)
            self.G = self.G.to_directed()
            self.p = defaultdict(dict)
            self.alpha = 0
            self.demandTime = defaultdict(dict)
            self.rebTime = defaultdict(dict)
            self.json_start = json_hr * 60
            self.tf = tf
            self.edges = list(self.G.edges) + [(i,i) for i in self.G.nodes]

                    
            for i,j in self.demand_input:
                self.demandTime[i,j] = defaultdict(int)
                self.rebTime[i,j] = 1
                
            for item in data["demand"]: 
                t,o,d,v,tt,p = item["time_stamp"], item["origin"], item["destination"], item["demand"], item["travel_time"], item["price"]
                if json_regions!= None and (o not in json_regions or d not in json_regions):
                    continue
                if (o,d) not in self.demand_input:
                    self.demand_input[o,d],self.p[o,d],self.demandTime[o,d] = defaultdict(float), defaultdict(float),defaultdict(float)
                    
                self.demand_input[o,d][(t-self.json_start)//json_tstep] += v*demand_ratio
                self.p[o,d][(t-self.json_start)//json_tstep] += p*v*demand_ratio
                self.demandTime[o,d][(t-self.json_start)//json_tstep] += tt*v*demand_ratio/json_tstep
            
            
            for o,d in self.edges:
                for t in range(0,tf*2):
                    if t in self.demand_input[o,d]:
                        self.p[o,d][t] /= self.demand_input[o,d][t]                    
                        self.demandTime[o,d][t] /= self.demand_input[o,d][t]
                        self.demandTime[o,d][t] = max(int(round(self.demandTime[o,d][t])),1)
                    else:
                        self.demand_input[o,d][t] = 0
                        self.p[o,d][t] = 0
                        self.demandTime[o,d][t] = 0
            
            for item in data["rebTime"]:
                hr,o,d,rt = item["time_stamp"], item["origin"], item["destination"], item["reb_time"]
                if json_regions!= None and (o not in json_regions or d not in json_regions):
                    continue
                if varying_time:
                    t0 = int((hr*60 - self.json_start)//json_tstep)
                    t1 = int((hr*60 + 60 - self.json_start)//json_tstep)
                    for t in range(t0,t1):
                        self.rebTime[o,d][t] = max(int(round(rt/json_tstep)),1)
                else:
                    if hr == json_hr:
                        for t in range(0,tf+1):
                            self.rebTime[o,d][t] = max(int(round(rt/json_tstep)),1)
            
            for item in data["totalAcc"]:
                hr, acc = item["hour"], item["acc"]
                if hr == json_hr+int(round(json_tstep/2*tf/60)):
                    for n in self.G.nodes:
                        self.G.nodes[n]['accInit'] = int(acc/len(self.G) * self.fleet_scale)
            self.tripAttr = self.get_random_demand()
                
        
        
        
        
    def get_random_demand(self, reset = False):        
        # generate demand and price
        # reset = True means that the function is called in the reset() method of AMoD enviroment,
        #   assuming static demand is already generated
        # reset = False means that the function is called when initializing the demand
        
        demand = defaultdict(dict)
        price = defaultdict(dict)        
        tripAttr = []
        
        # converting demand_input to static_demand
        # skip this when resetting the demand
        # if not reset:
        if self.is_json:
            for t in range(0,self.tf*2):
                for i,j in self.edges:                
                    if (i,j) in self.demand_input and t  in self.demand_input[i,j]:
                        demand[i,j][t] = np.random.poisson(self.demand_input[i,j][t])
                        price[i,j][t] = self.p[i,j][t]
                    else:
                        demand[i,j][t] = 0
                        price[i,j][t] = 0
                    tripAttr.append((i,j,t,demand[i,j][t],price[i,j][t]))
        else:
            self.static_demand = dict()            
            region_rand = (np.random.rand(len(self.G))*self.alpha*2+1-self.alpha) 
            if type(self.demand_input) in [float, int, list, np.array]:
                
                if type(self.demand_input) in [float, int]:            
                    self.region_demand = region_rand * self.demand_input  
                else:
                    self.region_demand = region_rand * np.array(self.demand_input)
                for i in self.G.nodes:
                    J = [j for _,j in self.G.out_edges(i)]
                    prob = np.array([np.math.exp(-self.rebTime[i,j][0]*self.trip_length_preference) for j in J])
                    prob = prob/sum(prob)
                    for idx in range(len(J)):
                        self.static_demand[i,J[idx]] = self.region_demand[i] * prob[idx]
            elif type(self.demand_input) in [dict, defaultdict]:
                for i,j in self.edges:
                    self.static_demand[i,j] = self.demand_input[i,j] if (i,j) in self.demand_input else self.demand_input['default']
                    
                    self.static_demand[i,j] *= region_rand[i]
            else:
                raise Exception("demand_input should be number, array-like, or dictionary-like values")
            
            # generating demand and prices
            if self.fix_price:
                p = self.p
            for t in range(0,self.tf*2):
                for i,j in self.edges:                
                    demand[i,j][t] = np.random.poisson(self.static_demand[i,j]*self.demand_ratio[i,j][t])
                    if self.fix_price:
                        price[i,j][t] = p[i,j]
                    else:
                        price[i,j][t] = min(3,np.random.exponential(2)+1)*self.demandTime[i,j][t]
                    tripAttr.append((i,j,t,demand[i,j][t],price[i,j][t]))

        return tripAttr

