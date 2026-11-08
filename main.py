from __future__ import print_function
import argparse
import os
import tqdm
from tqdm import trange
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.envs.amod_env import Scenario, AMoD
from src.algos.a2c_gnn import A2C
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum

def str2bool(v):
    # argparse type=bool treats any non-empty string as True, so "--flag False" -> True.
    # this parses the actual truthiness of the string instead.
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1', 'yes', 'y'):
        return True
    if v.lower() in ('false', 'f', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'expected a boolean value, got {v!r}')


parser = argparse.ArgumentParser(description='A2C-GNN')

# Simulator parameters
parser.add_argument('--json', type=str, default='nyc4x4',
                    help='json used for data')
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--demand_ratio', type=float, default=0.5, metavar='S',
                    help='demand_ratio (default: 0.5)')
parser.add_argument('--fleet_scale', type=float, default=1.0, metavar='S',
                    help='multiplier on initial fleet size (<1 binds supply, default: 1.0)')
parser.add_argument('--json_hr', type=int, default=7, metavar='S',
                    help='json_hr (default: 7)')
parser.add_argument('--json_tsetp', type=int, default=3, metavar='S',
                    help='minutes per timestep (default: 3min)')
parser.add_argument('--beta', type=float, default=0.5, metavar='S',
                    help='cost of rebalancing (default: 0.5)')


# Model parameters
parser.add_argument('--test', type=str2bool, default=False,
                    help='activates test mode for agent evaluation')
parser.add_argument('--directory', type=str, default='saved_files',
                    help='defines directory where to save files')
parser.add_argument('--max_episodes', type=int, default=16000, metavar='N',
                    help='number of episodes to train agent (default: 16k)')
parser.add_argument('--max_steps', type=int, default=60, metavar='N',
                    help='number of steps per episode (default: T=60)')
parser.add_argument('--CUDA', type=str2bool, default=True,
                    help='disables CUDA training')
parser.add_argument('--reward', type=str, default="revenue",
                    help='reward mode')
parser.add_argument('--baseline', type=str2bool, default=False,
                    help='non-RL baseline to evaluate instead of a trained policy: "none" = never '
                         'rebalance (serve only from the zone a vehicle is already in). '
                         'empty = use the trained RL policy. only used with --test True')
parser.add_argument('--early_stop', type=str2bool, default=False,
                    help='stop early when the smoothed train reward stops improving')
parser.add_argument('--patience', type=int, default=500, metavar='N',
                    help='episodes without improvement before stopping (default: 500)')
parser.add_argument('--min_delta', type=float, default=0.01, metavar='S',
                    help='relative improvement required to count, as a fraction of |best| (default: 0.01 = 1%%)')
parser.add_argument('--es_window', type=int, default=100, metavar='N',
                    help='moving-average window for early stopping (default: 100)')

#image saving
parser.add_argument('--save', type=str2bool, default=True,
                    help='disables saving reward file')

args = parser.parse_args()


# SAVE NAME NEED TO INCLUDE LAMBDA THO WITH LAMBA UNKNOWN CAN 
save_name = args.reward+"_"+str(args.demand_ratio)+"_"+str(args.fleet_scale)+"_"+str(args.seed)


if args.baseline:
    # baselines are policy-free, so key them by regime only (not by reward mode) and keep them
    # in a separate namespace so they never overwrite trained-agent logs
    save_name = "baseline_"+str(args.demand_ratio)+"_"+str(args.fleet_scale)

# --seed only reaches numpy (the Poisson demand draws) via Scenario; seed torch too so that
# policy init and Dirichlet action sampling are controlled. without this, runs are not
# reproducible and changing --seed does not actually vary the policy.

if args.test:
    args.seed = 42

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# set device
# would set mps but doesnt work with torch geometric
if (torch.cuda.is_available() and args.CUDA):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Define AMoD Simulator Environment
scenario = Scenario(json_file="data/scenario_"+args.json+".json", sd=args.seed, demand_ratio=args.demand_ratio, json_hr=args.json_hr, json_tstep=args.json_tsetp, fleet_scale=args.fleet_scale)
env = AMoD(scenario, beta=args.beta, reward_mode=args.reward)
# ensure per-json output dirs exist (torch.save won't create parent dirs)
os.makedirs(f"./{args.directory}/ckpt/{args.json}", exist_ok=True)
os.makedirs(f"./{args.directory}/rl_logs/{args.json}", exist_ok=True)
# Initialize A2C-GNN

model = A2C(env=env, input_size=21, device=device).to(device)

if not args.test:
    #######################################
    #############Training Loop#############
    #######################################

    #Initialize lists for logging
    log = {'train_reward': [],
           'train_served_demand': [],
           'train_reb_cost': [],
           'train_revenue': [],
           'train_profit': []}
    
    train_episodes = args.max_episodes #set max number of training episodes
    T = args.max_steps #set episode length
    epochs = trange(train_episodes) #epoch iterator
    best_reward = -np.inf #set best reward
    model.train() #set model in train mode
    # early stopping state: watch a moving average of train reward (raw episode reward is far
    # too noisy), and require a RELATIVE improvement so one threshold works across reward modes
    # whose scales differ by orders of magnitude (revenue ~1e3 vs worst ~1e1).
    best_ma = -np.inf
    episodes_since_improve = 0
    stopped_at = train_episodes

    for i_episode in epochs:
        obs = env.reset() #initialize environment
        episode_reward = 0
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        episode_revenue = 0
        episode_operating_cost = 0
        for step in range(T):
            # take matching step (Step 1 in paper)
            obs, paxreward, done, info = env.pax_step(paxAction=env.greedy_matching())
            episode_reward += paxreward
            # use GNN-RL policy (Step 2 in paper)
            action_rl = model.select_action(obs)
            # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
            desiredAcc = {env.region[i]: int(action_rl[i] *dictsum(env.acc,env.time+1))for i in range(len(env.region))}
            # solve minimum rebalancing distance problem (Step 3 in paper)
            rebAction = solveRebFlow(env, desiredAcc)
            # Take action in environment
            new_obs, rebreward, done, info = env.reb_step(rebAction)
            episode_reward += rebreward
            # Store the transition in memory
            model.rewards.append(paxreward + rebreward)
            # track performance over episode
            episode_served_demand += info['served_demand']
            episode_rebalancing_cost += info['rebalancing_cost']
            episode_revenue += info['revenue']
            episode_operating_cost += info['operating_cost']  # includes rebalancing cost
            # stop episode if terminating conditions are met
            if done:
                break
        # perform on-policy backprop
        model.training_step()

        # Send current statistics to screen
        epochs.set_description(f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}")
        # Checkpoint best performing model
        if episode_reward >= best_reward:

            # TODO check the save logic and able to change the name of save and load in
            model.save_checkpoint(path=f"./{args.directory}/ckpt/"+args.json+"/"+save_name+".pth")
            best_reward = episode_reward
        # Log KPIs
        log['train_reward'].append(episode_reward)
        log['train_served_demand'].append(episode_served_demand)
        log['train_reb_cost'].append(episode_rebalancing_cost)
        log['train_revenue'].append(episode_revenue)
        log['train_profit'].append(episode_revenue - episode_operating_cost)
        model.log(log, path=f"./{args.directory}/rl_logs/"+args.json+"/"+save_name+".pth")

        # early stopping on the smoothed train reward
        if args.early_stop and len(log['train_reward']) >= args.es_window:
            ma = float(np.mean(log['train_reward'][-args.es_window:]))
            if best_ma == -np.inf:
                best_ma = ma
                episodes_since_improve = 0
            elif ma > best_ma + args.min_delta * abs(best_ma):
                best_ma = ma
                episodes_since_improve = 0
            else:
                episodes_since_improve += 1
            if episodes_since_improve >= args.patience:
                stopped_at = i_episode + 1
                print(f"\nEarly stop at episode {stopped_at}: no >{args.min_delta:.1%} improvement "
                      f"in {args.patience} episodes (best {args.es_window}-ep avg = {best_ma:.2f})")
                break

    # Plot reward and revenue per episode (twin axes: reward and revenue can be on
    # different scales in non-revenue modes, e.g. served/worst)
    if args.save:
        window = min(50, len(log['train_reward']))
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(log['train_reward'], color='lightblue', label='Reward')
        smoothed_r = np.convolve(log['train_reward'], np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(log['train_reward'])), smoothed_r, color='darkblue', label=f'Reward ({window}-ep avg)')
        ax1.set_ylabel('Reward', color='darkblue')

        ax2 = ax1.twinx()
        ax2.plot(log['train_revenue'], color='lightcoral', label='Revenue')
        smoothed_v = np.convolve(log['train_revenue'], np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(log['train_revenue'])), smoothed_v, color='darkred', label=f'Revenue ({window}-ep avg)')
        ax2.plot(log['train_profit'], color='lightgreen', label='Profit')
        smoothed_p = np.convolve(log['train_profit'], np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(log['train_profit'])), smoothed_p, color='darkgreen', label=f'Profit ({window}-ep avg)')
        ax2.set_ylabel('Revenue / Profit')

        ax1.set_xlabel('Episode')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
        plt.title(f'Training Performance ({save_name})')
        plt.tight_layout()
        plt.savefig(f"./{args.directory}/{save_name}.png", dpi=150)
        print(f"Plot saved to ./{args.directory}/{save_name}.png")

else:
    # Load pre-trained model (baselines are policy-free, so there is no checkpoint to load)
    #TODO load with user specified name
    if not args.baseline:
        model.load_checkpoint(path=f"./{args.directory}/ckpt/"+args.json+"/"+save_name+".pth")
    else:
        print(f"evaluating non-RL baseline: {args.baseline}")
    test_episodes = args.max_episodes #set number of evaluation episodes
    T = args.max_steps #set episode length (must match training horizon)
    epochs = trange(test_episodes) #epoch iterator
    model.eval()
    #Initialize lists for logging
    log = {'test_reward': [],
           'test_served_demand': [],
           'test_reb_cost': [],
           'test_revenue': [],
           'test_profit': [],
           'test_service_rate': [],
           'test_worst_ratio': [],
           'test_avg_ride_length': [],
           # per-zone served/demand for EVERY episode (one list of len(region) each). logging
           # both (rather than the ratio) lets us pool correctly post-hoc as
           # sum(served)/sum(demand) instead of averaging per-episode ratios, and keeps the
           # absolute volumes for the spatial comparison figure.
           'test_zone_served': [],
           'test_zone_demand': [],
           'test_zone_ids': list(env.region)}
    for episode in epochs:
        episode_reward = 0
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        episode_revenue = 0
        episode_operating_cost = 0
        episode_total_demand = 0
        episode_ride_time = 0
        # accumulate per-zone service over the WHOLE episode, so the worst-zone ratio is
        # "worst-served zone across the morning", not an average of noisy per-step minima
        zone_served = {n: 0.0 for n in env.region}
        zone_demand = {n: 0.0 for n in env.region}
        obs = env.reset()
        for step in range(T):
            # take matching step (Step 1 in paper)
            obs, paxreward, done, info = env.pax_step(paxAction=env.greedy_matching())
            episode_reward += paxreward
            for n in env.region:
                zone_served[n] += env.servedZone[n]
                zone_demand[n] += env.demandZone[n]
            if args.baseline:
                # non-RL baseline: never rebalance. vehicles only serve demand originating in
                # whatever zone they already happen to be sitting in.
                rebAction = [0.0] * len(env.edges)
            else:
                # use GNN-RL policy (Step 2 in paper)
                action_rl = model.select_action(obs)
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                desiredAcc = {env.region[i]: int(action_rl[i] *dictsum(env.acc,env.time+1))for i in range(len(env.region))}
                # solve minimum rebalancing distance problem (Step 3 in paper)
                rebAction = solveRebFlow(env, desiredAcc)
            # Take action in environment
            new_obs, rebreward, done, info = env.reb_step(rebAction)
            episode_reward += rebreward
            # track performance over episode
            episode_served_demand += info['served_demand']
            episode_rebalancing_cost += info['rebalancing_cost']
            episode_revenue += info['revenue']
            episode_operating_cost += info['operating_cost']  # includes rebalancing cost
            episode_total_demand += info['total_demand']
            episode_ride_time += info['ride_time']
            if done:
                break
        # episode-level derived metrics
        served_zones = [zone_served[n]/zone_demand[n] for n in env.region if zone_demand[n] > 0]
        worst_ratio = min(served_zones) if served_zones else 0.0
        service_rate = episode_served_demand/episode_total_demand if episode_total_demand > 0 else 0.0
        # ride length in minutes (demandTime is in timesteps)
        avg_ride_length = (episode_ride_time/episode_served_demand*args.json_tsetp) if episode_served_demand > 0 else 0.0

        epochs.set_description(f"Episode {episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}")
        # Log KPIs
        log['test_reward'].append(episode_reward)
        log['test_served_demand'].append(episode_served_demand)
        log['test_reb_cost'].append(episode_rebalancing_cost)
        log['test_revenue'].append(episode_revenue)
        log['test_profit'].append(episode_revenue - episode_operating_cost)
        log['test_service_rate'].append(service_rate)
        log['test_worst_ratio'].append(worst_ratio)
        log['test_avg_ride_length'].append(avg_ride_length)
        log['test_zone_served'].append([zone_served[n] for n in env.region])
        log['test_zone_demand'].append([zone_demand[n] for n in env.region])
        model.log(log, path=f"./{args.directory}/rl_logs/{args.json}/{save_name}_test.pth")

    # summary across evaluation episodes -> one row of the results table
    print(f"\n=== EVAL  json={args.json}  reward={args.reward}  "
          f"demand_ratio={args.demand_ratio}  fleet_scale={args.fleet_scale}  "
          f"({len(log['test_reward'])} episodes) ===")
    for k_ in ['test_reward', 'test_revenue', 'test_profit', 'test_served_demand',
               'test_service_rate', 'test_worst_ratio', 'test_reb_cost', 'test_avg_ride_length']:
        v = np.array(log[k_])
        print(f"{k_.replace('test_',''):>18}: {v.mean():12.3f}  +/- {v.std():.3f}")
