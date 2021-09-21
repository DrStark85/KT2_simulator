# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 01:26:07 2021

@author: olajo
"""

import numpy as np
import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
# import matplotlib.pyplot as plt
# %matplotlib inline
# ximport collections

# Importing statblocks, filling nans and returning floats into ints (columns with nan are made into floats when importing from csv)
operative_statblocks = pd.read_csv('KT2_operative_statblocks.csv').fillna(0)
operative_statblocks.iloc[:,3:11] = operative_statblocks.iloc[:,3:11].values.astype(int)
# float_cols = operative_statblocks.iloc[3:11].values
# float_cols = float_cols.astype(int)
weapon_statblocks = pd.read_csv('KT2_weapon_statblocks.csv').fillna(0)
weapon_statblocks.iloc[:,4:20] = weapon_statblocks.iloc[:,4:20].values.astype(int)
print(operative_statblocks.iloc[:,0:4])
print(weapon_statblocks.iloc[:,0:4])


# Basic function for rolling any number of d6, returns number of normal successes and number of ones, optionally also number of crits
def d6(num_dice, threshold, crit_threshold=0):
    roll = np.random.randint(1, 7, num_dice)
    n = (roll >= threshold).sum()
    n_ones = (roll == 1).sum()
    if crit_threshold != 0:
        n_crit = (roll >= crit_threshold).sum()
        n_normal = n - n_crit
        return np.array([n_normal, n_ones, n_crit])
    else:
        return [n, n_ones]

def simulator(operative, target, operative_first=1, FF=1, turns_max=4, random_order=1):
    
    turn_counter = 0
    operative_HP = operative['W']
    target_HP = target['W']
    while operative_HP > 0 and target_HP > 0 and turn_counter < turns_max: # This loops over turns
        turn_counter+=1

        # print('Round ' + str(round_counter))
        
        round_marker = 0
        # If operative goes first, make its round now
        if operative_first == 1:
            round_marker +=1
            whose_turn = 'op'
            dmg = dmg_calculator(operative, target, FF, operative_HP, target_HP) # Damage dealt here is only calculated turn by turn, this is why in CC, HP has to be checked after every strike to see if any fighter is down and if so, break
            target_HP += -dmg[0]
            operative_HP += -dmg[1]
            if target_HP <= 0 or operative_HP <= 0:
                break
        
        # Make the target's round
        round_marker += 1
        whose_turn = 'ta'
        dmg = dmg_calculator(target, operative, FF, target_HP, operative_HP) # Damage dealt here is only calculated turn by turn, this is why in CC, HP has to be checked after every strike to see if any fighter is down and if so, break
        operative_HP += -dmg[0]
        target_HP += -dmg[1]
        if target_HP <= 0 or operative_HP <= 0:
            break
        
        # If operative goes last, make its round now
        if operative_first == 0:
            round_marker += 1
            whose_turn = 'op'
            dmg = dmg_calculator(operative, target, FF, operative_HP, target_HP) # Damage dealt here is only calculated turn by turn, this is why in CC, HP has to be checked after every strike to see if any fighter is down and if so, break
            target_HP += -dmg[0]
            operative_HP += -dmg[1]
            if target_HP <= 0 or operative_HP <= 0:
                break
        
        if random_order == 1: # If who acts first is to be randomized after the first turn, operative_first is randomized here
            order_dice = np.random.randint(1, 7, 2)
            if order_dice[0] - operative_first < order_dice[1]:
                operative_first = 1 - operative_first
        
        
        # if operative_first == 1: # Ensuring that if one goes to 0 or below first, it does not return dmg
        #     target_HP = target_HP - dmg[0] # Dealing any damage to opponent
        #     operative_HP = operative_HP - dmg[2]*3 # Dealing any "hot" damage to self
        #     if target_HP <= 0 or operative_HP <= 0:
        #         break
        #     operative_HP = operative_HP - dmg[1]
        #     target_HP = target_HP - dmg[3]*3
        # else:
        #     operative_HP = operative_HP - dmg[1]
        #     target_HP = target_HP - dmg[3]*3
        #     if operative_HP <=0 or target_HP <= 0:
        #         break
        #     target_HP = target_HP - dmg[0]
        #     operative_HP = operative_HP - dmg[2]*3
        
        # print('Operative HP')
        # print(operative_HP)
        # print('Target HP')
        # print(target_HP)
    # Calculating a measure of success - the amount of HP left when other part is at <= 0 HP (if none are at <= 0 HP, set the output to 0)
    if target_HP <= 0:
        result = max(0, operative_HP)
    elif operative_HP <= 0:
        result = -max(0, target_HP)
    else:
        result = -999 # As this is far away from the x-axis limits, it will not be seen in the graph
    
    # return [max(operative_HP, 0), max(target_HP, 0), turn_counter, max(operative_HP, 0)-max(target_HP, 0)]
    return [max(operative_HP, 0), max(target_HP, 0), str(turn_counter) + '-' + str(round_marker) + whose_turn, result]
       
def hit_calculator(A, WBS, crit, Relentless, Ceaseless, Balanced, Rending, Hot):
    hits = d6(A, WBS, crit)
    if Relentless:
        reroll = d6(A-hits[0]-hits[2], WBS, crit)
        hits[0] = hits[0] + reroll[0]
        hits[1] = reroll[1]
        hits[2] = hits[2] + reroll[2]
    if Ceaseless:
        reroll = d6(hits[1], WBS, crit)
        hits[0] = hits[0] + reroll[0]
        hits[1] = reroll[1]
        hits[2] = hits[2] + reroll[2]
    if Balanced and hits[0] + hits[2] < A:
        reroll = d6(1, WBS, crit)
        hits[0] = hits[0] + reroll[0]
        hits[1] = max(0, hits[1]-1) + reroll[1]
        hits[2] = hits[2] + reroll[2]
    if Rending and hits[0] >= 1 and hits[2] >= 1:
        hits[0] = hits[0] - 1
        hits[2] = hits[2] + 1
    
    if Hot == 1:
        hot_misfires = hits[1]
    else:
        hot_misfires = 0
    
    return hits, hot_misfires

def save_calculator(hits, AP, P, NoCover, Def, Sv, Inv, cover): # Returns two arrays, the first containing hits after Def and the second containing hits after Inv
    # print('hits)')
    # print(hits)
    if hits[2] >= 1: # If any critical hits, increase AP by P
        AP = AP + P
    def_dice = Def - AP # Defence dice is reduced by AP
     # The actual number of cover dice used is limited by remaining def dice and no more are used than there are normal hits, also checking whether the attack has NoCover:
    cover_actual = min(cover, def_dice, hits[0])*(1-NoCover)
    # (There might be some edge cases where it is desirable to roll despite being in cover and having some regular hits, i.e. in case of good armour and debilitating crit effects, these are not taken into account here.)
    
    def_dice = def_dice - cover_actual # Removing dice used for cover from defence dice pool
    def_total = d6(def_dice, Sv, 6)
    def_total[0] = def_total[0] + cover_actual
    # print(def_total)
    # print('def saves')
    # print(hits)
    hits_after_def = save_hit_removal(hits, def_total)
    # print(hits)
    # print(hits_after_def)
    if Inv > 1 and Inv < 7: # If having an Inv save, calculate hits using it as well:
        inv_dice = Def
        cover_actual = min(cover, inv_dice, hits[0])*(1-NoCover)
        inv_dice = inv_dice - cover_actual # Removing dice used for cover from defence dice pool
        inv_total = d6(inv_dice, Inv, 6)
        inv_total[0] = inv_total[0] + cover_actual
        # print('inv saves')
        # print(hits)
        hits_after_inv = save_hit_removal(hits, inv_total)
    else: # If target has no Inv save, the number of hits when using it remains the same
        hits_after_inv = hits
    return hits_after_def, hits_after_inv
    
def save_hit_removal(hits, saves):
    # print('hits')
    # print(hits)
    # print('saves')
    # print(saves)
    while hits[0]+hits[2] > 0 and saves[0]+saves[2] > 0:
        if hits[2] > 0 and saves[2] > 0: # Firstly remove crits, one for one
            hits[2] += -1
            saves[2] += -1
        elif hits[0] > 0 and saves[0] > 0: # Secondly remove normal hits, one for one
            hits[0] += -1
            saves[0] += -1
        elif hits[2] > 0 and saves[0] > 1: # If none of the above can be done and there is at least one crit and at least two normal saves, remove one for two
            hits[2] += -1
            saves[0] += -2
        elif hits[0] > 0 and saves[2] > 0: # If it is not possible to use a critical save to remove a critical hit, use it to remove a normal hit instead
            hits[0] += -1
            saves[2] += -1
        elif hits[2] > 0 and saves[0] == 1: # To avoid an eternal loop in case there are crits left and only one normal save
            break
        # print('hits')
        # print(hits)
        # print('saves')
        # print(saves)
    return hits
    
def parry_calculator(attacker_hits, defender_hits, attacker_defensive, attacker_shield, attacker_normal_damage, attacker_crit_damage, attacker_stun, defender_stunned, defender_HP, defender_brutal):
    strike = 0 # 0 means parry, 1 means strike
    dmg = 0 # Damage inflicted by attacker, starting at 0
    if attacker_defensive == 0 or (attacker_stun and defender_stunned == 0 and defender_hits[0] > 0): # If attacker is aggressive, it will always choose to deal damage
        strike = 1
    else:
        if (attacker_crit_damage >= defender_HP and attacker_hits[2] > 0) or (attacker_normal_damage >= defender_HP and attacker_hits[0] > 0): # Even if defensive, if striking will lead to winning immediately, do so
            strike = 1
        elif attacker_hits[2] > 0 and defender_hits[2] > 0: # If defensive, try to remove a critical hit
            attacker_hits[2] += -1
            defender_hits[2] += -1
            if attacker_shield and defender_hits[2] > 0: # If wielding a shield, another defender crit can be removed
                defender_hits[2] += -1
            elif attacker_shield and defender_hits[0] > 0: # If wielding a shield and defender has no crits left, a normal hit can be removed instead
                defender_hits[0] += -1
        elif attacker_hits[0] > 0 and defender_hits[0] > 0 and defender_brutal == 0: # If defensive and can't remove critical hits, try to remove a regular hit instead (does not work if defender's weapon has Brutal rule)
            attacker_hits[0] += -1
            defender_hits[0] += -1
            if attacker_shield and defender_hits[0] > 0: # If wielding a shield, another defender normal hit can be removed
                defender_hits[0] += -1
        elif attacker_hits[2] > 0 and defender_hits[0] > 0: # If none of above, try to remove a normal hit using a critical hit
            attacker_hits[2] += -1
            defender_hits[0] += -1
            if attacker_shield and defender_hits[0] > 0: # If wielding a shield, another defender normal hit can be removed
                defender_hits[0] += -1
        else: # If defensive but only has normal hits left and defender has only crits, or if defender has no hits left at all, or if attacker only has normal hits left and defender's weapon has Brutal rule, then attack instead
            strike = 1
    
    if strike == 1:
        if attacker_hits[2] > 0: # If attacker has crits left, use them first
            attacker_hits[2] += -1
            if attacker_stun and defender_stunned == 0 and defender_hits[0] > 0: # If the attacker's weapon can stun, and the defender isn't stunned yet in this round, the defender becomes stunned (loses 1 normal hit if possible). Note that losses of APL are not taken into account here.
                defender_hits[0] += -1
                defender_stunned = 1
            dmg = attacker_crit_damage
        elif attacker_hits[0] > 0: # If no crits left, use normal hits
            attacker_hits[0] += -1
            dmg = attacker_normal_damage
    return attacker_hits, defender_hits, dmg, defender_stunned
    
def dmg_calculator(a,d, FF, a_HP, d_HP): # Whoever is acting first is labelled attacker, 'a', and the other is labelled defender, 'd'
    dmg_to_defender = 0
    dmg_to_attacker = 0
        
    if FF: # If simulating a firefight (rather than close combat)
        
        # Calculating hits for attacker
        attacker_hits, attacker_hot_misfires = hit_calculator(a['A'], a['WBS'], a['Crit'], a['Relentless'], a['Ceaseless'], a['Balanced'], a['Rending'], a['Hot'])
        # print(attacker_hits)
        
        # MW from crits is handled here, before save/parry (could in theory be done together with other damage, but see below)
        dmg_to_defender += attacker_hits[2]*a['CritMW']
        
        attacker_hits_after_save = save_calculator(attacker_hits, a['AP'], a['P'], a['NoCover'], d['DF'], d['Sv'], d['Inv'], d['cover'])
        # For some bizarre reason the row above also sets attacker_hits to the same value as attacker_hits_after_save. This is why MWs are handled before the above row, and not together with other damage.
        # print(attacker_hits)
        # print(attacker_hits_after_save)
        dmg_to_defender += attacker_hits_after_save[0][0]*a['D'] + attacker_hits_after_save[0][2]*a['CD']
        dmg_to_attacker += attacker_hot_misfires * 3
        
    else: # If simulating close combat
        
        # In CC both attacker and defender attack multiple times, starting with the attacker
        attacker_hits, attacker_hot_misfires = hit_calculator(a['A'], a['WBS'], a['Crit'], a['Relentless'], a['Ceaseless'], a['Balanced'], a['Rending'], a['Hot'])
        defender_hits, defender_hot_misfires = hit_calculator(d['A'], d['WBS'], d['Crit'], d['Relentless'], d['Ceaseless'], d['Balanced'], d['Rending'], d['Hot'])

        while attacker_hits[0] + attacker_hits[2] + defender_hits[0] + defender_hits[2] > 0: # If there are any hits left to allocate, run hit allocation
            # In the start of each round, both the attacker and the defender are un-stunned
            a_stunned = 0
            d_stunned = 0
            
            if attacker_hits[0] + attacker_hits[2] > 0: # If attacker has hits left, allocate one of them
                attacker_hits, defender_hits, dmg, defender_stunned = parry_calculator(attacker_hits, defender_hits, a['defensive'], a['shield'], a['D'], a['CD'], a['Stun'], d_stunned, d_HP, d['Brutal'])
                dmg_to_defender += dmg
                if dmg_to_defender >= d_HP: # If damage from operator equals or exceeds target HP, stop hit allocation
                    break
            if defender_hits[0] + defender_hits[2] > 0: # If target has hits left, allocate one of them
                defender_hits, attacker_hits, dmg, attacker_stunned = parry_calculator(defender_hits, attacker_hits, d['defensive'], d['shield'], d['D'], d['CD'], d['Stun'], a_stunned, a_HP, a['Brutal'])
                dmg_to_attacker += dmg
                if dmg_to_attacker >= a_HP: # If damage from target equals or exceeds operator HP, stop hit allocation
                    break
            
    # Calculating dmg (As Inv saves currently don't work properly, damage is only calculated using regular saves)
    # print(o.hits[0][0])
    # print(o.hits[0][1])
    # print(t.hits[0][0])
    # print(t.hits[0][1])
    # print(dmg_to_defender)
    # print(dmg_to_attacker)
    
    return dmg_to_defender, dmg_to_attacker
    
# Old data, used for non-web version of script
# # Setting operative data
# operative_row = 1
# operative_weapon_row = 60
# operative_cover = {'cover': 1}
# operative_defensive = {'defensive': 1}
# operative_shield = {'shield': 0}

# # Setting target data
# target_row = 1 # 17
# target_weapon_row = 63 # 97
# target_cover = {'cover': 0}
# target_defensive = {'defensive': 0}
# target_shield = {'shield': 0}

# # Setting scenario and simulation data
# operative_first = 1
# FF = 0
# turns_max = 4
# random_order = 1 # After first turn, randomizes who will go first (assumes that the operatives will be the first ones activated by each player)
# num_simulations = 30
# print_inputs = 1

# # Creating dictionaries containing operative and target data
# operative = {**operative_statblocks.iloc[operative_row,:].to_dict(), **weapon_statblocks.iloc[operative_weapon_row,:].to_dict(), **operative_cover, **operative_defensive, **operative_shield}
# target = {**operative_statblocks.iloc[target_row,:].to_dict(), **weapon_statblocks.iloc[target_weapon_row,:].to_dict(), **target_cover, **target_defensive, **target_shield}

# # Creating and printing output strings with info on simulation set-up
# operative_details = 'Operative has '
# if operative['cover'] > 0:
#     operative_details += 'cover level ' + str(operative['cover']) + ', '
# if operative['defensive']:
#     operative_details += 'defensive focus in hand-to-hand'
# else:
#     operative_details += 'aggressive focus in hand-to-hand'
# if operative['shield']:
#     operative_details += ' and storm shield'
# operative_details += '.'

# target_details = 'Target has '
# if target['cover'] > 0:
#     target_details += 'cover level ' + str(target['cover']) + ', '
# if target['defensive']:
#     target_details += 'defensive focus in hand-to-hand'
# else:
#     target_details += 'aggressive focus in hand-to-hand'
# if target['shield']:
#     target_details += ' and storm shield'
# target_details += '.'

# if operative_first:
#     first = 'Operative'
# else:
#     first = 'Target'
# if random_order:
#     ordering = 'random'
# else:
#     ordering = 'fixed'
# simulation_details = first + ' acts first in the initial round, ordering is then ' + ordering + ', simulation runs for up to ' + str(turns_max) + ' turns, ' + str(num_simulations) + ' simulations are run.'

# if print_inputs == 1:
#     print('Operative: ' + operative['Operative name'] + ', ' + operative['Weapon name'])
#     print(operative_details)
#     print('')
#     print('Target: ' + target['Operative name'] + ', ' + target['Weapon name'])
#     print(target_details)
#     print('')
#     print(simulation_details)
#     print('')

# # Running the simulation
# data = pd.DataFrame([simulator(operative, target, operative_first, FF, turns_max, random_order)[2:4] for a in range(num_simulations)], columns=['Turn', 'Result'])
# data_freq = data.value_counts().reset_index(name='Count')
# data_freq['Count'] = data_freq['Count'] / num_simulations
# print(data_freq)
# # plt.hist(data[:,1], bins = 20)
# # print(data)
# # print(data.iterrows())

# # Printing outcomes
# print('Operator victories: ' + str((data.loc[:,'Result']>0).sum()/num_simulations))
# print('None down:          ' + str((data.loc[:,'Result']==-999).sum()/num_simulations))
# print('Both down:          ' + str((data.loc[:,'Result']==0).sum()/num_simulations))
# print('Operator losses:    ' + str(((data.loc[:,'Result']<0).sum()-(data.loc[:,'Result']==-999).sum())/num_simulations))

# # Sorting results by turn and creating histogram
# data_sorted = {}
# for x in data.iterrows():
#     if x[1][0] not in data_sorted:
#         data_sorted[x[1][0]] = []
#     data_sorted[x[1][0]].append(x[1][1])
# # print(data_sorted)
# # print(sorted(data_sorted.keys()))
# # for i in sorted(data_sorted.keys()):
# #     print(data_sorted[i])
# df = pd.concat([pd.DataFrame(data_sorted[i], columns=['Turn '+str(i)]) for i in sorted(data_sorted.keys())], axis=1)
# # df = pd.concat([pd.DataFrame(a, columns=[f'x{i}']) for i, a in enumerate([x1, x2, x3], 1)], axis=1)
# # print(df)
# df.plot.hist(stacked=True, bins=range(-target['W'], operative['W']+1, 1), grid=True).legend(loc='upper right', bbox_to_anchor=(1.3, 1))


# Defining starting values for dropdown selections in app
faction_options = [{'label': i, 'value': i} for i in sorted(set(operative_statblocks.loc[:,'Faction']))]
faction_options = [{'label': 'All','value': 'All'}] + faction_options

fireteam_options = [{'label': i, 'value': i} for i in sorted(set(operative_statblocks.loc[:,'Fire team']))]
fireteam_options = [{'label': 'All','value': 'All'}] + fireteam_options

operative_options = [{'label': i, 'value': j} for i,j in sorted(set(zip(operative_statblocks.loc[:,'Operative name'], operative_statblocks.index)))]
# operative_options = [{'label': '','value': ''}] + operative_options

weapon_options = [{'label': i, 'value': j} for i,j in sorted(set(zip(weapon_statblocks.loc[:,'Weapon name'], operative_statblocks.index)))]
# weapon_options = [{'label': '','value': ''}] + weapon_options

run_app = 1
    
app = dash.Dash(__name__)

# Setting up app layout
app.layout = html.Div([
    html.Div(style={'padding': '30px'}, children = 'Kill Team 2.0 simulator'),
    html.Div(style={'display': 'inline-block'}, children=[
        
        # Simulation inputs
        html.Div(style={'padding': '10px', 'width': '200px'}, children = [
            html.Label('Simulation type'),
            dcc.RadioItems(
                id = 'FF',
                options=[
                      {'label': 'Shooting', 'value': 1},
                      {'label': 'Hand-to-hand', 'value': 0}
                ],
                value = 1,
                labelStyle={'display': 'block'},
            ),
            html.Label('Who goes first?'),
            dcc.RadioItems(
                id = 'operative_first',
                options=[
                    {'label': 'Operative', 'value': 1},
                    {'label': 'Target', 'value': 0}
                ],
                value = 1,
                labelStyle={'display': 'block'},
            ),
            html.Label('Max number of turns simulated'),
            dcc.Slider(
                id = 'turns_max',
                min=1,
                max=4,
                marks={i: str(i) for i in range(1, 5)},
                value=1,
            ),
            html.Div(style={'padding': '10px'}),
            html.Label('Random activation order after first turn?'),
            dcc.RadioItems(
                id = 'random_order',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                value = 1,
                labelStyle={'display': 'block'},
            ),
            html.Div(style={'padding': '10px'}),
            html.Label('Number of simulations'),
            dcc.Input(
                id = 'num_simulations',
                type = 'number',
                value = 10000,
            ),
        ]),
        
        # Operator inputs
        html.Div(style={'padding': '10px', 'display': 'inline-block', 'width': '300px'}, children = [
            html.Label('Operative faction'),
            dcc.Dropdown(
                id = 'o_faction',
                options = faction_options,
                value = 'All'                   
            ),
            html.Label('Operative fire team'),
            dcc.Dropdown(
                id = 'o_fireteam',
                options = fireteam_options,
                value = 'All'                   
            ),
            html.Label('Operative type'),
            dcc.Dropdown(
                id = 'o_type',
                options = operative_options,
                value = -1                  
            ),
            html.Label('Operative weapon'),
            dcc.Dropdown(
                id = 'o_weapon',
                options = weapon_options,
                value = -1              
            ),
            html.Label('Operative cover'),
            dcc.Dropdown(
                id = 'o_cover',
                options = [
                    {'label': '0', 'value': 0},
                    {'label': '1', 'value': 1},
                    {'label': '2', 'value': 2},
                ],
                value = 0
            ),
            html.Label('Operative HtH tactic'),
            dcc.RadioItems(
                id = 'o_defensive',
                options = [
                    {'label': 'Aggressive', 'value': 0},
                    {'label': 'Defensive', 'value': 1},
                ],
                value = 0,
                labelStyle={'display': 'block'},
            ),
            html.Label('Operative has Storm shield?'),
            dcc.RadioItems(
                id = 'o_shield',
                options = [
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0},
                ],
                value = 0,
                labelStyle={'display': 'block'},
            ),
        ]),
        
        # Target inputs
        html.Div(style={'padding': '10px', 'display': 'inline-block', 'width': '300px'}, children = [
            html.Label('Target faction'),
            dcc.Dropdown(
                id = 't_faction',
                options = faction_options,
                value = 'All'                   
            ),
            html.Label('Target fire team'),
            dcc.Dropdown(
                id = 't_fireteam',
                options = fireteam_options,
                value = 'All'                   
            ),
            html.Label('Target type'),
            dcc.Dropdown(
                id = 't_type',
                options = operative_options,
                value = -1                 
            ),
            html.Label('Target weapon'),
            dcc.Dropdown(
                id = 't_weapon',
                options = weapon_options,
                value = -1                  
            ),
            html.Label('Target cover'),
            dcc.Dropdown(
                id = 't_cover',
                options = [
                    {'label': '0', 'value': 0},
                    {'label': '1', 'value': 1},
                    {'label': '2', 'value': 2},
                ],
                value = 0,
            ),
            html.Label('Target HtH tactic'),
            dcc.RadioItems(
                id = 't_defensive',
                options = [
                    {'label': 'Aggressive', 'value': 0},
                    {'label': 'Defensive', 'value': 1},
                ],
                value = 0,
                labelStyle={'display': 'block'},
            ),
            html.Label('Target has Storm shield?'),
            dcc.RadioItems(
                id = 't_shield',
                options = [
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0},
                ],
                value = 0,
                labelStyle={'display': 'block'},
            ),
        ]),
        
        # Start button
        html.Div(style={'padding': '10px', 'width': '290px'}, children = [
            html.Button('Run simulation', id='start_button', n_clicks=0),
        ]),
    ]),
    
    # Graphic for plotting
    html.Div(style={'padding': '10px', 'display': 'inline-block'}, id = 'plot', children = [
        dcc.Graph(id='simulation_graphic'),
        html.Div(id = 'eval'),
    ]),
])

# Callback for running simulation
@app.callback(
    Output('simulation_graphic', 'figure'),
    Output('eval', 'children'),
    Input('start_button', 'n_clicks'),
    State('FF', 'value'),
    State('operative_first', 'value'),
    State('turns_max', 'value'),
    State('random_order', 'value'),
    State('num_simulations', 'value'),
    State('o_type', 'value'),
    State('o_weapon', 'value'),
    State('o_cover', 'value'),
    State('o_defensive', 'value'),
    State('o_shield', 'value'),
    State('t_type', 'value'),
    State('t_weapon', 'value'),
    State('t_cover', 'value'),
    State('t_defensive', 'value'),
    State('t_shield', 'value'),)
def update_graph(n_clicks, FF, operative_first, turns_max, random_order, num_simulations, operative_row, operative_weapon_row, operative_cover, operative_defensive, operative_shield, target_row, target_weapon_row, target_cover, target_defensive, target_shield):
    print(operative_row, operative_weapon_row, target_row, target_weapon_row)
    
    # Limiting num_simulations to below 1e6, to avoid too long running times
    if num_simulations > 1e6:
        num_simulations = 1e6
    
    # Only run if inputs have been set, by testing their values
    if operative_row > -1 and operative_weapon_row > -1 and target_row > -1 and target_weapon_row > -1:
        
        # Creating dictionaries containing operative and target data
        # operative = {**operative_statblocks.iloc[operative_row,:].to_dict(), **weapon_statblocks.iloc[operative_weapon_row,:].to_dict(), **operative_cover, **operative_defensive, **operative_shield}
        # target = {**operative_statblocks.iloc[target_row,:].to_dict(), **weapon_statblocks.iloc[target_weapon_row,:].to_dict(), **target_cover, **target_defensive, **target_shield}
        operative = {**operative_statblocks.iloc[operative_row,:].to_dict(), **weapon_statblocks.iloc[operative_weapon_row,:].to_dict(), **{'cover': operative_cover}, **{'defensive': operative_defensive}, **{'shield': operative_shield}}
        target = {**operative_statblocks.iloc[target_row,:].to_dict(), **weapon_statblocks.iloc[target_weapon_row,:].to_dict(), **{'cover': target_cover}, **{'defensive': target_defensive}, **{'shield': target_shield}}
        
        # Running simulation
        data = pd.DataFrame([simulator(operative, target, operative_first, FF, turns_max, random_order)[2:4] for a in range(num_simulations)], columns=['Turn', 'Result'])
        
        # Reordering output into frequency table, giving name to count column
        data_turncounts = data.value_counts().reset_index(name='Frequency').sort_values('Turn')
        data_turncounts.loc[:,'Frequency'] = data_turncounts.loc[:,'Frequency'] / num_simulations
        print(data_turncounts)
        data_maxcounts = data.loc[:,'Result'].value_counts().reset_index(name='Frequency')
        max_bar_size = max(data_maxcounts['Frequency']) / num_simulations
        
        # # Ordering simulation outcomes by turn
        # data_sorted = {}
        # for x in data.iterrows():
        #     if x[1][0] not in data_sorted:
        #         data_sorted[x[1][0]] = []
        #     data_sorted[x[1][0]].append(x[1][1])
        # df = pd.concat([pd.DataFrame(data_sorted[i], columns=['Turn '+str(i)]) for i in sorted(data_sorted.keys())], axis=1)
        
        # Creating figure object
        fig = px.bar(data_turncounts, x="Result", y="Frequency", color="Turn", title="Simulated outcomes").update_traces(width=1).update_layout(xaxis_range=[-target['W']-1,operative['W']+1], yaxis_range=[0, max_bar_size*1.05], xaxis_dtick=1, xaxis_fixedrange=True, yaxis_fixedrange=True)
        
        o_victories = (data.loc[:,'Result']>0).sum()/num_simulations
        none_down = (data.loc[:,'Result']==-999).sum()/num_simulations
        both_down = (data.loc[:,'Result']==0).sum()/num_simulations
        o_losses = ((data.loc[:,'Result']<0).sum()-(data.loc[:,'Result']==-999).sum())/num_simulations
        o_victories_avg_W = (data.loc[data.loc[:,'Result']>0,'Result']).sum() / (o_victories * num_simulations)
        o_losses_avg_W = -(data.loc[(data.loc[:,'Result']<0) & (data.loc[:,'Result']!=-999),'Result']).sum() / (o_losses * num_simulations)
        o_victories_W_minus_o_losses_W = ((data.loc[data.loc[:,'Result']>0,'Result']).sum() - (data.loc[(data.loc[:,'Result']<0) & (data.loc[:,'Result']!=-999),'Result']).sum()) / num_simulations
        evaldata = [
            html.P('Operator victories: ' + str(o_victories)),
            html.P('None down:          ' + str(none_down)),
            html.P('Both down:          ' + str(both_down)),
            html.P('Operator losses:    ' + str(o_losses)),
            html.P('Operator victories - avg W left: ' + str(o_victories_avg_W)),
            html.P('Operator losses - avg W left: ' + str(o_losses_avg_W)),
            html.P('Op W at wins - Ta W at losses: ' + str(o_victories_W_minus_o_losses_W))
        ]
    else:
        fig = px.bar()
        evaldata = [
            html.P('Operator victories: '),
            html.P('None down: '),
            html.P('Both down: '),
            html.P('Operator losses: '),
            html.P('Operator victories - avg W left: '),
            html.P('Operator losses - avg W left: '),
            html.P('Op W at wins - Ta W at losses: ')
        ]
    
    return fig, evaldata

    
# Callbacks for setting fire teams and operator types based on faction
@app.callback(
Output('o_fireteam', 'options'),
Output('o_type', 'options'),
Input('o_faction', 'value'),)
def filter_operative_on_faction(o_faction):
    if o_faction == 'All':
        o_fireteam = fireteam_options
        o_type = operative_options
    else:
        o_fireteam = [{'label': i, 'value': i} for i in sorted(set(operative_statblocks.loc[operative_statblocks['Faction'] == o_faction,'Fire team']))]
        o_fireteam = [{'label': 'All','value': 'All'}] + o_fireteam
        o_type = [{'label': i, 'value': j} for i,j in sorted(set(zip(operative_statblocks.loc[operative_statblocks['Faction'] == o_faction,'Operative name'],operative_statblocks.index[operative_statblocks['Faction'] == o_faction])))]
    return o_fireteam, o_type
@app.callback(
Output('t_fireteam', 'options'),
Output('t_type', 'options'),
Input('t_faction', 'value'),)
def filter_target_on_faction(t_faction):
    if t_faction == 'All':
        t_fireteam = fireteam_options
        t_type = operative_options
    else:
        t_fireteam = [{'label': i, 'value': i} for i in sorted(set(operative_statblocks.loc[operative_statblocks['Faction'] == t_faction,'Fire team']))]
        t_fireteam = [{'label': 'All','value': 'All'}] + t_fireteam
        t_type = [{'label': i, 'value': j} for i,j in sorted(set(zip(operative_statblocks.loc[operative_statblocks['Faction'] == t_faction,'Operative name'],operative_statblocks.index[operative_statblocks['Faction'] == t_faction])))]
    return t_fireteam, t_type

# Callbacks for setting operative types based on fire team
@app.callback(
Output('o_type', 'options'),
Input('o_fireteam', 'value'),
State('o_faction', 'value'),)
def filter_operative_on_fireteam(o_fireteam, o_faction):
    if o_fireteam == 'All':
        if o_faction == 'All':
            o_type = operative_options
        else:
            o_type = [{'label': i, 'value': j} for i,j in sorted(set(zip(operative_statblocks.loc[operative_statblocks['Faction'] == o_faction,'Operative name'],operative_statblocks.index[operative_statblocks['Faction'] == o_faction])))]
    else:
        o_type = [{'label': i, 'value': j} for i,j in sorted(set(zip(operative_statblocks.loc[operative_statblocks['Fire team'] == o_fireteam,'Operative name'],operative_statblocks.index[operative_statblocks['Fire team'] == o_fireteam])))]
    return o_type
@app.callback(
Output('t_type', 'options'),
Input('t_fireteam', 'value'),
State('t_faction', 'value'),)
def filter_target_on_fireteam(t_fireteam, t_faction):
    if t_fireteam == 'All':
        if t_faction == 'All':
            t_type = operative_options
        else:
            t_type = [{'label': i, 'value': j} for i,j in sorted(set(zip(operative_statblocks.loc[operative_statblocks['Faction'] == t_faction,'Operative name'],operative_statblocks.index[operative_statblocks['Faction'] == t_faction])))]
    else:
         t_type = [{'label': i, 'value': j} for i,j in sorted(set(zip(operative_statblocks.loc[operative_statblocks['Fire team'] == t_fireteam,'Operative name'],operative_statblocks.index[operative_statblocks['Fire team'] == t_fireteam])))]
    return t_type

# Callbacks for setting weapons based on faction and combat type
@app.callback(
Output('o_weapon', 'options'),
Input('o_faction', 'value'),
Input('FF', 'value'),)
def filter_operative_weapon(o_faction, FF):
    if FF == 1:
        combat_type = 'FF'
    else:
        combat_type = 'HtH'
    if o_faction == 'All':
        o_weapon = [{'label': i, 'value': j} for i,j in sorted(set(zip(weapon_statblocks.loc[weapon_statblocks['FF/HtH'] == combat_type,'Weapon name'],weapon_statblocks.index[weapon_statblocks['FF/HtH'] == combat_type])))]
    else:
        o_weapon = [{'label': i, 'value': j} for i,j in sorted(set(zip(weapon_statblocks.loc[(weapon_statblocks['Faction'] == o_faction) & (weapon_statblocks['FF/HtH'] == combat_type),'Weapon name'],weapon_statblocks.index[(weapon_statblocks['Faction'] == o_faction) & (weapon_statblocks['FF/HtH'] == combat_type)])))]
    return o_weapon
@app.callback(
Output('t_weapon', 'options'),
Input('t_faction', 'value'),
Input('FF', 'value'),)
def filter_target_weapon(t_faction, FF):
    if FF == 1:
        combat_type = 'FF'
    else:
        combat_type = 'HtH'
    if t_faction == 'All':
        t_weapon = [{'label': i, 'value': j} for i,j in sorted(set(zip(weapon_statblocks.loc[weapon_statblocks['FF/HtH'] == combat_type,'Weapon name'],weapon_statblocks.index[weapon_statblocks['FF/HtH'] == combat_type])))]
    else:
        t_weapon = [{'label': i, 'value': j} for i,j in sorted(set(zip(weapon_statblocks.loc[(weapon_statblocks['Faction'] == t_faction) & (weapon_statblocks['FF/HtH'] == combat_type),'Weapon name'],weapon_statblocks.index[(weapon_statblocks['Faction'] == t_faction) & (weapon_statblocks['FF/HtH'] == combat_type)])))]
    return t_weapon

# Running server
if __name__ == '__main__' and run_app:
    app.run_server(debug=False)
    