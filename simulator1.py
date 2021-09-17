# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 01:26:07 2021

@author: olajo
"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# %matplotlib inline
# ximport collections

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

# Importing statblocks, filling nans and returning floats into ints (columns with nan are made into floats when importing from csv)
operative_statblocks = pd.read_csv('KT2_operative_statblocks.csv').fillna(0)
operative_statblocks.iloc[:,3:11] = operative_statblocks.iloc[:,3:11].values.astype(int)
# float_cols = operative_statblocks.iloc[3:11].values
# float_cols = float_cols.astype(int)
weapon_statblocks = pd.read_csv('KT2_weapon_statblocks.csv').fillna(0)
weapon_statblocks.iloc[:,4:20] = weapon_statblocks.iloc[:,4:20].values.astype(int)
# print(operative_statblocks.iloc[:,2:12])
# print(weapon_statblocks.iloc[:,3:10])

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
    
def parry_calculator(attacker_hits, defender_hits, attacker_defensive, attacker_shield, attacker_normal_damage, attacker_crit_damage, defender_HP, defender_brutal):
    strike = 0 # 0 means parry, 1 means strike
    dmg = 0 # Damage inflicted by attacker, starting at 0
    if attacker_defensive == 0: # If attacker is aggressive, it will always choose to deal damage
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
            dmg = attacker_crit_damage
        elif attacker_hits[0] > 0: # If no crits left, use normal hits
            attacker_hits[0] += -1
            dmg = attacker_normal_damage
    return attacker_hits, defender_hits, dmg
    
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
            if attacker_hits[0] + attacker_hits[2] > 0: # If attacker has hits left, allocate one of them
                attacker_hits, defender_hits, dmg = parry_calculator(attacker_hits, defender_hits, a['defensive'], a['shield'], a['D'], a['CD'], d_HP, d['Brutal'])
                dmg_to_defender += dmg
                if dmg_to_defender >= d_HP: # If damage from operator equals or exceeds target HP, stop hit allocation
                    break
            if defender_hits[0] + defender_hits[2] > 0: # If target has hits left, allocate one of them
                defender_hits, attacker_hits, dmg = parry_calculator(defender_hits, attacker_hits, d['defensive'], d['shield'], d['D'], d['CD'], a_HP, a['Brutal'])
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
    

# Setting operative data
operative_row = 1
operative_weapon_row = 63
operative_cover = {'cover': 0}
operative_defensive = {'defensive': 1}
operative_shield = {'shield': 1}

# Setting target data
target_row = 1 # 17
target_weapon_row = 63 # 97
target_cover = {'cover': 0}
target_defensive = {'defensive': 0}
target_shield = {'shield': 0}

# Setting scenario and simulation data
operative_first = 0
FF = 0
turns_max = 4
random_order = 1 # After first turn, randomizes who will go first (assumes that the operatives will be the first ones activated by each player)
num_simulations = 10000

# Creating dictionaries containing operative and target data
operative = {**operative_statblocks.iloc[operative_row,:].to_dict(), **weapon_statblocks.iloc[operative_weapon_row,:].to_dict(), **operative_cover, **operative_defensive, **operative_shield}
target = {**operative_statblocks.iloc[target_row,:].to_dict(), **weapon_statblocks.iloc[target_weapon_row,:].to_dict(), **target_cover, **target_defensive, **target_shield}
print(operative)
print(target)

# data = np.array([simulator(operative, target, operative_first, FF, turns_max)[2:4] for a in range(num_simulations)])
data = pd.DataFrame([simulator(operative, target, operative_first, FF, turns_max, random_order)[2:4] for a in range(num_simulations)], columns=['Turn', 'Result'])

# plt.hist(data[:,1], bins = 20)
# print(data)
# print(data.iterrows())
print('Operator victories: ' + str((data.loc[:,'Result']>0).sum()/num_simulations))
print('None down:          ' + str((data.loc[:,'Result']==-999).sum()/num_simulations))
print('Both down:          ' + str((data.loc[:,'Result']==0).sum()/num_simulations))
print('Operator losses:    ' + str(((data.loc[:,'Result']<0).sum()-(data.loc[:,'Result']==-999).sum())/num_simulations))

data_sorted = {}
for x in data.iterrows():
    if x[1][0] not in data_sorted:
        data_sorted[x[1][0]] = []
    data_sorted[x[1][0]].append(x[1][1])
# print(data_sorted)
# print(sorted(data_sorted.keys()))
# for i in sorted(data_sorted.keys()):
#     print(data_sorted[i])
df = pd.concat([pd.DataFrame(data_sorted[i], columns=['Turn '+str(i)]) for i in sorted(data_sorted.keys())], axis=1)
# df = pd.concat([pd.DataFrame(a, columns=[f'x{i}']) for i, a in enumerate([x1, x2, x3], 1)], axis=1)
# print(df)
df.plot.hist(stacked=True, bins=range(-target['W'], operative['W']+1, 1), grid=True).legend(loc='upper right', bbox_to_anchor=(1.3, 1))


# data_sorted = collections.OrderedDict(sorted(data_sorted.items()))
# print(data_sorted)
# plt.hist(data_sorted.values(), bins = 20, stacked=True, density=True)

# data_test = [np.array([1, 3, 2, 4]), np.array([1, 2, 7, ]), np.array([3, 4, 7])]
# data_test = [[1, 3, 2, 4], [1, 2, 7, ], [3, 4, 7]]
# print(data_test)
# plt.figure()
# plt.hist(data_test, stacked=True)
# plt.show()

# mu, sigma = 200, 25
# x = mu + sigma*np.random.randn(1000,3)


# x1 = mu + sigma*np.random.randn(990,1)
# x2 = mu + sigma*np.random.randn(980,1)
# x3 = mu + sigma*np.random.randn(1000,1)
# n, bins, patches = plt.hist(x, 30, stacked=True, density = True)

# #Stack the data
# plt.figure()
# plt.hist([x1,x2,x3], bins, stacked=True, density=True)
# plt.show()