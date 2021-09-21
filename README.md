# KT2_simulator
Simulator for analysis of different aspects within the miniature game Warhammer 40000 Kill Team

Some disclaimers:
- I am far from a professional developer, just dabbling in Python.
- This is a hobby project which has grown organically without any real plan.
- At the moment the code is fully workable and incorporates a bunch of settings, but it's in an unfinished state with very little optimization and barely cleaned up at all.
- This repository is probably only relevant for people who either a) are playing Games Workshop's miniature wargame Kill Team v. 2.0 or b) want a good laugh at how a home-taught programmer like me is writing code.

Included files:
simulator1.py is the script containing the logic for the simulation itself, split into a bunch of functions which I've tried to give some clarifying comments for.
KT2_operative_statblocks.csv and KT2_weapon_statblocks.csv contain the tables for different operative and weapon stats, in order to run them against each other. They currently only include a few Marine and Imperial Guard statlines and I will hopefully add more in the future, but you can make your own version of course. If you have more complete tables you can of course rewrite the data import in simulator1.py in order to handle yours instead.

Graphical interface:
The scripts now generates a web-based graphical interface which can be viewed in a browser (Firefox works for me, none others tested). Thanks to responsive features in the interface, this makes it considerably easier to alter various parameters and gives a fairly easy way of overviewing the current scenario.

Inputs:

1. The simulator requires you to set the operator (your miniature) and the target (the opponent's miniature). Note that these two could basically be used interchangeably, but it helps using clear names for them. This is done by entering which rows in the statblock data will be used as well as how many cover dice each has, which tactic they will use in hand-to-hand (aggressive or defensive) and whether they have a storm shield.

2. Other inputs are:
- Who will go first (operative_first set to 1 or 0).
- Whether simulating a firefight or hand-to-hand (FF set to 1 or 0).
- How many turns (or "turning points" for those who buy into the newspeak of the rulebook) the simulation will maximally run. If you just want to see the results of one turn of shooting, finding out the likelihood of taking down a target in one shooting action or something such, just set this to 1.
- Whether the turn order will be randomized after the first. Setting random_order to 0 means that the same order as for the first turn is maintained, i.e. if you set operative_first to 1 the operative will act first in every turn. Setting random_order to 1 means that it is randomized in the same way as determining who starts activations in each turn so it's what it would look like if both players activate these figures as their first activations in the turn.
- How many simulations to run. On my not particularly new computer running 10000 simulations takes a few seconds.

Outputs:

1. The script prints all data on the operative and target, then prints the probabilities of different outcomes as well as average wounds in the scenarios where the operator/target has won. It also prints a "weighted sum": (Operator win ratio * Num wounds when operator has won - Target win ratio * Num wounds when target has won) / num_simulations, which is an attempt at a "total score" to allow easy comparisons between e.g. different weapons.

2. Most importantly, the script creates a histogram of different outcomes turn by turn. This provides a lot of information but can sometimes become a bit messy. The x axis is how many wounds the operative has left in each simulation after the target has been defeated, negative numbers mean that the target won and shows how many wounds it has left instead. Rounds are named as follows: A-Bxy. A is the turn number, B is whether it's in the first or second figure's activation and xy is either 'op' or 'ta' to indicate whether it's in the operative's or the target's activation. This is because if using random activation order, the operative might be attacking first in the first turn but last in the second or vice versa.

Notes on logic

The script assumes that both figures attack in each turn. There are of course situations where one figure takes its activation and then the other one attacks it, robbing the target of the chance to attack back in its activation, and this is something which could be added with an option for this. In shooting, one shoots in its activation and then the other shoots back. In hand-to-hand one takes its action attacking (and then both fight, with the action one dealing the first strike/parry) and then the next (where the other is then first to deal strikes/parrys).

Striking or parrying also contains some set logic. If the 'defensive' parameter is set to 0 for operative or target, it will act aggressively and seek to strike with every attack. If it's set to 1 it instead tries to do this in decreasing priority:
1. If striking will lead to the opponent going down immediately, then strike.
2. Try to remove a critical hit using its own critical hit.
3. Try to remove a normal hit using its own normal hit.
4. Try to remove a normal hit using a critical hit.
5. Strike.

This can certainly be improved on but seems to give rather plausible results in my simulations.

Features, missing features etc

Currently, Inv saves are not implemented. This is because the decision on whether to use these or regular saves depends not only on what is known at that stage (number of defence dice to use, cover if any) but also on the prediction of best likely outcome from factors such as remaining wounds, hits and damage of weapons etc. For instance, with a 2+/5++ save, if the opponent has AP1 and has scored no normal hits but two critical hits and you need to save both in order for the model to remain your best chance is using Inv hoping for two sixes or one six and two fives, but if there instead are enough wounds left to survive one crit it's better to roll normal saves and probably get two successes to remove one of the crits. Naturally, implementing logic for handling this is somewhat complex and I haven't solved it yet.

The following weapon special rules are currently implemented:
Lethal (implemented by giving each weapon a Crit rating, which is 6 for weapons without Lethal)
AP
Relentless
Ceaseless
Balanced
P
Rending
CritMW
Brutal
NoCover
Hot
Stun

Any special rules related to the operator rather than its weapon are currently not implemented. These are quite diverse and sometimes probably rather difficult to implement so I will most likely not be able to cover them entirely. I may make some attempt at including some of the more straightforward or important ones.

Possible future features

- Done! A graphical interface would be nice, to not have to type into the code every time a new operative, target or setting is chosen.
- Done! Implementing Stun.
- Implementing some operator-specific rules.
- Make complete tables of operators and weapons.
- Make each operator only able to select weapons allowed for that operator.
- Improved parry logic.
- Implementing Inv saves (probably some very simplified decision process, if that is possible).
- DONE! Printing the average W left for the operative/target in those simulation runs where each side has won.
- Etc etc.
