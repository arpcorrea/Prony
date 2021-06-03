# Objective
This code calculates the Prony Series out of frequency sweep tests. 

## How to run
Paste your frequency, storage and loss array data in the "frequency", "storage" and "loss" variables respectively. 

## Comments
The code is not yet 100% fully autonomously, so right now, it runs perfectly with 2 Maxwell elements per decade, and 3 decades are considered, generating 6 Maxwell elements.
Each Maxwell element has 2 unknowns: stiffness and relaxation time, summing up 12 unkowns.
The fitting functions are, so far, hard-coded. G1,t1... G6,t6 are manually set, which requires upgrades.

## Fitting Bounds 
All variables must be positive, thus stiffness unknowns, which ocupy even positions in the unknown solution, vary between 0 (lower bound) and infinity (upper bound). Relaxation unkowns bounds, which ocupy odd positions in the unknown solution, are defined in function of the number of maxwell elements per decade.

## Fitting Curves
Curve fitting can be done over the storage function, loss function or even the summation of both. For the current specific case, best curve fitting with both data points, storage and loss, was found on the storage fitting function. The evaluation of which fuction fits better is evaluated by the user.
