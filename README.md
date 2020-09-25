# Generalised Iverter and Rotational Approaches for Quantum Walks
Code for quantum walk implementation on IBM Q

## Generalised Inverters Approach
Contains methods and functions for simulating quantum walks with the generalised CNOT approach. The quantum walk state space can contain n qubits, with n capped differently with every machine that runs the simulation.

The size of the workspace and the number of gates in the circuit vary, with the growth rate following the relevant equations in the paper.

The file also contains functions for the calculation of the variance of the quantum walk. These functions assume that the walker on the cycle does not cross the initial position. With sufficiently large cycle, the variance can be calculated for as many coin-flips as the machine can handle.

## Rotations Approach
Contains methods for simulating quantum walks with the rotational approach. The circuits are generated according to the relevant analysis in the paper.

The size of the workspace and the number of gates in the circuit vary, with the growth rate following the relevant equations in the paper.

## How to Run
A small example of the steps necessary to create and run the quantum circuits:
- in order to create the circuits, first run the `walk(n,t)` method, with n the number of qubits and t the number of steps; for the rotational circuit run the `rotWalk(n,t)`.
- to simulate the quantum circuits and get the probability distribution, run the `simulateQW(n, t, rep)` method or `simulateQWRot(n, t, rep)` for the rotations
- to run on the quantum computer, run `runqcQW(n, t, rep)` or `runqcQWRot(n, t, rep)` for the rotations

The rest of the methods necessary for the analysis presented on the paper do not contribute anything to the circuits themselves and are self explanatory. 
