# Quantum walk
Code for quantum walk implementation on IBM Q

## Generalised Inverters Approach
Contains methods and functions for simulating quantum walks with the generalised CNOT approach. The quantum walk state space can contain $n$ qubits, with $n$ capped differently with every machine that runs the simulation.

The size of the workspace and the number of gates in the circuit vary, with the growth rate following the relevant equations in the paper.

The file also contains functions for the calculation of the variance of the quantum walk. These functions assume that the walker on the cycle does not cross the initial position. With sufficiently large cycle, the variance can be calculated for as many coin-flips as the machine can handle.

## Rotations Approach
Contains methods for simulating quantum walks with the rotational approach. The circuits are generated according to the relevant analysis in the paper.

The size of the workspace and the number of gates in the circuit vary, with the growth rate following the relevant equations in the paper. 