import numpy as np
from numpy import pi as pi
import time

# importing QISKit
from qiskit import Aer, IBMQ
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit import IBMQ, BasicAer
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from qiskit.providers.aer import StatevectorSimulator

# Generalized Toffoli gate using rotations for n control and 1 target qubits
def genTofRot(n, q_reg, q_coin, q_circ, *q):
    """This is a function that returns a generalized CNOT implemented with rotations"""
    delta = pi/2
    phi = pi/2
    theta = pi/2
    
    # Get the Phi(delta) operator
    pr = getPhiOperator(n+1)
    
    if (len(q) > 3):        
        # Create the circuit
        l = list([q_reg[x] for x in range(n-1, -1, -1)])
        l.append(q_coin[0])
        q_circ.append(pr, l)
        q_circ.crz(phi, q[-2], q[-1]) # Matrix A
        q_circ.cu3(theta, 0, 0, q[-2], q[-1]) # Controlled rotation
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, *q[:-2], q[-1]) # Call the function again
        q_circ.cu3(-theta, 0, 0, q[-2], q[-1]) # Matrix B
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, *q[:-2], q[-1]) # Call the function again
        q_circ.crz(-phi, q[-2], q[-1]) # Matrix C
        
    elif (len(q) == 3):
        q_circ.ccx(*q) # If there is only 2 control qubits, do a Toffoli gate
    
    elif (len(q) == 2):
        q_circ.cx(*q) # If there is only 1 control qubit, do a CNOT
    
    return q_circ

# Define Increment function with rotations
def rot_inc(q_circ, q_reg, q_coin, n):
    """This is the increment circuit with rotations"""
    if (n==20):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16], q_reg[17],
                                      q_reg[18], q_reg[19])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16], q_reg[17],
                                      q_reg[18])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16], q_reg[17])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-10, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-11, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-12, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-13, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-14, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-15, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-16, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-17, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==19):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16], q_reg[17],
                                      q_reg[18])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16], q_reg[17])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-10, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-11, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-12, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-13, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-14, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-15, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-16, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==18):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16], q_reg[17])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-10, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-11, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-12, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-13, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-14, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-15, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==17):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-10, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-11, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-12, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-13, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-14, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==16):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-10, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-11, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-12, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-13, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==15):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-10, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-11, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-12, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==14):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-10, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-11, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==13):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-10, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==12):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==11):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==10):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==9):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==8):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==7):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==6):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==5):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==4):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==3):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    
    q_circ.ccx(q_coin[0], q_reg[0], q_reg[1])
    q_circ.cx(q_coin[0], q_reg[0])
    q_circ.barrier()
    
    return q_circ
    
# Define Decrement function with rotations
def rot_dec(q_circ, q_reg, q_coin, n):
    """This is the decrement circuit with rotations"""
    q_circ.x(q_coin[0]) # Reverse the coin to decrease

    # Circuit
    for i in range(0, n-1):
        q_circ.cx(q_coin[0], q_reg[i]) # Invert the registers necessary
    
    if (n==20):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16], q_reg[17],
                                      q_reg[18], q_reg[19])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16], q_reg[17],
                                      q_reg[18])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16], q_reg[17])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-10, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-11, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-12, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-13, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-14, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-15, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-16, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-17, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==19):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16], q_reg[17],
                                      q_reg[18])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16], q_reg[17])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-10, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-11, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-12, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-13, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-14, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-15, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-16, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==18):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16], q_reg[17])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-10, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-11, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-12, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-13, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-14, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-15, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==17):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-10, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-11, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-12, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-13, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-14, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==16):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-10, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-11, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-12, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-13, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==15):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-10, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-11, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-12, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==14):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-10, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-11, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==13):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11], q_reg[12])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-10, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==12):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10], q_reg[11])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-9, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==11):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9], q_reg[10])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-8, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==10):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8], 
                                      q_reg[9])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-7, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==9):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7], q_reg[8])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-6, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==8):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6], q_reg[7])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-5, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==7):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5], q_reg[6])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-4, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==6):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4], q_reg[5])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-3, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==5):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-2, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==4):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2], q_reg[3])
        q_circ = genTofRot(n-1, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    elif (n==3):
        q_circ = genTofRot(n, q_reg, q_coin, q_circ, q_coin[0], q_reg[0], q_reg[1], q_reg[2])
    
    q_circ.ccx(q_coin[0], q_reg[0], q_reg[1])
    q_circ.cx(q_coin[0], q_reg[0])
    
    for i in range(0, n-1):
        q_circ.cx(q_coin[0], q_reg[i]) # Uncompute the registers necessary

    q_circ.x(q_coin[0]) # Uncompute the reversion of the coin
    q_circ.barrier()
    
    return q_circ

def getPhiOperator(n):
    '''Returns the Phi(delta) operator.'''
    # Create the matrix representation
    N = 2**n
    d = [[0] * N for i in range(N)]
    for i in range(0, N-2):
        for j in range (0, N-2):
            if (i==j):
                d[i][j] = 1

    d[N-2][N-2] = 0.+1.j
    d[N-1][N-1] = 0.+1.j
    
    # Create the phase operator
    pr = Operator(d)
    
    return pr

# Create the Quantum Circuit
def rotWalk(n, t):
    '''Function that generates the quantum circuit for the walk, using rotation gates instead
    of generalised CNOT. Number of qubits for the state space, n, number of coin flips, t, and
    returns quantum circuit and number of gates.'''
    q_reg = QuantumRegister(n, 'q')
    q_coin = QuantumRegister(1, 'coin')
    c_reg = ClassicalRegister(n, 'c')
    qwalk_circuit = QuantumCircuit(q_reg, q_coin, c_reg)

    def runQWC(qwalk_circuit, steps, n):
    
        for i in range(steps):
            qwalk_circuit.h(q_coin[0])
            qwalk_circuit = rot_inc(qwalk_circuit, q_reg, q_coin, n)
            qwalk_circuit = rot_dec(qwalk_circuit, q_reg, q_coin, n)

        qwalk_circuit.measure(q_reg, c_reg)

        return qwalk_circuit

    steps = t
    qwalk_circuit = runQWC(qwalk_circuit, steps, n)
    
    return qwalk_circuit

def calcSimRuntime(n, t):
    '''Returns the simulation runtime of the quantum walk circuit for specific n and t.'''
    print("Number of qubits:", n)
    tm = 0
    circ = rotWalk(n, t)
    start = time.time()
    simulate = execute(circ, backend=Aer.get_backend("qasm_simulator"), shots=1)
    end = time.time()
    tm = end - start
    
    return tm

def simulateQWRot(n, t, rep):
    '''Runs the simulation and returns the probability distribution in the form of a dictionary.
    n - number of qubits, t - number of coin-flips, rep - number of repetitions of the experiment.'''
    # Create the circuit
    qwcirc = rotWalk(n, t)
    print(qwcirc.size())
    qwcirc.draw(output='mpl')
    
    # Run walk on local simulator
    backend = Aer.get_backend("qasm_simulator")
    
    start = time.time()
    simulate = execute(qwcirc, backend=backend, shots=rep).result()
    end = time.time()
    counts = simulate.get_counts()
    print("Runtime:", end-start)
    
    return counts

def runqcQWRot(n, t, rep):
    '''Runs the quantum walk on the real machine and returns the probability distribution in the form of a dictionary.
    n - number of qubits, t - number of coin-flips, rep - number of repetitions of the experiment.'''
    provider = IBMQ.load_account()
    device = provider.get_backend('ibmq_16_melbourne')
    
    # Create the circuit
    qwcirc = walk(n, t)
    print(qwcirc.size())
    
    # Run walk on quantum machine
    job = execute(qwcirc, backend=device, shots=rep)
    job_monitor(job)
    results = job.result()
    countsqc = results.get_counts()
    countsqc = dict(OrderedDict(sorted(countsqc.items())))

    return countsqc