import os
import os.path
from IPython.display import display
# from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from math import log
import time
import warnings

# importing QISKit
from qiskit import Aer, IBMQ
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit import IBMQ, BasicAer
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from qiskit.providers.aer import StatevectorSimulator
from qiskit.tools.visualization import plot_state_city

# Toffoli gate for n qubits
def ncx(circ, tgt, ctrl, anc):
    '''Create a generalised CNOT gate with more than 2 control qubits.'''
    # Compute
    circ.ccx(ctrl[0], ctrl[1], anc[0])
    for i in range(2, len(ctrl)):
        circ.ccx(ctrl[i], anc[i-2], anc[i-1])
    
    # CNOT
    circ.cx(anc[len(ctrl)-2], tgt)

    # Uncompute
    for i in range(len(ctrl)-1, 1, -1):
        circ.ccx(ctrl[i], anc[i-2], anc[i-1])
    circ.ccx(ctrl[0], ctrl[1], anc[0])
    
    return circ

# Define increment function   
def inc(q_circuit, q_reg, q_anc, q_coin, n):
    '''Increment circuit for n-qubit states. Specifficaly, n=logN.''' 
    for i in range(n, 2, -1):
        tgt = q_reg[i-1] # Target qubit is the MSQ
        ctrl = []
        for j in range(0, i-1):
            ctrl.append(q_reg[j])
        ctrl.append(q_coin[0])
        q_circuit = ncx(q_circuit, tgt, ctrl, q_anc)
    
    q_circuit.ccx(q_coin[0], q_reg[0], q_reg[1])
    q_circuit.cx(q_coin[0], q_reg[0])
    q_circuit.barrier()
    
    return q_circuit

# Define decrement function
def dec(q_circuit, q_reg, q_anc, q_coin, n):
    '''Decrement circuit for n-qubit states. Specifficaly, n=logN.'''
    q_circuit.x(q_coin[0]) # Reverse the coin to decrease

    # Circuit
    for i in range(0, n-1):
        q_circuit.cx(q_coin[0], q_reg[i]) # Invert the registers necessary

    for i in range(n, 2, -1):
        tgt = q_reg[i-1]
        ctrl = []
        for j in range(0, i-1):
            ctrl.append(q_reg[j])
        ctrl.append(q_coin[0])
        q_circuit = ncx(q_circuit, tgt, ctrl, q_anc)
    
    q_circuit.ccx(q_coin[0], q_reg[0], q_reg[1])
    q_circuit.cx(q_coin[0], q_reg[0])

    for i in range(0, n-1):
        q_circuit.cx(q_coin[0], q_reg[i]) # Uncompute the inverted registers

    q_circuit.x(q_coin[0]) # Uncompute the reversion of the coin

    q_circuit.barrier()
    
    return q_circuit

# Function that generates the quantum circuit. Also returns the number of gates
# within the quantum circuit
def walk(n, t):
    '''Function that generates the quantum circuit for the quantum walk. Arguments are given
    as the number of qubits needed to describe the state space, n, and the time steps (i.e coin 
    flips or iterations of the walk), t.'''
    # Create the Quantum Circuit
    q_reg = QuantumRegister(n, 'q')
    q_coin = QuantumRegister(1, 'coin')
    q_anc = QuantumRegister(n-1, 'anc')
    c_reg = ClassicalRegister(n, 'c')
    qwalk_circuit = QuantumCircuit(q_reg, q_anc, q_coin, c_reg)

    def runQWC(qwalk_circuit, steps):
        for i in range(steps):
            qwalk_circuit.h(q_coin[0])
            qwalk_circuit.barrier()
            qwalk_circuit = inc(qwalk_circuit, q_reg, q_anc, q_coin, n)
            qwalk_circuit = dec(qwalk_circuit, q_reg, q_anc, q_coin, n)

        qwalk_circuit.measure(q_reg, c_reg)

        return qwalk_circuit

    steps = t
    qwalk_circuit = runQWC(qwalk_circuit, steps)
    
    return qwalk_circuit

def calcSimRuntime(n, t):
    '''Returns the simulation runtime of the quantum walk circuit for specific n and t.'''
    print("Number of qubits:", n)
    tm = 0
    circ = walk(n, t)
    start = time.time()
    simulate = execute(circ, backend=Aer.get_backend("qasm_simulator"), shots=1)
    end = time.time()
    tm = end - start
    
    return tm

def simulateQW(n, t, rep):
    '''Runs the simulation and returns the probability distribution in the form of a dictionary.
    n - number of qubits, t - number of coin-flips, rep - number of repetitions of the experiment.'''
    # Create the circuit
    qwcirc = walk(n, t)
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

def runqcQW(n, t, rep):
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

def sv(n, t):
    '''Creates the state vector for the quantum walk.'''
    # Create the circuit
    qwcirc = walk(n, t)
    
    # Select the StatevectorSimulator from the Aer provider
    sv_simulator = Aer.get_backend('statevector_simulator')

    # Execute and get counts
    result = execute(qwcirc, sv_simulator).result()
    statevector = result.get_statevector(qwcirc)

    # statevector

    b = []
    for k in range(0,len(statevector)):
        b.append(k)

    i = iter(statevector)
    j = iter(b)
    dct = dict(zip(j, i))

    # Create the dictionary without all the    
    qstate = {}
    for i in range(0,len(dct)):
        qstate[format(i, '#08b')[2:]] = dct[i]

    return qstate

def gateQStateList(dct):
    '''Method that returns the states and the probabilities of the entire experiment (dictionary)
    as a list containing tuples [(x1,p1),...,(xB,pB)]'''
    lst = list(dct.items())
    for i in range(0,len(lst)):
        t = list(lst[i])
        lst[i] = tuple(t)
    
    return lst

def getNonZeros(dct):
    '''Method that returns the positions of the state space that have non-zero amplitudes in
    the form of a dictionary.'''
    non_zeros = dict()

    for key in dct.keys():
        if (dct[key] != (0+0j)):
            non_zeros[key] = dct[key]
            
    return non_zeros

def getProbabilities(dct):
    '''Methods that calculates the probabilities of each position of the state space from
    the amplitudes of the quantum state. Returns a dictionary that contains the states with
    removed the ancilla and coin qubits.'''
    dct = getNonZeros(dct)
    probs = dict()
    dct1 = dict()
    dct2 = dict()
    
    for key in dct.keys():
        if (key[0] == '0'):
            dct1[key] = dct[key]
        elif (key[0] == '1'):
            dct2[key] = dct[key]

    # The two states with the same last 6 qubits (state space) will have their amplitudes squared
    # and sumed to give the probability for the state.
    for key_1 in dct1.keys():
        probs[key_1[6:len(key_1)]] = dct1[key_1]**2
        for key_2 in dct2.keys():
            if (key_1[6:len(key_1)] == key_2[6:len(key_2)]):
                probs[key_1[6:len(key_1)]] = dct1[key_1]**2 + dct2[key_2]**2
            else:
                if key_2[6:len(key_2)] not in probs:
                    probs[key_2[6:len(key_2)]] = dct2[key_2]**2
                    
    dec = [0]*len(probs.keys())
    probs_dec = dict()
    
    for key in probs.keys():
        probs_dec[int(key, 2)] = probs[key]
    
    return probs_dec

def getMean(dct):
    '''Return the mean of the states of the quantum walk. Additionally return the
    dictionary of the probabilities as a list so it can be used'''
    lst = gateQStateList(dct) # Make the dct to list with tuples
    mean = 0
    x_i = 0
    p_i = 0
        
    for i in range(0, len(lst)):
        t = list(lst[i]) # Make the tuple as a list, so we can take x and p separately
        x_i = t[0] # Take the position
        p_i = t[1].real
        mean = mean + x_i*p_i
        
    return mean,lst

def qStateVar(dct):
    '''Method that calculates the variance of the quantum state. Needs to get as input
    the dictionary with the probabilities, not the amplitudes.'''
    dct = getProbabilities(dct) # Get the probabilities dictionary with decimal positions
    mean,lst = getMean(dct) # Calculate the expected value for the quantum walk
    
    # Calculate the variance
    x_i = 0
    p_i = 0
    var = 0
    
    for i in range(0, len(lst)):
        t = list(lst[i]) # Make the tuple as a list, so we can take x and p separately
        x_i = t[0] # Take the position
        p_i = t[1].real # Take the probability as a real number (just due to error of np.complex)
        var = var + ((x_i - mean)**2)*p_i
    
    return var

def qStVarSd(N, t):
    '''Produce the variance and standard deviation for a quantum walk of N states and 
    t coin flips. Works for a single input of t'''
    n = int(np.log2(N))
    
    qwalk = walk(n, t) # Generate the quantum walk circuit for n qubits
    qstate = getQState(qwalk) # Get the quantum state vector
    var = qStateVar(qstate)
    sd = np.sqrt(var)
        
    return var,sd

def qStVarSdMult(N, t):
    '''Method to run the quantum walk and get all the quantum states for the desired
    coin flips, t. Coin flips need to be defined as a list.'''
    states = [{}]*len(t)
    var = [0]*len(t)
    sd = [0]*len(t)
    qstate = [{}]*len(t)
    n = int(np.log2(N))
    
    for i in range(0, len(t)):
        qwalk = walk(n, t[i]) # Generate the quantum walk circuit for n qubits
        qstate[i] = getQState(qwalk) # Get the quantum state vector

    for i in range(0, len(t)):
        var[i] = qStateVar(qstate[i])
        sd[i] = np.sqrt(var[i])
        
    return var,sd

def showProbStates(N, t):
    '''Method that returns the quantum states after a certain number of coin flips along
    with the probability of each quantum state to appear with no ancillas and/or coins.
    NOTE: this specific method does not work with a list of coin flips, t.'''
    n = int(np.log2(N))
    
    qwalk = walk(n, t)
    qstate = getQState(qwalk)
    dct = getProbabilities(qstate)
    
    kleidia = list(dct.keys())
    times = list(dct.values())
    sum = 0
    for i in range(0,len(kleidia)):
        print(kleidia[i], ":", times[i])
        sum = sum + times[i]
        
    print("The sum of the probabilities is:", sum)
    
def theorVar(t):
    '''Method that calculates the theoretical variance of a quantum walk on an N-cycle
    given the number of coin flips, t.'''
    s = (np.sqrt(2)-1)/np.sqrt(2)
    var = (t**2)*s*(1-s)
    sd = np.sqrt(var)
    
    return var,sd

def getTheorVar(t):
    '''Return the theoretical variance calculated for the quantum walk on a cycle of any
    size.'''
    var = [0]*len(t)
    sd = [0]*len(t)
    
    for i in range(0, len(t)):
        var[i],sd[i] = theorVar(t[i])
        
    return var,sd