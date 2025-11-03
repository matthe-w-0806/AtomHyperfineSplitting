import math
import numpy as np

mu_B = 9.2740100657 * 10 ** -24
hbar = 1.05 * 10 ** -34

                            
class Potassium:

    g_I = 0.000176490 
    g_S = 2.0023193043737
    g_L = 0.99998627

    def __init__(self) -> None:
        pass

class _2S12(Potassium):

    g_J = 2.00229421
    A_hfs = -285.731 * 10 ** 6
    B_hfs = 0
    IsoShift = 125.58 * 10 ** 6

    def __init__(self) -> None:
        super().__init__()

class _2P32(Potassium):

    g_J = 1.334102228
    A_hfs = -7.48 * 10 ** 6
    B_hfs = - 3.23 * 10 ** 6
    IsoShift = 0

    def __init__(self) -> None:
        super().__init__()

p = _2S12()

def IdotJ(J,I,mj1,mi1,mj2,mi2):
    if mj1 == mj2 and mi1 == mi2:
        return mj1*mi1
    elif mj2 == mj1+1 and mi2 == mi1-1:
        return 0.5*((J-mj1)*(J+mj1+1)*(I+mi1)*(I-mi1+1))**0.5
    elif mj2 == mj1-1 and mi2 == mi1+1:
        return 0.5*((J+mj1)*(J-mj1+1)*(I-mi1)*(I+mi1+1))**0.5
    else:
        return 0
    
def f(J,I,mj1,mi1,mj2,mi2):
    if mj1 == mj2 and mi1 == mi2:
        return 0.5 * (1.5 * mi1 ** 2 - I * (I+1)) * (3 * mj1 ** 2 - J * (J+1))
    if mj2 == mj1-1 and mi2 == mi1+1:
        return 0.75 * (2 * mj1 -1) * (2 * mi1+1) * ((J + mj1) * (J - mj1 + 1) * (I - mi1) * (I + mi1 + 1)) ** 0.5
    if mj2 == mj1+1 and mi2 == mi1-1:
        return 0.75 * (2 * mj1 -1) * (2 * mi1+1) * ((J + mj1) * (J - mj1 + 1) * (I - mi1) * (I + mi1 + 1)) ** 0.5
    if mj2 == mj1-2 and mi2 == mi1+2:
        return 0.75 * ((J+mj1)*(J+mj1+1)*(J-mj1+1)*(J-mj1+2)*(I-mi1)*(I-mi1-1)*(I+mi1+1)*(I+mi1+2)) ** 0.5
    if mj2 == mj1+2 and mi2 == mi1-2:
        return 0.75 * ((J-mj1)*(J-mj1-1)*(J+mj1+1)*(J+mj1+2)*(I+mi1)*(I+mi1-1)*(I-mi1+1)*(I-mi1+2)) ** 0.5
    else:
        return 0

def dirac_delta(i,j):
    if i == j:
        return i
    else:
        return 0

def H_element(J,I,mj1,mi1,mj2,mi2,B):
    return p.A_hfs * IdotJ(J,I,mj1,mi1,mj2,mi2) + p.B_hfs * f(J,I,mj1,mi1,mj2,mi2) / (2*I*J*(2*I-1)*(J-1)) + mu_B / hbar * B * (dirac_delta(mi1,mi2) * p.g_I + dirac_delta(mj1,mj2) * p.g_J)

def H_matrix(J,I,B):
    N = int((2*I+1) * (2*J+1))
    H = np.zeros((N,N))
    _J = np.arange(J,-J-1,-1)
    _I = np.arange(I,-I-1,-1) #Arange doesn't include endpoint, add extra step
    substate = []
    for mj in _J:
        for mi in _I:
            substate.append((mj,mi,len(substate)))
    
    for (mj1,mi1,ROW) in substate:
        for (mj2,mi2,COLUMN) in substate:
            H[ROW,COLUMN] = H_element(J,I,mj1,mi1,mj2,mi2,B)

    return H

H = H_matrix(0.5,4,0)
H = H_matrix(0.5,4,0)
print(np.linalg.eig(H)[0])

import matplotlib.pyplot as plt

J = 0.5
I = 4
B_vals = np.linspace(0.0, 0.1, 201)  # Tesla
N = int((2*I+1)*(2*J+1))

eigvals = np.zeros((len(B_vals), N))

for i, B in enumerate(B_vals):
    H = H_matrix(J, I, B)
    w = np.linalg.eigvalsh(H)
    eigvals[i, :] = np.sort(w)

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot()

for n in range(N):
    ax.plot(B_vals, eigvals[:, n], lw=1)
plt.xlabel("B (T)")
plt.ylabel("Eigenvalues (Hz)")
plt.title("Eigenvalues of H vs magnetic field B")
plt.grid(True)
plt.tight_layout()
plt.show()
