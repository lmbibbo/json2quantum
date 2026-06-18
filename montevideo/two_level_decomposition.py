import numpy as np
import json


def convert_1to2(U: np.ndarray, destination_file = None):
    """
    Given a circuit represented by a unitary matrix U (this is, a circuit with abstraction level = 1),
    generate a circuit with abstraction level = 2 (this is, a circuit with generalized toffoli gates and 
    multi controlled single qubit gates)
    """

    # Check that U is unitary and a valid quantum circuit (this is, the dimension of U is a power of 2)
    if U.shape[0] != U.shape[1] or not is_unitary(U):
        raise ValueError("U must be a unitary square matrix")

    dimensions = U.shape[0]
    qubit_count = int(np.log2(dimensions))

    if (2**qubit_count != dimensions):
        raise ValueError("U must be a 2^n times 2^n matrix")

    _, matrices_json = two_level_decomposition(U, dimensions)

    circuit = {
        "abstraction_level": 2,
        "qubit_count": qubit_count,
        "ancilla_qubits": [],
        "operations": []
    }

    for matrix_json in matrices_json:
        sub_circuit = toffoli_decomposition(matrix_json["states"][0], matrix_json["states"][1], qubit_count, matrix_json["matrix"])
        append_sub_circuit(circuit, sub_circuit)

    if destination_file:
        with open(destination_file, "w") as f:
            json.dump(circuit, f, indent=2)
    
    return circuit


def append_sub_circuit(circuit, sub_circuit):
    """
    Given a circuit and a sub_circuit, append the sub_circuit into the circuit.
    circuit and sub_circuit must have the same qubit_count
    """

    if (sub_circuit["qubit_count"] != circuit["qubit_count"]):
        raise ValueError("Both circuits must have the same number of qubits")

    for operation in sub_circuit["operations"]:
        circuit["operations"].append(operation)




def two_level_decomposition(U: np.ndarray, dimensions: int) -> list:
    """
    Given a matrix U, return the two-level matrix decomposition.
    The decomposition is represented by an array of python dictionaries of the form:
    {"states" = [state1, state2], matrix = Ui}
    """


    if U.shape[0] != U.shape[1]:
        raise ValueError("Non-square matrix")
    if U.shape[0] <= 1:
        raise ValueError("Unitary matrix should have dimensions >= 2")

    n = U.shape[0]

    # Total matrix count: n*(n-1)//2 + 1, stored at indices 1..n*(n-1)//2+1
    # Index 0 is left as None to preserve the same 1-based indexing as Octave
    total = n * (n - 1) // 2 
    matrices = [None] * (total + 1)
    # matrices_json = {}
    matrices_json = []

    A = U.astype(complex).copy()

    # Zero out the first column of A below the diagonal
    for i in range(2, n + 1):             # i = 2..n  (Octave 1-based)
        Ui = np.eye(n, dtype=complex)
        Ui_2by2 = np.eye(2, dtype = complex)
        a = A[0, 0]

        

        if abs(A[i - 1, 0]) > 1e-5:
            b = A[i - 1, 0]
            norm_ab = np.linalg.norm([a, b])
            Ui[0,     0    ] =  np.conj(a) / norm_ab
            Ui[i - 1, i - 1] = -a          / norm_ab
            Ui[0,     i - 1] =  np.conj(b) / norm_ab
            Ui[i - 1, 0    ] =  b          / norm_ab

            Ui_2by2[0, 0] =  np.conj(a) / norm_ab
            Ui_2by2[1, 1] = -a          / norm_ab
            Ui_2by2[0, 1] =  np.conj(b) / norm_ab
            Ui_2by2[1, 0] =  b          / norm_ab

        else:
            Ui[0, 0] = np.conj(a)



        json_Ui = {"states": [dimensions - n, dimensions - n + i - 1], "matrix": Ui_2by2}
        

        Mi = np.eye(dimensions, dtype=complex)
        Mi[dimensions - n : dimensions,
           dimensions - n : dimensions] = Ui

        # matrices_json[f"U{i-1}"] = json_Ui
        matrices_json.append(json_Ui)
        matrices[i - 1] = Mi              # Octave: matrices(:,:, i-1)
        A = Ui @ A


    if n == 3:
        # matrices_json["U3"] = {"states": [dimensions - 2, dimensions - 1], "matrix": A[1:, 1:].conj().T}
        matrices_json.append({"states": [dimensions - 2, dimensions - 1], "matrix": A[1:, 1:].conj().T})

        Mi = np.eye(dimensions, dtype=complex)
        Mi[dimensions - 2 : dimensions,
           dimensions - 2 : dimensions] = A[1:, 1:].conj().T
        
        matrices[3] = Mi  
        
        return matrices, matrices_json

        

    else:
        # Recursive call on the (n-1)×(n-1) lower-right sub-matrix
        if n == 3:
            print_matrix(A[1:, 1:])
        recursive_matrices, recursive_matrices_json = two_level_decomposition(A[1:, 1:], dimensions)

        # Map recursive index j -> current index (n-1)+j
        # mirrors Octave: v = n : (size(recursive_matrices)(3) + n - 1)
        rec_count = len(recursive_matrices) - 1   # valid entries (index 0 is None)
        for j in range(1, rec_count + 1):
            matrices[n - 1 + j] = recursive_matrices[j]
            # matrices_json[f"U{n-1+j}"] = recursive_matrices_json[f"U{j}"]
            matrices_json.append(recursive_matrices_json[j-1])


    return matrices, matrices_json



def toffoli_decomposition(state1: int, state2: int, qubit_count: int, U: np.ndarray):
    """
    Given a 2 by 2 matrix U and two states in which the matrix U act, return a JSON with the Toffoli decomposition
    """

    bin_state1 = bin(state1)[2:]
    bin_state1 = list("0"*(qubit_count - len(bin_state1)) + bin_state1)   # Fill bin_state1 with zeros on the left

    bin_state2 = bin(state2)[2:]
    bin_state2 = list("0"*(qubit_count - len(bin_state2)) + bin_state2)  # Fill bin_state2 with zeros on the left


    circuit = {
        "qubit_count": qubit_count,
        "operations": []
        }


    current_state = bin_state1
    difference = [q for q in range(qubit_count) if current_state[q] != bin_state2[q]]
    while len(difference) > 1:

        q = difference[0] # Get first qubit in wich the current_state and bin_state2 differ:

        # Define generalized toffoli gate:
        generalized_toffoli = {
        "type": "generalized_toffoli",
        "targets": [q],
        "controls": [
            {"qubit": t, "state": current_state[t]}
            for t in range(qubit_count) if t != q
        ],
        "unitary": {
            "matrix": [
                [0, 1],
                [1, 0]
            ]
            }
        }

        # Add generalized toffoli to circuit:
        circuit["operations"].append(generalized_toffoli)

        # update current state:
        current_state[q] = str(1 - int(current_state[q])) 

        # re-compute difference between states
        difference = [q for q in range(qubit_count) if current_state[q] != bin_state2[q]]


    k = len(circuit["operations"]) # total numbers of generalized toffoli gates before the controlled U gate


    # Append U matrix to the circuit
    q = difference[0]
    controlled_U = {
        "type": "controlled_unitary",
        "targets": [q],
        "controls": [
            {"qubit": t, "state": current_state[t]}
            for t in range(qubit_count) if t != q
        ],
        "unitary": {
            "matrix":  complex_matrix_to_json(U)
        }
    }


    circuit["operations"].append(controlled_U)



    # Append the generalized toffoli gates in reverse order:
    for i in range(k):
        circuit["operations"].append(circuit["operations"][k - i - 1])
        


    return circuit


def is_unitary(U, tol=1e-10):
    """Helper method to check if matrix U is unitary within a given tolerance"""
    U_dagger = U.conj().T
    identity = np.eye(U.shape[0])

    return np.allclose(U_dagger @ U, identity, atol=tol)


def complex_matrix_to_json(matrix):
    return [
        [
            {
                "real": float(x.real),
                "imag": float(x.imag)
            }
            for x in row
        ]
        for row in matrix
    ]


# Constants for color print
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_matrix(matrix):

    rows = matrix.shape[0]
    columns = matrix.shape[1]

    number_size = max([len(f"{matrix[r, c]:.2f}") for r in range(rows) for c in range(columns)])

    for r in range(rows):
        for c in range(columns):
            element = ""
            non_zero = False

            if matrix[r, c] != 0:
                element = f"{matrix[r, c]:.2f}"
                non_zero = True
                # print(bcolors.FAIL + f"{matrix[r, c]:.2f}" + bcolors.ENDC, end = " "*4)

            elif matrix[r, c] == 0:
                element = "0"
                # print("0", end = " "*4)

            spacing_left = ((number_size - len(element)) // 2)
            spacing_rigth = spacing_left + (number_size - len(element)) % 2

            element = " "*spacing_left + element + " "*spacing_rigth
            print(element, end = " "*4)
            


        print("\n", end = "") 