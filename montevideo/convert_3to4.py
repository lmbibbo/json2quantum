import numpy as np
import json


def convert_3to4(circuit_spec: dict | str, destination_file = None):
    

    if isinstance(circuit_spec, str):
        spec = json.loads(circuit_spec)
    else:
        spec = circuit_spec  # already a dict

    qubit_count = spec["qubit_count"]


    # Transform multicontrolled U to toffoli and controlled U:

    circuit = {
        "abstraction_level": 3,
        "qubit_count": qubit_count, # qubit_count only counts qubits from principal register
        "ancilla_qubits": [qubit_count],
        "operations": []
    }

    for operation in spec["operations"]:
        if operation["type"] == "controlled_unitary":

            

            # Define generalized toffoli gate:
            generalized_toffoli = {
            "type": "generalized_toffoli",
            "targets": [qubit_count], # Ancilla qubit
            "controls": operation["controls"],
            "unitary": {
                "matrix": [
                    [0, 1],
                    [1, 0]
                ]
                }
            }


            controlled_unitary = {
                "type": "singy_controlled_unitary",
                "targets": operation["targets"],
                "controls": [
                    {"qubit": qubit_count, "state": 0}
                ],
                "unitary": operation["unitary"]
            }


            circuit["operations"].append(generalized_toffoli)
            circuit["operations"].append(controlled_unitary)
            circuit["operations"].append(generalized_toffoli)



        elif operation["type"] == "generalized_toffoli":

            circuit["operations"].append(operation)


    if destination_file:
        with open(destination_file, "w") as f:
            json.dump(circuit, f, indent=2)


    return circuit


