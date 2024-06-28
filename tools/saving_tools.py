import json
import numpy as np

def save_dict_to_json(data, filename):
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    # Convert the dictionary to a serializable format
    serializable_data = convert_to_serializable(data)

    # Save the serializable dictionary to a JSON file
    with open(filename, 'w') as file:
        json.dump(serializable_data, file)


def load_dict_from_json(filename):
    # Load the dictionary from the JSON file
    with open(filename, 'r') as file:
        loaded_data = json.load(file)

    return loaded_data
