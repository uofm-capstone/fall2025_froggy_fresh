import os
import json
from datetime import datetime

def load_run_data(folder_path):
    # Return a dictionary of datetime keys and frog count values from all run JSONs
    run_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")]
    graph_data = {}
    for run in run_files:
        with open(run, 'r') as file:
            data = json.load(file)
            if "runDate" in data and "frogs" in data:
                # TODO: Once the Camera CSV Column PR is approved, add in camera number filtering 
                run_time = datetime.strptime(data["runDate"], "%Y-%m-%dT%H_%M_%S")
                graph_data[run_time] = data["frogs"]
    return graph_data


if __name__ == "__main__":
    load_run_data(os.path.join(os.path.expanduser("~"), "Documents", "Leapfrog", "runs"))