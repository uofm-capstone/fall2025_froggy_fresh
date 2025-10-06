import os
import json
import datetime

def load_run_data(folder_path):
    run_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")]
    for run in run_files:
        with open(run, 'r') as file:
            data = json.load(file)
            if "runDate" in data and "frogs" in data:
                run_time = datetime.strptime(data["runDate"], "%Y-%m-%dT%H_%M_%S")
                print(run_time)

if __name__ == "__main__":
    load_run_data(os.path.join(os.path.expanduser("~"), "Documents", "Leapfrog", "runs"))