from typing import Tuple
from decouple import config
import numpy as np
from qci_client import QciClient
import json

class DiracSolution:
    def __init__(self, q_matrix: np.ndarray):
        self.qubo_data = {
            'file_name': "smallest_objective.json",
            'file_config': {'qubo': {"data": q_matrix}}
        }
        self.api_url = config("QCI_URL", default="https://api.qci-prod.com")
        self.api_token = config("QCI_TOKEN", default="YOUR TOKEN") # type: ignore
        self.qclient = QciClient(url=self.api_url, api_token=self.api_token) #type: ignore

        print("Initialized parameters!!!")
    def error_status(self, job_response):
        print("Checking for errors ...")
        try:
            if job_response['status'] == "ERROR": # type: ignore
                return job_response['status'], job_response['job_info']['results']['error'] # type: ignore
            else:
                return False
        except KeyError:
            return "Error: Unable to retrieve error status information from the job response"


    def solve(self) -> Tuple[np.ndarray , np.ndarray] | None:
        print("Uploading file ....")
        response_json = self.qclient.upload_file(file=self.qubo_data)
        print("Starting job ....")
        file_path = "fca/assets/output.json"
        with open(file_path, "w") as f:
            f.write('hello world')
        job_body = self.qclient.build_job_body(job_type="sample-qubo",
                                  qubo_file_id=response_json['file_id'],
                                  job_params={
                                        "device_type": "dirac-1",
                                        "num_samples": 5,
                                        "relaxation_schedule": 1
                                      }
                                )
        print("Getting a response ....")
        job_response = self.qclient.process_job(job_body=job_body)

        with open(file_path, "w") as json_file:
            json.dump(job_response, json_file, indent=4)

        if type(response := self.error_status(job_response)) != bool:
            print(response)
            return None
        
        all_solutions, all_energies = np.asarray(job_response["results"]["solutions"]), np.asarray(job_response["results"]["energies"])
        print("Got the solution")
        if len(all_energies) == 0 or len(all_solutions) == 0:
            return np.ndarray([]), np.ndarray([])
        energy = all_energies[0]
        min_index = 0
        for i in range(1, len(all_energies)):
            if all_energies[i] < energy:
                min_index = i
                energy = all_energies[i]

        return all_solutions[min_index], energy


