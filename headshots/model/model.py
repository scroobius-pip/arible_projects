import copy
import json
import os
import time
import uuid
from multiprocessing import Process
from typing import Dict
import random
import websocket
from model.helpers import (
    convert_outputs_to_base64,
    # convert_request_file_url_to_path,/
    convert_image_urls_to_paths,
    fill_template,
    get_images,
    setup_comfyui,
)

side_process = None
original_working_directory = os.getcwd()


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._model = None
        self.ws = None
        self.json_workflow = None
        self.server_address = "127.0.0.1:8188"
        self.client_id = str(uuid.uuid4())

    def load(self):
        # Start the ComfyUI server
        global side_process
        if side_process is None:
            side_process = Process(
                target=setup_comfyui,
                kwargs=dict(
                    original_working_directory=original_working_directory,
                    data_dir=self._data_dir,
                ),
            )
            side_process.start()
            print("ComfyUI process started")

        # Load the workflow file as a python dictionary
        with open(
            os.path.join(self._data_dir, "comfy_ui_workflow.json"), "r"
        ) as json_file:
            self.json_workflow = json.load(json_file)

        # Connect to the ComfyUI server via websockets
        socket_connected = False
        while not socket_connected:
            try:
                self.ws = websocket.WebSocket()
                self.ws.connect(
                    "ws://{}/ws?clientId={}".format(self.server_address, self.client_id)
                )
                socket_connected = True
            except Exception as e:
                print(
                    f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"
                )
                print("Error connecting to ComfyUI server:", e)
                # print("Could not connect to comfyUI server. Trying again...")
                time.sleep(5)

        print("Truss has successfully connected to the ComfyUI server!")

    def predict(self, model_input: Dict) -> Dict:

        ref_image_urls = model_input["ref_image_urls"]
        if len(ref_image_urls) < 5:
            remaining = 5 - len(ref_image_urls)
            for i in range(remaining):
                ref_image_urls.append(ref_image_urls[0])

        random.shuffle(ref_image_urls)

        prompts = model_input["prompts"]
        negative_prompt = model_input["negative_prompt"]

        ref_image_paths, tempfiles = convert_image_urls_to_paths(ref_image_urls)
        json_workflow = copy.deepcopy(self.json_workflow)
        template_values = {f"ref_{i}": value for i, value in enumerate(ref_image_paths)}
        template_values["negative_prompt"] = negative_prompt

        results = []
        for prompt in prompts:
            template_values["positive_prompt"] = prompt
            json_workflow = fill_template(json_workflow, template_values)
         
            try:
                outputs = get_images(
                    self.ws, json_workflow, self.client_id, self.server_address
                )
                for node_id in outputs:
                    for item in outputs[node_id]:
                        file_name = item.get("filename")
                        file_data = item.get("data")
                        output = convert_outputs_to_base64(
                            node_id=node_id, file_name=file_name, file_data=file_data
                        )
                        results.append(output)

            except Exception as e:
                print("Error occurred while running Comfy workflow: ", e)

        for file in tempfiles:
            file.close()

        return results
