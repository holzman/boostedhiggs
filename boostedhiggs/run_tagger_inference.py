"""
Methods for running the tagger inference.
Author(s): Raghav Kansal, Cristina Mantilla Suarez, Melissa Quinnan, Farouk Mokhtar
"""

import json

# import time
from typing import Dict

import numpy as np
import tritonclient.grpc as triton_grpc
import tritonclient.http as triton_http
from coffea.nanoevents.methods.base import NanoEventsArray
from tqdm import tqdm

from .get_tagger_inputs import get_pfcands_features, get_svs_features

# import onnxruntime as ort


# adapted from https://github.com/lgray/hgg-coffea/blob/triton-bdts/src/hgg_coffea/tools/chained_quantile.py
class wrapped_triton:
    def __init__(self, model_url: str, batch_size: int, out_name: str = "softmax__0") -> None:
        fullprotocol, location = model_url.split("://")
        _, protocol = fullprotocol.split("+")
        address, model, version = location.split("/")

        self._protocol = protocol
        self._address = address
        self._model = model
        self._version = version

        self._batch_size = batch_size
        self._out_name = out_name

    def __call__(self, input_dict: Dict[str, np.ndarray]) -> np.ndarray:
        if self._protocol == "grpc":
            client = triton_grpc.InferenceServerClient(url=self._address, verbose=False)
            triton_protocol = triton_grpc
        elif self._protocol == "grpcs":
            client = triton_grpc.InferenceServerClient(url=self._address, verbose=False, ssl=True)
            triton_protocol = triton_grpc
        elif self._protocol == "http":
            client = triton_http.InferenceServerClient(
                url=self._address,
                verbose=False,
                concurrency=12,
            )
            triton_protocol = triton_http
        else:
            raise ValueError(f"{self._protocol} does not encode a valid protocol (grpc or http)")

        # manually split into batches for gpu inference
        input_size = input_dict[list(input_dict.keys())[0]].shape[0]
        # print(f"size of input (number of events) = {input_size}")

        outs = [
            self._do_inference(
                {key: input_dict[key][batch : batch + self._batch_size] for key in input_dict},
                triton_protocol,
                client,
            )
            for batch in tqdm(range(0, input_dict[list(input_dict.keys())[0]].shape[0], self._batch_size))
        ]

        return np.concatenate(outs) if input_size > 0 else outs

    def _do_inference(self, input_dict: Dict[str, np.ndarray], triton_protocol, client) -> np.ndarray:
        # Infer
        inputs = []

        for key in input_dict:
            input = triton_protocol.InferInput(key, input_dict[key].shape, "FP32")
            input.set_data_from_numpy(input_dict[key])
            inputs.append(input)

        output = triton_protocol.InferRequestedOutput(self._out_name)

        request = client.infer(
            self._model,
            model_version=self._version,
            inputs=inputs,
            outputs=[output],
        )

        return request.as_numpy(self._out_name)


def runInferenceTriton(
    tagger_resources_path: str,
    events: NanoEventsArray,
    fj_idx_lep,
    model_name: str = "ak8_MD_vminclv2ParT_manual_fixwrap",
) -> dict:
    # total_start = time.time()
    # print(f"Running tagger inference with model {model_name}")

    with open(f"{tagger_resources_path}/triton_config_{model_name}.json") as f:
        triton_config = json.load(f)

    with open(f"{tagger_resources_path}/{triton_config['model_name']}.json") as f:
        tagger_vars = json.load(f)

    pversion, out_name = {
        "particlenet_hww_inclv2_pre2": ["ParticleNet", "output__0"],
        "particlenet_hww_inclv2_pre2_noreg": ["PN_v2_noreg", "softmax__0"],
        "ak8_MD_vminclv2ParT_manual_fixwrap": ["ParT", "softmax"],
        "ak8_MD_vminclv2ParT_manual_fixwrap_all_nodes": ["ParT", "softmax"],
    }[model_name]

    triton_model = wrapped_triton(triton_config["model_url"], triton_config["batch_size"], out_name=out_name)

    fatjet_label = "FatJet"
    pfcands_label = "FatJetPFCands"
    svs_label = "FatJetSVs"

    # prepare inputs for the fatjet
    tagger_inputs = []

    feature_dict = {
        **get_pfcands_features(tagger_vars, events, fj_idx_lep, fatjet_label, pfcands_label),
        **get_svs_features(tagger_vars, events, fj_idx_lep, fatjet_label, svs_label),
    }

    for input_name in tagger_vars["input_names"]:
        for key in tagger_vars[input_name]["var_names"]:
            np.expand_dims(feature_dict[key], 1)

    if out_name == "softmax":
        tagger_inputs = {
            f"{input_name}": np.concatenate(
                [np.expand_dims(feature_dict[key], 1) for key in tagger_vars[input_name]["var_names"]],
                axis=1,
            )
            for i, input_name in enumerate(tagger_vars["input_names"])
        }
    else:
        tagger_inputs = {
            f"{input_name}__{i}": np.concatenate(
                [np.expand_dims(feature_dict[key], 1) for key in tagger_vars[input_name]["var_names"]],
                axis=1,
            )
            for i, input_name in enumerate(tagger_vars["input_names"])
        }

    # run inference on the fat jet
    # try:
    tagger_outputs = triton_model(tagger_inputs)
    # except Exception:
    #     print("---can't run inference due to error with the event or the server is not running--")
    #     return {}

    # get the list of output labels defined in `model_name.json` and replace label_ by prob
    output_names = [x.replace("label_", "prob").replace("_", "") for x in tagger_vars["output_names"]]

    pnet_vars = {}
    if pversion == "ParticleNet":  # missing softmax for that model (unfortunately)
        import scipy

        # last index is mass regression
        tagger_outputs[:, :-1] = scipy.special.softmax(tagger_outputs[:, :-1], axis=1)

    for i, output_name in enumerate(output_names):
        pnet_vars[f"fj_{pversion}_{output_name}"] = tagger_outputs[:, i]

    return pnet_vars
