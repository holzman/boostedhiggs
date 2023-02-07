"""
Methods for running the tagger inference.
Author(s): Raghav Kansal, Cristina Mantilla Suarez, Melissa Quinnan, Farouk Mokhtar
"""

import json
from typing import Dict

import awkward as ak
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
            for batch in tqdm(
                range(
                    0,
                    input_dict[list(input_dict.keys())[0]].shape[0],
                    self._batch_size,
                )
            )
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
    # print(f"Running tagger inference with model {model_name}")

    with open(f"{tagger_resources_path}/triton_config_{model_name}.json") as f:
        triton_config = json.load(f)

    with open(f"{tagger_resources_path}/{triton_config['model_name']}.json") as f:
        tagger_vars = json.load(f)

    pversion, out_name = {
        "05_10_ak8_ttbarwjets": ["PN_UCSD", "softmax__0"],
        "particlenet_hww_inclv2_pre2": ["PN_v2", "output__0"],
        "particlenet_hww_inclv2_pre2_noreg": ["PN_v2_noreg", "softmax__0"],
        "ak8_MD_vminclv2ParT_manual_fixwrap": ["ParT", "softmax"],
    }[model_name]

    triton_model = wrapped_triton(
        triton_config["model_url"],
        triton_config["batch_size"],
        out_name=out_name,
    )

    fatjet_label = "FatJet"
    pfcands_label = "FatJetPFCands"
    svs_label = "FatJetSVs"
    # jet_label = "ak8"

    # prepare inputs for both fat jets
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

    # run inference for both fat jets
    tagger_outputs = []

    try:
        tagger_outputs = triton_model(tagger_inputs)
    except ValueError:
        print("---can't run inference due to error with the event or the server is not running--")
        return {}

    if out_name == "output__0":
        import scipy

        mass = tagger_outputs[:, -1]
        tagger_outputs = scipy.special.softmax(tagger_outputs[:, :-1], axis=1)
        np.append(tagger_outputs, mass)

    # print(f"Inference took {time_taken:.1f}s")

    if model_name == "05_10_ak8_ttbarwjets":
        pnet_vars = {
            f"fj_{pversion}_ttbar": tagger_outputs[:, 0:1],
            f"fj_{pversion}_wjets": tagger_outputs[:, 2],
            f"fj_{pversion}_HVV_elenuqq": tagger_outputs[:, 3],
            f"fj_{pversion}_HVV_munuqq": tagger_outputs[:, 4],
            f"fj_{pversion}_HVV_taunuqq": tagger_outputs[:, 5],
        }
    else:
        output_names = [x.replace("label_", "prob").replace("_", "") for x in tagger_vars["output_names"]]

        pnet_vars = {}
        for i, output_name in enumerate(output_names):
            pnet_vars[f"fj_{pversion}_{output_name}"] = tagger_outputs[:, i]

        derived_vars = {
            f"fj_{pversion}_probQCD": np.sum(tagger_outputs[:, 23:28], axis=1),
            f"fj_{pversion}_probTopb": np.sum(tagger_outputs[:, 28:], axis=1),
            f"fj_{pversion}_probHWWelenuqq": np.sum(tagger_outputs[:, 6:8], axis=1),
            f"fj_{pversion}_probHWWmunuqq": np.sum(tagger_outputs[:, 8:10], axis=1),
        }

        pnet_vars = {**pnet_vars, **derived_vars}

        # also add pku vars of that jet
        if pversion == "ParT":
            jet = ak.firsts(events[fatjet_label][fj_idx_lep])
            pku_vars = {
                f"fj_PKU_{pversion}_{output_name}": jet[f"inclParTMDV1_{output_name}"] for output_name in output_names
            }
            pku_vars[f"fj_PKU_{pversion}_mass"] = jet["inclParTMDV1_mass"]

            pnet_vars = {**pnet_vars, **pku_vars}

    # print(f"Total time taken: {time.time() - total_start:.1f}s")
    return pnet_vars
