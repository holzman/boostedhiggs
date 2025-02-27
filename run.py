#!/usr/bin/python

import argparse
import json
import os
import pickle as pkl
import time

import pandas as pd
import uproot
from coffea import nanoevents, processor

nanoevents.PFNanoAODSchema.warn_missing_crossrefs = False


def main(args):
    # make directory for output
    if not os.path.exists("./outfiles"):
        os.makedirs("./outfiles")

    if args.channels:
        channels = args.channels.split(",")

    # if --macos is specified in args, process only the files provided
    if args.macos:
        files = {}
        files[args.sample] = [f"rootfiles2/rootfiles/{args.sample}/file{i+1}.root" for i in range(1)]

    # if --local is specified in args, process only the args.sample provided
    elif args.local:
        files = {}
        with open(f"fileset/pfnanoindex_{args.pfnano}_{args.year}.json", "r") as f:
            files_all = json.load(f)
            for subdir in files_all[args.year]:
                for key, flist in files_all[args.year][subdir].items():
                    if key in args.sample:
                        files[key] = ["root://cmseos.fnal.gov/" + f for f in flist]

    else:
        # get samples
        if "metadata" in args.config:
            with open(args.config, "r") as f:
                files = json.load(f)
        else:
            if not args.config or not args.configkey:
                raise Exception("No config or configkey provided for condor jobs")

            # hopefully this step is avoided in condor jobs that have metadata.json
            from condor.file_utils import loadFiles

            files, _ = loadFiles(args.config, args.configkey, args.year, args.pfnano, args.sample.split(","))

            # print(files)

    if not files:
        print("Did not find files.. Exiting.")
        exit(1)

    # build fileset with files to run per job
    fileset = {}
    starti = args.starti
    job_name = "/" + str(starti * args.n)
    if args.n != -1:
        job_name += "-" + str(args.starti * args.n + args.n)
    for sample, flist in files.items():
        if args.sample:
            if sample not in args.sample.split(","):
                continue
        if args.n != -1:
            fileset[sample] = flist[args.starti * args.n : args.starti * args.n + args.n]
        else:
            fileset[sample] = flist

    print(
        len(list(fileset.keys())),
        "Samples in fileset to be processed: ",
        list(fileset.keys()),
    )
    print(fileset)

    # define processor
    year = args.year.replace("APV", "")
    yearmod = ""
    if "APV" in args.year:
        yearmod = "APV"

    if args.processor == "hww":
        from boostedhiggs.hwwprocessor import HwwProcessor

        p = HwwProcessor(
            year=year,
            yearmod=yearmod,
            channels=channels,
            inference=args.inference,
            output_location="./outfiles" + job_name,
            region=args.region,
        )

    elif args.processor == "vh":
        from boostedhiggs.vhprocessor import vhProcessor

        p = vhProcessor(
            year=year,
            yearmod=yearmod,
            channel=channels[0],
            inference=args.inference,
            output_location=f"./outfiles/{job_name}",
        )

    elif args.processor == "lumi":
        from boostedhiggs.lumi_processor import LumiProcessor

        p = LumiProcessor(year=args.year, yearmod=yearmod, output_location=f"./outfiles/{job_name}")

    elif args.processor == "zll":
        from boostedhiggs.zll_processor import ZllProcessor

        p = ZllProcessor(
            year=year,
            yearmod=yearmod,
            channels=channels,
            inference=args.inference,
            output_location="./outfiles" + job_name,
        )
    elif args.processor == "input":
        # define processor
        from boostedhiggs.inputprocessor import InputProcessor

        p = InputProcessor(args.label, inference=args.inference, output_location=f"./outfiles/{job_name}")
    else:
        from boostedhiggs.trigger_efficiencies_processor import TriggerEfficienciesProcessor

        p = TriggerEfficienciesProcessor(year=args.year)

    tic = time.time()
    if args.executor == "dask":
        from coffea.nanoevents import NanoeventsSchemaPlugin
        from distributed import Client
        from lpcjobqueue import LPCCondorCluster

        cluster = LPCCondorCluster(
            ship_env=True,
            transfer_input_files="boostedhiggs",
        )
        client = Client(cluster)
        nanoevents_plugin = NanoeventsSchemaPlugin()
        client.register_worker_plugin(nanoevents_plugin)
        cluster.adapt(minimum=1, maximum=30)

        print("Waiting for at least one worker")
        client.wait_for_workers(1)

        # does treereduction help?
        executor = processor.DaskExecutor(status=True, client=client, treereduction=2)

    else:
        uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

        if args.executor == "futures":
            executor = processor.FuturesExecutor(status=True)
        else:
            executor = processor.IterativeExecutor(status=True)

    nanoevents.PFNanoAODSchema.mixins["SV"] = "PFCand"

    run = processor.Runner(
        executor=executor,
        savemetrics=True,
        schema=nanoevents.PFNanoAODSchema,
        chunksize=args.chunksize,
    )

    out, metrics = run(fileset, "Events", processor_instance=p)

    elapsed = time.time() - tic
    print(f"Metrics: {metrics}")
    print(f"Finished in {elapsed:.1f}s")

    # dump to pickle
    filehandler = open("./outfiles/" + job_name + ".pkl", "wb")
    pkl.dump(out, filehandler)
    filehandler.close()

    # merge parquet
    if args.processor == "hww" or args.processor == "vh" or args.processor == "zll" or args.processor == "input":
        for ch in channels:
            data = pd.read_parquet("./outfiles/" + job_name + ch + "/parquet")
            data.to_parquet("./outfiles/" + job_name + "_" + ch + ".parquet")
            # remove old parquet files
            os.system("rm -rf ./outfiles/" + job_name + ch)
    elif args.processor == "input":
        # merge parquet
        data = pd.read_parquet(f"./outfiles/{job_name}/parquet")
        data.to_parquet(f"./outfiles/{job_name}.parquet")

        # make directory for output rootfiles
        if not os.path.exists("./outfiles/outroot"):
            os.makedirs("./outfiles/outroot")

        os.system(f"mv ./outfiles/{job_name}/outroot ./outfiles/outroot/{job_name}")

        # remove unmerged parquet files
        os.system("rm -rf ./outfiles/" + job_name)

        # in cmsenv
        # hadd outfiles/job_name.root outfiles/outroot/job_name/*
        # rm -r outfiles/outroot


if __name__ == "__main__":
    # e.g.
    # noqa: run locally on lpc (hww mc) as: python run.py --year 2017 --processor hww --pfnano v2_2 --n 1 --starti 0 --config samples_inclusive.yaml --key mc
    # noqa: run locally on lpc (vh) as: python run.py --year 2018 --sample HZJ_HToWW_M-125 --processor vh --pfnano v2_2 --n 1 --starti 0 --config samples_vh.yaml --key mc --channels lep --executor iterative
    # noqa: run locally on lpc (hww trigger) as: python run.py --year 2017 --processor trigger --pfnano v2_2 --n 1 --starti 0 --sample GluGluHToWW_Pt-200ToInf_M-125 --local --channels ele --config samples_inclusive.yaml --key mc

    # noqa: run locally on single file (hww): python run.py --year 2017 --processor hww --pfnano v2_2 --n 1 --starti 0 --sample GluGluHToWW_Pt-200ToInf_M-125 --local --channels ele --config samples_inclusive.yaml --key mc
    # noqa: run locally on on single file (zll): python run.py --year 2017 --processor zll --pfnano v2_2 --n 1 --starti 0 --sample GluGluHToWW_Pt-200ToInf_M-125 --local --channels ele --config samples_inclusive.yaml --key mc

    # python run.py --processor input --macos --sample GluGluHToWW_Pt-200ToInf_M-125 --n 1 --starti 0 --label H --inference
    # python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 1 --starti 0 --label H --inference

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", dest="year", default="2017", help="year", type=str)
    parser.add_argument("--starti", dest="starti", default=0, help="start index of files", type=int)
    parser.add_argument("--n", dest="n", default=-1, help="number of files to process", type=int)
    parser.add_argument("--config", dest="config", default=None, help="path to datafiles", type=str)
    parser.add_argument("--key", dest="configkey", default=None, help="config key: [data, mc, ... ]", type=str)
    parser.add_argument("--sample", dest="sample", default=None, help="specify sample", type=str)
    parser.add_argument("--processor", dest="processor", required=True, help="processor", type=str)
    parser.add_argument("--chunksize", dest="chunksize", default=10000, help="chunk size in processor", type=int)
    parser.add_argument("--channels", dest="channels", default=None, help="channels separated by commas")
    parser.add_argument("--region", dest="region", default="signal", help="specify region for selections", type=str)

    parser.add_argument(
        "--executor",
        type=str,
        default="futures",
        choices=["futures", "iterative", "dask"],
        help="type of processor executor",
    )
    parser.add_argument(
        "--pfnano",
        dest="pfnano",
        type=str,
        default="v2_2",
        help="pfnano version",
    )
    parser.add_argument("--macos", dest="macos", action="store_true")
    parser.add_argument("--local", dest="local", action="store_true")
    parser.add_argument("--inference", dest="inference", action="store_true")
    parser.add_argument("--no-inference", dest="inference", action="store_false")

    parser.add_argument("--label", dest="label", default="H", help="jet label for inputskimmer", type=str)

    parser.set_defaults(inference=False)
    args = parser.parse_args()

    main(args)
