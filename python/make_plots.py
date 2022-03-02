#!/usr/bin/python

from axes import axis_dict
import pickle as pkl
import pyarrow.parquet as pq
import pyarrow as pa
import awkward as ak
import numpy as np
import pandas as pd
import json
import os
import glob
import shutil
import pathlib
from typing import List, Optional

import argparse
from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection

import hist as hist2
import matplotlib.pyplot as plt
import mplhep as hep
from hist.intervals import clopper_pearson_interval

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch ")

# define the axes for the different variables to be plotted

# define samples
signal_by_ch = {
    'ele': 'GluGluHToWWToLNuQQ',
    'mu': 'GluGluHToWWToLNuQQ',
    'had': 'GluGluHToWWTo4q',
}
# signal_by_ch = {
#     'ele': 'GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8',
#     'mu': 'GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8',
#     'had': 'GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8',
# }
data_by_ch = {
    'ele': 'SingleElectron',
    'mu': 'SingleMuon',
    'had': 'JetHT',
}
add_samples = {
    'SingleElectron': 'SingleElectron',
    'SingleMuon': 'SingleMuon',
    'JetHT': 'JetHT',
    'QCD': 'QCD_Pt',
    'DYJets': 'DYJets',
    'ZQQ': 'ZJetsToQQ',
    'WQQ': 'WJetsToQQ',
    'SingleTop': 'ST',
    'TTbar': 'TT',
    'WJetsLNu': 'WJetsToLNu',
}


def get_simplified_label(sample):   # get simplified "alias" names of the samples for plotting purposes
    f = open('configs/simplified_labels.json')
    name = json.load(f)
    f.close()
    if sample in name.keys():
        return str(name[sample])
    else:
        return sample


def get_sum_sumgenweight(idir, year, sample):
    pkl_files = glob.glob(f'{idir}/{sample}/outfiles/*.pkl')  # get the pkl metadata of the pkl files that were processed
    sum_sumgenweight = 1
    for file in pkl_files:
        # load and sum the sumgenweight of each
        with open(file, 'rb') as f:
            metadata = pkl.load(f)
        sum_sumgenweight = sum_sumgenweight + metadata[sample][year]['sumgenweight']
    return sum_sumgenweight


def make_hist(idir, odir, vars_to_plot, samples, years, channels, pfnano):  # makes histograms and saves in pkl file
    hists = {}  # define a placeholder for all histograms
    for year in years:
        # Get luminosity of year
        f = open('../fileset/luminosity.json')
        luminosity = json.load(f)
        f.close()
        print(f'Processing samples from year {year} with luminosity {luminosity[year]}')

        hists[year] = {}

        for ch in channels:  # initialize the histograms for the different channels and different variables
            hists[year][ch] = {}

            for var in vars_to_plot[ch]:
                sample_axis = hist2.axis.StrCategory([], name='samples', growth=True)

                hists[year][ch][var] = hist2.Hist(
                    sample_axis,
                    axis_dict[var],
                )

        xsec_weight_by_sample = {}
        # loop over the processed files and fill the histograms
        for ch in channels:
            for sample in samples[year][ch]:
                print('Processing sample', sample)
                is_data = False
                for key in data_by_ch.values():
                    if key in sample:
                        is_data = True
                if not is_data and sample not in xsec_weight_by_sample.keys():
                    pkl_dir = f'{idir}/{sample}/outfiles/*.pkl'
                    pkl_files = glob.glob(pkl_dir)  # get list of files that were processed
                    if not pkl_files:  # skip samples which were not processed
                        print('- No processed files found...', pkl_dir, 'skipping sample...', sample)
                        continue

                    # Find xsection
                    if args.pfnano:
                        f = open('../fileset/xsec_pfnano.json')
                    else:
                        f = open('../fileset/xsec.json')
                    xsec = json.load(f)
                    f.close()
                    xsec = eval(str((xsec[sample])))

                    # Get sum_sumgenweight of sample
                    sum_sumgenweight = get_sum_sumgenweight(idir, year, sample)

                    # Get overall weighting of events
                    xsec_weight = (xsec * luminosity[year]) / (sum_sumgenweight)  # each event has (possibly a different) genweight... sumgenweight sums over events in a chunk... sum_sumgenweight sums over chunks
                    xsec_weight_by_sample[sample] = xsec_weight
                elif sample in xsec_weight_by_sample.keys():
                    xsec_weight = xsec_weight_by_sample[sample]
                else:
                    xsec_weight = 1

                parquet_files = glob.glob(f'{idir}/{sample}/outfiles/*_{ch}.parquet')  # get list of parquet files that have been processed

                for parquet_file in parquet_files:
                    try:
                        data = pq.read_table(parquet_file).to_pandas()
                    except:
                        if not is_data:
                            print('Not able to read data: ', parquet_file, ' should remove evts from scaling')
                        continue

                    for var in vars_to_plot[ch]:
                        if var not in data.keys():
                            # print(f'- No {var} for {year}/{ch} - skipping')
                            continue

                        # we can make further selections before filling the hists here
                        # data = data[data['ht'] > 300]

                        variable = data[var].to_numpy()
                        try:
                            event_weight = data['weight'].to_numpy()
                        except:
                            event_weight = 1  # for data

                        # filling histograms
                        single_sample = None
                        for single_key, key in add_samples.items():
                            if key in sample:
                                single_sample = single_key

                        if single_sample is not None:
                            hists[year][ch][var].fill(
                                samples=single_sample,  # combining all events under one name
                                var=variable,
                                weight=event_weight * xsec_weight,
                            )
                        else:
                            hists[year][ch][var].fill(
                                samples=sample,
                                var=variable,
                                weight=event_weight * xsec_weight,
                            )

    # store the hists variable
    with open(f'{odir}/hists.pkl', 'wb') as f:  # saves the hists objects
        pkl.dump(hists, f)


def make_stack(odir, vars_to_plot, years, channels, pfnano, logy=True, add_data=True):
    # load the hists
    with open(f'{odir}/hists.pkl', 'rb') as f:
        hists = pkl.load(f)
        f.close()

    # make the histogram plots in this directory
    # TODO: we will want combined plots for all years later too
    for year in years:
        if logy:
            if not os.path.exists(f'{odir}/hists_{year}_log'):
                os.makedirs(f'{odir}/hists_{year}_log')
        else:
            if not os.path.exists(f'{odir}/hists_{year}'):
                os.makedirs(f'{odir}/hists_{year}')
        for ch in channels:
            for var in vars_to_plot[ch]:
                if hists[year][ch][var].shape[0] == 0:     # skip empty histograms (such as lepton_pt for hadronic channel)
                    continue

                # get histograms
                h = hists[year][ch][var]

                # get samples existing in histogram
                samples = [h.axes[0].value(i) for i in range(len(h.axes[0].edges))]
                signal_label = signal_by_ch[ch]
                data_label = data_by_ch[ch]

                # data
                data = None
                if data_label in samples:
                    data = h[{"samples": data_label}]

                # signal
                signal = h[{"samples": signal_label}]
                print(signal)
                if not logy:
                    signal = signal * 10  # if not log, scale the signal

                # everything else (background)
                bkg_labels = [label for label in samples if (label and label != data_label and label != signal_label)]
                bkg = [h[{"samples": label}] for label in bkg_labels]

                if bkg is None:
                    print('No background samples to plot besides the signal')
                    return

                # print(data,signal,bkg)

                if add_data and data:
                    fig, (ax, rax) = plt.subplots(nrows=2,
                                                  ncols=1,
                                                  figsize=(8, 8),
                                                  tight_layout=True,
                                                  gridspec_kw={"height_ratios": (3, 1)},
                                                  sharex=True
                                                  )
                    fig.subplots_adjust(hspace=.07)
                    data_err_opts = {
                        'linestyle': 'none',
                        'marker': '.',
                        'markersize': 12.,
                        'elinewidth': 2,
                    }
                    hep.histplot(data,
                                 ax=ax,
                                 histtype="errorbar",
                                 color="k",
                                 yerr=True,
                                 label=get_simplified_label(data_label),
                                 **data_err_opts
                                 )
                    rax.errorbar(
                        x=[data.axes.value(i)[0] for i in range(len(data.values()))],
                        y=data.values() / np.sum([b.values() for b in bkg], axis=0),
                        fmt="ko",
                    )
                else:
                    fig, ax = plt.subplots(1, 1)

                hep.histplot(bkg,
                             ax=ax,
                             stack=True,
                             sort='yield',
                             histtype="fill",
                             label=[get_simplified_label(bkg_label) for bkg_label in bkg_labels],
                             )

                hep.histplot(signal,
                             ax=ax,
                             label=get_simplified_label(signal_label),
                             color='red'
                             )

                if logy:
                    ax.set_yscale('log')
                    ax.set_ylim(0.1)
                ax.set_title(f'{ch} channel')
                ax.legend()

                hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
                hep.cms.text("Work in Progress", ax=ax)

                if logy:
                    print('Saving to ', f'{odir}/hists_{year}_log/{var}_{ch}.pdf')
                    plt.savefig(f'{odir}/hists_{year}_log/{var}_{ch}.pdf')
                else:
                    print('Saving to ', f'{odir}/hists_{year}/{var}_{ch}.pdf')
                    plt.savefig(f'{odir}/hists_{year}/{var}_{ch}.pdf')
                plt.close()


def make_norm(idir, vars_to_plot, years, channels, logy=True):
    for year in years:
        if not os.path.exists(f'{odir}/hists_{year}'):
            os.makedirs(f'{odir}/hists_{year}')
        for ch in channels:
            for var in vars_to_plot[ch]:
                if hists[year][ch][var].shape[0] == 0:
                    continue

                fig, ax = plt.subplots(1, 1)
                hep.histplot([x for x in hists[year][ch][var].stack(0)[1:]],
                             ax=ax,
                             stack=False,
                             density=True,
                             )
                hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
                hep.cms.text("Work in Progress", ax=ax)
                plt.savefig(f'{odir}/hists_{year}/{var}_{ch}_density.pdf')
                plt.close()


def main(args):
    odir = args.odir
    if not os.path.exists(odir):
        os.makedirs(odir)

    years = args.years.split(',')
    channels = args.channels.split(',')

    # get samples to make histograms
    f = open(args.samples)
    json_samples = json.load(f)
    f.close()

    # build samples
    samples = {}
    for year in years:
        samples[year] = {}
        for ch in channels:
            samples[year][ch] = []
            for key, value in json_samples[year][ch].items():
                if value == 1:
                    samples[year][ch].append(key)

    # get variables to plot
    f = open(args.vars)
    variables = json.load(f)
    f.close()
    vars_to_plot = {}
    for ch in variables.keys():
        vars_to_plot[ch] = []
        for key, value in variables[ch].items():
            if value == 1:
                vars_to_plot[ch].append(key)

    if not args.hist:
        # make the histograms and save in pkl files
        make_hist(args.idir, odir, vars_to_plot, samples, years, channels, args.pfnano)

    if not args.noplot:
        # plot all process in stack
        make_stack(odir, vars_to_plot, years, channels, args.pfnano, logy=True)
        make_stack(odir, vars_to_plot, years, channels, args.pfnano, logy=False)


if __name__ == "__main__":
    # e.g.
    # run locally as: python make_plots.py --year 2017 --vars configs/vars.json --channels ele,mu,had --idir ../results/ --odir hists --pfnano --samples configs/samples_pfnano.json
    # run on lpc as: python make_plots.py --year 2017 --vars configs/vars.json --channels ele,mu,had --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/Mar1_2017/ --odir hists --pfnano --samples configs/samples_pfnano.json

    parser = argparse.ArgumentParser()
    parser.add_argument('--years',      dest='years',       default='2017',                        help="year")
    parser.add_argument('--vars',       dest='vars',        default="configs/vars.json",           help='path to json with variables to be plotted')
    parser.add_argument('--samples',    dest='samples',     default="configs/samples_pfnano.json", help='path to json with samples to be plotted')
    parser.add_argument('--channels',   dest='channels',    default='ele,mu,had',                  help='channels for which to plot this variable')
    parser.add_argument('--odir',       dest='odir',        default='hists',                       help="tag for output directory")
    parser.add_argument('--idir',       dest='idir',        default='../results/',                 help="input directory with results")
    parser.add_argument("--pfnano",     dest='pfnano',      action='store_true',                   help="Run with pfnano")
    parser.add_argument("--hist",       dest='hist',        action='store_true',                   help="Make hists")
    parser.add_argument("--noplot",     dest='noplot',      action='store_true',                   help="Do not plot")

    args = parser.parse_args()

    main(args)
