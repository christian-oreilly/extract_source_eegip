#!/usr/bin/env python3

import mne
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm
from mne.io.eeglab.eeglab import _check_load_mat, _get_info
import xarray as xr
from collections import OrderedDict
import json
import argparse
import os
from shutil import copyfile, rmtree

def get_bem_artifacts(template, montage_name="HGSN129-montage.fif"):
    montage = mne.channels.read_dig_fif(str(Path(subjects_dir) / template / "montages" / montage_name))
    trans = mne.channels.compute_native_head_t(montage)
    bem_model = mne.read_bem_surfaces(str(Path(subjects_dir) / template / "bem" / f"{template}-5120-5120-5120-bem.fif"))
    bem_solution = mne.read_bem_solution(
        str(Path(subjects_dir) / template / "bem" / f"{template}-5120-5120-5120-bem-sol.fif"))
    surface_src = mne.read_source_spaces(str(Path(subjects_dir) / template / "bem" / f"{template}-oct-6-src.fif"))

    return montage, trans, bem_model, bem_solution, surface_src


def preprocess(raw, notch_width=None, line_freq=50.0):
    if notch_width is None:
        notch_width = np.array([1.0, 0.1, 0.01, 0.1])

    notch_freqs = np.arange(line_freq, raw.info["sfreq"] / 2.0, line_freq)
    raw.notch_filter(notch_freqs, picks=["eeg", "eog"], fir_design='firwin',
                     notch_widths=notch_width[:len(notch_freqs)], verbose=None)


def mark_bad_channels(raw, file_name, mark_to_remove=("manual", "rank")):
    raw_eeg = _check_load_mat(file_name, None)
    info, _, _ = _get_info(raw_eeg)
    print("############ file_name", file_name, type(file_name))
    print("############ raw_eeg", raw_eeg.keys(), type(raw_eeg))
    chan_info = raw_eeg.marks["chan_info"]

    mat_chans = np.array(info["ch_names"])
    assert (len(chan_info["flags"][0]) == len(mat_chans))

    if len(np.array(chan_info["flags"]).shape) > 1:
        ind_chan_to_drop = np.unique(np.concatenate([np.where(flags)[0] for flags, label in zip(chan_info["flags"],
                                                                                                chan_info["label"])
                                                     if label in mark_to_remove]))
    else:
        ind_chan_to_drop = np.where(chan_info["flags"])[0]

    bad_chan = [chan for chan in mat_chans[ind_chan_to_drop]]

    raw.info['bads'].extend(bad_chan)


def add_bad_segment_annot(raw, file_name, mark_to_remove=("manual",)):
    raw_eeg = _check_load_mat(file_name, None)
    time_info = raw_eeg.marks["time_info"]

    if len(np.array(time_info["flags"]).shape) > 1:
        ind_time_to_drop = np.unique(np.concatenate([np.where(flags)[0] for flags, label in zip(time_info["flags"],
                                                                                                time_info["label"])
                                                     if label in mark_to_remove]))
    else:
        ind_time_to_drop = np.where(time_info["flags"])[0]

    ind_starts = np.concatenate(
        [[ind_time_to_drop[0]], ind_time_to_drop[np.where(np.diff(ind_time_to_drop) > 1)[0] + 1]])
    ind_ends = np.concatenate([ind_time_to_drop[np.where(np.diff(ind_time_to_drop) > 1)[0]], [ind_time_to_drop[-1]]])
    durations = (ind_ends + 1 - ind_starts) / raw.info["sfreq"]
    onsets = ind_starts / raw.info["sfreq"]

    for onset, duration in zip(onsets, durations):
        raw.annotations.append(onset, duration, description="bad_lossless_qc")


def remove_rejected_ica_components(raw, file_name, inplace=True):
    raw_eeg = _check_load_mat(file_name, None)
    mark_to_remove = ["manual"]
    comp_info = raw_eeg.marks["comp_info"]

    if len(np.array(comp_info["flags"]).shape) > 1:
        ind_comp_to_drop = np.unique(np.concatenate([np.where(flags)[0] for flags, label in zip(comp_info["flags"],
                                                                                                comp_info["label"])
                                                     if label in mark_to_remove]))
    else:
        ind_comp_to_drop = np.where(comp_info["flags"])[0]

    if inplace:
        mne.preprocessing.read_ica_eeglab(file_name).apply(raw, exclude=ind_comp_to_drop)
    else:
        mne.preprocessing.read_ica_eeglab(file_name).apply(raw.copy(), exclude=ind_comp_to_drop)


def preprocessed_raw(path, line_freq, montage=None, verbose=False, rename_channel=True):
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=verbose)
    raw.set_montage(montage, verbose=verbose)

    preprocess(raw, line_freq=line_freq, notch_width=np.array([0.0, 0.1, 0.01, 0.1]))

    raw = raw.filter(1, None, fir_design='firwin', verbose=verbose)

    mark_bad_channels(raw, path)
    add_bad_segment_annot(raw, path)
    remove_rejected_ica_components(raw, path, inplace=True)

    raw = raw.interpolate_bads(reset_bads=True, verbose=verbose)

    if rename_channel:
        raw.rename_channels({ch: ch2 for ch, ch2 in chan_mapping.items() if ch in raw.ch_names})

    raw.set_channel_types({ch: "eog" for ch in eog_channels if ch in raw.ch_names})

    return raw


def process_events_london(raw, age):
    annot_sample = []
    annot_id = []
    freq = raw.info["sfreq"]

    if age == "06":
        rst0 = []
        rst1 = []
        for a in raw.annotations:
            if a["description"] == 'Rst0':
                rst0.append(a["onset"])
            if a["description"] == 'Rst1':
                rst1.append(a["onset"])

        if len(rst0) == 0 and len(rst1) == 0:
            return None

        annot_sample = np.concatenate([np.arange(start, stop - 0.999, 1.0) for start, stop in zip(rst0, rst1)])
        annot_sample = (annot_sample * freq).astype(int)
        annot_id = [1] * len(annot_sample)

    elif age == "12":
        # annots = [OrderedDict((("onset", 0), ("duration", 0), ("description", "base"), ('orig_time', None)))]
        # annots.extend([a for a in raw.annotations
        #               if a["description"] in ["eeg1", "eeg2", "eeg3"]])
        annots = [a for a in raw.annotations if a["description"] in ["eeg1", "eeg2", "eeg3"]]
        if len(annots) == 0:
            return None

        if raw.annotations[-1]["onset"] > annots[-1]["onset"]:
            end = np.min([raw.annotations[-1]["onset"], annots[-1]["onset"] + 50.])
            annots.append(OrderedDict((("onset", end), ("duration", 0),
                                       ("description", "end"), ('orig_time', None))))

        for annot, next_annot in zip(annots[:-1], annots[1:]):
            annot_sample.append(np.arange(int(annot["onset"] * freq),
                                          int(next_annot["onset"] * freq),
                                          int(tmax * freq)))
            annot_id.extend(event_id[("london", age)][annot["description"]] * np.ones(len(annot_sample[-1])))

        annot_sample = np.concatenate(annot_sample)

    else:
        raise ValueError("Invalid value for session.")

    return np.array([annot_sample, [0] * len(annot_sample), annot_id], dtype=int).T


def process_events_washington(raw):
    annot_sample = []
    annot_id = []
    freq = raw.info["sfreq"]

    # COV AND VIDEOS
    # annots = [OrderedDict((("onset", 0), ("duration", 0), ("description", "cov"), ('orig_time', None)))]
    # annots.extend([a for a in raw.annotations if a["description"] in ["Toys", "EndM", "Socl"]])
    annots = [a for a in raw.annotations if a["description"] in ["Toys", "EndM", "Socl"]]
    if len(annots) == 0:
        return None

    annots.append(OrderedDict((("onset", annots[-1]["onset"] + 50.), ("duration", 0),
                               ("description", "end"), ('orig_time', None))))

    for annot, next_annot in zip(annots[:-1], annots[1:]):
        if annot["description"] == "EndM":
            continue

        annot_sample.append(np.arange(int(annot["onset"] * freq),
                                      int(next_annot["onset"] * freq),
                                      int(tmax * freq)))
        id_ = event_id["washington"]["videos"][annot["description"]]
        annot_id.extend(id_ * np.ones(len(annot_sample[-1])))

    annot_sample = np.concatenate(annot_sample)

    return np.array([annot_sample, [0] * len(annot_sample), annot_id], dtype=int).T


def process_events(raw, dataset, age):
    if dataset == "london":
        return process_events_london(raw, age)
    if dataset == "washington":
        return process_events_washington(raw)


def process_epochs(raw, dataset, age, events):
    freq = raw.info["sfreq"]
    if dataset == "london":
        filtered_event_id = {key: val for key, val in event_id["london"][age].items() if val in events[:, 2]}
        if len(filtered_event_id):
            # "tmax = tmax[dataset] - 1.0 / freq" because MNE is inclusive on the last point and we don't want that
            # "baseline=None" because the baseline is corrected by a 1Hz high-pass on the raw data
            return mne.Epochs(raw, events, filtered_event_id, tmin=tmin,
                              tmax=tmax - 1.0 / freq, baseline=None,
                              preload=True, reject_by_annotation=True)
        return None

    elif dataset == "washington":

        filtered_event_id = {key: val for key, val in event_id[dataset]["videos"].items() if val in events[:, 2]}
        if len(filtered_event_id):
            return mne.Epochs(raw, events, filtered_event_id, tmin=tmin,
                              tmax=tmax - 1.0 / freq, baseline=None,
                              preload=True, reject_by_annotation=True)
        return None


def process_sources(epochs, trans, surface_src, bem_solution, template, config):
    fwd = mne.make_forward_solution(epochs.info, trans, surface_src, bem_solution,
                                    **config["mne"]["make_forward_solution"])
    noise_cov = mne.compute_covariance(epochs,
                                       **config["mne"]["compute_covariance"])
    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov,
                                                              **config["mne"]["minimum_norm"]["make_inverse_operator"])
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator,
                                                 **config["mne"]["minimum_norm"]["apply_inverse_epochs"])
    anat_label = mne.read_labels_from_annot(template, subjects_dir=subjects_dir,
                                            **config["mne"]["read_labels_from_annot"])
    label_ts = mne.extract_label_time_course(stcs, anat_label, surface_src,
                                             **config["mne"]["extract_label_time_course"])
    return label_ts, anat_label


def validate_models(config_path):
    with Path(config_path).open('r') as f:
        config = json.load(f)

    for key in config["global"]:
        globals()[key] = config["global"][key]


    for template, age in template_ages.items():
        mne.datasets.fetch_infant_template(age, subjects_dir=subjects_dir)

        montage, trans, bem_model, bem_solution, surface_src = get_bem_artifacts(template)
        montage.ch_names = ["E" + str(int(ch_name[3:])) for ch_name in montage.ch_names]
        montage.ch_names[128] = "Cz"

        info = mne.create_info(montage.ch_names, sfreq=256, ch_types="eeg")
        raw = mne.io.RawArray(np.zeros((len(montage.ch_names), 1)), info, copy=None, verbose=False).set_montage(montage)

        fig = mne.viz.plot_alignment(raw.info, trans=trans, subject=template,
                                     subjects_dir=subjects_dir, surfaces='head',
                                     show_axes=True, dig="fiducials", eeg="projected",
                                     coord_frame='mri', mri_fiducials=True,
                                     src=surface_src, bem=bem_solution)
        time.sleep(1.0)
        fig.plotter.off_screen = True
        mne.viz.set_3d_view(figure=fig, azimuth=135, elevation=80, distance=0.6)
        time.sleep(1.0)
        fig.plotter.screenshot(f"coregistration_{template}_1.png")

        mne.viz.set_3d_view(figure=fig, azimuth=45, elevation=80, distance=0.6)
        time.sleep(1.0)
        fig.plotter.screenshot(f"coregistration_{template}_2.png")

        mne.viz.set_3d_view(figure=fig, azimuth=270, elevation=80, distance=0.6)
        time.sleep(1.0)
        fig.plotter.screenshot(f"coregistration_{template}_3.png")


def save_config_example(config_path):
    config = {
        "global": dict(tmax=0.8,
                       tmin=-0.2,
                       # As per https://github.com/methlabUZH/automagic/wiki/How-to-start
                       # These number are coherent with the map shown in geodesic-sensor-net.pdf
                       eog_channels=["E1", "E8", "E14", "NAS", "E21", "E25", "E32", "E125", "E126", "E127", "E128"],
                       sites=["london", "washington"],
                       line_freqs={"london": 50.0, "washington": 60.0},
                       template_dict={"06": "ANTS6-0Months3T",
                                      "12": "ANTS12-0Months3T",
                                      "18": "ANTS18-0Months3T"},
                       template_ages={"ANTS6-0Months3T": "6mo",
                                      "ANTS12-0Months3T": "12mo",
                                      "ANTS18-0Months3T": "18mo"},
                       event_id={
                           "london": {
                               "06": {"rst": 1},
                               "12": {"base": 0, "eeg1": 1, "eeg2": 2, "eeg3": 3}
                           },
                           "washington": {
                               "cov": {"cov": 0},
                               "videos": {"cov": 0, "Toys": 1, "Socl": 2},
                               "videos_only": {"Toys": 1, "Socl": 2},
                               "all": {"cov": 0, "Toys": 1, "Socl": 2}
                           }
                       }
                       ),
        "mne": {
            "make_forward_solution": dict(mindist=0),
            "compute_covariance": dict(tmax=0.0, method="auto"),
            "minimum_norm": {
                "make_inverse_operator": dict(loose=0.0),
                "apply_inverse_epochs": dict(method="eLORETA",
                                             lambda2=0.1,
                                             pick_ori=None,
                                             return_generator=True)
            },
            "read_labels_from_annot": dict(parc='aparc'),
            "extract_label_time_course": dict(mode='mean_flip', allow_empty=True, return_generator=False)
        }
    }

    with Path(config_path).open('w') as f:
        json.dump(config, f, indent=4)


def compute_sources(config_path, derivatives_name, overwrite=False):
    with Path(config_path).open('r') as f:
        config = json.load(f)

    for key in config["global"]:
        globals()[key] = config["global"][key]

    for template, age in template_ages.items():
        mne.datasets.fetch_infant_template(age, subjects_dir=subjects_dir)

    for dataset in sites:
        # Save a copy of the configuration file at the root of the derivatives.
        derivatives_root_in = bids_root / dataset / "derivatives" / "lossless"
        derivatives_root_out = derivatives_root_in / "derivatives" / derivatives_name
        if derivatives_root_out.exists():
            if overwrite:
                rmtree(derivatives_root_out)
            else:
                raise RuntimeError(f"The derivatives {derivatives_root_out} already exists. You can call the program "
                                   f"using --overwrite if you are sure you want to overwrite the current "
                                   f"data at this path.")


        derivatives_root_out.mkdir(parents=True, exist_ok=True)
        with (derivatives_root_out / "config.json").open('w') as f:
            json.dump(config, f, indent=4)

        # Save a copy of this program at the root of the derivatives.
        prog_path = Path(os.getcwd()) / __file__
        copyfile(prog_path, derivatives_root_out / prog_path.name)

        for age, template in config["global"]["template_dict"].items():
            montage, trans, bem_model, bem_solution, surface_src = get_bem_artifacts(template)
            montage.ch_names = ["E" + str(int(ch_name[3:])) for ch_name in montage.ch_names]
            montage.ch_names[128] = "Cz"
            file_pattern = f"sub-s*/ses-m{age}/eeg/sub-*_eeg_qcr.set"
            for path in tqdm(list(derivatives_root_in.glob(file_pattern))):
                raw = preprocessed_raw(path, line_freqs[dataset], montage, rename_channel=False)
                raw.set_montage(montage)
                subject_no = Path(path).name[5:8]

                events = process_events(raw, dataset, age)
                if events is None:
                    continue
                epochs = process_epochs(raw, dataset, age, events)

                label_ts, anat_label = process_sources(epochs, trans, surface_src, bem_solution, template, config)

                source_file_name = path.name[:-4] + "_source_labels.nc"
                out_path = derivatives_root_out / f"sub-s{subject_no}" / f"ses-m{age}" / "eeg" / source_file_name
                out_path.parent.mkdir(exist_ok=True, parents=True)
                sources_xr = xr.DataArray(np.array(label_ts),
                                         dims=("epoch", "region", "time"),
                                         coords={"epoch": np.arange(len(label_ts)),
                                                 "region": [label.name for label in anat_label],
                                                 "time": epochs.times})
                sources_xr.to_netcdf(out_path)
                break
            break
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract sources for EEG-IP.')

    parser.add_argument('--get_example_config', dest='get_example_config', action="store_true",
                        help=('Save an example of configuration file. If no path is provided as'
                        ' an argument, it will save the configuration file in the current'
                        ' working directory with the file name config.json.'))

    parser.add_argument('--validate_head_models', dest='validate', action='store_true',
                        help='Generate validation images for the head models.')

    parser.add_argument('--derivatives_name', dest='derivatives_name', default="sources",
                        help='Name of the derivative where to save the sources.')

    parser.add_argument('--bids_root', dest='out_root_path', default='/project/def-emayada/eegip',
                        help='Root of the BIDS project.')

    parser.add_argument('--config_path', dest='config_path', default="config.json",
                        help='Path to the configuration file.')

    parser.add_argument('--overwrite', dest='overwrite', action="store_true",
                        help='Overwrite the output derivatives if it already exists.')

    parser.add_argument('--fs_subjects_dir', dest='subjects_dir', default="./fs_models/",
                        help=('Directory where the head templates can be found. If they corresponding templates are '
                              'absent, they will be downloaded in that directory.'))

    args = parser.parse_args()

    locals()["subjects_dir"] = args.subjects_dir
    locals()["bids_root"] = Path(args.out_root_path)

    if args.validate:
        validate_models(args.config_path)
    elif args.get_example_config:
        save_config_example(args.config_path)
    else:
        compute_sources(args.config_path, args.derivatives_name, args.overwrite)
