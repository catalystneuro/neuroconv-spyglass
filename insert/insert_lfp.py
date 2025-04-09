import datajoint as dj
from pathlib import Path
import numpy as np

dj_local_conf_path = "/Users/pauladkisson/Documents/CatalystNeuro/JadhavConv/jadhav-lab-to-nwb/src/jadhav_lab_to_nwb/spyglass_mock/dj_local_conf.json"
dj.config.load(dj_local_conf_path)  # load config for database connection info

import spyglass.common as sgc  # this import connects to the database
import spyglass.data_import as sgi
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from pynwb import NWBHDF5IO

# LFP Imports
import spyglass.lfp as sglfp
from spyglass.utils.nwb_helper_fn import estimate_sampling_rate
from pynwb.ecephys import ElectricalSeries, LFP


def insert_lfp(nwbfile_path: Path):
    """
    Insert LFP data from an NWB file into a spyglass database.

    Parameters
    ----------
    nwbfile_path : Path
        The path to the NWB file to insert.
    """
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    lfp_file_name = sgc.AnalysisNwbfile().create(nwb_copy_file_name)
    analysis_file_abspath = sgc.AnalysisNwbfile().get_abs_path(lfp_file_name)

    raw_io = NWBHDF5IO(nwbfile_path, "r")
    raw_nwbfile = raw_io.read()
    lfp_eseries = raw_nwbfile.processing["ecephys"]["LFP"].electrical_series["ElectricalSeriesLFP"]
    eseries_kwargs = {
        "data": lfp_eseries.data,
        "timestamps": lfp_eseries.timestamps,
        "description": lfp_eseries.description,
    }

    # Create dynamic table region and electrode series, write/close file
    analysis_io = NWBHDF5IO(path=analysis_file_abspath, mode="a", load_namespaces=True)
    analysis_nwbfile = analysis_io.read()

    # get the indices of the electrodes in the electrode table
    electrodes_table = analysis_nwbfile.electrodes.to_dataframe()
    lfp_electrode_indices = electrodes_table.index.tolist()

    electrode_table_region = analysis_nwbfile.create_electrode_table_region(
        lfp_electrode_indices, "filtered electrode table"
    )
    eseries_kwargs["name"] = "filtered data"
    eseries_kwargs["electrodes"] = electrode_table_region
    es = ElectricalSeries(**eseries_kwargs)
    lfp_object_id = es.object_id
    ecephys_module = analysis_nwbfile.create_processing_module(name="ecephys", description="ecephys module")
    ecephys_module.add(LFP(electrical_series=es))
    analysis_io.write(analysis_nwbfile, link_data=False)
    analysis_io.close()

    sgc.AnalysisNwbfile().add(nwb_copy_file_name, lfp_file_name)

    lfp_electrode_group_name = "lfp_electrode_group"
    sglfp.lfp_electrode.LFPElectrodeGroup.create_lfp_electrode_group(
        nwb_file_name=nwb_copy_file_name,
        group_name=lfp_electrode_group_name,
        electrode_list=lfp_electrode_indices,
    )
    lfp_sampling_rate = estimate_sampling_rate(eseries_kwargs["timestamps"][:1_000_000])
    key = {
        "nwb_file_name": nwb_copy_file_name,
        "lfp_electrode_group_name": lfp_electrode_group_name,
        "interval_list_name": "raw data valid times",
        "lfp_sampling_rate": lfp_sampling_rate,
        "lfp_object_id": lfp_object_id,
        "analysis_file_name": lfp_file_name,
    }
    sglfp.ImportedLFP.insert1(key, allow_direct_insert=True)
    sglfp.lfp_merge.LFPOutput.insert1(key, allow_direct_insert=True)

    raw_io.close()


def test_lfp(nwbfile_path: Path):
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    lfp_electrical_series = (sglfp.ImportedLFP & {"nwb_file_name": nwb_copy_file_name}).fetch_nwb()[0]["lfp"]
    spyglass_lfp_data = np.asarray(lfp_electrical_series.data[:100])
    with NWBHDF5IO(nwbfile_path, "r") as io:
        nwbfile = io.read()
        nwb_lfp_data = np.asarray(
            nwbfile.processing["ecephys"]["LFP"].electrical_series["ElectricalSeriesLFP"].data[:100]
        )
    np.testing.assert_array_equal(spyglass_lfp_data, nwb_lfp_data)

def main():
    nwbfile_path = Path("/Volumes/T7/CatalystNeuro/Spyglass/raw/mock_lfp.nwb")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)

    if sgc.Session & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.Session & {"nwb_file_name": nwb_copy_file_name}).delete()
    if sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    if sgc.ProbeType & {"probe_type": "my_probe_type"}:
        (sgc.ProbeType & {"probe_type": "my_probe_type"}).delete()

    sgi.insert_sessions(str(nwbfile_path), rollback_on_fail=True, raise_err=True)
    insert_lfp(nwbfile_path=nwbfile_path)
    test_lfp(nwbfile_path=nwbfile_path)


if __name__ == "__main__":
    main()
    print("Done!")
