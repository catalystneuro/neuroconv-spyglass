import datajoint as dj
from pathlib import Path
from pynwb import NWBHDF5IO
import numpy as np
from tqdm import tqdm

dj_local_conf_path = "/Users/pauladkisson/Documents/CatalystNeuro/JadhavConv/jadhav-lab-to-nwb/src/jadhav_lab_to_nwb/spyglass_mock/dj_local_conf.json"
dj.config.load(dj_local_conf_path)  # load config for database connection info

# spyglass.common has the most frequently used tables
import spyglass.common as sgc  # this import connects to the database

# spyglass.data_import has tools for inserting NWB files into the database
import spyglass.data_import as sgi
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

# Spike sorting specific imports
from spyglass.spikesorting.spikesorting_merge import (
    SpikeSortingOutput,
)
import spyglass.spikesorting.v1 as sgs
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
from spyglass.spikesorting.analysis.v1.group import UnitSelectionParams
from spyglass.spikesorting.analysis.v1.unit_annotation import UnitAnnotation


def main():
    nwbfile_path = Path("/Volumes/T7/CatalystNeuro/Spyglass/raw/mock_sorting.nwb")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)

    if sgc.Session & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.Session & {"nwb_file_name": nwb_copy_file_name}).delete()
    if sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()

    sgi.insert_sessions(str(nwbfile_path), rollback_on_fail=True, raise_err=True)
    annotation_to_type = {
        "custom_label": "label",
        "custom_quantification": "quantification",
    }
    insert_sorting(nwbfile_path=nwbfile_path, annotation_to_type=annotation_to_type)
    test_sorting(nwbfile_path=nwbfile_path)


def insert_sorting(nwbfile_path: Path, annotation_to_type: dict):
    """
    Insert spike sorting data from an NWB file into a spyglass database.

    This function adds UnitAnnotation data from the units table in the NWB file to the UnitAnnotation table in the
    spyglass database. The annotations are added as labels or quantifications depending on the type of annotation.

    Parameters
    ----------
    nwbfile_path : Path
        The path to the NWB file to insert.
    annotation_to_type : dict
        A dictionary mapping annotation names to their types (label or quantification).
    """
    io = NWBHDF5IO(nwbfile_path, "r")
    nwbfile = io.read()
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    merge_id = str((SpikeSortingOutput.ImportedSpikeSorting & {"nwb_file_name": nwb_copy_file_name}).fetch1("merge_id"))

    UnitSelectionParams().insert_default()
    group_name = "all_units"
    SortedSpikesGroup().create_group(
        group_name=group_name,
        nwb_file_name=nwb_copy_file_name,
        keys=[{"spikesorting_merge_id": merge_id}],
    )
    group_key = {
        "nwb_file_name": nwb_copy_file_name,
        "sorted_spikes_group_name": group_name,
    }
    group_key = (SortedSpikesGroup & group_key).fetch1("KEY")
    _, unit_ids = SortedSpikesGroup().fetch_spike_data(group_key, return_unit_ids=True)

    for unit_key in tqdm(unit_ids, desc="Inserting Unit Annotations"):
        unit_id = unit_key["unit_id"]
        for annotation, annotation_type in annotation_to_type.items():
            annotation_value = nwbfile.units.get((unit_id, annotation))
            annotation_key = {
                **unit_key,
                "annotation": annotation,
                annotation_type: annotation_value,
            }
            UnitAnnotation().add_annotation(annotation_key, skip_duplicates=True)
    io.close()


def test_sorting(nwbfile_path: Path):
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    with NWBHDF5IO(nwbfile_path, "r") as io:
        nwbfile = io.read()
        units_table = nwbfile.units.to_dataframe()
    group_key = {
        "nwb_file_name": nwb_copy_file_name,
        "sorted_spikes_group_name": "all_units",
    }
    group_key = (SortedSpikesGroup & group_key).fetch1("KEY")
    spikes_spyglass = SortedSpikesGroup().fetch_spike_data(group_key)
    spikes_nwb = [unit.spike_times for _, unit in units_table.iterrows()]
    for nwb_unit, spyglass_unit in zip(spikes_nwb, spikes_spyglass):
        np.testing.assert_array_equal(nwb_unit, spyglass_unit)


if __name__ == "__main__":
    main()
    print("Done!")
