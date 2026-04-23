from ..utils.constants import BASE_DATA_PATH
from ..utils.custom_typing import (
    BRASItem,
    BRASItemSWFTRoom,
    MaterialType,
    MeshDataType,
    MeshType,
)
from ..utils.utils import is_inside
from .spatial_model import SWFTRoom
from pathlib import Path
from pyroomacoustics.acoustics import OctaveBandsFactory
from pyroomacoustics.parameters import Material, constants
from soundfile import write, read
from torch.utils.data import Dataset
from typing import cast
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pyroomacoustics as pra
import torch
import torchaudio


def generate_void_json_file() -> None:
    """Generate a template JSON file for source and receiver positions.

    Creates a JSON file with entries for all scenes in the BRAS dataset with
    placeholder dictionaries for source and receiver positions of different
    loudspeaker types (Genelec8020c and Dodecahedron).

    Raises:
        IOError: If the JSON file cannot be written to the target directory.
    """
    dataset = BRASBenchmarkDataset()
    keys = list(dataset.mesh_dict.keys())
    initial_dict = {}
    for key in keys:
        initial_dict[key] = {
            "Genelec8020c": {"source_positions": {}, "receiver_positions": {}},
            "Dodecahedron": {"source_positions": {}, "receiver_positions": {}},
        }
    json_file_path = os.path.join(
        BASE_DATA_PATH, "BRAS", "source_receiver_positions.json"
    )
    with open(json_file_path, "w") as json_file:
        json.dump(initial_dict, json_file, indent=4)


class BRASBenchmarkDataset(Dataset):
    """PyTorch Dataset for the BRAS Benchmark acoustic dataset.

    Provides access to room impulse responses (RIRs) from the BRAS Benchmark
    dataset, including mesh data, material properties, source/receiver positions,
    and audio waveforms. Supports filtering by scene and material estimate type.

    Attributes:
        base_data_path (Path): Path to the BRAS dataset directory.
        ignore_keys (list[str]): Scene keys to exclude from the dataset.
        mesh_dict (dict): Dictionary mapping scene names to mesh data.
        material_properties (dict): Dictionary of material absorption/scattering properties.
        wav_file_list (list): List of relative paths to RIR WAV files.
        source_receiver_positions (dict): Mapping of source and receiver positions by scene.
    """

    def __init__(
        self,
        base_data_path: Path = Path(BASE_DATA_PATH) / "BRAS",
        ignore_keys: list[str] = [],
        material_types: str = "initial",
    ) -> None:
        """Initialize the BRAS Benchmark dataset.

        Args:
            base_data_path: Path to the BRAS dataset directory. Defaults to
                BASE_DATA_PATH/BRAS.
            ignore_keys: Scene keys to exclude from dataset loading.
                Examples: ["CR1"], ["Genelec"], ["CR1", "CR3"]. Defaults to [].
            material_types: Type of material absorption estimates to use.
                Must be either "initial" (default estimation) or "fitted"
                (fitted to measured RIRs). Defaults to "initial".

        Raises:
            AssertionError: If material_types is not "fitted" or "initial".
            FileNotFoundError: If required dataset files are not found.
        """
        self.base_data_path = base_data_path
        self.ignore_keys = ignore_keys
        assert material_types in [
            "fitted",
            "initial",
        ], "material_types must be either 'fitted' or 'initial'"
        self.mesh_data_json_file = os.path.join(base_data_path, "mesh_data.json")
        self.mesh_dict = self.load_all_meshes()
        self.material_properties_dir = os.path.join(
            base_data_path,
            f"3_surface_descriptions/3 Surface descriptions/_csv/{material_types}_estimates",
        )
        self.material_properties = self.load_all_material_properties()
        self.wav_file_list = self.get_wav_file_list()
        with open(
            os.path.join(base_data_path, "source_receiver_positions.json"), "r"
        ) as json_file:
            self.source_receiver_positions = cast(
                dict[str, dict[str, dict[str, dict[str, list[float]]]]],
                json.load(json_file),
            )

    def __len__(self) -> int:
        """Return the number of RIR samples in the dataset.

        Returns:
            int: Number of WAV files in the dataset.
        """
        return len(self.wav_file_list)

    def __getitem__(self, idx: int) -> BRASItem:
        """Retrieve a RIR sample and associated metadata.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            BRASItem: Dictionary containing:
                - "scene_name" (str): Name of the acoustic scene.
                - "loudspeaker" (str): Loudspeaker identifier.
                - "microphone" (str): Microphone identifier.
                - "loudspeaker_type" (str): Type of loudspeaker used.
                - "mesh" (MeshType): Mesh geometry data.
                - "source_position" (np.ndarray): Source coordinates [x, y, z].
                - "receiver_position" (np.ndarray): Receiver coordinates [x, y, z].
                - "material_properties" (dict): Material absorption/scattering data.
                - "waveform" (Tensor): RIR audio waveform.
                - "sample_rate" (int): Sample rate of the RIR.
                - "wav_file_name" (str): Base name of the RIR file.

        Raises:
            IndexError: If idx is out of bounds.
            FileNotFoundError: If the audio file is not found.
        """
        wav_file_rel_path = self.wav_file_list[idx]
        wav_file_name = cast(
            str, os.path.splitext(os.path.basename(wav_file_rel_path))[0]
        )
        file_path = os.path.join(self.base_data_path, wav_file_rel_path)

        split = wav_file_name.split("_")
        split.pop(1)
        scene_name = split[0]
        if scene_name == "CR1":
            scene_name = scene_name + "_" + split[1]
            split.pop(1)
        split.pop(0)
        loudspeaker_nb = split[0]
        microphone_nb = split[1]
        loudspeaker_type = split[2]

        source_position = self.source_receiver_positions[scene_name][loudspeaker_type][
            "source_positions"
        ].get(loudspeaker_nb, None)
        receiver_position = self.source_receiver_positions[scene_name][
            loudspeaker_type
        ]["receiver_positions"].get(microphone_nb, None)

        source_position = np.array(source_position, dtype=np.float32)
        receiver_position = np.array(receiver_position, dtype=np.float32)
        mesh = self.mesh_dict[scene_name]
        mesh = self.check_mesh_consistency(mesh, source_position)

        waveform, sample_rate = read(file_path)
        waveform = torch.from_numpy(waveform.T).unsqueeze(0).float()
        return {
            "scene_name": scene_name,
            "loudspeaker": loudspeaker_nb,
            "microphone": microphone_nb,
            "loudspeaker_type": loudspeaker_type,
            "mesh": mesh,
            "source_position": source_position,
            "receiver_position": receiver_position,
            "material_properties": self.material_properties,
            "waveform": waveform,
            "sample_rate": sample_rate,
            "wav_file_name": wav_file_name,
        }

    def get_wav_file_list(self) -> list[str]:
        """Retrieve list of all RIR WAV file paths in the dataset.

        Recursively searches the base data directory for WAV files containing
        "_RIR_" in the filename, excluding binaural RIRs and ignored scenes.

        Returns:
            list[str]: Relative paths to RIR WAV files.
        """
        wav_file_list = []
        for root, dirs, files in os.walk(self.base_data_path):
            for file in files:
                if file.endswith(".wav") and "_RIR_" in file:  # Discard Binaural RIRs
                    if any(key in file for key in self.ignore_keys):
                        continue
                    else:
                        relative_path = os.path.relpath(
                            os.path.join(root, file), self.base_data_path
                        )
                        wav_file_list.append(relative_path)
        return wav_file_list

    def load_all_meshes(self) -> dict[str, MeshType]:
        """Load all mesh geometries from the mesh data JSON file.

        Reads mesh vertices, faces, and material assignments from a JSON file
        and organizes them into a structured dictionary.

        Returns:
            dict[str, MeshType]: Mapping of scene names to mesh data containing
                vertices (np.ndarray), faces (np.ndarray), and material assignments.

        Raises:
            FileNotFoundError: If the mesh data JSON file is not found.
            json.JSONDecodeError: If the JSON file is malformed.
        """
        with open(self.mesh_data_json_file, "r") as json_file:
            mesh_dict: MeshDataType = json.load(json_file)
            for key in mesh_dict.keys():
                vertices = np.array(mesh_dict[key]["vertices"])
                faces = np.array(mesh_dict[key]["faces"])
                material_assignment = mesh_dict[key]["material_assignment"]
                mesh: MeshType = {
                    "vertices": vertices,
                    "faces": faces,
                    "material_assignment": material_assignment,
                }
                mesh_dict[key] = mesh
        return mesh_dict

    def load_all_material_properties(self) -> dict[str, MaterialType]:
        """Load material absorption and scattering coefficients from CSV files.

        Scans the material properties directory for CSV files containing
        frequency-dependent absorption and scattering data.

        Returns:
            dict[str, MaterialType]: Mapping of material names to their properties
                (frequency array, absorption coefficients, scattering coefficients).

        Raises:
            FileNotFoundError: If the material properties directory doesn't exist.
        """
        material_properties: dict[str, MaterialType] = {}
        for root, dirs, files in os.walk(self.material_properties_dir):
            for file in files:
                if file.endswith(".csv"):
                    material_name = os.path.splitext(file)[0]
                    file_path = os.path.join(root, file)
                    if "CR" in material_name:
                        properties = np.genfromtxt(file_path, delimiter=",")
                        material: MaterialType = {
                            "frequency": properties[0],
                            "absorption": properties[1],
                            "scattering": properties[2],
                        }
                        material_properties[material_name] = material
        return material_properties

    def check_mesh_consistency(
        self,
        mesh: MeshType,
        inside_point: np.ndarray,
    ) -> MeshType:
        """Verify and fix mesh orientation using a point known to be inside.

        Uses the generalized winding number algorithm to detect if the mesh
        normals are correctly oriented. If the inside point is detected outside,
        reverses all face winding orders to fix the orientation.

        Args:
            mesh: Mesh data containing vertices, faces, and material assignment.
            inside_point: A point known to be inside the mesh (e.g., source or receiver).

        Returns:
            MeshType: The same mesh with corrected orientation if needed.

        Raises:
            AssertionError: If the mesh is not consistent even after fixing, or if
                the outside point is detected inside the mesh.
        """
        vertices = mesh["vertices"]
        faces = mesh["faces"]
        triangles = vertices[faces]
        # we do not do a clone here, as we will modify
        # the faces in-place if we need to flip the normals.
        # This way, we avoid flipping the normals twice
        # if the mesh is already consistent.

        check_point = np.expand_dims(inside_point, axis=0)
        bbox = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        outside_point = np.array([bbox[0][0] - 1, bbox[0][1] - 1, bbox[0][2] - 1])[
            None, :
        ]
        # We use a relatively high tolerance here,
        # as the meshes in the dataset are not perfect,
        # and we want to avoid flipping the normals if it is not necessary.
        tol = 0.1

        # Check if an outside point is detected outside the mesh.
        # If not, the mesh might not be a closed manifold.
        assert not is_inside(
            triangles, outside_point, tol=tol
        ), "Mesh is not consistent, the outside point is inside the mesh."

        # Check if an inside point is detected inside.
        # If not, the mesh might have flipped normals.
        if not is_inside(triangles, check_point, tol=tol):
            for i in range(len(faces)):
                faces[i] = faces[i][::-1]
            triangles = vertices[faces]

            # Double check if the inside point is now detected inside the mesh.
            # If not, the mesh might not be a closed manifold
            # (probably redundant with the first test, but we keep it for safety)
            assert is_inside(
                triangles, check_point, tol=tol
            ), "Mesh is not consistent, and flipping the normals did not solve the issue."
        # Mesh has been modified in-place, but we return it
        # for clarity and consistency with the rest of the code.
        return mesh


class BRASBenchmarkToSWFTRoom(Dataset):
    """Converts BRAS Benchmark dataset to SWFT rooms with pyroomacoustics backend.

    This dataset adapter transforms BRAS mesh geometries and material properties
    into SWFTRoom objects using pyroomacoustics for audio simulation. Rooms are
    cached in memory to avoid redundant computations when accessing the same
    scene multiple times.

    Attributes:
        dataset (BRASBenchmarkDataset): Underlying BRAS dataset.
        cached_rooms (dict): Cache of SWFTRoom objects indexed by scene name.
        fs (int): Sampling frequency for pyroomacoustics rooms.
        pra_max_order (int): Maximum reflection order for ISM.
        octave_bands: Octave band factory for material resampling.
        material_library (dict): Converted pyroomacoustics Material objects.
    """

    def __init__(
        self,
        base_data_path: Path = Path(BASE_DATA_PATH) / "BRAS",
        ignore_keys: list[str] = [],
        fs: int = 32000,
        pyroomacoustics_max_order: int = 5,
        material_types: str = "initial",
    ) -> None:
        """Initialize the BRAS to SWFT room converter.

        Args:
            base_data_path: Path to the BRAS dataset directory. Defaults to
                BASE_DATA_PATH/BRAS.
            ignore_keys: Scene keys to exclude. Defaults to [].
            fs: Sampling frequency for pyroomacoustics room objects.
                Must be a power-of-2 multiple of 125 Hz. Defaults to 32000 Hz.
            pyroomacoustics_max_order: Maximum reflection order for image source
                model in pyroomacoustics. Defaults to 5.
            material_types: Material estimate type ("initial" or "fitted").
                Defaults to "initial".

        Raises:
            ValueError: If sampling frequency is not a power-of-2 multiple of 125 Hz.
        """
        self.dataset = BRASBenchmarkDataset(
            base_data_path=base_data_path,
            ignore_keys=ignore_keys,
            material_types=material_types,
        )
        self.cached_rooms = {}
        if np.floor(np.log2(fs / 125)) != np.log2(fs / 125):
            raise ValueError(
                "Sampling frequency must be a power of 2 multiple of 125 Hz."
            )
        self.fs = fs
        self.pra_max_order = pyroomacoustics_max_order
        self.octave_bands = OctaveBandsFactory(
            fs=self.fs,
            n_fft=constants.get("octave_bands_n_fft"),
            keep_dc=constants.get("octave_bands_keep_dc"),
            base_frequency=constants.get("octave_bands_base_freq"),
        )
        self.material_library = self.build_material_library(
            self.dataset.material_properties
        )

    def __len__(self) -> int:
        """Return the number of RIR samples in the dataset.

        Returns:
            int: Number of samples in the underlying BRAS dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> BRASItemSWFTRoom:
        """Retrieve a BRAS item with an associated SWFT room object.

        Extends the BRASBenchmarkDataset item with a cached SWFTRoom object
        constructed from the mesh and material properties. Rooms are cached
        by scene name to avoid redundant computations.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            BRASItemSWFTRoom: Dictionary containing all BRASItem fields plus:
                - "swft_room" (SWFTRoom): The converted room object.

        Raises:
            IndexError: If idx is out of bounds.
        """
        data = self.dataset[idx]
        scene_name = data["scene_name"]
        original_fs = data["sample_rate"]
        rir_gt = data["waveform"]
        resampled_rir_gt = torchaudio.transforms.Resample(
            orig_freq=original_fs, new_freq=self.fs
        )(rir_gt)
        data["waveform"] = resampled_rir_gt
        data["sample_rate"] = self.fs

        data = cast(BRASItemSWFTRoom, data)

        if scene_name in self.cached_rooms:
            data["swft_room"] = self.cached_rooms[scene_name]
            return data
        else:
            mesh = data["mesh"]
            material_assignment = mesh["material_assignment"]
            material_properties = self.dataset.material_properties
            pra_room = self.get_pra_room(mesh, material_assignment, material_properties)
            swft_room = SWFTRoom(pra_room)
            self.cached_rooms[scene_name] = swft_room
            data["swft_room"] = swft_room
            return data

    def get_pra_room(
        self,
        mesh: MeshType,
        material_assignment: list[str],
        material_properties: dict[str, MaterialType],
    ) -> "pra.Room":
        """Build a pyroomacoustics Room from mesh geometry and materials.

        Converts mesh faces and material assignments into pyroomacoustics walls
        and constructs a Room object with the specified sampling frequency and
        maximum reflection order.

        Args:
            mesh: Mesh data with vertices, faces, and material assignment.
            material_assignment: List mapping face indices to material names.
            material_properties: Dictionary of material properties.

        Returns:
            pra.Room: Pyroomacoustics room object configured with the mesh walls
                and material properties.
        """
        vertices = mesh["vertices"]
        faces = mesh["faces"]

        walls = self.build_walls_from_faces(
            vertices[faces], material_assignment, material_properties
        )
        room = pra.Room(walls, fs=self.fs, max_order=self.pra_max_order)

        return room

    def build_material_library(
        self, material_properties: dict[str, MaterialType]
    ) -> dict:
        """Convert material properties to pyroomacoustics Material objects.

        Resamples material absorption and scattering coefficients to octave bands
        matching the pyroomacoustics configuration.

        Args:
            material_properties: Dictionary mapping material names to their
                frequency-dependent absorption and scattering data.

        Returns:
            dict: Mapping of material names to pyroomacoustics Material objects.
        """
        material_library = {}
        for material_name, properties in material_properties.items():
            center_freqs = properties["frequency"]
            absorption = properties["absorption"]
            scattering = properties["scattering"]

            material = Material(
                energy_absorption={
                    "description": material_name,
                    "coeffs": absorption,
                    "center_freqs": center_freqs,
                },
                scattering={
                    "description": material_name,
                    "coeffs": scattering,
                    "center_freqs": center_freqs,
                },
            )
            material.resample(self.octave_bands)
            material_library[material_name] = material
        return material_library

    def build_walls_from_faces(
        self,
        faces: np.ndarray,
        material_assignment: list[str],
        material_properties: dict[str, MaterialType],
    ) -> list:
        """Construct pyroomacoustics walls from mesh faces.

        Converts each triangular face and its assigned material into a
        pyroomacoustics wall object with appropriate absorption and scattering.

        Args:
            faces: Array of triangular faces from the mesh geometry.
            material_assignment: List mapping face indices to material names.
            material_properties: Dictionary of material properties.

        Returns:
            list: List of pyroomacoustics Wall objects.
        """
        walls = []
        for i, face in enumerate(faces):
            material_name = material_assignment[i]
            material = self.material_library[material_name]
            vertices = face.copy()
            wall = pra.wall_factory(
                corners=vertices.T,
                absorption=material.absorption_coeffs,
                scattering=material.scattering_coeffs,
                name="wall_" + str(i),
            )
            walls.append(wall)
        return walls


def try_dataset():
    dataset = BRASBenchmarkDataset(ignore_keys=["CR1", "CR3", "CR4"])
    print(f"Number of meshes in the dataset: {len(dataset)}")
    data = dataset[0]
    print("Data keys:", data.keys())
    print("wav file name:", data["wav_file_name"])
    print("Scene name:", data["scene_name"])
    print("Loudspeaker:", data["loudspeaker"])
    print("Microphone:", data["microphone"])
    print("Loudspeaker type:", data["loudspeaker_type"])
    print("Mesh vertices shape:", data["mesh"]["vertices"].shape)
    print("Mesh faces shape:", data["mesh"]["faces"].shape)
    print("Source position:", data["source_position"])
    print("Receiver position:", data["receiver_position"])
    print("Material properties keys:", data["material_properties"].keys())
    print("Waveform shape:", data["waveform"].shape)
    print("Sample rate:", data["sample_rate"])

    rir, sr = data["waveform"], data["sample_rate"]
    time = torch.arange(rir.shape[1]) / sr
    plt.plot(time, rir[0].numpy())
    plt.title("RIR Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    stft = torch.stft(rir, n_fft=512, hop_length=256, return_complex=True)
    magnitude = torch.abs(stft)
    plt.figure()
    plt.imshow(10 * torch.log10(magnitude[0] + 1e-10), aspect="auto", origin="lower")
    plt.title("RIR Spectrogram")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")

    freq_bandwidth = sr / 2 / 10
    mag_bands = np.zeros((10, stft.shape[2]))
    for i in range(10):
        freq_mask = (i * freq_bandwidth <= torch.fft.fftfreq(stft.shape[1], 1 / sr)) & (
            torch.fft.fftfreq(stft.shape[1], 1 / sr) < (i + 1) * freq_bandwidth
        )
        mag_bands[i] = magnitude[0][freq_mask].mean(dim=0).numpy()

    edr = 10 * np.log10(
        (mag_bands**2)[:, ::-1].cumsum(axis=1)[:, ::-1]
        / (mag_bands**2).sum(axis=1, keepdims=True)
    )
    print(edr.shape)
    freq_bands = [
        f"{i*freq_bandwidth:.1f}-{(i+1)*freq_bandwidth:.1f}Hz" for i in range(10)
    ]
    plt.figure()
    for i in range(10):
        plt.plot(edr[i], label=freq_bands[i])
    plt.legend()
    plt.title("Energy Decay Relief")
    plt.xlabel("Time Frames")
    plt.ylabel("EDR (dB)")
    plt.show()


def try_BRAStoSWFTRoom():
    # This allows to load material absorptions starting from 20Hz.
    # The standard default value for pyroomacoustics is 125 Hz.
    constants.set("octave_bands_base_freq", 31.25)
    swft_dataset = BRASBenchmarkToSWFTRoom(fs=44100, material_types="initial")
    data = swft_dataset[1]

    room_object = cast(SWFTRoom, data["swft_room"])
    receiver_position = data["receiver_position"]
    source_position = data["source_position"]
    rir_gt = data["waveform"]
    rir_sr = data["sample_rate"]
    raw_mesh = data["mesh"]

    print("Raw mesh faces shape:", raw_mesh["faces"].shape)
    print("SWFT room walls number:", len(room_object.room.walls))

    walls = room_object.room.walls
    wall = walls[0]
    wall_name = wall.name
    wall_number = int(wall_name.split("_")[1])
    wall_material_name = data["mesh"]["material_assignment"][wall_number]
    print("Wall name:", wall_name)
    print("Wall material name:", wall_material_name)
    print("Wall absorption coefficients:", wall.absorption)
    print("Wall scattering coefficients:", wall.scatter)
    print("Room center_freqs:", room_object.center_freqs)

    print("Data keys:", data.keys())
    print("Scene name:", data["scene_name"])
    print("SWFT room:", data["swft_room"])
    room_object.room.plot()
    plt.plot(
        receiver_position[0],
        receiver_position[1],
        receiver_position[2],
        "ro",
        label="Receiver",
    )
    plt.plot(
        source_position[0], source_position[1], source_position[2], "go", label="Source"
    )
    plt.legend()
    plt.title("SWFT Room Geometry with Source and Receiver Positions")

    freq_response_at_mic = room_object.get_frequency_response_at_point(
        receiver_position
    )
    rt60_at_mic = room_object.get_rt60_profile()

    plt.figure()
    plt.plot(
        room_object.center_freqs,
        freq_response_at_mic,
        marker="o",
    )
    plt.title("Frequency Response at Microphone Position")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xscale("log")

    plt.figure()
    plt.plot(room_object.center_freqs, rt60_at_mic, marker="o")
    plt.title("RT60 Profile at Microphone Position")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("RT60 (s)")
    plt.xscale("log")

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=rir_sr, n_fft=512, hop_length=256, n_mels=64
    )
    mel_spec = mel(rir_gt)
    magnitude = torch.abs(mel_spec)
    plt.figure()
    plt.imshow(10 * torch.log10(magnitude[0] + 1e-6), aspect="auto", origin="lower")
    plt.title("RIR Mel Spectrogram")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")

    plt.figure()
    t = torch.arange(rir_gt.shape[1]) / data["sample_rate"]
    plt.plot(t, rir_gt[0].numpy())
    plt.title("RIR Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.show()
