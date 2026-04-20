![Python Version](https://img.shields.io/badge/python-3.14-blue)
[![Tests](https://github.com/TAYLOR-SWFT/Taylor-SWFT/actions/workflows/tests.yml/badge.svg)](https://github.com/TAYLOR-SWFT/Taylor-SWFT/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Taylor SWFT

## Fast Discrete Statistical Wave Field Theory Using Taylor Expansion For Late Reverberation

# Install

Download the repository:
```sh
git clone https://github.com/TAYLOR-SWFT/Taylor-SWFT.git
```
Install the package:
```sh
pip install -e .[test] # or simply . if you do not run tests.
```
or with `conda`:
```sh
conda env create -f conda_env.yaml
```

# Run our experiments

You can download the BRAS benchmark dataset [here](https://depositonce.tu-berlin.de/items/38410727-febb-4769-8002-9c710ba393c4/full). In the paper, we used rooms from CR1 to CR4. Put them in the `data` folder (or any other one) following this structure:

```
data/BRAS/
├── 1_scene_descriptions-CR1
│   └── 1 Scene descriptions
│       └── CR1 coupled rooms (laboratory and reverberation chamber)
│           ├── BRIRs
│           │   └── wav
│           ├── Geometry
│           │   └── rooms
│           ├── Pictures
│           │   └── Details
│           └── RIRs
│               └── wav
├── 1_scene_descriptions-CR2
│   └── 1 Scene descriptions
│       └── CR2 small room (seminar room)
│           ├── BRIRs
│           │   └── wav
│           ├── Geometry
│           ├── Pictures
│           │   └── Details
│           └── RIRs
│               └── wav
├── 1_scene_descriptions-CR3
│   └── 1 Scene descriptions
│       └── CR3 medium room (chamber music hall)
│           ├── BRIRs
│           │   └── wav
│           ├── Geometry
│           ├── Pictures
│           │   └── Details
│           └── RIRs
│               └── wav
├── 1_scene_descriptions-CR4
│   └── 1 Scene descriptions
│       └── CR4 large room (auditorium)
│           ├── BRIRs
│           │   └── wav
│           ├── Geometry
│           ├── Pictures
│           │   └── Details
│           └── RIRs
│               └── wav
└── 3 Surface descriptions
│   ├── _csv
│   │   ├── fitted_estimates
│   │   └── initial_estimates
│   ├── _descr
│   ├── _img
│   └── _plots
├── mesh_data.json
└── source_receiver_positions.json
```
_For any question about `mesh_data.json` and `source_receiver_positions.json`, please contact us._

When everything is installed, you can run `pytest` to ensure the installation is correct and then run the `main` file. 
_For faster runs you can lower the number of wanted sources for the ISM baseline._

# Reference
If this work has been useful to you, please cite this paper:
```
@article{,
  author  = ,
  title   = ,
  journal = ,
  volume  = ,
  number  = ,
  pages   = ,
  year    = ,
  month   = ,
  issn    = ,
  doi     = ,
  url     = ,
  eprint  = 
}
```

# Contact

For any question, please contact [Louis Lalay](mailto:louis.lalay@telecom-paris.fr) or [Marius Rodrigues](mailto:marius.rodrigues@telecom-paris.fr).

# License
This work is licensed under the terms of the MIT license.