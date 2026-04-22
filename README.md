# Frontier Dataset Generation

## Setup
1. Clone the repository
   
   ```bash
   git clone https://github.com/siddharthmaram/Frontier-Dataset-Generation.git
   cd Frontier-Dataset-Generation
   ```
2. Create conda environment

   ```bash
   conda env create -f environment.yml
   conda activate habitat
   ```
3. Install other libraries

   ```bash
   pip install pyyaml
   ```
   
5. Download the HM3D dataset (train and val splits along with semantics) and add the path to `config.yaml`
6. `camera_height` can be changed in `config.yaml`, and `AGENT_HEIGHT` variable can be changed in `generate_dataset.py` (Default is 1.5 meters for both).

## Run

To run the code, use the command
```bash
python generate_dataset.py --driver
```

To run the code for a specific scene, use the command
```bash
python generate_dataset.py --scene <scene_name>
```

To start from a specific scene, use the command
```bash
python generate_dataset.py --driver --start-index 0
```

