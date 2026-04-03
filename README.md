# BEST-study

Code repository for the BEST study series in breast ultrasound analysis.

This repository has been reorganized for clarity and accessibility. The update focuses on non-functional cleanup:
- unified top-level documentation
- explicit module-level navigation
- standardized license and code availability notes
- repository housekeeping files for cleaner version control

No algorithm logic was changed during this cleanup.

## Repository Modules

### 1) BMM and ICM model
Path: `Benign_Malignant_Model&Invasive_Carcinoma_in_Situ_Carcinoma_Model`

Purpose:
- BMM: benign vs malignant classification
- ICM: invasive carcinoma vs ductal carcinoma in situ classification

Main entry points:
- `main.py` for training/inference workflows
- `eval.py` for evaluation

### 2) Breast Intelligent Recognition Device
Path: `Breast_Intelligent_Recognition_Device`

Purpose:
- Multi-model breast imaging classification workflow

Main entry points:
- `mainBI.py` for training and evaluation
- `requirements.txt` for dependencies

### 3) Molecular Subtype Model (MSM)
Path: `Molecula_Subtype_Model`

Purpose:
- Molecular subtype classification from ultrasound images

Main entry point:
- `MSM.py`

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/nayutayuki/BEST-study.git
   cd BEST-study
   ```
2. Enter the target module directory.
3. Follow each module README for environment setup and commands.

## Documentation Index

- Repository structure: `docs/REPOSITORY_STRUCTURE.md`
- Code availability statement: `docs/CODE_AVAILABILITY_STATEMENT.md`

## License

This project is released under the MIT License. See `LICENSE`.
