# Repository Structure

This file provides a standardized view of the repository layout.

## Top-level Layout

```text
BEST-study/
|-- LICENSE
|-- README.md
|-- .gitignore
|-- docs/
|   |-- CODE_AVAILABILITY_STATEMENT.md
|   `-- REPOSITORY_STRUCTURE.md
|-- Benign_Malignant_Model&Invasive_Carcinoma_in_Situ_Carcinoma_Model/
|   |-- README.md
|   |-- main.py
|   |-- eval.py
|   |-- cfgs/
|   |-- datasets/
|   |-- models/
|   `-- utils/
|-- Breast_Intelligent_Recognition_Device/
|   |-- README.md
|   |-- mainBI.py
|   |-- requirements.txt
|   |-- data_utils/
|   |-- models/
|   `-- BC_data/
`-- Molecula_Subtype_Model/
    |-- README.md
    `-- MSM.py
```

## Module Notes

### Benign_Malignant_Model&Invasive_Carcinoma_in_Situ_Carcinoma_Model
- Binary classification tasks for BMM and ICM.
- Config-driven training process (`cfgs/*.yaml`).

### Breast_Intelligent_Recognition_Device
- Multi-architecture classification workflows.
- Organized by data utilities and model implementations.

### Molecula_Subtype_Model
- Molecular subtype classification implementation.
- Compact runnable script (`MSM.py`).

## Organization Principles

- Keep model code and utilities grouped within each module.
- Keep module-level README files as the first source of run instructions.
- Keep repository-level governance and code-availability information at top level (`README.md`, `docs/`, `LICENSE`).
