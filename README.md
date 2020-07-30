# mimic cxr

### MIMIC Physionet CXR:
This is a collection of chest X-ray images containing diagnosis of different chest Xray infections.

### TODO
 * DONE: Train a simple classification model ✔️ 
 * Add data description
 * Add data visualization examples
 * Show train/test metrics with tensorboard

## File strucutre

├── notebooks
│   ├── Full_data_prep.ipynb
│   └── Pneumonia_data_prep.ipynb
├── output
│   ├── image_paths.csv
│   ├── pneumonia_df.csv
│   ├── pneumonia_images_and_labels.csv
│   └── pneumonia_images_and_labels_modified.csv
├── README.md
├── src
│   ├── data.py
│   └── model.py
│
└── utils
    ├── data.py
    └── generate_imgcsv.py

    