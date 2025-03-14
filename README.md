# TexSense - Analysis of Outdoor Running Kinetics
This repository contains analysis code and kinetic data of 26 participants during outdoor running at six different speeds. The analysis includes time normalization, calculation of mean steps, and statistical analysis using SPM (Statistical Parametric Mapping). Additionally, we provide a Jupyter notebook for plotting individual data and creating reports.

The presented code and data are used in following publications:
 - Höschler, L., Halmich, C., Čigoja, S., Ullirch, M., Koelewijn, A.D., Schwameder, H. (2025). Joint Kinetics and Ground Reaction Forces During Outdoor Running: A Wearable Sensor Study. 

The project TexSense is funded within the context of WISS 2025, der Wissenschafts- und Innovationsstrategie 2025, by the federal state of Salzburg.

## Directory Structure
```
root/
│
├── particpants.csv            # Participant characteristics
├── data/                      # Data directory
│     ├── subject_speed.csv    # .csv files containing kinetic data
│     ...
├── spm_statistics.ipynb       # Jupyter Notebook for performing SPM analysis
├── subject_report.ipynb       # Jupyter Notebook for generating subject reports
│
├── utils.py                   # Utility functions for data processing
├── plots.py                   # Functions for plotting the data
│
├── README.md                  # Project README file
├── requirements.txt           # Project dependencies
├── LICENSE_code.md            # License information (code)
└── LICENSE_data.md            # License information (data)
```

## Data

The `data/` directory contains CSV files for each subject. Each file is named in the format `subject_speed.csv`, where `subject` is the subject ID and `speed` is the running speed in km/h. The `participants.csv` file contains characteristics of the subjects.

### Processing Information
Each file contains 3D joint kinetics (joint moments of hip, knee & ankle) as well as 3D GRFs of multiple individual stance phases during unrestricted outdoor running on a straight level tartan track. These metrics were determined using lower-body IMU data in combination with a pre-trained Convolutional Neural Network. The average relative error of this approach is 5 % (across all metrics and for full gait cycle estimations). The data is already cleaned for outliers.

## Installation
To run the code in this repository, you need to have Python installed along with the following packages:
- pandas==1.5.3
- numpy==1.26.4
- scipy
- scikit-fda>=0.9.1
- matplotlib
- spm1d
- tqdm
- seaborn>=0.12.2

You can install the required packages from the ```requirements.txt``` using pip:

```pip install -r requirements.txt```

## Plotting
The ```subject_report.ipynb``` provides an easy Jupyter Notebook to explore individual biomechanical responses to increased running speeds.

## SPM Analysis
The ```spm_statistics.ipynb``` provides a Jupyter Notebook to perform a repeated ANOVA analysis of the kinetic time series data to investigate the influence of running speed. Post-Hoc paired t-test were applied. For further details please contact us or refer to our manuscript.

## License
Please note that this repository is dual-licensed.

The dataset is available under the terms of the Creative Commons BY-NC 4.0 license.

The provided example code is licensed under the MIT license. 

Please see the respective LICENSE files for more details.

## Contact
Lucas Höschler, [lucas.hoeschler@googlemail.com](mailto:lucas.hoeschler@googlemail.com)