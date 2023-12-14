# CellDeathPred: a deep learning framework for ferroptosis and apoptosis prediction based on cell painting

Application of contrastive learning for cell painting dataset
[[`nature CDD`](https://www.nature.com/articles/s41420-023-01559-y)] [[`bioRxiv`](https://www.biorxiv.org/content/10.1101/2023.03.14.532633v1)]

## Testing dataset:

The image data used in this work and described in Figure 1 and Figure 2 are available at [zenodo](https://zenodo.org/records/8375591)

<img width="657" alt="Screenshot 2023-09-25 at 13 10 41" src="https://github.com/peng-lab/CellDeathPred/assets/67750721/745703b7-4d69-446a-b9e1-d76d4ebf4789">

Figure 1: A Schematic overview of the cell death inducers used for this study. HT-1080 cells were seeded and treated with 7 apoptosis inducers, 7 ferroptosis inducers (FINs), and DMSO as solvent control. Cells treated with apoptosis inducers execute the apoptotic program by activating caspases. Treatment of cells with FINs results in lipid peroxide accumulation due to the limited GPX4 activity and hence induce ferroptosis. B Results of the dose-response (20-point) viability assay with apoptosis inducers in HT-1080 cells. 24 h and 72 h incubation time. Cellular ATP levels were measured using luminescence signals. Values indicate mean ± SD (n = 6, technical replicates). C Same as in B here for treatment with FINs.

<img width="639" alt="Screenshot 2023-09-25 at 13 12 26" src="https://github.com/peng-lab/CellDeathPred/assets/67750721/a2e6a154-a7f4-4aa5-bb0e-fc5a9a65a934">

Figure 2: Schematic overview of the data generation process. A HT-1080 cells were treated with five different concentrations of apoptosis and ferroptosis inducers. ATP measurement (left) and cell painting (right) experiments were conducted in parallel. Staurosporine (STS) and RSL3 data are shown as representative data for apoptosis and ferroptosis inducers, respectively. Values indicate mean values ± SD of six technical replicates. The cells were imaged with a ×40 objective. The different organelles (nuclei, golgi apparatus, actin cytoskeleton, mitochondria, endoplasmatic reticulum) were imaged using four different fluorescence channels. B The data from the viability assay were annotated with the images from the cell painting experiment. Only if viability was in the range of 80–30% the images were used for model training. Three experiments were performed. Experiments 1 and 2 were used for training the CellDeathPred model. Experiment 3 to test the model.

Each image has the following metadata associated with it: row, column, well, plate, field, concentration, atp value, treatment, cell death type. This information is contained in the .csv files.\
Because, we select wells with ATP values that fall into the range [0.3–0.8] we have two .csv files: df_exp8_all.csv and df_exp8_range2.csv. The first contains all the images while the second only those that are inside the range respectively.
