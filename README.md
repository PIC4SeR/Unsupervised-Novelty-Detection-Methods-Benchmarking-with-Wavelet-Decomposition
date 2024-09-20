
# Reference Code for the Paper: "Unsupervised Novelty Detection Methods Benchmarking with Wavelet Decomposition"

This repository contains the reference code for the paper titled **"Unsupervised Novelty Detection Methods Benchmarking with Wavelet Decomposition"**, which will be published as a proceeding in the **8th International Conference on System Reliability and Safety (ICSRS)**, held in Sicily, Italy, from **November 20-22, 2024**.

- Preprint link: [https://arxiv.org/abs/2409.07135](https://arxiv.org/abs/2409.07135)
- Conference link: [https://www.icsrs.org/](https://www.icsrs.org/)
- Author contact: [https://orcid.org/0009-0006-2741-8054](https://orcid.org/0009-0006-2741-8054)

## Setup Instructions for Linux-Based Machine

Follow these steps to set up and run the project on a Linux-based system.

### Prerequisites

Before starting, ensure you have the following installed:

1. **Git**: Download and install Git by following instructions from the [github official website](https://github.com/git-guides/install-git).
   
2. **Docker**: Install Docker from the [docker official website](https://docs.docker.com/engine/install/).

3. **PDF Viewer**: Ensure you have a PDF viewer installed to open PDF files.

### Step-by-Step Instructions

1. **Pull the Docker Image**: Download the pre-built Docker image.
   ```bash
   docker pull arielpriarone/unsupervised-novelty-detection-methods-benchmarking-with-wavelet-decomposition:latest
   ```

2. **Tag the Docker Image**: Tag the downloaded image for easier reference.
   ```bash
   docker image tag arielpriarone/unsupervised-novelty-detection-methods-benchmarking-with-wavelet-decomposition:latest undbwwd:latest
   ```

3. **Clone the Git Repository**: Clone the repository containing the project code.
   ```bash
   git clone https://github.com/PIC4SeR/Unsupervised-Novelty-Detection-Methods-Benchmarking-with-Wavelet-Decomposition.git UNDBwWD
   ```

4. **Navigate to the Project Directory**: Change the directory to the cloned repository.
   ```bash
   cd UNDBwWD
   ```

5. **Run the Docker Container**: Start the container with necessary settings.
   ```bash
   docker run -it -v $(pwd):/code --runtime=nvidia --gpus all undbwwd:latest
   ```

6. **Create Database Directory**: Set up a directory for MongoDB.
   ```bash
   mkdir -p /data/db
   ```

7. **Set Directory Permissions**: Change the ownership of the directory.
   ```bash
   chown `id -u` /data/db
   ```

8. **Start MongoDB**: Run MongoDB in the background.
   ```bash
   mongod --fork --logpath /var/log/mongodb/mongod.log
   ```

9. **Navigate to the Code Directory**: Go to the code folder inside the container.
   ```bash
   cd code
   ```

10. **Run the Demo Script**: Execute the demo script.
   ```bash
   python3 demo.py
   ```

11. **Go to the Results Directory**: Navigate to the results directory.
   ```bash
   cd results
   ```

12. **Compile the LaTeX File**: Use `pdflatex` to generate the table.
   ```bash
   pdflatex table.tex
   ```

13. **Exit the Container**: Press `Ctrl+D` to exit the Docker container.

14. **Open the Results**: Use a PDF viewer to open the results.
     ```bash
     xdg-open Correlation_table.pdf
     xdg-open Fig5.pdf
     xdg-open Fig7.pdf
     xdg-open table.pdf
     ```

---

Follow these steps to set up the environment, run the project, and view the results.



## Customizing Script Settings

The `demo.py` script includes several configuration flags that allow you to customize its behavior. You can modify these settings to control which steps are executed when running the script.

To change the script settings, open the `demo.py` file in a text editor and locate the following section:

    load_timeseries_flag    = True	# if True, the timeseries are loaded to MongoDB
    run_FA_flag             = True	# if True, the feature agent is run
    train_flag              = False	# if True, the models are trained
    NormalizeNovMetric      = True	# if True, the novelty metric is normalized
    Generate_PCA            = False	# if True, the PCA models are generated
    Optimize_Autoencoder    = False	# if True, the Autoencoder models are optimized
    Fit_Autoencoder         = False	# if True, the Autoencoder models are fitted
    plot_patches            = False	# if True, the variance is visualized as a patch in the plot

### Configuration Flags Description

-   **`load_timeseries_flag`**:
    
    -   **Purpose**: Loads the time series data into MongoDB.
    -   **Set to `True`**: If you want to load the time series data. Useful for the initial setup.
    -   **Set to `False`**: If the data is already loaded and you want to skip this step.
-   **`run_FA_flag`**:
    
    -   **Purpose**: Executes the feature agent process.
    -   **Set to `True`**: To run the FA, which is necessary for generating feature sets.
    -   **Set to `False`**: To skip the FA process the features are already loded in the MongoDB database
-   **`train_flag`**:
    
    -   **Purpose**: Trains the machine learning models used in the analysis.
    -   **Set to `True`**: If you need to train the models from scratch.
    -   **Set to `False`**: If you prefer to use pre-trained models.
-   **`NormalizeNovMetric`**:
    
    -   **Purpose**: Normalizes the novelty metric for consistent scaling.
    -   **Set to `True`**: To normalize the novelty scores.
    -   **Set to `False`**: To use raw novelty scores without normalization.
-   **`Generate_PCA`**:
    
    -   **Purpose**: Generates Principal Component Analysis (PCA) models.
    -   **Set to `True`**: If you need to create PCA models for dimensionality reduction.
    -   **Set to `False`**: If PCA models are already generated.
-   **`Optimize_Autoencoder`**:
    
    -   **Purpose**: Optimizes the hyperparameters of Autoencoder models.
    -   **Set to `True`**: To perform hyperparameter optimization on Autoencoders.
    -   **Set to `False`**: To skip optimization and use existing parameters.
-   **`Fit_Autoencoder`**:
    
    -   **Purpose**: Fits the Autoencoder models to the data.
    -   **Set to `True`**: If you need to fit the Autoencoders after optimization.
    -   **Set to `False`**: If the Autoencoders are already fitted.
-   **`plot_patches`**:
    
    -   **Purpose**: Visualizes variance as patches in the generated plot.
    -   **Set to `True`**: To include variance patches in your visualizations.
    -   **Set to `False`**: To generate plots without variance patches.
