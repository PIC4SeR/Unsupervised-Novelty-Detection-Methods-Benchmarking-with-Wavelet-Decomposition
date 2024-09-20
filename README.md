
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
