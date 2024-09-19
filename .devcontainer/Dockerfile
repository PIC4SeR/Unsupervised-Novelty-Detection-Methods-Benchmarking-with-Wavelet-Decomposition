FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    python3-pip python3-pyqt5 git python3-pyqt5.qtsvg python3-tk texlive-extra gnupg curl

RUN apt-get update && \
apt-get -y install sudo

RUN curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
    sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg \
    --dearmor

RUN echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

RUN apt-get update && apt-get install -y mongodb-org

RUN pip install tensorflow pandas numpy matplotlib seaborn scikit-learn \
    optuna ipykernel ipympl tensorflow-probability tf-keras opencv-python \
    ultralytics absl-py alabaster arrow asttokens Babel certifi charset-normalizer \
    click colorama contourpy cycler Cython decorator distlib dnspython docutils \
    executing filelock fonttools idna imagesize immutabledict iniconfig ipython \
    jedi Jinja2 joblib k-means-constrained kiwisolver llvmlite markdown-it-py \
    MarkupSafe matplotlib matplotlib-inline mdurl meson meson-python numba numpy \
    numpydoc ortools packaging pandas parso pbr pillow pipenv pipenv-to-requirements \
    platformdirs pluggy prompt_toolkit protobuf pure-eval Pygments pymongo \
    pyparsing pyproject-metadata PyQt5 PyQt5-Qt5 PyQt5-sip pyrsistent pyserial \
    pytest python-dateutil pytz PyWavelets PyYAML requests rich scikit-learn==1.5.0 \
    scipy seaborn shellingham six snowballstemmer Sphinx sphinxcontrib-applehelp \
    sphinxcontrib-devhelp sphinxcontrib-htmlhelp sphinxcontrib-jsmath sphinxcontrib-qthelp \
    sphinxcontrib-serializinghtml ssqueezepy stack-data tabulate threadpoolctl tikzplotlib \
    traitlets typer types-python-dateutil typing_extensions tzdata urllib3 virtualenv \
    wcwidth webcolors onnx onnxruntime optuna-dashboard

RUN apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super