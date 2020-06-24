# Docker file for data_analysis_pipeline_eg
# Kenneth Foo, Feb, 2020

# use rocker/tidyverse as the base image
FROM rocker/tidyverse

# Install R
RUN apt-get update && \ 
    apt-get install r-base r-base-dev -y

# Install conda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    /opt/conda/bin/conda update -n base -c defaults conda

# put anaconda python in path
ENV PATH="/opt/conda/bin:${PATH}"

# For rocker/rstudio authentication stuff
CMD ["/bin/bash"]

RUN conda install -y -c anaconda docopt \
                        selenium \
                        altair vega_datasets && \
    conda install -c conda-forge  xgboost \
                                  lightgbm \
                                  bayesian-optimization && \
    pip install scikit-learn==0.22.1
                                  
# Install chromium and chromedriver
RUN apt install -y chromium && apt-get install -y libnss3 && apt-get install unzip

RUN wget -q "https://chromedriver.storage.googleapis.com/79.0.3945.36/chromedriver_linux64.zip" -O /tmp/chromedriver.zip \
    && unzip /tmp/chromedriver.zip -d /usr/bin/ \
    && rm /tmp/chromedriver.zip && chown root:root /usr/bin/chromedriver && chmod +x /usr/bin/chromedriver

# R packages (docopt, testthat, knitr are already somehow included in tidyverse)
RUN Rscript -e "install.packages(c('caret', 'ggridges', 'ggcorrplot'))"