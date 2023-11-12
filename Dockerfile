
FROM python:3.9

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pandas
RUN pip install tables
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install tabulate

WORKDIR /app

RUN git clone https://github.com/openfisca/openfisca-core.git
RUN git clone https://github.com/openfisca/openfisca-france.git

# better practice to add the specific tag of the repo ?
# RUN git clone --branch <tag_or_branch> https://github.com/example/repo.git


WORKDIR /app/openfisca-core 
RUN make install-deps install-edit

WORKDIR /app/openfisca-france
RUN make install

WORKDIR /app/codes

CMD ["/bin/bash"]




