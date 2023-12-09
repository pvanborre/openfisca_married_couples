
FROM python:3.9

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pandas
RUN pip install tables
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install tabulate
RUN pip install statsmodels

WORKDIR /app

RUN git clone -n https://github.com/openfisca/openfisca-core.git && \
    cd openfisca-core && \
    git checkout 9b160f96b7eaae7f796ba7bf2fef5434830acf42

WORKDIR /app

RUN git clone -n https://github.com/openfisca/openfisca-france.git && \
    cd openfisca-france && \
    git checkout 5e900ca5bad464f389ce8718bb8fd4bd18f1d6c8


# for reproductibility best practice is to specify a commit ID 
# I put commit IDs of December, 8th 2023


WORKDIR /app/openfisca-core 
RUN make install-deps install-edit

WORKDIR /app/openfisca-france
RUN make install

WORKDIR /app/codes

CMD ["/bin/bash"]




