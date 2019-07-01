# traces-seconds

 Methods to synthetize traces in seconds for cloud computing research.

This repository contains the software to generate workload traces in seconds from traces
in hours. It is aimed to obtaining traces to compare allocation strategies for cloud
computing.

The input and output traces can be obtained from <https://zenodo.org/record/3264869>.
They have to be copied to the directory [traces](traces) in order for the notebooks in
this repository to work. You can use these commands:

    wget https://zenodo.org/record/3264869/files/wc98_brown.csv.gz?download=1
    wget https://zenodo.org/record/3264869/files/wc98_constant.csv.gz?download=1
    wget https://zenodo.org/record/3264869/files/wc98_pink.csv.gz?download=1
    wget https://zenodo.org/record/3264869/files/wc98_real.csv.gz?download=1
    wget https://zenodo.org/record/3264869/files/wc98_smooth.csv.gz?download=1
    wget https://zenodo.org/record/3264869/files/wc98_uniform.csv.gz?download=1
    wget https://zenodo.org/record/3264869/files/wc98_white.csv.gz?download=1
    wget https://zenodo.org/record/3264869/files/wiki_brown.csv.gz?download=1
    wget https://zenodo.org/record/3264869/files/wiki_constant.csv.gz?download=1
    wget https://zenodo.org/record/3264869/files/wiki_pink.csv.gz?download=1
    wget https://zenodo.org/record/3264869/files/wiki_smooth.csv.gz?download=1
    wget https://zenodo.org/record/3264869/files/wiki_uniform.csv.gz?download=1
    wget https://zenodo.org/record/3264869/files/wiki_white.csv.gz?download=1
    wget https://zenodo.org/record/3264869/files/wikipedia_2014_hours.csv.gz?download=1

This are the contents of the repository:

* [1_Trace_Generation](1_Trace_Generation.ipynb): is a jupyter notebook with the trace
synthetis code. It uses as real [traces](traces) the files in the trace directory for the
[World Cup 98](https://zenodo.org/record/3264869/files/wc98_real.csv.gz?download=1) and the
[Wikipedia in 2014](https://zenodo.org/record/3264869/files/wikipedia_2014_hours.csv.gz?download=1).

* [2_Evaluation](2_Evaluation.ipynb): is a jupyter notebook that applies
[Malloovia](https://github.com/asi-uniovi/malloovia) to the previous traces and analyzes
them.

These notebooks generate figures in the directory [figs](figs).

The explanation of the software and the traces can be obtained in the paper _Influence
of the trace resolution and length in the cost optimization process in cloud computing_,
published in the 2019 International Symposium on Performance Evaluation of Computer and
Telecommunication Systems (SPECTS 2019).
