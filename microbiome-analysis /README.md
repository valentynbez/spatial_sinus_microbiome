In this folder there are 3 analysis files.  
1. `sinus-qiime2-preprocessing` - preprocessing of sequencing data
2. `downstream-analysis` - the analysis after retrieving data from `QIIME2`  
    It relies on `QIIME2` generated artefacts:
    - `alpha diverity`: calculated alpha-diversity (Faith PD) measure for samples
    - `taxonomy`: collapsed taxonomy tables 
    - `distance matrices`: matrices of Bray-Curtis, weighted and unweighted UniFrac distances between samples
    - `metadata`: or, so-called `mapping file`
3. `taxa-barplots.ipynb` - code for Figure 5 and Figure 6  
Relies on taxonomy collapsed to a *Class* level - `data/taxonomy/level-7.csv`
