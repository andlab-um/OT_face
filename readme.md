# Oxytocin and Self-other distinction <img src="https://raw.githubusercontent.com/andlab-um/OT_face/main/demo.png" align="right" width="561px">

[![GitHub repo size](https://img.shields.io/github/languages/code-size/andlab-um/OT_face?color=brightgreen&label=repo%20size&logo=github)](https://github.com/andlab-um/OT_face)
[![DOI](https://img.shields.io/badge/DOI-10.1093%2Fcercor%2Fbhac167-blue)](https://doi.org/10.1093/cercor/bhac167)<br />
[![Twitter URL](https://img.shields.io/twitter/url?label=%40ANDlab3&style=social&url=https%3A%2F%2Ftwitter.com%2Flizhn7)](https://twitter.com/ANDlab3)

**Code and data for: <br />**
**Wang, Y., Wang, R., & Wu, H. (2022). The role of oxytocin in modulating self–other distinction in human brain: A pharmacological fMRI study.** *Cerebral Cortex*. <br />
[DOI: 10.1093/cercor/bhac167](https://doi.org/10.1093/cercor/bhac167).
___

## Highlights


## Data


- The behavior data for the experiment are in `beha_data.csv`
- Psychometric data for the experiment are in `psych_data.csv`
- All fMRI data with descriptions of the variables is put in openfmri

## Models

MVPA modelling and analysis (including figure 2-3) is under `./models`

## Structure

**This repository contains:**
```
root
 ├── data               # Preprocessed regression data & psychometric data & morphometry data & rs-FC data
 │    ├── dra 
 │    ├── imq 
 │    ├── mms
 │    └── rsfc
 ├── bs_log             # Bootstrapping reuslts
 │    ├── IMQ-MMS
 │    ├── IMQ-rsFC
 │    └── rsFC-MMS
 ├── util.py            # To provide various basic function
 ├── bootstrap.py       # To conduct subject-wise bootstrapping 
 ├── eval.py            # To lay the foundation for later analyses
 ├── evaluation.ipynb   # To test Hypothesis 1 and Hypothesis 2
 ├── dra_data_pre.ipynb # To pave the way for dyadic regression analysis (dra)
 ├── dra.ipynb          # To test Hypothesis 3 and plot figure 7
 ├── pipeline.ipynb     # To analyse the effects of different pipelines and plot figure 8
 ├── figure_2.ipynb     # To plot figure 2
 ├── figure_3.ipynb     # To plot figure 3
 ├── figure_9.ipynb     # To plot figure 9
 ├── LICENSE
 └── README.md
```
**Note**: File directories are marked by "\<\>". Before running the codes, change the directories to the path of corresponding locations. <br />

___

## Citation

    @article{wang2022role,
      title={The role of oxytocin in modulating self--other distinction in human brain: a pharmacological fMRI study},
      author={Wang, Yuanchen and Wang, Ruien and Wu, Haiyan},
      journal={Cerebral Cortex},
      year={2022}
    }
    
___
