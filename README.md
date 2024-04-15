[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KULL-Centre/_2023_Pesce_IDPdesign/blob/main/IDPDesigner.ipynb)

# Design of intrinsically disordered protein variants with diverse structural properties
Supporting data and code for:

*Pesce, F., Bremer, A., Tesei, G., Hopkins, J. B., Grace, C. R., Mittag, T., & Lindorff-Larsen, K. (2023). Design of intrinsically disordered protein variants with diverse structural properties. bioRxiv.*

Molecular dynamics simulations are available on [ERDA](https://erda.ku.dk/archives/2bef5e8ad566d5204dd34ec6a316896b/published-archive.html).
Data also available on [ZENODO](https://doi.org/10.5281/zenodo.10972882).

## List of content:
- **CODE**: Python code to reproduce simulations (single- and multi-chain) and run the design algorithm (using Rg or contact maps as structural targets).
- **EVOLUTION_DATA**: Data produced by running the design algorithm.
- **NMR**: Raw PFG-NMR measurements of A1-LCD and V1 at 34°C.
- **SAXS**: Solvent subtracted and averaged SEC-SAXS measurements of A1-LCD and V2–V5.
- **120centroids_data.pkl**: Sequences, Rg and sequence features of 120 representative A1-LCD swap variants generated by designing more compact variants.
- **Figures.ipynb**: Jupyter notebook to reproduce the figures in the main text of the paper.
- **IDPDesigner.ipynb**: Google Colab notebook to design IDP sequences with a specific scaling exponent.
- **V1-5_WT_csat.xlsx**: Measured saturation concentration values for A1-LCD and V1–V5.
- **exp_sequences.fasta**: Sequences of A1-LCD and V1–V5 experimentally characterized.
- **no_expression_sequences.fasta**: Sequences of swap variants of A1-LCD that did not express in *E. coli*.
