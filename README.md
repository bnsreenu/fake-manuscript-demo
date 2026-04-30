# fake-manuscript-demo

**A controlled demonstration of how easily convincing fake scientific manuscripts can be assembled using AI tools, across two completely different scientific domains. Built and shared for educational purposes to highlight gaps in current scientific publishing integrity systems.**

---

## What this is

This repository contains the complete pipelines used to generate two fake but realistic-looking scientific manuscripts, shared as a public demonstration of a real and growing problem in scientific publishing.

**This is not a how-to guide. It is a demonstration of a gap.**

Both manuscripts were built in a total of approximately 10 hours. Every component was fabricated.

---

## Two demonstrations. Two domains. Same problem.

### Demonstration 1: Digital Pathology (`pathology/`)

**Fake manuscript title:** *Hypoxia-associated elongation of tumor-associated macrophage nuclei predicts poor overall survival in triple-negative breast cancer*

**What was fabricated:**

| Component | How it was faked |
|---|---|
| H&E histology images | Generated with ChatGPT image generation using carefully crafted prompts |
| Nuclear segmentation | Real StarDist pipeline run on fake images, producing real p-values |
| Patient cohort | 47 synthetic TNBC patients with realistic demographics, HIF-1α scores, survival times |
| Survival analysis | Real Kaplan-Meier curves and Cox regression on synthetic data |
| Tissue source | Methods name real biorepositories (NCI BCLR, CHTN) and real equipment (Hamamatsu NanoZoomer S360). None of it is real. None of it is checked at submission. |
| References | 14 references, 12 of which are real papers correctly cited |

**Key result:** Mann-Whitney p = 1.1×10⁻¹², Cohen's d = 0.72. Statistically rigorous. Biologically fabricated.

---

### Demonstration 2: Battery Materials (`battery_materials/`)

**Fake manuscript title:** *Intragranular crack propagation in high-nickel NMC811 cathode particles quantified by X-ray microscopy correlates with rate-dependent electrochemical degradation*

**What was fabricated:**

| Component | How it was faked |
|---|---|
| XRM 3D volume renderings | Generated with ChatGPT image generation using prompts specifying instrument, contrast, and phase segmentation |
| Particle morphology data | Synthetic dataset of 67 particles (32 low-cycle, 35 high-cycle) with realistic crack density, porosity, delamination distributions anchored to published XRM literature |
| Electrochemical data | Capacity fade curves from a physically calibrated degradation model; EIS spectra from a Randles circuit model; dQ/dV curves with correct NMC811 phase transition peaks at experimentally verified voltages |
| Instrument and methods | Names real instruments (ZEISS Xradia Versa 620, Neware BTS4000, BioLogic SP-200, Dragonfly 2022.2) and real reagents (Umicore NMC811, Timcal Super C65, Solvay PVDF 5130). None used. None checked. |
| References | 15 references, all real papers correctly cited |

**Key result:** Crack density vs capacity retention Pearson r = −0.941, p < 0.0001 across 67 particles. Physically plausible. Entirely synthetic.

---

## Repository structure

```
fake-manuscript-demo/
│
├── pathology/
│   ├── nuclear_morphology_analysis.py   # StarDist segmentation + shape analysis pipeline
│   ├── synthetic_cohort.py              # Synthetic patient cohort + KM + Cox regression
│   └── bhattiprolu_manuscript.pdf       # The fake pathology manuscript (do not submit)
│
├── battery_materials/
│   ├── xrm_particle_analysis.py         # Synthetic XRM particle morphology + statistics
│   ├── electrochemical_data.py          # Capacity fade, EIS, dQ/dV generation
│   └── battery_manuscript.pdf           # The fake battery manuscript (do not submit)
│
└── README.md
```

---

## How to run the code

### Requirements: pathology

```bash
pip install numpy pandas matplotlib scikit-image stardist csbdeep scipy lifelines
```

For SSL certificate issues in corporate environments:

```bash
pip install lifelines --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

**Nuclear morphology analysis:**

1. Place your H&E images in a `figures/` folder inside `pathology/`
2. Edit `IMAGE_CONFIGS` at the top of `nuclear_morphology_analysis.py` to point to your image paths
3. Run:

```bash
python pathology/nuclear_morphology_analysis.py
```

Results saved to `nuclear_morphology_results/`

**Synthetic patient cohort:**

```bash
python pathology/synthetic_cohort.py
```

Results saved to `synthetic_cohort_results/`

---

### Requirements: battery materials

```bash
pip install numpy pandas matplotlib scipy
```

**XRM particle morphology:**

```bash
python battery_materials/xrm_particle_analysis.py
```

Results saved to `xrm_results/`

**Electrochemical data:**

```bash
python battery_materials/electrochemical_data.py
```

Results saved to `electrochemical_results/`

---

## What a reviewer would have needed to catch this

This table applies to both manuscripts.

| Review stage | Would it be caught? | Why |
|---|---|---|
| Journal submission portal | No | No automated AI image detection at submission |
| Handling editor | Unlikely | Hypothesis is plausible, writing is fluent, statistics appear rigorous |
| Peer review | Possibly | Requires domain expert to recognize subtle AI image artifacts under close inspection |
| Post-publication | Possibly | If someone attempted to replicate the data or contact the named biorepositories or instrument facilities |
| Statistical audit | No | The statistics are mathematically correct on synthetic inputs |
| Reference check | Mostly no | References are real papers. Two minor citation content errors were deliberately left in the pathology manuscript as an exercise for the reader. |

The battery manuscript is arguably harder to catch than the pathology manuscript. The electrochemical data is generated from physically correct mathematical models. The dQ/dV curves show the H2-H3 phase transition peak suppression at the experimentally correct voltage, with the correct rate of diminishment with cycling. A battery expert would recognize this as the canonical NMC811 degradation signature and would have no reason to question it.

---

## What this is not

This repository is not:

- A template for committing scientific fraud
- A claim that either manuscript was submitted anywhere
- A demonstration that all AI-assisted science is fraudulent

Neither manuscript was submitted to any journal or preprint server. The named biorepositories, instrument vendors, and reagent suppliers were not contacted. No actual patient data, cell data, or XRM tomography data was used or misrepresented.

---

## The question worth asking

The question is not whether someone will use AI to fabricate a scientific manuscript.

The question is whether we will know when they do.

Possible directions for the field to consider:

- Mandatory cryptographic provenance metadata embedded in research images at the point of acquisition
- AI image generation detection integrated into journal submission portals
- Mandatory deposition of raw data for image-based and materials characterization studies
- Registered reports that pre-commit to methodology before data collection
- Cross-referencing of named biorepositories, instrument facilities, and reagent lot numbers at submission

---

## Author

**Sreenivas Bhattiprolu (DigitalSreeni)**
DigitalSreeni LLC., Antioch, CA, USA

YouTube: [DigitalSreeni](https://www.youtube.com/@DigitalSreeni)

This demonstration was built and shared in the spirit of open science and responsible AI literacy. If you found it useful or alarming, both reactions are appropriate.

---

## License

The code in this repository is released under the MIT License. The manuscript PDFs are shared for educational reference only and must not be submitted to any journal or preprint server.

---

## Citation

If you reference this demonstration in your own work:

```
Bhattiprolu, S. (2025). fake-manuscript-demo: A controlled demonstration of AI-generated 
scientific manuscript fabrication across multiple domains. GitHub. 
https://github.com/bnsreenu/fake-manuscript-demo
```
