# fake-manuscript-demo

**A controlled demonstration of how easily a convincing fake scientific manuscript can be assembled using AI tools. Built and shared for educational purposes to highlight gaps in current scientific publishing integrity systems.**

---

## What this is

This repository contains the complete pipeline used to generate a fake but realistic-looking scientific manuscript, shared as a public demonstration of a real and growing problem in scientific publishing.

Every component of this manuscript was fabricated:

- **H&E histology images**: generated using ChatGPT image generation with carefully crafted prompts
- **Nuclear segmentation and statistics**: a real Python pipeline (StarDist) run on the fake images, producing real p-values from fake data
- **Patient cohort**: 47 synthetic TNBC patients with realistic demographics, survival times, and biomarker values
- **Survival analysis**: real Kaplan-Meier curves and Cox regression on the synthetic cohort
- **Manuscript**: full journal-style paper with title, abstract, introduction, methods, results, discussion, figure legends, ethics statement, and 14 references (12 of which are real papers, correctly cited)
- **Tissue source**: the methods section names real biorepositories (NCI BCLR, CHTN) and real equipment (Hamamatsu NanoZoomer S360). None of it is real. None of it is checked at submission.

Total time to build: approximately 4 hours.

---

## Why this was built

This is not a how-to guide. It is a demonstration of a gap.

Current scientific publishing integrity systems are designed to catch:
- Plagiarism (text similarity detection)
- Duplicate or manipulated images (tools like ImageTwin, Proofig)
- Statistical anomalies in reported data

They are not designed to catch:
- AI-generated images that have never existed before
- Synthetic patient cohorts
- Manuscripts where every individual component passes scrutiny but the underlying data is fabricated

The goal of this demonstration is to make that gap concrete and visible, so that the scientific community, journal editors, and publishers can have an informed conversation about what safeguards are needed.

---

## Repository contents

| File | Description |
|---|---|
| `nuclear_morphology_analysis.py` | StarDist-based nuclear segmentation pipeline. Segments nuclei in H&E images, computes aspect ratio and shape metrics, generates violin plots and Mann-Whitney statistics. |
| `synthetic_cohort.py` | Generates the synthetic n=47 TNBC patient cohort. Produces Kaplan-Meier survival curves, Cox proportional hazards regression, forest plot, and Table 1. Requires the `lifelines` package. |
| `manuscript.pdf` | The complete fake manuscript in journal submission format. Includes all figures as placeholders with captions. Do not cite this. Do not submit this. |

---

## How to run the code

### Requirements

```bash
pip install numpy pandas matplotlib scikit-image stardist csbdeep scipy lifelines
```

For SSL certificate issues in corporate environments:

```bash
pip install lifelines --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

### Nuclear morphology analysis

1. Place your H&E images in a `figures/` folder
2. Edit the `IMAGE_CONFIGS` dictionary at the top of `nuclear_morphology_analysis.py` to point to your image paths
3. Run:

```bash
python nuclear_morphology_analysis.py
```

Results are saved to `nuclear_morphology_results/`

### Synthetic cohort

```bash
python synthetic_cohort.py
```

Results are saved to `synthetic_cohort_results/`

---

## What a reviewer would have needed to catch this

| Review stage | Would it be caught? | Why |
|---|---|---|
| Journal submission portal | No | No automated AI image detection at submission |
| Handling editor | Unlikely | Hypothesis is plausible, writing is fluent, statistics appear rigorous |
| Peer review | Possibly | Requires expert recognition of subtle AI image artifacts |
| Post-publication | Possibly | If someone attempted to replicate the patient cohort |
| Statistical audit | No | The statistics are mathematically correct; the data they describe is fake |

---

## What this is not

This repository is not:

- A template for committing scientific fraud
- A claim that this specific manuscript was submitted anywhere
- A demonstration that all AI-assisted science is fraudulent

The manuscript was never submitted to any journal. The fabricated tissue sources (NCI BCLR, CHTN) were not contacted. No actual patient data was used or misrepresented.

---

## The question worth asking

The question is not whether someone will use AI to fabricate a scientific manuscript.

The question is whether we will know when they do.

Possible directions for the field to consider:

- Mandatory cryptographic provenance metadata embedded in research images at acquisition
- AI-generation detection tools integrated into journal submission portals
- Mandatory raw data deposition for image-based studies
- Registered reports that pre-commit to methodology before data collection

---

## Author

**Sreenivas Bhattiprolu (DigitalSreeni)**
DigitalSreeni LLC., Antioch, CA, USA

YouTube: [DigitalSreeni](https://www.youtube.com/c/DigitalSreeni)

This demonstration was built and shared in the spirit of open science and responsible AI literacy. If you found it useful or alarming, both reactions are appropriate.

---

## License

The code in this repository is released under the MIT License. The manuscript PDF is shared for educational reference only and must not be submitted to any journal or preprint server.

---

## Citation

If you reference this demonstration in your own work:

```
Bhattiprolu, S. (2025). fake-manuscript-demo: A controlled demonstration of AI-generated 
scientific manuscript fabrication. GitHub. [https://github.com/DigitalSreeni/fake-manuscript-demo](https://github.com/bnsreenu/fake-manuscript-demo/blob/main/bhattiprolu_manuscript.pdf)
```
