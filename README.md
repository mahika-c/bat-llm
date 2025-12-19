# LLM for Egyptian Fruit Bat Dataset

### Mahika Calyanakoti, Xiaoshen Ma, Shruti Agarwal

## ABSTRACT:

We propose an “audio-language” modeling pipeline for Egyptian fruit bat social vocalizations that

* (1) encodes raw bat audio into discrete units (“bio-tokens”)

* (2) learns mappings between acoustic sequences and rich social labels (emitter, addressee, context, actions)

* (3) generates textual summaries or predicts the next vocalization conditioned on interaction context.

Using ESP’s public bioacoustics resources and bat datasets, we will compare self-supervised audio encoders plus discretizers (k-means/VQ-VAE) against mel-spectrogram transformers for tokenization and modeling. We will evaluate on multi-task classification, sequence modeling (next-call prediction), and audio-to-text captioning of calls. The deliverable is a Python library + report with ablations on tokenization granularity, encoder choice, and context conditioning. (See ESP’s NatureLM-audio and library for background and tooling.)

## BACKGROUND:

Egyptian fruit bats are highly social, interact in the dark, and rely on rich vocal communication, including demonstrated vocal learning and context-specific calls. Annotated corpora include identities, addressees, behavioral context, and actions before/after calls. These datasets have been curated and shared by Yovel et al. and widely used in bioacoustics research. Recent “audio-language” models (ex. ESP’s NatureLM-audio) suggest that general-purpose LLM-style prompting can unify species ID, captioning, and behavior inference, motivating a bat-specific pipeline that learns discrete acoustic units and aligns them with textual labels.

## Repository overview

This repo contains an end‑to‑end pipeline for **audio‑language modeling on Egyptian fruit bat vocalizations**

- **Data & preprocessing**
  - `99a_Data_Preprocessing.ipynb`, `99b_Data_Preprocessing.ipynb`: original, full‑dataset preprocessing scripts (documentary only, don't need to run them for the 10k workflows).
  - `01_Download_Dataset.ipynb`: downloads and unzips the public Egyptian fruit bat 10k dataset into `starter_code/data/` (audio + `annotations.csv`).
   - `02_Create_PyTorch_DataLoaders.ipynb`: constructs PyTorch datasets and data loaders from raw audio and emitter labels for supervised experiments.
  - `03_Check_Annotations_Subset.ipynb`: sanity‑checks that `data/annotations.csv` is a consistent subset of `annotations_filenames.csv`.
  - `04_EDA_and_Baseline_Features.ipynb`: exploratory data analysis plus creation of derived audio features (downsampled audio and mel‑spectrograms under `derived/`).

- **Tokenization / representation learning**
  - `04_EDA_and_Baseline_Features.ipynb`: generates mel‑spectrogram features that downstream tokenizers (ex. VQ‑VAE) can consume.
  - `05_Tokenization_Strategies.ipynb`: builds three types of representations:
    - wav2vec2 / HuBERT frame embeddings + k‑means to produce discrete “bio‑tokens” (saved in `derived/tokens/k_means`),
    - a VQ‑VAE over mel‑spectrograms to learn a discrete codebook (tokens saved in `derived/tokens/vqvae`),
    - continuous AST embeddings pooled per call (saved in `derived/ast_features`).
    - Does energy-based chunking / spectral segmentation too for discrete tokens.

- **Supervised Classification**
  - `06_Baseline_Classification.ipynb`: trains simple logistic‑regression baselines on top of AST (Audio Spectrogram Transformer) embeddings for emitter and context classification.
  - `07_Improved_Classification.ipynb`: improves on the 06 baselines by combining AST embeddings with discrete token histograms, and using `GridSearchCV` to tune logistic‑regression hyperparameters separately for emitter and context tasks.

- **Sequence modeling / language modeling**
  - `08_Next_Token_Language_Model.ipynb`: trains a small Transformer **next‑token language model** over the discrete acoustic tokens (wav2vec2+k‑means or VQ‑VAE codes), reporting validation cross‑entropy, perplexity, and token‑level accuracy, and providing a simple sampling routine for qualitative evaluation. 
  - `09_Decoder_Next_Token_LM.ipynb`: extends the next-token LM with **decoding and qualitative evaluation**, including experiments with k-means (baseline and RLE) and VQ-VAE tokens. This notebook implements audio reconstruction for VQ-VAE-based models and qualitative diagnostics (e.g., RMS energy plots and repetition-controlled sampling) to assess generated outputs.
## Generated audio samples (qualitative)

We include two example audio samples generated from the trained next-token language model using **VQ-VAE tokens**, decoded back into waveform audio. These samples are intended for **qualitative inspection only**, illustrating that the model produces non-degenerate audio with plausible temporal structure.

- `single_VQ_VAE_audio_sample.wav`
- `stitched_VQ_VAE_audio.wav`

Note: k-means–based models (including RLE) do not admit direct waveform decoding; audio generation is therefore only supported for VQ-VAE–based models.


## How to run the pipeline (10k subset)

1. **Download data**: run `starter_code/01_Download_Dataset.ipynb` to populate `data/annotations.csv` and `data/audio/`.
2. **Inspect / EDA**: run `04_EDA_and_Baseline_Features.ipynb` to explore the dataset and create mel features in `derived/`.
3. **Tokenization / embeddings**: run `05_Tokenization_Strategies.ipynb` to generate:
   - AST embeddings in `derived/ast_features/`,
   - discrete tokens in `derived/tokens/k_means/` and `derived/tokens/vqvae/`.
4. **Classification baselines**:
   - `06_Baseline_Classification.ipynb` for simple AST‑based logistic regression,
   - `07_Improved_Classification.ipynb` for grid‑searched, feature‑rich classifiers.
5. **Next‑token modeling**: run `08_Next_Token_Language_Model.ipynb` to train and evaluate a Transformer LM on token sequences and sample continuations.

All notebooks are designed to be run from within `starter_code/` with the same working directory, and they write all derived artifacts under `starter_code/derived/` so the original `data/` folder remains unchanged. Need to run the files yourself since they are too big to be stored in the github repo (see gitignore file).
