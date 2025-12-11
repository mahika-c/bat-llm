# LLM for Egyptian Fruit Bat Dataset

### Mahika Calyanakoti, Xiaoshen Ma, Shruti Agarwal

## INSTRUCTIONS:

To download data:

Run first cell of 01_Download_Dataset.ipynb to get the smaller 10k dataset

## ABSTRACT:

We propose an “audio-language” modeling pipeline for Egyptian fruit bat social vocalizations that

* (1) encodes raw bat audio into discrete units (“bio-tokens”)

* (2) learns mappings between acoustic sequences and rich social labels (emitter, addressee, context, actions)

* (3) generates textual summaries or predicts the next vocalization conditioned on interaction context.

Using ESP’s public bioacoustics resources and bat datasets, we will compare self-supervised audio encoders plus discretizers (k-means/VQ-VAE) against mel-spectrogram transformers for tokenization and modeling. We will evaluate on multi-task classification, sequence modeling (next-call prediction), and audio-to-text captioning of calls. The deliverable is a Python library + report with ablations on tokenization granularity, encoder choice, and context conditioning. (See ESP’s NatureLM-audio and library for background and tooling.)

## BACKGROUND:

Egyptian fruit bats are highly social, interact in the dark, and rely on rich vocal communication, including demonstrated vocal learning and context-specific calls. Annotated corpora include identities, addressees, behavioral context, and actions before/after calls. These datasets have been curated and shared by Yovel et al. and widely used in bioacoustics research. Recent “audio-language” models (ex. ESP’s NatureLM-audio) suggest that general-purpose LLM-style prompting can unify species ID, captioning, and behavior inference, motivating a bat-specific pipeline that learns discrete acoustic units and aligns them with textual labels.
