# Human activity classification with FMCW radar

EE4775 Object Classification with Radar (TU Delft). Classify radar recordings of
six daily activities (walking, sitting down, standing up, picking up, drinking,
falling) from their micro-Doppler signatures, and measure how well the model
travels from young lab subjects to elderly people in care homes.

Group members: Akram Chakrouni, Adam El Haddouchi, Ilyaas Shousha

## Deliverables
- `notebooks/` - the runnable build (start at `notebooks/README.md`).
- `models/` - the trained models (`svm_ds1`, `cnn_ds1`, `svm_lab`).
- `figures/` - figures used in the 1-pager and presentation.

## Result in one line
On clean single-site data, interpretable physical features into an SVM match a
fine-tuned ResNet18 at ~0.96 subject-independent accuracy. Transferring from the
lab to elderly care sites drops accuracy to ~0.77, and the CNN's extra capacity
does not help; the gap is driven by population and radar geometry, not model
size.

## Layout
```
notebooks/     five notebooks (01 index, 02 DSP+cache, 03 SVM, 04 CNN, 05 transfer)
notebooks/radar_pipeline.py   the one verified implementation, imported by all
models/        saved trained models
figures/       figures for the 1-pager and presentation
```

## Reproduce
See `notebooks/README.md` for environment setup. Then run the five notebooks in
order with the `Python (radar)` kernel. Notebook 02 builds the spectrogram cache
once (a few minutes); the rest load it and run in seconds, except the CNN fits
which take under a minute each on the Apple-silicon GPU.
