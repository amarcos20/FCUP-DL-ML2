# Literature: murmur detection on CirCor DigiScope

Reference papers for the dataset, the PhysioNet Challenge 2022 and published architectures to compare with our models.
WA = weighted accuracy (Challenge metric). "Hidden test" = official Challenge test set (not public), so these numbers are
**not directly comparable** with ours. Fair comparison = re-implementing the architecture in our pipeline and split.

## Dataset and Challenge

| file | reference | notes |
|---|---|---|
| `00_Oliveira2022_CirCor_dataset.pdf` | Oliveira et al., *The CirCor DigiScope Dataset: From Murmur Detection to Murmur Classification*, IEEE JBHI 26(6), 2022. doi:10.1109/JBHI.2021.3137048 | dataset paper (cite in the report) |
| `01_Reyna2023_PhysioNet2022_overview_PLOS.pdf` | Reyna et al., *Heart murmur detection from phonocardiogram recordings: The George B. Moody PhysioNet Challenge 2022*, PLOS Digit Health 2(9), 2023. doi:10.1371/journal.pdig.0000324 | Challenge overview, metric definition |

## Architectures (PhysioNet Challenge 2022 entries, hidden-test WA from the official table)

| file | reference | architecture | input | WA |
|---|---|---|---|---|
| `02_Lu2022_HearHeart_lightweight_CNN.pdf` | Lu et al. (HearHeart), CinC 2022 | lightweight 2D CNN, no segmentation, weighted loss, CV ensemble | mel spectrogram (128 bands, 25–2000 Hz) + SpecAugment | **0.780** (1st) |
| `03_McDonald2022_CUED_parallel_HSMM_CinC.pdf`, `03b_McDonald2024_CUED_parallel_HSMM_PLOS.pdf` | McDonald, Gales, Agarwal (CUED_Acoustics), CinC 2022; PLOS Digit Health 2024 (doi:10.1371/journal.pdig.0000436) | RNN frame-wise heart-state predictions → two parallel HSMMs (normal vs murmur) | log-spectrogram; uses segmentation labels for training | 0.776 (2nd) |
| `04_Walker2023_Dual_Bayesian_ResNet.pdf` | Walker, Krones et al. (PathToMyHeart), CinC 2022 / arXiv:2305.16691 | two MC-dropout Bayesian ResNet50, binary tasks Present-vs-rest and Unknown-vs-rest | log-mel, overlapping windows | 0.771 (4th) |
| `05_Lee2022_CAU_UMN_LCNN.pdf` | Lee et al. (CAU_UMN), CinC 2022 | Light CNN (LCNN) + peak-interval embedding | log-mel / STFT / CQT | 0.767 (5th) |
| `06_Nejedly2022_ISIBrno_CNN_GRU_expert_features.pdf` | Nejedly et al. (ISIBrno-AIMT), CinC 2022 | CNN + GRU (CRNN) with segmentation and murmur heads | STFT | 0.755 (9th) |
| `07_SmartBeatIT2022_BiLSTM.pdf` | Monteiro, Fred, Plácido da Silva (SmartBeatIT), CinC 2022 | bidirectional LSTM | temporal features | 0.757 |
| `12_Xia2022_MetaHeart_ResNet_focal_loss.pdf` | Xia et al. (MetaHeart), CinC 2022 | residual network + class-distinguished focal loss | spectrogram | 0.72 (18th) |

## Other architectures (public data, own splits)

| file | reference | architecture | WA |
|---|---|---|---|
| `08_Elola2023_murmur_grading_ResNet_attention.pdf` | Elola et al., IEEE JBHI 27(8), 2023. doi:10.1109/JBHI.2023.3275039 | ensemble of 15 ResNets with channel-wise attention (murmur **grading**: absent/soft/loud) | — (different task) |
| `09_Panah2023_wav2vec2_murmur.pdf` | Panah & Hines, EUSIPCO 2023 | wav2vec 2.0 fine-tuned on raw audio | 0.80 |
| `10_Niizumi2024_pretrained_audio_representations.pdf` | Niizumi et al., arXiv:2404.17107, 2024 | pre-trained audio models: CNN14 (PANNs), BYOL-A, AST, M2D | M2D 0.832, CNN14 0.582 |
| `11_Patwa2023_wavelet_scattering_1DCNN.pdf` | Patwa, Rahman, Al-Naffouri, arXiv:2303.11423 | wavelet scattering + 1D CNN (vs LSTM, C-RNN) | 0.837 (own CV) |

## Links
- Official results: https://physionetchallenges.org/2022/results/
- All Challenge papers: https://physionet.org/files/challenge-2022/1.0.0/papers/index.html
- All Challenge source code: https://physionet.org/files/challenge-2022/1.0.0/sources
- Dual Bayesian ResNet code: https://github.com/Benjamin-Walker/heart-murmur-detection
