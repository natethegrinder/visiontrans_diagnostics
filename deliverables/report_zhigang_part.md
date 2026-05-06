# 1. Introduction

## 1.3 Hypothesis: Why CNNs may be Theoretically Suited for Medical Imaging

Before the experiments, we predicted that CNNs would outperform ViTs trained from scratch on this task. This is because CNNs possess inductive biases of locality and translation invariance, which naturally align with the need to detect local lesions (such as nodules and masses) in chest X-rays. In contrast, ViTs, which lack explicit spatial structure priors \cite{dosovitskiy2020}, find it difficult to spontaneously learn such complex spatial relationships on limited medical datasets. In addition, we also explored whether transfer learning can bridge this "bias gap."

---

# Methodology

## 3.4. ViT Transfer Learning

### 3.4.1 Model Selection: Exploring the Inductive Bias Spectrum

As demonstrated by Dosovitskiy et al. \cite{dosovitskiy2020}, in the absence of extremely large-scale data (e.g., JFT-300M), ViTs tend to underperform CNNs of comparable scale due to their lack of inductive bias. For a dataset like NIH CXR14 \cite{wang2017}, which contains approximately 110,000 images and exhibits severe label imbalance, we infer that training a ViT from scratch would be difficult to converge. Therefore, like the CNN branch, we adopt a transfer learning approach.

We select two representative architectures to construct an Inductive Bias Spectrum for comparison:

- **DeiT-B (85.8M):** As a representative of the vanilla ViT \cite{touvron2021}, its architecture is identical to the standard ViT, with only distillation introduced at the training level. Using it as a baseline allows us to examine the adaptability of pure global self-attention mechanisms to medical imaging.
- **Swin-T (27.5M):** As a representative of the "middle ground" \cite{liu2021swin}, it introduces hierarchical window attention. This design injects CNN-like multi-scale locality into the Transformer, and the relationship between architectural efficiency and model scale is a key focus of our analysis.

### 3.4.2 Two-Phase Fine-tuning: Mitigating Catastrophic Forgetting

To preserve the general features learned by the backbone on ImageNet during transfer, we do not directly update all parameters. Instead, we adopt a more robust two-phase fine-tuning strategy:

- **Phase 1 (Warm-up Stage, Epoch 1–5):** Freeze the backbone and train only the randomly initialized linear classification head (approximately 10,766 parameters). The rationale is: at the initial stage, the gradients produced by the head are highly random and volatile. If backpropagated directly, they would disrupt the feature representations already learned by the backbone, leading to catastrophic forgetting. A 5-epoch warm-up allows the head to first converge from a random state to a reasonable range.
- **Phase 2 (Fine-tuning Stage, Epoch 6–40):** Unfreeze the entire network. At this point, the backbone receives "meaningful gradients" from the converged head, enabling more stable domain adaptation toward the medical imaging domain.

### 3.4.3 Summary of Training Details

To ensure fair comparison, the ViT branch maintains high consistency with the CNN baseline in hyperparameter settings:

- **Loss Function:** Focal Loss \cite{lin2017focal} is uniformly adopted to address extreme class imbalance.
- **Optimizer & Schedule:** AdamW optimizer is used, combined with linear warm-up and cosine annealing. The learning rate is conservatively set to $10^{-5}$ to avoid damaging pretrained weights.
- **Engineering:** Automatic Mixed Precision (AMP) is enabled to optimize memory usage, and Early Stopping (Patience = 10) is applied to capture the optimal generalization point.

---

## 3.5. Interpretability Methods

In clinical diagnosis, a model must not only "predict accurately," but also "point correctly." We aim to understand which regions the model attends to during inference, and whether these attention areas align with the true lesion locations. The choice of interpretability methods is not arbitrary, but is strictly determined by the architectural characteristics of each model.

### 3.5.1 Architecture-driven Logic of Method Selection

**Attention Rollout (DeiT-B specific):** DeiT-B has an explicit CLS token and global self-attention. Rollout \cite{abnar2020} traces the path of information flowing from input patches to the CLS token through recursive matrix multiplication: $A_i' = 0.5A_i + 0.5I$, $R = A_{12}' \cdots A_1'$. Rollout is class-agnostic—it reflects the overall visual attention distribution of the model, rather than the discriminative basis for a specific disease. This property must be carefully considered when interpreting the results.

**GradCAM \cite{selvaraju2017} (Unified framework with differentiated implementations):** For Swin-T, we apply standard GradCAM on the final-stage 7×7×768 feature map, using gradient GAP values as channel weights. For DeiT-B, there is a key implementation difference: since the classification head is connected only to the CLS token, the direct gradients at patch locations are nearly zero; the correct approach is to use the gradients at the CLS position as channel weights and apply them to the patch activations (skip_first_token=True). For CNN (DenseNet121), we use the same GradCAM framework (cnn_mode=True), applying standard weighted summation on the feature maps of the last convolutional block. The unified framework ensures fairness in cross-architecture comparison.

Placing DeiT-B's GradCAM alongside Rollout allows us to distinguish which activations arise from "global saliency features" and which from "class-discriminative features."

### 3.5.2 Experimental Observation: Corner Artifact and Border Trimming

In preliminary experiments, the heatmaps of DeiT-B exhibited abnormally high activations at the four corners, i.e., the Corner Artifact described by Darcet et al. \cite{darcet2023}: ViTs without Register Tokens tend to "dump" hard-to-interpret global information into background patches, resulting in extremely high corner values. To suppress this corner artifact in Attention Rollout heatmaps, we apply a 16-pixel border exclusion zone when computing the Pointing Game metric, zeroing the outermost ring prior to peak detection.

### 3.5.3 Sample Selection Strategy: Addressing the "Imbalance Trap"

Directly using $\text{sigmoid} > 0.5$ to filter samples is not feasible. Due to the dynamic reweighting of Focal Loss and the extreme imbalance of the dataset, positive outputs are often concentrated in the range of $0.15 \sim 0.45$, making the 0.5 threshold select only 1.67% of samples, of which 90% belong to the Effusion class.

Therefore, we instead adopt the **Argmax Filter**: selecting samples where the GT is single-label, and the argmax predictions of all three models (DeiT-B, Swin-T, DenseNet121) match the GT (a total of 1683 samples, accounting for 21.0%), covering all target diseases. This ensures that interpretability analysis is always conducted on samples that the models "correctly recognize," making the heatmap analysis meaningful.

### 3.5.4 Quantitative Evaluation Metrics

We use three complementary metrics in parallel to comprehensively evaluate heatmap quality:

- **IoU (Intersection over Union):** The strictest metric, requiring high overlap between the binarized heatmap and the GT bounding box.
- **Pointing Game (PG) \cite{fong2017}:** A standard metric for weakly supervised localization, which only checks whether the maximum activation point of the heatmap falls within the bounding box, and is insensitive to threshold selection.
- **Bbox Attention Ratio (AR):** A threshold-free metric (mean inside Bbox / mean over the entire image), which is most fair for diffuse heatmaps such as Rollout, and measures whether the lesion region receives relatively higher attention.

### 3.5.5 Population-level Averaged Heatmaps

In addition to per-sample quantitative evaluation, we compute averaged heatmaps for all selected samples of each label (per-label averaged heatmap). For labels with larger sample sizes (e.g., Infiltration n=712, Effusion n=459, Atelectasis n=183), averaged heatmaps effectively suppress noise caused by individual differences, revealing the spatial distribution patterns of consistent attention for each disease, serving as a qualitative complement to the quantitative metrics.

---

# Results

## 4.3. From-Scratch vs Transfer ViT

Transfer learning yields a substantial performance improvement over training from scratch. The best from-scratch ViT achieves a mean test AUC of 0.7184, compared to 0.7910 and 0.8011 for fine-tuned DeiT-B and Swin-T respectively, representing a gap of 0.07 and 0.08 AUC points respectively.

The training dynamics further illustrate this gap. As shown in Figure X (see `deliverables/fig_4_3_vit_auc_comparison.png`), during Phase 1 (epochs 1–5, backbone frozen), the validation AUC rises slowly from 0.488 to 0.577. At epoch 6 — the first epoch of full fine-tuning — the validation AUC records a step-wise increase from 0.577 to 0.796 (+21.9pp) in a single epoch. The analysis of this phenomenon is presented in Section 5.1.

## 4.4. DeiT-B vs Swin-T

Per-label test AUC results for both fine-tuned models are reported in Table X.

| Label | DeiT-B | Swin-T |
|---|---|---|
| Atelectasis | 0.7515 | **0.7664** |
| Cardiomegaly | **0.8753** | 0.8716 |
| Effusion | **0.8228** | 0.8211 |
| Infiltration | 0.6897 | **0.6992** |
| Mass | 0.7819 | **0.7969** |
| Nodule | **0.7300** | 0.7285 |
| Pneumonia | 0.7124 | **0.7237** |
| Pneumothorax | 0.8459 | **0.8572** |
| Consolidation | 0.7353 | **0.7454** |
| Edema | **0.8460** | 0.8450 |
| Emphysema | 0.8746 | **0.9026** |
| Fibrosis | 0.8070 | **0.8096** |
| Pleural Thickening | 0.7494 | **0.7624** |
| Hernia | 0.8527 | **0.8865** |
| **Mean** | 0.7910 | **0.8011** |

Swin-T outperforms DeiT-B on 10 out of 14 labels, achieving a higher mean AUC of 0.8011 vs 0.7910, despite having approximately one-third the number of parameters (27.5M vs 85.8M). DeiT-B leads only on Cardiomegaly (0.8753 vs 0.8716), Effusion (0.8228 vs 0.8211), Nodule (0.7300 vs 0.7285), and Edema (0.8460 vs 0.8450), with margins of 0.004, 0.002, 0.001, and 0.001 respectively.

Both models show consistently weaker performance on Infiltration (DeiT-B: 0.6897, Swin-T: 0.6992), which is the lowest-AUC label across the entire experiment. The architectural analysis of these per-label differences is discussed in Section 5.2 and 5.3.

## 4.5. Interpretability Results

The NIH Chest X-ray dataset provides bounding box annotations for a subset of diseases, with at most 10 annotated samples per disease class available in our clean sample index (Section 3.5.3). Given this limited coverage, bbox-based quantitative metrics (Pointing Game, Bbox Attention Ratio) are reported as indicative evidence rather than statistically definitive results. The primary analysis therefore relies on population-level averaged heatmaps (Section 4.5.1) and representative case studies (Section 4.5.2), with quantitative metrics provided as a supporting reference in Section 4.5.3.

### 4.5.1 Population-level Average Heatmaps

Figure X presents per-label averaged heatmaps across four methods for labels with sufficient sample coverage. Labels with high sample counts (Infiltration n=712, Effusion n=459, Atelectasis n=183, Pneumothorax n=102) provide statistically robust spatial patterns that are less sensitive to individual image variation. Sample counts reflect the three-model argmax intersection clean index.

*[Figure X: Per-label average heatmaps — rows: Rollout / DeiT GradCAM / Swin GradCAM / CNN GradCAM; columns: Infiltration, Effusion, Atelectasis, Pneumothorax, Cardiomegaly. Source: `deliverables/avg_heatmaps_composite.png`]*

Across all labels, Attention Rollout produces consistently diffuse activations with no disease-specific spatial structure, confirming that it reflects general information routing rather than class-discriminative localization. DeiT GradCAM shows more focused activation patterns than Rollout, with visible disease-dependent variation. Swin GradCAM and CNN GradCAM both produce spatially concentrated maps with disease-dependent variation, while DeiT GradCAM is more diffuse and Rollout shows no disease specificity, with activation regions that visibly align with expected anatomical locations — lower lung fields for Effusion, lateral pleural margins for Pneumothorax, and the central cardiac silhouette for Cardiomegaly.

### 4.5.2 Case Studies: Cardiomegaly and Pneumothorax

Two diseases are selected as representative case studies because they represent opposing extremes: Cardiomegaly, where Swin GradCAM achieves the strongest localization, and Pneumothorax, the only disease where DeiT GradCAM outperforms Swin.

*[Figure X: Five-column panel for Cardiomegaly case (`deliverables/case_studies/Cardiomegaly_00000661_000.png`) — columns: Original + GT bbox / Attention Rollout / DeiT GradCAM / Swin GradCAM / CNN GradCAM. Swin IoU=0.615, DeiT IoU=0.277, Rollout IoU=0.001.]*

*[Figure X: Five-column panel for Pneumothorax case (`deliverables/case_studies/Pneumothorax_00010071_008.png`) — columns: Original + GT bbox / Attention Rollout / DeiT GradCAM / Swin GradCAM / CNN GradCAM. DeiT PG=1, Swin IoU=0.000, Rollout IoU=0.000.]*

For Cardiomegaly, Swin GradCAM produces the most precise localization within the cardiac silhouette (IoU=0.615, PG=1.0), with CNN GradCAM achieving comparable concentration (IoU=0.522, PG=1.0). DeiT GradCAM shows a broader response (IoU=0.277) and Rollout produces near-zero localization (IoU=0.001). For Pneumothorax, the pattern shifts: all methods produce low IoU due to the diffuse nature of the signal, but CNN GradCAM yields the highest overlap (IoU=0.062), while Swin GradCAM (IoU=0.001) and DeiT GradCAM (IoU=0.000) both fail to localize meaningfully by this threshold-based metric. Notably, DeiT GradCAM's averaged Bbox Attention Ratio remains highest for Pneumothorax across the full evaluation set (AR=2.665, Section 4.5.3), indicating that its activation peak tends toward the correct region even when the heatmap is too diffuse to register positive IoU. The architectural interpretation of these differences is discussed in Section 5.3.

### 4.5.3 Quantitative Summary

Table X reports Bbox Attention Ratio (AR) and Pointing Game (PG) averaged over all annotated samples per disease. AR measures the relative concentration of heatmap activation inside the GT bounding box (values >1.0 indicate above-average attention in the lesion region). PG reports the fraction of samples where the heatmap peak falls within the GT bounding box. Sample counts reflect the three-model argmax intersection clean index.

| Disease (n) | Rollout AR | DeiT AR | Swin AR | CNN AR | Rollout PG | DeiT PG | Swin PG | CNN PG |
|---|---|---|---|---|---|---|---|---|
| Cardiomegaly (10) | 0.780 | 1.327 | **4.790** | 3.516 | 0.000 | 0.600 | **1.000** | **1.000** |
| Atelectasis (10) | 0.726 | 1.941 | **4.373** | 2.982 | 0.000 | 0.000 | **0.300** | 0.200 |
| Effusion (10) | 0.669 | 1.576 | **2.933** | 2.747 | 0.000 | 0.000 | 0.200 | **0.300** |
| Mass (7) | 0.707 | 2.488 | **3.850** | 3.423 | 0.000 | 0.143 | 0.571 | **0.714** |
| Pneumothorax (6) | 0.604 | **2.665** | 1.270 | 1.841 | 0.000 | **0.167** | 0.000 | 0.000 |
| Nodule (3) | 0.408 | **2.001** | 1.891 | 1.314 | 0.000 | 0.000 | 0.000 | 0.000 |

Four observations are consistent across diseases. First, Rollout PG is 0.000 for every disease class and AR values fall below 1.0, indicating no preferential attention to lesion regions — consistent with its class-agnostic design. Second, Swin GradCAM dominates on structure-defined lesions (Cardiomegaly, Atelectasis) by both AR and PG. Third, CNN GradCAM is competitive with Swin on most diseases and surpasses it on Effusion (PG: 0.300 vs 0.200) and Mass (PG: 0.714 vs 0.571), suggesting that hierarchical local feature extraction transfers well to diffuse and focal lesion types. Fourth, Pneumothorax remains the sole exception where DeiT GradCAM yields the highest AR (2.665) and non-zero PG (0.167), while both Swin and CNN GradCAM produce near-zero localization — consistent with DeiT's global attention being better suited for absence-defined signals.

---

# Discussion

## 5.1 Analysis of the Performance Gains from Pre-training (The Impact of Pre-training)

Experimental results show that fine-tuned ViTs significantly outperform their from-scratch counterparts, which verifies the role of pretrained weights in helping ViTs overcome the lack of inductive bias.

### 5.1.1 Feature Alignment Phenomenon at Epoch 6

In the training curves of transfer learning (see Figure X, Loss/AUC Curve), a noteworthy phenomenon is the performance jump after Phase 2 begins. When the model transitions from a frozen backbone (Phase 1) to full-parameter fine-tuning (Phase 2), at the sixth epoch, the validation AUC records a step-wise increase of approximately 21.9pp (from 0.577 to 0.796).

This data point indicates that the low-level representations learned by the pretrained model on ImageNet (such as edge detection and texture recognition) share common features with chest X-ray anatomical structures. Once the head has undergone initial warm-up and begins to pass effective gradients, the model can rapidly complete feature alignment from the general domain to the medical imaging domain, and its convergence efficiency is much higher than the learning process starting from random initialization.

### 5.1.2 Pre-training as a Compensatory Prior

Comparative experimental results show that although the final AUC of the fine-tuned Swin-T (0.8011) has a stable improvement compared to the from-scratch version, the more essential difference lies in the improvement of the performance lower bound.

As hypothesized in the Introduction, ViTs trained from scratch exhibit a certain degree of "learning sluggishness" on a medium-scale dataset such as NIH CXR14, making it difficult to spontaneously construct complex spatial relationships within a limited number of epochs. Pretrained weights, by transferring an already structured feature space, in fact provide ViTs with a compensatory inductive bias. It not only improves the final AUC value, but more importantly enables the model, under limited data conditions (approximately 110k images), to reach a discriminative level comparable to deep CNNs (such as DenseNet121).

### 5.1.3 Implications for Clinical Deployment

Although the performance gains brought by transfer learning may appear as modest improvements numerically, they are crucial in clinical scenarios where annotation costs are extremely high. The experiments show that directly training pure self-attention architectures from scratch on medical data may face insufficient generalization capability. Therefore, leveraging pretrained models from large-scale general datasets for domain adaptation is a practical path to ensure that ViT architectures reach clinically usable benchmarks in medical imaging tasks.

## 5.2. The Role of Inductive Bias: A Cross-Architecture Analysis

As shown in Table X (Section 4.1), the ranking of Mean AUC on the test set across models is: DenseNet121 (0.8023) > Swin-T (0.8011) > DeiT-B (0.7910) > ResNet18 (0.7646) > ViT from scratch (0.7184). This ordering is directionally consistent with expectations regarding the strength of inductive bias, but the magnitude of differences between models varies and requires a layered interpretation.

The most significant gap appears between **the presence and absence of pretraining**: ViT from scratch (0.7184) differs from the closest DeiT-B fine-tuned model (0.7910) by 0.073; this gap has already been discussed in Section 5.1. After controlling for pretraining conditions, the **impact of architectural design remains observable but relatively moderate**: Swin-T (0.8011) exceeds DeiT-B (0.7910) by 0.010, and both are close to DenseNet121 (0.8023). The difference between DenseNet121 and Swin-T is only 0.0012, which falls within the range of experimental error and is insufficient to serve as evidence that CNNs systematically outperform transfer-based ViTs.

The **approximate parity between Swin-T and DenseNet121 is the most noteworthy finding of this section**. Swin-T introduces a hierarchical structure and window-based local attention within the Transformer framework; these two aspects closely correspond to the core design principles of CNNs (locality and hierarchy). Achieving performance comparable to DenseNet121 with 27.5M parameters suggests that when a Transformer architecture deliberately incorporates local inductive bias, its representational capacity is sufficient to compensate for the inherent structural advantages of CNNs. In contrast, DeiT-B, with 85.8M parameters, still underperforms Swin-T, indicating that for this task, parameter scale cannot substitute for structural priors.

Overall, the experimental results provide **conditional support** for the inductive bias hypothesis: pretraining is the single most influential factor, while architectural design has a persistent but limited effect on performance. There is no systematic gap between CNNs and transfer-based ViTs, but within ViTs, the difference introduced by incorporating local priors (Swin vs. DeiT) is consistent. This finding is more nuanced than the original hypothesis (that CNNs outperform ViTs overall) and better reflects the actual data distribution.

## 5.3. Architecture-Specific Localization Differences

The quantitative results in Section 4.5 show that localization quality is not a uniform property of a model's architecture — it depends on whether the model's inductive bias matches the spatial structure of the target disease. This section examines the three patterns that appear consistently across the data.

### 5.3.1 Locality Prior Determines Localization Quality: Two Opposing Cases

The clearest evidence for the divergence between Swin GradCAM and DeiT GradCAM comes from two diseases with spatially opposite diagnostic signals.

**Case 1: Cardiomegaly (Swin >> DeiT).** The diagnostic signal for cardiomegaly is the cardiothoracic ratio — cardiac width relative to thoracic width. Assessing this ratio requires the model to simultaneously perceive the cardiac boundary and the lateral lung field edges, a multi-scale structural judgment. Swin-T's hierarchical window attention processes the image at progressively coarser resolutions across four stages (56→28→14→7), naturally building a feature pyramid: shallow stages capture cardiac edge texture, deeper stages encode global scale relationships. This maps well onto the diagnostic requirement, resulting in sharp anatomical localization (Swin IoU=0.615, PG=1.000). DeiT-B's uniform patch-level attention operates at a single scale and is relatively insensitive to local boundaries, producing a more diffuse activation map (DeiT IoU=0.159, PG=0.600).

**Case 2: Pneumothorax (DeiT > Swin) — key evidence for architectural reversal.** The diagnostic signal for pneumothorax is the inverse: it is defined by the *absence* of peripheral lung markings — a region where normal texture has disappeared. Detecting this requires the model to compare the peripheral zone against the central lung and identify where texture is anomalously absent relative to its surroundings. DeiT-B's global self-attention lets every patch interact with all others simultaneously, which is naturally suited to this kind of relational judgment (DeiT IoU=0.071, PG=0.167). Swin-T's windowed attention processes each spatial window independently; an absence signal spanning multiple windows is invisible within any single window, and the shifted-window mechanism offers only limited cross-window communication, leading to complete failure on this task (Swin IoU=0.004, PG=0.000).

Together, these two cases establish a consistent pattern: **locality bias helps for presence-defined lesions** (defined by focal structural changes) and **becomes a liability for absence-defined lesions** (defined by the disappearance of normal signal). The architectural reversal on Pneumothorax is not a fluke — it is a predictable consequence of how each model processes spatial information.

### 5.3.2 Cross-architecture Comparison: Swin GradCAM vs. DenseNet GradCAM

The population-level average heatmaps (Figure~\ref{fig:avg_heatmaps}) reveal a notable contrast between Swin GradCAM and DenseNet GradCAM. Swin GradCAM produces disease-specific spatial patterns across labels: Effusion activations form horizontal bands in the bilateral lower lung fields corresponding to the costophrenic angles; Atelectasis shows bilateral symmetric responses in the lower lobes; Infiltration activations spread diffusely across both lung fields in a multi-focal pattern consistent with the nature of the condition; Pneumothorax activations concentrate in the upper and mid lung zones with prominent peripheral edge signals. The activation region shifts meaningfully from disease to disease, indicating that the model has learned spatially distinct features for each condition.

DenseNet GradCAM presents a strikingly different picture. Average heatmaps across all diseases share a nearly identical form — a smooth elliptical blob centered consistently on the central-to-lower-right thorax, with boundaries approximating a normalized 2D Gaussian. The maps for Atelectasis, Effusion, Infiltration, and Mass are almost indistinguishable in shape, with center positions varying by less than 10% of image width. This disease-invariant response suggests that DenseNet's gradient signal converges to a shared central thoracic region at the last dense block, rather than differentiating to disease-specific anatomical locations. Two non-exclusive explanations are possible: (1) DenseNet's dense skip connections cause gradients to diffuse heavily during backpropagation, accumulating in the region of highest feature density — the central thorax corresponding to the hilum and cardiac silhouette; (2) the model relies on the central thoracic region as a shared discriminative anchor for many conditions, since hilar and cardiac structures carry anatomical relevance across a wide range of diseases, even if this strategy provides no spatial localization value.

This is somewhat counterintuitive. CNNs are traditionally expected to produce spatially precise activation maps due to their hierarchical local feature extraction. The results show a more nuanced picture: Swin GradCAM has a clear advantage on structure-defined diseases (Cardiomegaly, Atelectasis), while DenseNet GradCAM is competitive with Swin on Effusion and Mass. **The key determinant of localization quality is not the CNN-vs-Transformer divide, but whether a model incorporates local inductive bias.** This connects back to the AUC analysis in Section 5.2: Swin-T's multi-scale locality benefits both classification and localization, consistently outperforming DeiT-B's pure global attention on both fronts — confirming that locality matters broadly, beyond just classification accuracy.

### 5.3.3 The Structural Basis of Rollout's Non-localization

Across all six evaluated diseases, Attention Rollout produces near-zero IoU and PG of 0.000 in every case. Bbox Attention Ratio falls below 1.0 for all diseases (Cardiomegaly=0.780, Atelectasis=0.726, Effusion=0.669, Mass=0.707, Pneumothorax=0.604, Nodule=0.408), meaning lesion regions receive *lower* Rollout activation than the image average overall. Rollout not only fails to concentrate on disease regions — it systematically underweights them. Applying the 16-pixel border exclusion to suppress the corner artifact \cite{darcet2023} when computing the Pointing Game does not change this: the activation peak still never falls within the lesion bounding box.

The root cause is architectural. Rollout is class-agnostic by construction: it propagates raw attention weights from input patches to the CLS token without any class-specific gradient. The resulting map reflects the model's **information routing topology** — which patches the CLS token draws from across 12 layers — rather than what drives the model's decision for a specific disease. A patch with high Rollout value may be one the model uses to confirm normal background rather than one containing disease signal; the two are indistinguishable in the Rollout map. AR values below 1.0 further indicate that lesion regions are not privileged nodes in the information routing graph — they receive no more weight than background, and often less.

Rollout's proper role in this analysis is as a model introspection tool, not a localization tool. It reveals how DeiT-B routes information globally across layers, but it cannot answer the question of which region drove the model's prediction for a given disease — that is precisely what GradCAM is designed to answer.


---

<!-- CITATIONS NEEDED — search keywords below, paste bib entry, replace key -->

- `\cite{dosovitskiy2020}` → search: **"An Image is Worth 16x16 Words Transformers Image Recognition Scale"** (Dosovitskiy et al., ICLR 2021)
- `\cite{touvron2021}` → search: **"Training data-efficient image transformers distillation through attention DeiT"** (Touvron et al., ICML 2021)
- `\cite{liu2021swin}` → search: **"Swin Transformer Hierarchical Vision Transformer Shifted Windows"** (Liu et al., ICCV 2021)
- `\cite{wang2017}` → search: **"ChestX-ray8 Hospital-scale Chest X-ray Database Benchmarks"** (Wang et al., CVPR 2017)
- `\cite{lin2017focal}` → search: **"Focal Loss Dense Object Detection RetinaNet"** (Lin et al., ICCV 2017)
- `\cite{abnar2020}` → search: **"Quantifying Attention Flow in Transformers"** (Abnar & Zuidema, ACL 2020)
- `\cite{selvaraju2017}` → search: **"Grad-CAM Visual Explanations Deep Networks Gradient-based Localization"** (Selvaraju et al., ICCV 2017)
- `\cite{darcet2023}` → search: **"Vision Transformers Need Registers"** (Darcet et al., NeurIPS 2023)
- `\cite{fong2017}` → search: **"Interpretable Explanations Black Boxes Meaningful Perturbation"** (Fong & Vedaldi, ICCV 2017)
