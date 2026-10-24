# Research sources and reused implementation

The dataset authors' [paper](https://arxiv.org/html/2509.21561v2) evaluates Dinomaly
and DRAEM with SAM masking, native-resolution 256-pixel patches and, for the
transformer, rank-8 LoRA. It reports pixel AUROC, not image-classification accuracy.
No surgical-task code or task checkpoint was found in the authors' linked sources.

This integration reuses [Anomalib 2.2.0 Dinomaly](https://github.com/open-edge-platform/anomalib/tree/v2.2.0/src/anomalib/models/image/dinomaly),
based on the [official Dinomaly implementation](https://github.com/guojiajeremy/Dinomaly),
and [official MobileSAM](https://github.com/ChaoningZhang/MobileSAM) commit
`f706ad9c4eb7f219c00d9050e46328518ffb65d2`. These projects provide Apache-2.0
licensed implementations; upstream notices remain in their installed packages.
The paper's authors and these software authors retain their respective credits.

Adaptations: DINOv2-reg Small for available hardware; MobileSAM with deterministic
RGB-derived automatic prompts; 256-pixel tiles resized to 252 for stride14;
validation-only model selection. These deviations and the small balanced cohorts
mean this is not an exact reproduction of the paper or its reported metrics.

The custom adapter uses Anomalib's decoder, bottleneck, reconstruction loss and
anomaly-map computation. It freezes pretrained encoder weights, adds trainable
rank-8 qkv/output adapters and permits their gradients through the encoder.
Training uses normal patches, sampled equally by instrument and source image.
Class-specific thresholds maximize validation F1. The API returns stitched,
foreground-masked reconstruction maps in original-image coordinates.
