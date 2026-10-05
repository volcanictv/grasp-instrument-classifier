"""End-to-end GraSP instrument pipeline: boxes in, instance masks, classes and uncertainty out.

    BoxSegmenter            ground-truth or detector boxes -> instance masks (fine-tuned SAM2, optionally with SAM3)
    EvidentialEnsembleClassifier (grasp_classifier.evidential)   mask -> class and an epistemic uncertainty score
    EvidentialTracker       SAM2 video propagation of an uncertain instance, classes fused over the frames
    InstrumentPipeline      the three stages together, plus the semantic map the benchmark's mIoU, IoU and mcIoU are computed on
"""

__version__ = "0.3.0"
