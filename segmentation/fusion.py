def fuse_predictions(clinical_probability, image_probability):
    """
    Prototype late-fusion of independent clinical and ultrasound predictions.

    Clinical probability = probability of CKD from clinical model.
    Image probability = probability of pathological finding from ultrasound.

    Note:
    The current clinical and ultrasound datasets are not patient-linked.
    Therefore this is a prototype fusion rule, not a patient-level
    multimodal model.
    """

    clinical_weight = 0.60
    image_weight = 0.40

    final_probability = (
        clinical_weight * clinical_probability
        + image_weight * image_probability
    )

    if final_probability >= 0.50:
        final_prediction = "CKD Detected"
    else:
        final_prediction = "Healthy"

    return {
        "clinical_probability": round(
            clinical_probability * 100, 2
        ),
        "image_probability": round(
            image_probability * 100, 2
        ),
        "final_probability": round(
            final_probability * 100, 2
        ),
        "final_prediction": final_prediction
    }