"""Zero Mouse Clicks — architectural plan sets from a marketing floor-plan image.

Small on-device segmentation models (ONNX) + rules. No cloud API, no LLM tokens.
"""
__version__ = "0.2.0"


def plan_set(image, out_pdf, area=None, project="Marketing floor plan", unit="Unit", debug=None, area_kind=None):
    """Read a floor-plan image and write an A3 PDF plan set.

    image   path to a JPG/PNG marketing floor plan
    out_pdf path of the PDF to write
    area    declared area in m²; None = try to read it from the image. Room dimensions printed on the
            plan («2.69 x 2.20», «12'4" x 11'0"») set the scale when they agree with each other;
            the area is then a cross-check
    area_kind  'private' (net spaces + partitions, default) or 'gross' (built area, with walls)
    debug   optional path of a JPG with the model mask and the vector model side by side
    Returns a dict with the vector model, the finished-state model and a summary in ``info``.
    """
    from .pipeline import run
    return run(image, area, project, unit, out_pdf, debug=debug, area_kind=area_kind)
