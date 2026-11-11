"""Zero Mouse Clicks — architectural plan sets from a marketing floor-plan image.

Small on-device segmentation models (ONNX) + rules. No cloud API, no LLM tokens.
"""
__version__ = "0.1.0"


def plan_set(image, out_pdf, area=None, project="Marketing floor plan", unit="Unit", debug=None):
    """Read a floor-plan image and write an A3 PDF plan set.

    image   path to a JPG/PNG marketing floor plan
    out_pdf path of the PDF to write
    area    declared private (net interior) area in m²; None = try to read it from the image
    debug   optional path of a JPG with the model mask and the vector model side by side
    Returns a dict with the vector model, the finished-state model and a summary in ``info``.
    """
    from .pipeline import run
    return run(image, area, project, unit, out_pdf, debug=debug)
