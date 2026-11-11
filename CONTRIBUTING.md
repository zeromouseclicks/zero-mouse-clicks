# Contributing

Thanks for trying Zero Mouse Clicks. Contributions in English or Spanish are welcome.

## Reporting a failing plan

1. Run with `--debug`. It writes `<out>_debug.jpg`: the input, what the models see, and the vector model.
2. Open an issue with what went wrong (a missing wall, a wrong room type, a bad scale…) and, if you can, the debug image.
3. **Only attach images you own or have the right to share.** Developer brochures and portal listings are copyrighted. If you cannot share the image, describe it (style, size in pixels, country) and share the debug image of the vector model only.

## Pull requests

- Keep rules explicit and readable: every decision should be traceable to one function.
- Run the example before opening a PR: `zmc examples/synthetic_54m2.png --area 54.4 --debug`.
- New model weights must be trained only on data you have the right to use (the synthetic generator is the default). State the data and recipe in `MODEL_CARD.md`.
- Translating the Spanish code comments to English is a welcome first contribution.

By contributing you agree that your contribution is licensed under the Apache License 2.0, as stated in section 5 of the [License](LICENSE).
