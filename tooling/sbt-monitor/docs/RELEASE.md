# Release procedure

1. Replace collective/placeholder citation and repository metadata with the final public project metadata.
2. Re-check the PyPI normalized name immediately before upload.
3. Run `pytest`.
4. Build wheel and sdist with `python -m build`.
5. Validate metadata with `python -m twine check dist/*`.
6. Install both artifacts into clean environments and rerun the smoke tests.
7. Generate `SHA256SUMS.txt`.
8. Upload first to TestPyPI, install from TestPyPI, then upload to PyPI.
9. Create the matching source-control release and permanent software archive DOI.

PyPI releases are immutable. Never reuse version `0.1.0` after a public upload.
