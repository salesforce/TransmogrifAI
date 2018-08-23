# Docs

[Sphinx](http://www.sphinx-doc.org) based docs site hosted on [ReadTheDocs](https://readthedocs.org/projects/transmogrifai).

## Running locally

If you wish to run the docs locally install the following dependencies:
```bash
pip install sphinx sphinx-autobuild recommonmark sphinx_rtd_theme
```

Then simply run:
```bash
cd docs
make html
sphinx-autobuild . _build/html
```

Browse to - http://localhost:8000