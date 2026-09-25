# Binder guide

The Binder badge opens `notebooks/00_quick_start.ipynb` directly. Choose
**Run > Run All Cells**. The synthetic demo runs the complete source modules but
does not launch full RHB extraction or four-fold training.

The environment is defined only by `binder/environment.yml`; `postBuild`
performs an editable install without downloads beyond environment resolution.
No notebook needs network access, local paths, or manual input after launch.

Update the badge's GitHub owner/repository placeholder before public upload.
