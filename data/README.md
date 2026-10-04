# Data — how to obtain it (not committed)

Raw imaging data are **not** stored in the repository. Everything below regenerates from the
public source plus the code in `src/`.

## LIDC-IDRI

- Source: TCIA — https://www.cancerimagingarchive.net/collection/lidc-idri/
- Download with the **NBIA Data Retriever** (~125 GB of DICOM; the annotation XMLs are bundled).
- Place it so that the patient folders sit directly under a single root. The `path` column of the
  split files in `outputs/splits/` is relative to the repository root and assumes this layout,
  with `<root>` being the repository root:
  ```
  <root>/LIDC-IDRI/manifest-<id>/LIDC-IDRI/LIDC-IDRI-0001/.../*.dcm
  ```
  Nothing reads that column (images are loaded through the preprocessed index), so another
  location works as long as `config.yaml` points at it.
- **pylidc configuration** (annotation engine). pylidc ships the annotation database and needs the
  DICOM root only to load pixels. Create the configuration file in your home directory: it is
  `pylidc.conf` on Windows (`C:\Users\<you>\pylidc.conf`) and `.pylidcrc` on Linux and macOS
  (`~/.pylidcrc`), which is what pylidc 0.2.3 looks for:
  ```
  [dicom]
  path = <root>/LIDC-IDRI/manifest-<id>/LIDC-IDRI
  warn = True
  ```
  Point `paths.lidc_dicom_root` and `paths.lidc_metadata_csv` in `config.yaml` at the same folder
  (editing them changes the configuration hash; see the README).
- **Verify completeness** before building anything:
  ```
  python scripts/verify_download.py --config config.yaml
  ```
  It confirms that every CT series resolves, that slice counts match `metadata.csv`, and that every
  annotated slice is present. The download used for the paper was complete: 1018 of 1018 scans.
