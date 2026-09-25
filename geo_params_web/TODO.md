# GeoParams Web Application TODO

This file tracks objectives specific to `geo_params_web`. The TODO convention
is documented in [`../docs/about-todos.md`](../docs/about-todos.md).

## Implementing

No objective is currently recorded as being implemented.

## To Implement Soon

No objective currently has a deadline.

## Ideas

### Standalone application

- [ ] Provide a Windows distribution without Docker or virtualization.
  - Users should not need to install Git, Python, PDM, WSL, or Docker.
  - Keep Docker available for development, reproduction, and server deployment.
- [ ] Bundle a private Python runtime with the application.
  - Evaluate PyInstaller `onedir` first, wrapped in a regular installer.
  - The runtime must not enter `PATH` or interfere with system Python installs.
  - Uninstalling the application should remove the runtime, not user data.
- [ ] Create a native launcher to coordinate local services.
  - Replace Nginx and `supervisord` in the Windows package.
  - Start Flask, Streamlit, and jobs on `127.0.0.1` only.
  - Select available ports and open the default browser automatically.
  - Consider a WebView window only after the browser-based flow is stable.
- [ ] Keep the installation, internal data, and user work separate.
  - Store configuration, caches, and logs outside the executable directory.
  - Do not request technical storage decisions during installation.
  - Provide actions in the application to open and change the data directory.
  - Preserve user data by default during updates and uninstallation.
- [ ] Introduce projects that can reference images at their original locations.
  - Store relative and absolute paths when applicable.
  - Record hashes, sizes, modification times, calibration, metadata, parameters,
    results, and application and algorithm versions.
  - Treat missing or changed images as offline until the user relocates or
    confirms them.
  - Verify relocated files by hash and never search entire partitions
    automatically.
- [ ] Offer explicit image-storage strategies.
  - **Link:** use the file at its current location without copying it.
  - **Copy into project:** make the project self-contained.
  - **Managed library:** copy and deduplicate content by hash.
  - Do not use hard links as a data-isolation mechanism.
- [ ] Keep reproducible derivatives separate from source images.
  - Thumbnails, tiles, and intermediate results may remain in a cache.
  - Clearing the cache must not invalidate projects or erase final results.
- [ ] Support project verification, consolidation, and export.
  - Integrity checks should detect missing or modified files.
  - Freezing an analysis should record the exact inputs and versions used.
  - Collecting or consolidating should copy linked media into a portable project.
  - Export should support results-only packages and optional source images.
  - Import should restore exported projects without their original paths.
- [ ] Prepare the distribution for nontechnical users.
  - Provide per-user installation, shortcuts, and a clear uninstaller.
  - Clearly show where projects and persistent data are stored.
  - Evaluate code signing to reduce Windows SmartScreen warnings.
  - Validate licenses and permissions before distributing public executables.

## Done

No objective has been archived in this TODO yet.
