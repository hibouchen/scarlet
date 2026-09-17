# Web platform

SCARLET can be offered to a team through JupyterHub. Users open JupyterLab in a browser while the Python kernel, SCARLET, and experimental data remain on the server.

## Shared runtime

The deployment uses a SCARLET Docker image tagged for a project release. Every user server is created from that exact image, which provides the same Python and package versions for the whole team. Each user has a separate persistent workspace, while experimental data can be mounted read-only from shared storage.

The deployment files are located in [`deployment/jupyterhub`](https://github.com/hibouchen/scarlet/tree/main/deployment/jupyterhub). They include GitHub OAuth authentication, one container per user, and an automatic copy of the tutorial notebook.

## Viewer status

The current viewer uses Qt and cannot run directly in a browser. The notebook workflow is therefore available first. A web viewer will require a dedicated browser interface that reuses the existing NeXus-reading and mask-writing services while replacing the Qt interface.

## Operations

Run the platform behind HTTPS and restrict access to named users. Keep shared raw data read-only for notebook containers, store user outputs separately, and publish a new image for each SCARLET release.
