# NeXus schemas

SCARLET distributes YAML schemas to validate NeXus/HDF5 files created or used by the workflow. The following command lists the exact names available in the installation:

```bash
scarlet schema list
```

The main schemas are:

| Schema | Role |
| --- | --- |
| `scarlet_nxsas_raw_v1.3_mono.yaml` | Single-wavelength SCARLET raw data using the `NXsas_raw` profile. |
| `scarlet_masks_v1.0.yaml` | Detector mask bundle. |
| `scarlet_refs_sub_v1.0.yaml` | References required for subtraction. |
| `scarlet_refs_norm_v1.0.yaml` | References required for normalization. |
| `scarlet_workflow_context_v1.0.yaml` | Workflow context, artifacts, logs, and file-exclusion history. |

Functional explanations of the profiles are kept with the package: [SCARLET schemas](https://github.com/hibouchen/scarlet/tree/main/src/scarlet/schemas).

For the raw-data structure and the collimation convention, see the [data format](format.md).

## Workflow exclusion history

Workflow-context files may contain `/entry/excluded_files`. Its parallel arrays
store the path, reason, source, optional run key, action, and UTC timestamp for
each exclusion or reinstatement event. The group is optional for compatibility
with workflow-context files written before exclusion tracking was introduced.

See [Workflow context](workflow.md) for the corresponding Python API.
