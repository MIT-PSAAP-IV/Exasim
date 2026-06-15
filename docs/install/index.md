# Installation

Pick the page that matches your platform.

| Platform | Page |
|---|---|
| macOS (Apple Silicon or Intel) | [`macos.md`](macos.md) |
| Linux x86_64, CPU only | [`linux-cpu.md`](linux-cpu.md) |
| Linux x86_64, NVIDIA GPU | [`linux-nvidia.md`](linux-nvidia.md) |
| Linux x86_64, AMD GPU | [`linux-amd.md`](linux-amd.md) |

All platforms share the dependency build steps in
[`common.md`](common.md). The per-platform
pages reference it.

## Repository folder name

The cloned folder must be named `Exasim`. Renaming it breaks the
Kokkos build paths.
