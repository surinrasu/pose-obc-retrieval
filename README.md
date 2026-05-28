# Pose-OBC Retrieval

![](./banner.png)

`pose-obc-retrieval` is a Rust CLI and web UI for retrieving oracle-bone glyphs from human pose. It trains a compact twin-tower retrieval model that embeds SpinePose style 37-keypoint pose features and raster glyph shape features into the same cosine-search space.

## Usage

The project uses [mise](https://mise.jdx.dev/) for the Rust toolchain and tasks.

```sh
mise trust
mise install

# Download SpinePose ONNX models used by the Burn runtime
mise run data:spinepose

# Pull packed pose-obc files from Git LFS and unpack them locally
mise run data:pose-obc

# Train the retrieval model and build the glyph embedding index
mise run train:retrieval
mise run retrieval:index

# Search by a dataset sample
SAMPLE=0 mise run retrieval:search

# Start the web UI
mise run serve:retrieval
```

The UI listens on `http://127.0.0.1:8080` by default. Use `ADDR=127.0.0.1:1234` to choose another address.

### Train

```sh
mise run train:retrieval -- --epochs 20 --batch-size 32
```

Outputs are written under `runs/retrieval/`:

- `last.mpk`: latest retrieval checkpoint
- `retrieval_config.json`: model dimensions used by the checkpoint
- `retrieval_training_report.json`: per-epoch loss report

For Metal:

```sh
mise run train:retrieval:metal
```

CUDA support will be added later, same applies below.

### Index

```sh
mise run retrieval:index
```

This writes `runs/retrieval/glyph_index.json`. The index stores candidate glyph metadata plus normalized embeddings, so repeated searches do not need to re-encode the glyph corpus.

Extracted pose and glyph feature will be cached under `runs/retrieval/feature_cache`. They are only valid for exactly same pair of source image and JSON. `

### Search

```sh
SAMPLE=0 mise run retrieval:search
IMAGE=/path/to/query.avif mise run retrieval:search
TOP_K=16 SAMPLE=0 mise run retrieval:search
```

### Serve

```sh
mise run serve:retrieval
mise run serve:retrieval:metal
mise run serve:retrieval:live
```

The live mode posts browser camera frames to the local service and returns the top glyph candidates for each frame.

## Data Layout

The retrieval dataset is expected to contain one or more `persona_*` directories. Image and glyph files are paired by filename.

```text
data/pose-obc/
  persona_01/
    images/
      0201_u516D.avif
    glyphs/
      0201_u516D.avif
    poses/
      0201_u516D.json
```

### Packed Dataset

Packed retrieval data is stored under `data/pose-obc-packed` as one directory per persona:

```text
data/pose-obc-packed/
  persona_01/
    images.avif
    glyphs.avif
    manifest.jsonl
    poses.jsonl
```

`images.avif` and `glyphs.avif` are multi-frame AVIF files. `manifest.jsonl` stays as normal Git text for reviewable diffs and records both source file hashes and packed-frame hashes; `poses.jsonl` and AVIF files are Git LFS objects.

```sh
cargo run --release -- dataset pack --force
cargo run --release -- dataset verify
cargo run --release -- dataset unpack --force
```

## Pose Model

The repository also contains a Lite-HRNet training path for COCO person-keypoint data with 37-keypoint labels.

```sh
mise run data:coco2017
mise run data:coco2017:generate-pose37
mise run train:coco2017
```

For Metal:

```sh
mise run train:coco2017:metal
```

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
