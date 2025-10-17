# 3D Octree Compression

A C++ implementation of super fast 3D image compression using octrees, with OpenGL visualization.

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/bfb98448-b7a4-4b69-8182-5201847749fa" alt="bunny_gif" width="100%" /><br/>
      <sub><b>Figure 1:</b> Octree layers with coloured leaf nodes</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/70507d24-57bd-4922-ac48-ced328385b10" alt="bunny_all_nodes" width="74.5%" /><br/>
      <sub><b>Figure 2:</b> Complete octree with coloured leaf nodes</sub>
    </td>
  </tr>
</table>

# How it Works

Octree compression works by recursively subdividing a 3D image into eight smaller cubes (octants). If all voxels within an octant have similar or identical values, the region is represented by a single node instead of individual voxels. This reduces the amount of data by efficiently summarizing uniform areas, allowing for compact storage and faster processing.

<p align="center">
  <img width="507" height="188" alt="image" src="https://github.com/user-attachments/assets/b7ccb38a-95a3-44cb-ac2c-31cfd53cc35f" />
</p>

# More Examples

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/2da7efdb-9001-4694-a547-ca2e373e788c" alt="teapot" width="100%" /><br/>
      <sub><b>Figure 1:</b> Teapot - layers (4,000 vertices)</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/358b368d-bdb6-421e-b0ee-58b403750d03" alt="teapot_full" width="100%" /><br/>
      <sub><b>Figure 2:</b> Teapot - complete (4,000 vertices)</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/419e896f-c78c-4a78-9a21-aa509adf0645" alt="horse" width="100%" /><br/>
      <sub><b>Figure 3:</b> Horse - layers (50,000 vertices)</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/86c70e34-728c-46be-bcb8-e2e0ad30dc98" alt="horse_full" width="100%" /><br/>
      <sub><b>Figure 4:</b> Horse - complete (50,000 vertices)</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/8a0894f0-4256-4f5f-b36d-1c8ac9fc7539" alt="dragon" width="100%" /><br/>
      <sub><b>Figure 5:</b> Dragon - layers (125,000 vertices)</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/3eba6490-e6e7-4777-b4b1-7d3a81493522" alt="dragon_full" width="100%" /><br/>
      <sub><b>Figure 6:</b> Dragon - complete (125,000 vertices)</sub>
    </td>
  </tr>
</table>

# Local Setup

## Dependencies

- C++17 compiler (`g++` or `clang++`)
- CMake 3.10+
- OpenGL 4.6
- GLFW3
- GLM
- GLAD (OpenGL loader)

### Install on Ubuntu/WSL

```bash
sudo apt update
sudo apt install build-essential cmake
sudo apt install libglfw3-dev libglm-dev
sudo apt install libgl1-mesa-dev libglu1-mesa-dev
```

## Setup

### Clone the repository

```bash
git clone https://github.com/koralkulacoglu/3d-octree-compression.git
cd 3d-octtree-compression
```

### Generate GLAD files

```bash
python3 -m venv venv
source venv/bin/activate
pip install glad
python -m glad --profile=core --api=gl=4.6 --generator=c --out-path=.
```

### Build

```bash
mkdir build && cd build
cmake ..
make -j
```

## Run

```bash
cd bin
./OctreeViewer
```

## Rebuild

```bash
cd build
rm -rf * && cmake .. && make -j
```
