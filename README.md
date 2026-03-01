# rust_rmbg

A high-performance image background removal tool written in Rust, featuring a modern GUI powered by [Slint](https://slint.dev/) and hardware-accelerated machine learning inference via [ONNX Runtime](https://onnxruntime.ai/).

## Features

- **Multi-Model Support**: Automatically scans for `.onnx` files in the current directory and allows switching between them via the UI dropdown.
- **BEN2 & RMBG Support**: Optimized preprocessing for different model flavors. BEN2 models are automatically detected (based on "ben2" in the filename) for correct normalization.
- **Improved Tensor Handling**: Built-in support for both `f32` and `f16` (Half-Precision) output tensors, ensuring compatibility with a wider range of optimized models.
- **Hardware Acceleration**: Leverages GPU acceleration (CUDA and DirectML) to perform fast local background removal.
- **Batch Processing**: Supports dragging and dropping multiple images at once, processing them sequentially with a real-time progress bar.
- **32-bit BMP Support**: Option to export results as 32-bit BMP files with an alpha channel, ideal for legacy software compatibility.
- **Drag & Drop Support**: Seamlessly select images by dragging and dropping them directly onto the application window (custom OLE Drag'n'Drop hooking on Windows).
- **Modern GUI**: A stylish and fast user interface built entirely with Slint.
- **Offline Processing**: Everything runs locally on your machine—no internet connection or cloud API keys required.

## Prerequisites

Before building and running the application, ensure you have the following requirements:

- **Rust toolchain** (1.75+ or later Recommended)
- **ONNX models**: Place your `.onnx` model files in the same directory as the executable. 
    - Supports **RMBG-2.0**, **InSPyReNet**, and **BEN2** models.
    - If using **BEN2**, ensure the filename contains "ben2" (case-insensitive, e.g., `BEN2_Base.onnx`) to trigger the correct normalization.

### 📥 Download
You can download the latest version from the [Releases Page](https://github.com/kirinonakar/rust_rmbg/releases).

### 🤖 Supported Models & Downloads

Download the `.onnx` files and place them in the application folder:

- **RMBG-2.0**: You have to register at https://huggingface.co/briaai/RMBG-2.0/ after huggingface login, then you can download the model.
  - [model.onnx](https://huggingface.co/briaai/RMBG-2.0/blob/main/onnx/model.onnx)
  - [model_fp16.onnx](https://huggingface.co/briaai/RMBG-2.0/blob/main/onnx/model_fp16.onnx)
- **BEN2**:
  - [BEN2_Base.onnx](https://huggingface.co/PramaLLC/BEN2/resolve/main/BEN2_Base.onnx)
- **InSPyReNet (Plus Ultra)**:
  - [model.onnx](https://huggingface.co/OS-Software/InSPyReNet-SwinB-Plus-Ultra-ONNX/blob/main/onnx/model.onnx)
  - [model_fp16.onnx](https://huggingface.co/OS-Software/InSPyReNet-SwinB-Plus-Ultra-ONNX/blob/main/onnx/model_fp16.onnx)

## Installation & Build

Clone the repository and build using Cargo:

```bash
git clone https://github.com/your-username/rust_rmbg.git
cd rust_rmbg
cargo build --release
```

Once built, move your `model_fp16.onnx` model file to `target/release/` and run the executable:
```bash
cargo run --release
```

## Usage

1. **Launch the application**: Run the `rust_rmbg` executable.
2. **Select Model**: Use the **"Model:"** dropdown to select from available `.onnx` files in the app folder.
3. **Configure Options**: Toggle the **"Add 32bit bmp with alpha channel"** checkbox if you need BMP output in addition to PNG.
4. **Select images**: Drag and drop one or multiple `.png`, `.jpg`, `.jpeg`, or `.bmp` files into the main window, or use the **"Select Image"** button.
4. **Processing**: The application will process files sequentially. A progress bar will indicate the current status and overall batch progress.
5. **Save**: Resulting images are saved automatically in the same folder as the original files:
   - Always saves as `[filename]_rmbg.png` (with transparency).
   - If enabled, also saves as `[filename]_rmbg.bmp` (32-bit with alpha channel).

## Technologies Used

- **GUI**: [Slint](https://crates.io/crates/slint)
- **Image Processing**: [image](https://crates.io/crates/image), [imageproc](https://crates.io/crates/imageproc)
- **Machine Learning**: [ort](https://crates.io/crates/ort) (ONNX Runtime wrapper for Rust)
- **Math/Tensors**: [ndarray](https://crates.io/crates/ndarray)
- **System Integration**: [windows-sys](https://crates.io/crates/windows-sys) (for low-level drag-drop hooks), [rfd](https://crates.io/crates/rfd) (for file dialogs)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
