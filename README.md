# rust_rmbg

A high-performance image background removal tool written in Rust, featuring a modern GUI powered by [Slint](https://slint.dev/) and hardware-accelerated machine learning inference via [ONNX Runtime](https://onnxruntime.ai/).

## Features

- **Hardware Acceleration**: Leverages GPU acceleration (CUDA and DirectML) to perform fast local background removal.
- **Drag & Drop Support**: Seamlessly select images by dragging and dropping them directly onto the application window (custom OLE Drag'n'Drop hooking on Windows).
- **High Quality Results**: Uses advanced RMBG models (such as RMBG-2.0  or InSPyReNet, utilizing `model_fp16.onnx`) to accurately mask out image backgrounds.
- **Modern GUI**: A stylish and fast user interface built entirely with Slint.
- **Offline Processing**: Everything runs locally on your machine—no internet connection or cloud API keys required.

## Prerequisites

Before building and running the application, ensure you have the following requirements:

- **Rust toolchain** (1.75+ or later Recommended)
- **ONNX model**: You need the ONNX background removal model file (e.g., `model_fp16.onnx` from RMBG-2.0 or InSPyReNet) placed in the same directory as the executable.

### 📥 Download
You can download the latest version from the [Releases Page](https://github.com/kirinonakar/rust_rmbg/releases).

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
2. **Select an image**: Drag and drop any `.png`, `.jpg`, or `.jpeg` file straight into the main window, or use the file picker dialog to choose an image.
3. **Processing**: The application will automatically process the image and extract the subject, replacing the background with transparency.
4. **Save**: The output image with the removed background will automatically be saved beside the original file with the suffix `_rmbg.png` (e.g., `image_rmbg.png`).

## Technologies Used

- **GUI**: [Slint](https://crates.io/crates/slint)
- **Image Processing**: [image](https://crates.io/crates/image), [imageproc](https://crates.io/crates/imageproc)
- **Machine Learning**: [ort](https://crates.io/crates/ort) (ONNX Runtime wrapper for Rust)
- **Math/Tensors**: [ndarray](https://crates.io/crates/ndarray)
- **System Integration**: [windows-sys](https://crates.io/crates/windows-sys) (for low-level drag-drop hooks), [rfd](https://crates.io/crates/rfd) (for file dialogs)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
