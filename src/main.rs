#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
slint::include_modules!();

use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use image::imageops::FilterType;
use ndarray::Array4;
use ort::{session::builder::GraphOptimizationLevel, session::Session};
use rfd::FileDialog;
use std::sync::Arc;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::fs::File;
use std::io::{Write, BufWriter};
use byteorder::{WriteBytesExt, LittleEndian};

#[cfg(target_os = "windows")]
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM, GetLastError};
#[cfg(target_os = "windows")]
use windows_sys::Win32::UI::Shell::{DragAcceptFiles, DragFinish, DragQueryFileW, HDROP};
#[cfg(target_os = "windows")]
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CallWindowProcW, ChangeWindowMessageFilterEx, SetWindowLongPtrW, GWLP_WNDPROC, 
    MSGFLT_ALLOW, WM_DROPFILES, WNDPROC,
};
#[cfg(target_os = "windows")]
use windows_sys::Win32::System::Ole::RevokeDragDrop;

static APP_WINDOW_HANDLE: OnceLock<slint::Weak<MainWindow>> = OnceLock::new();
#[cfg(target_os = "windows")]
static mut ORIGINAL_WNDPROC: WNDPROC = None;

#[cfg(target_os = "windows")]
unsafe extern "system" fn wnd_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    match msg {
        WM_DROPFILES => {
            let hdrop = wparam as HDROP;
            let mut path_buf = [0u16; 1024]; 
            let mut pt = windows_sys::Win32::Foundation::POINT { x: 0, y: 0 };
            
            unsafe {
                windows_sys::Win32::UI::Shell::DragQueryPoint(hdrop, &mut pt);
                
                let count = DragQueryFileW(hdrop, 0xFFFFFFFF, std::ptr::null_mut(), 0);
                let mut paths = Vec::new();
                for i in 0..count {
                    let len = DragQueryFileW(hdrop, i, path_buf.as_mut_ptr(), 1024);
                    if len > 0 {
                        paths.push(String::from_utf16_lossy(&path_buf[..len as usize]));
                    }
                }
                
                if !paths.is_empty() {
                    let paths_str = paths.join("|");
                    if let Some(weak) = APP_WINDOW_HANDLE.get() {
                        let weak_clone = weak.clone();
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(ui) = weak_clone.upgrade() {
                                ui.invoke_files_dropped(slint::SharedString::from(paths_str.as_str()));
                            }
                        });
                    }
                }
                DragFinish(hdrop);
            }
            return 0;
        }
        _ => {}
    }
    
    // 원래 윈도우 프로시저 호출 (체이닝)
    unsafe {
        if let Some(orig) = ORIGINAL_WNDPROC {
            CallWindowProcW(Some(orig), hwnd, msg, wparam, lparam)
        } else {
            windows_sys::Win32::UI::WindowsAndMessaging::DefWindowProcW(hwnd, msg, wparam, lparam)
        }
    }
}

fn main() -> Result<(), slint::PlatformError> {
    let ui = MainWindow::new()?;
    let ui_weak = ui.as_weak();

    // 전역 핸들 설정
    let _ = APP_WINDOW_HANDLE.set(ui_weak.clone());

    // Initialize ORT with GPU support
    let _ = ort::init()
        .with_name("rust_rmbg")
        .with_execution_providers([
            ort::ep::CUDA::default().build(),
            ort::ep::DirectML::default().build(),
        ])
        .commit();

    let model_path = "model_fp16.onnx";
    let session_opt = match Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_intra_threads(4)
        .unwrap()
        .commit_from_file(model_path)
    {
        Ok(s) => Some(Arc::new(std::sync::Mutex::new(s))),
        Err(e) => {
            eprintln!("Failed to load ONNX model: {:?}", e);
            ui.set_status_text(slint::SharedString::from(format!("Error loading model: {}", e)));
            None
        }
    };

    // Slint Window event handling for Drop
    let session_clone = session_opt.clone();
    // Let's print out what events exist by letting compiler fail if Drop is wrong
    // Slint's Window does not have an on_window_event. It has on_winit_window_event if the winit feature is enabled.
    // In slint 1.15, to handle drop we either need winit or try something else.
    // Let's remove the WindowEvent trap to avoid errors and rely on the fallback button for now 
    // or try using rfd drag and drop if possible. Actually, Slint UI natively supports files being dropped
    // on a `TouchArea` but wait, there's no `drop` event on TouchArea.
    /*
    ui.window().on_window_event(move |event| {
        // ...
    });
    */


    let ui_weak_clone = ui_weak.clone();
    ui.on_files_dropped(move |path| {
        let path_str = path.to_string();
        let ui = ui_weak_clone.unwrap();
        if ui.get_is_processing() {
            return;
        }
        let save_32bit = ui.get_save_32bit_bmp();
        
        let mut paths_to_process = Vec::new();
        if path_str.is_empty() {
            if let Some(files) = FileDialog::new().add_filter("Image", &["png", "jpg", "jpeg", "bmp"]).pick_files() {
                paths_to_process.extend(files);
            }
        } else {
            paths_to_process.extend(path_str.split('|').filter(|s| !s.is_empty()).map(PathBuf::from));
        }

        if paths_to_process.is_empty() {
            return;
        }

        ui.set_is_processing(true);
        ui.set_progress(0.0);
        
        let ui_thread_handle = ui_weak_clone.clone();
        let session_thread = session_clone.clone();
        
        std::thread::spawn(move || {
            let total = paths_to_process.len();
            for (i, p) in paths_to_process.into_iter().enumerate() {
                let current_file_name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
                
                let ui_weak_status = ui_thread_handle.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_weak_status.upgrade() {
                        ui.set_progress(i as f32 / total as f32);
                        ui.set_status_text(format!("Processing {}/{} : {}", i + 1, total, current_file_name).into());
                    }
                });

                let ui_weak_error = ui_thread_handle.clone();
                let res = match &session_thread {
                    Some(s) => process_single_image(&p, s, save_32bit),
                    None => {
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_weak_error.upgrade() {
                                ui.set_status_text("Error: Model not loaded".into());
                            }
                        });
                        Err("Model not loaded".to_string())
                    }
                };
                
                let out_path_opt = res.ok();
                let ui_weak_preview = ui_thread_handle.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_weak_preview.upgrade() {
                        if let Some(out_path) = out_path_opt {
                            if let Ok(slint_img) = slint::Image::load_from_path(&out_path) {
                                ui.set_preview_image(slint_img);
                                ui.set_has_image(true);
                            }
                        }
                    }
                });
            }

            let ui_weak = ui_thread_handle.clone();
            let _ = slint::invoke_from_event_loop(move || {
                if let Some(ui) = ui_weak.upgrade() {
                    ui.set_progress(1.0);
                    ui.set_is_processing(false);
                    ui.set_status_text(format!("Completed processing {} file(s).", total).into());
                }
            });
        });
    });

    #[cfg(target_os = "windows")]
    {
        let ui_handle_clone = ui_weak.clone();
        
        // Timer를 사용하여 이벤트 루프가 시작되고 300ms 뒤에 훅을 설치합니다.
        slint::Timer::single_shot(std::time::Duration::from_millis(300), move || {
            if let Some(ui) = ui_handle_clone.upgrade() {
                use raw_window_handle::{HasWindowHandle, RawWindowHandle};
                let window_handle = ui.window().window_handle();
                if let Ok(handle) = window_handle.window_handle() {
                    if let RawWindowHandle::Win32(h) = handle.as_raw() {
                        let hwnd = h.hwnd.get() as HWND;
                        println!("Slint HWND 획득 성공 (지연 실행): {:?}", hwnd);

                        unsafe {
                            // 핵심: 이벤트 루프가 덮어씌운 OLE 드래그 앤 드롭을 이 시점에서 빼앗아옵니다.
                            let hr = RevokeDragDrop(hwnd);
                            println!("RevokeDragDrop 실행 (S_OK=0 이면 정상): {}", hr);

                            // 관리자 권한 UIPI 우회 설정
                            ChangeWindowMessageFilterEx(hwnd, WM_DROPFILES, MSGFLT_ALLOW, std::ptr::null_mut());
                            ChangeWindowMessageFilterEx(hwnd, 0x0049, MSGFLT_ALLOW, std::ptr::null_mut()); 
                            ChangeWindowMessageFilterEx(hwnd, 0x004A, MSGFLT_ALLOW, std::ptr::null_mut());
                            
                            // 드래그 앤 드롭 활성화
                            DragAcceptFiles(hwnd, 1);
                            println!("DragAcceptFiles 설정 완료");

                            // WndProc 교체 (Subclassing)
                            let prev_proc = SetWindowLongPtrW(
                                hwnd,
                                GWLP_WNDPROC,
                                wnd_proc as *const () as isize,
                            );
                            
                            if prev_proc != 0 {
                                println!("WndProc 후킹 성공. 이전 주소: 0x{:X}", prev_proc);
                                type WndProcFn = unsafe extern "system" fn(HWND, u32, WPARAM, LPARAM) -> LRESULT;
                                ORIGINAL_WNDPROC = Some(core::mem::transmute::<isize, WndProcFn>(prev_proc));
                            } else {
                                println!("경고: SetWindowLongPtrW 실패. 에러 코드: {}", GetLastError());
                            }
                        }
                    }
                }
            }
        });
    }

    println!("이벤트 루프 시작");
    ui.run()
}

fn process_single_image(path: &Path, session: &Arc<std::sync::Mutex<Session>>, save_32bit: bool) -> Result<PathBuf, String> {
    // 1. Load image
    let img = image::open(path).map_err(|e| format!("Error loading image: {}", e))?;

    let (width, height) = img.dimensions();
    
    // 2. Preprocess: Resize to 1024x1024, Normalize
    let mean = [0.485, 0.456, 0.406];
    let std_dev = [0.229, 0.224, 0.225];

    let resized = img.resize_exact(1024, 1024, FilterType::Triangle);
    let mut input_tensor = Array4::<f32>::zeros((1, 3, 1024, 1024));
    for (x, y, pixel) in resized.to_rgb8().enumerate_pixels() {
        // RMBG input is RGB (BCHW)
        input_tensor[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 / 255.0 - mean[0]) / std_dev[0];
        input_tensor[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 / 255.0 - mean[1]) / std_dev[1];
        input_tensor[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 / 255.0 - mean[2]) / std_dev[2];
    }

    // 3. Inference
    let input_vec = input_tensor.into_raw_vec();
    let input_value = ort::value::Tensor::from_array((vec![1_i64, 3_i64, 1024_i64, 1024_i64], input_vec)).unwrap();
    let mut session_guard = session.lock().unwrap();
    let outputs = session_guard.run(ort::inputs![input_value]).map_err(|e| format!("Inference Error: {}", e))?;

    // 4. Postprocess
    let (_shape, mask_data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| format!("Tensor extraction error: {}", e))?;
    
    // Min-Max Scaling
    let mut mask_min = f32::MAX;
    let mut mask_max = f32::MIN;
    for &v in mask_data {
        if v < mask_min { mask_min = v; }
        if v > mask_max { mask_max = v; }
    }
    let mask_range = mask_max - mask_min;

    // Output mask is (1, 1, 1024, 1024)
    let mask_img = ImageBuffer::from_fn(1024, 1024, |x, y| {
        let v = mask_data[(y * 1024 + x) as usize];
        let normalized = if mask_range > 0.0 { (v - mask_min) / mask_range } else { 0.0 };
        let val = (normalized * 255.0).clamp(0.0, 255.0) as u8;
        Luma([val])
    });

    // Resize mask back to original size
    let mask_resized = DynamicImage::ImageLuma8(mask_img).resize_exact(width, height, FilterType::Triangle).into_luma8();

    // Apply mask to original image
    let mut rgba_img = img.to_rgba8();
    for (x, y, pixel) in rgba_img.enumerate_pixels_mut() {
        let mask_val = mask_resized.get_pixel(x, y)[0];
        pixel[3] = mask_val; // replace alpha with mask
    }

    // Save output
    let out_path = path.with_file_name(format!("{}_rmbg.png", path.file_stem().unwrap().to_string_lossy()));
    rgba_img.save(&out_path).map_err(|e| format!("Error saving png: {}", e))?;

    // Save 32bit BMP if checked
    if save_32bit {
        let bmp_path = path.with_file_name(format!("{}_rmbg.bmp", path.file_stem().unwrap().to_string_lossy()));
        let rgba_raw = rgba_img.into_raw();
        save_32bit_bmp_from_data(width, height, &rgba_raw, &bmp_path).map_err(|e| format!("Error saving BMP: {}", e))?;
    }

    Ok(out_path)
}

fn save_32bit_bmp_from_data(width: u32, height: u32, rgba_data: &[u8], output_path: &Path) -> std::io::Result<()> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    // BMP File Header (14 bytes)
    writer.write_all(b"BM")?;
    let header_size = 14 + 40; // FileHeader + BITMAPINFOHEADER
    let pixel_data_offset = header_size as u32;
    let file_size = pixel_data_offset + (width * height * 4);
    
    writer.write_u32::<LittleEndian>(file_size)?;
    writer.write_u16::<LittleEndian>(0)?; // Reserved 1
    writer.write_u16::<LittleEndian>(0)?; // Reserved 2
    writer.write_u32::<LittleEndian>(pixel_data_offset)?;

    // BITMAPINFOHEADER (40 bytes)
    writer.write_u32::<LittleEndian>(40)?; // biSize
    writer.write_i32::<LittleEndian>(width as i32)?; // biWidth
    writer.write_i32::<LittleEndian>(height as i32)?; // biHeight (positive for bottom-up)
    writer.write_u16::<LittleEndian>(1)?; // biPlanes
    writer.write_u16::<LittleEndian>(32)?; // biBitCount
    writer.write_u32::<LittleEndian>(0)?; // biCompression (BI_RGB)
    writer.write_u32::<LittleEndian>(0)?; // biSizeImage
    writer.write_i32::<LittleEndian>(0)?; // biXPelsPerMeter
    writer.write_i32::<LittleEndian>(0)?; // biYPelsPerMeter
    writer.write_u32::<LittleEndian>(0)?; // biClrUsed
    writer.write_u32::<LittleEndian>(0)?; // biClrImportant

    // Pixel Data (Bottom-Up)
    // rgba_data is expected to be [R, G, B, A, ...]
    for y in (0..height).rev() {
        for x in 0..width {
            let offset = ((y * width + x) * 4) as usize;
            writer.write_u8(rgba_data[offset + 2])?; // B
            writer.write_u8(rgba_data[offset + 1])?; // G
            writer.write_u8(rgba_data[offset])?;     // R
            writer.write_u8(rgba_data[offset + 3])?; // A
        }
    }

    writer.flush()?;
    Ok(())
}

