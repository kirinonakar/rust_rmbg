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
use half::f16;

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

    // Initialize ORT
    let _ = ort::init()
        .with_name("rust_rmbg")
        .commit();

    // Find onnx models in current directory
    let mut onnx_models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(".") {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                if ext == "onnx" {
                    if let Some(name) = entry.path().file_name() {
                        onnx_models.push(name.to_string_lossy().into_owned());
                    }
                }
            }
        }
    }
    onnx_models.sort();

    let model_entries: Vec<slint::SharedString> = onnx_models.iter().map(|s| slint::SharedString::from(s)).collect();
    let initial_model = onnx_models.first().cloned().unwrap_or_default();
    
    ui.set_model_entries(slint::ModelRc::new(slint::VecModel::from(model_entries)));
    ui.set_active_model(initial_model.clone().into());

    let current_session = Arc::new(std::sync::Mutex::new(None::<Session>));
    let is_loading = Arc::new(std::sync::Mutex::new(false));
    
    // Load initial model if exists
    if !initial_model.is_empty() {
        let session_clone = current_session.clone();
        let is_loading_clone = is_loading.clone();
        let ui_weak_for_thread = ui_weak.clone();
        let model_name_str = initial_model.to_string();
        
        ui.set_status_text(format!("Loading model: {}...", model_name_str).into());
        *is_loading_clone.lock().unwrap() = true;
        
        let execution_mode_str = ui.get_execution_mode().to_string();
        
        std::thread::spawn(move || {
            let mut builder = Session::builder()
                .unwrap()
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .unwrap()
                .with_intra_threads(4)
                .unwrap();
                
            if execution_mode_str == "GPU (Default)" {
                if let Ok(b) = builder.clone().with_execution_providers([
                    ort::ep::CUDA::default().build(),
                    ort::ep::DirectML::default().build(),
                ]) {
                    builder = b;
                }
            }
                
            let res = builder.commit_from_file(&model_name_str);
            
            let _ = slint::invoke_from_event_loop(move || {
                *is_loading_clone.lock().unwrap() = false;
                if let Some(ui) = ui_weak_for_thread.upgrade() {
                    match res {
                        Ok(s) => {
                            *session_clone.lock().unwrap() = Some(s);
                            ui.set_status_text("Model loaded".into());
                        }
                        Err(e) => {
                            ui.set_status_text(format!("Failed to load model: {}", e).into());
                        }
                    }
                }
            });
        });
    } else {
        ui.set_status_text("No .onnx models found in current directory".into());
    }

    // Handle model selection changes
    let ui_weak_model = ui_weak.clone();
    let session_model = current_session.clone();
    let is_loading_model = is_loading.clone();
    ui.on_model_selected(move |model_name| {
        if let Some(ui) = ui_weak_model.upgrade() {
            let session_clone = session_model.clone();
            let is_loading_clone = is_loading_model.clone();
            let ui_weak_for_thread = ui_weak_model.clone();
            let model_name_str = model_name.to_string();
            
            let execution_mode_str = ui.get_execution_mode().to_string();
            ui.set_status_text(format!("Loading model: {}...", model_name_str).into());
            *is_loading_clone.lock().unwrap() = true;
            
            std::thread::spawn(move || {
                // 이전 모델 세션을 먼저 해제하여 메모리(VRAM)에서 깨끗이 비웁니다.
                *session_clone.lock().unwrap() = None;

                let mut builder = Session::builder()
                    .unwrap()
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .unwrap()
                    .with_intra_threads(4)
                    .unwrap();
                    
                if execution_mode_str == "GPU (Default)" {
                    if let Ok(b) = builder.clone().with_execution_providers([
                        ort::ep::CUDA::default().build(),
                        ort::ep::DirectML::default().build(),
                    ]) {
                        builder = b;
                    }
                }

                let res = builder.commit_from_file(&model_name_str);
                
                let _ = slint::invoke_from_event_loop(move || {
                    *is_loading_clone.lock().unwrap() = false;
                    if let Some(ui) = ui_weak_for_thread.upgrade() {
                        match res {
                            Ok(s) => {
                                *session_clone.lock().unwrap() = Some(s);
                                ui.set_status_text(format!("Model {} loaded", model_name_str).into());
                            }
                            Err(e) => {
                                ui.set_status_text(format!("Error loading {}: {}", model_name_str, e).into());
                            }
                        }
                    }
                });
            });
        }
    });

    let ui_weak_exec = ui_weak.clone();
    let session_exec = current_session.clone();
    let is_loading_exec = is_loading.clone();
    ui.on_execution_mode_changed(move |mode| {
        if let Some(ui) = ui_weak_exec.upgrade() {
            let session_clone = session_exec.clone();
            let is_loading_clone = is_loading_exec.clone();
            let ui_weak_for_thread = ui_weak_exec.clone();
            
            let model_name_str = ui.get_active_model().to_string();
            let mode_str = mode.to_string();
            
            if model_name_str.is_empty() { return; }
            
            ui.set_status_text(format!("Switching to {} mode...", mode_str).into());
            *is_loading_clone.lock().unwrap() = true;
            
            std::thread::spawn(move || {
                *session_clone.lock().unwrap() = None;
                
                let mut builder = Session::builder()
                    .unwrap()
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .unwrap()
                    .with_intra_threads(4)
                    .unwrap();
                    
                if mode_str == "GPU (Default)" {
                    if let Ok(b) = builder.clone().with_execution_providers([
                        ort::ep::CUDA::default().build(),
                        ort::ep::DirectML::default().build(),
                    ]) {
                        builder = b;
                    }
                }
                
                let res = builder.commit_from_file(&model_name_str);
                
                let _ = slint::invoke_from_event_loop(move || {
                    *is_loading_clone.lock().unwrap() = false;
                    if let Some(ui) = ui_weak_for_thread.upgrade() {
                        match res {
                            Ok(s) => {
                                *session_clone.lock().unwrap() = Some(s);
                                ui.set_status_text(format!("Model {} loaded in {} mode", model_name_str, mode_str).into());
                            }
                            Err(e) => {
                                ui.set_status_text(format!("Error switching {}: {}", model_name_str, e).into());
                            }
                        }
                    }
                });
            });
        }
    });

    // Slint Window event handling for Drop
    let session_clone = current_session.clone();
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
        if let Some(ui) = ui_weak_clone.upgrade() {
            let path_str = path.to_string();
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
            let is_loading_thread = is_loading.clone();
            let active_model_for_thread = ui.get_active_model().to_string();
        
        std::thread::spawn(move || {
            let active_model = active_model_for_thread;
            let total = paths_to_process.len();
            let mut oom_error_occurred = false;
            let mut other_error_occurred = false;
            
            for (i, p) in paths_to_process.into_iter().enumerate() {
                let current_file_name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
                
                let ui_weak_status = ui_thread_handle.clone();
                let file_name_for_ui = current_file_name.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_weak_status.upgrade() {
                        ui.set_progress(i as f32 / total as f32);
                        ui.set_status_text(format!("Processing {}/{} : {}", i + 1, total, file_name_for_ui).into());
                    }
                });

                let ui_weak_error = ui_thread_handle.clone();
                let res = {
                    while *is_loading_thread.lock().unwrap() {
                        let ui_weak_wait = ui_thread_handle.clone();
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_weak_wait.upgrade() {
                                ui.set_status_text("Waiting for model to finish loading...".into());
                            }
                        });
                        std::thread::sleep(std::time::Duration::from_millis(500));
                    }
                    let mut session_guard = session_thread.lock().unwrap();
                    if let Some(sess) = &mut *session_guard {
                        process_single_image(&p, sess, &active_model, save_32bit)
                    } else {
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_weak_error.upgrade() {
                                ui.set_status_text("Error: Model not loaded".into());
                            }
                        });
                        Err("Model not loaded".to_string())
                    }
                };
                
                if let Err(ref e) = res {
                    eprintln!("Error processing {}: {}", current_file_name, e);
                    let err_msg = e.to_lowercase();
                    if err_msg.contains("memory") || err_msg.contains("alloc") || err_msg.contains("oom") || err_msg.contains("resource") {
                        oom_error_occurred = true;
                    } else {
                        other_error_occurred = true;
                    }
                }
                
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
                
                if oom_error_occurred {
                    break;
                }
            }

            let ui_weak_final = ui_thread_handle.clone();
            let _ = slint::invoke_from_event_loop(move || {
                if let Some(ui) = ui_weak_final.upgrade() {
                    ui.set_progress(1.0);
                    ui.set_is_processing(false);
                    if oom_error_occurred {
                        ui.set_status_text("out of memory, please use smaller model".into());
                    } else if other_error_occurred {
                        ui.set_status_text("out of memory, please use smaller model".into());
                    } else {
                        ui.set_status_text(format!("Completed processing {} file(s).", total).into());
                    }
                }
            });
        });
    }
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

enum ModelFlavor {
    Rmbg,
    Ben2,
}

fn detect_flavor(model_path: &str) -> ModelFlavor {
    let lower = model_path.to_lowercase();
    if lower.contains("ben2") {
        println!("Detected BEN2 flavor for model: {}", model_path);
        ModelFlavor::Ben2
    } else {
        println!("Detected RMBG flavor for model: {}", model_path);
        ModelFlavor::Rmbg
    }
}

fn process_single_image(path: &Path, session: &mut Session, model_name: &str, save_32bit: bool) -> Result<PathBuf, String> {
    // 1. Load image
    let img = image::open(path).map_err(|e| format!("Error loading image: {}", e))?;

    let (width, height) = img.dimensions();
    println!("Processing image: {}x{}", width, height);
    
    let flavor = detect_flavor(model_name);
    let resized = img.resize_exact(1024, 1024, FilterType::Triangle);
    let mut input_tensor = Array4::<f32>::zeros((1, 3, 1024, 1024));
    
    match flavor {
        ModelFlavor::Rmbg => {
            let mean = [0.485, 0.456, 0.406];
            let std_dev = [0.229, 0.224, 0.225];
            for (x, y, pixel) in resized.to_rgb8().enumerate_pixels() {
                input_tensor[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 / 255.0 - mean[0]) / std_dev[0];
                input_tensor[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 / 255.0 - mean[1]) / std_dev[1];
                input_tensor[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 / 255.0 - mean[2]) / std_dev[2];
            }
        }
        ModelFlavor::Ben2 => {
            for (x, y, pixel) in resized.to_rgb8().enumerate_pixels() {
                input_tensor[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
                input_tensor[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
                input_tensor[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
            }
        }
    }

    // 3. Inference
    let input_vec = input_tensor.into_raw_vec();
    let input_shape = vec![1_i64, 3_i64, 1024_i64, 1024_i64];
    let input_value = ort::value::Tensor::from_array((input_shape, input_vec)).unwrap();
    
    let input_name = session.inputs()[0].name().to_string();

    let outputs = session.run(ort::inputs![input_name => input_value]).map_err(|e| {
        let err_msg = e.to_string().to_lowercase();
        if err_msg.contains("memory") || err_msg.contains("alloc") || err_msg.contains("oom") || err_msg.contains("resource") {
            "out of memory, please use smaller model".to_string()
        } else {
            format!("Inference Error: {}", e)
        }
    })?;

    // 4. Postprocess
    let (shape, mut mask_data): (Vec<usize>, Vec<f32>) = if let Ok((shape, data)) = outputs[0].try_extract_tensor::<f32>() {
        (shape.iter().map(|&v| v as usize).collect(), data.to_vec())
    } else {
        let (shape, data) = outputs[0].try_extract_tensor::<f16>().map_err(|e| format!("Tensor extraction error: {}", e))?;
        (shape.iter().map(|&v| v as usize).collect(), data.iter().map(|&v| v.to_f32()).collect())
    };
    
    println!("Output tensor shape: {:?}", shape);

    let mut raw_min = f32::MAX;
    let mut raw_max = f32::MIN;
    for &v in &mask_data {
        if v.is_finite() {
            if v < raw_min { raw_min = v; }
            if v > raw_max { raw_max = v; }
        }
    }
    println!("Raw Mask stats: min={}, max={}", raw_min, raw_max);

    // 텐서 값이 Logit(- 범위 포함)인지, 아니면 이미 [0, 1] 확률이나 [0, 255] 픽셀값인지 감지합니다.
    if raw_min < -0.1 {
        println!("Output detected as Logits. Applying Math Sigmoid...");
        for v in &mut mask_data {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
    } else if raw_max > 2.0 {
        // [0, 255] 형태로 스케일된 값일 경우. 이 때는 단순히 최대 범위로 선형 Min-Max 보간.
        println!("Output detected as pre-scaled values [> 2.0]. Normalizing to [0, 1]...");
        let range = if raw_max > raw_min { raw_max - raw_min } else { 1.0 };
        for v in &mut mask_data {
            *v = (*v - raw_min) / range;
        }
    }

    // 형태에서 마스크의 가로, 세로 크기를 안전하게 추출 (주로 마지막 2개 차원)
    let mut dims: Vec<u32> = shape.iter().filter(|&&d| d > 1).map(|&d| d as u32).collect();
    let (mask_h, mask_w) = if dims.len() >= 2 {
        let w = dims.pop().unwrap();
        let h = dims.pop().unwrap();
        (h, w)
    } else if dims.len() == 1 {
        let s = dims[0];
        (s, s)
    } else {
        (1024, 1024)
    };

    let mask_img = ImageBuffer::from_fn(mask_w, mask_h, |x, y| {
        // 단일 채널인 경우 그대로 쓰고 멀티 채널이면 마지막 채널에 접근하게 할 수 있지만 
        // 일반적으로 0번째 채널이 주요 마스크입니다.
        let v = mask_data[(y * mask_w + x) as usize];
        
        // 주의: 여기서 전체 이미지에 대해 다시 동적 Min-Max Scaling 을 하면 안 됩니다!
        // 이미 확률 [0, 1] 로 변환되었으므로 그대로 [0, 255] 로 변환하여 알파값의 손실을 방지합니다.
        let val = (v * 255.0).clamp(0.0, 255.0) as u8;
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

