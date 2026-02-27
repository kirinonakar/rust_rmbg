slint::include_modules!();

use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use image::imageops::FilterType;
use ndarray::Array4;
use ort::{session::builder::GraphOptimizationLevel, session::Session};
use rfd::FileDialog;
use std::sync::Arc;
use std::path::PathBuf;
use std::sync::OnceLock;

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
                
                let len = DragQueryFileW(hdrop, 0, path_buf.as_mut_ptr(), 1024);
                if len > 0 {
                    let path = String::from_utf16_lossy(&path_buf[..len as usize]);
                    
                    if let Some(weak) = APP_WINDOW_HANDLE.get() {
                        let weak_clone = weak.clone();
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(ui) = weak_clone.upgrade() {
                                ui.invoke_files_dropped(slint::SharedString::from(path.as_str()));
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
    let session = match Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_intra_threads(4)
        .unwrap()
        .commit_from_file(model_path)
    {
        Ok(s) => Arc::new(std::sync::Mutex::new(s)),
        Err(e) => {
            eprintln!("Failed to load ONNX model: {:?}", e);
            ui.set_status_text(slint::SharedString::from(format!("Error loading model: {:?}", e)));
            // We will continue but it won't work
            Arc::new(std::sync::Mutex::new(Session::builder().unwrap().commit_from_memory(&[]).unwrap())) // dummy
        }
    };

    // Slint Window event handling for Drop
    let session_clone = session.clone();
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
        if path_str.is_empty() {
            if let Some(file) = FileDialog::new().add_filter("Image", &["png", "jpg", "jpeg"]).pick_file() {
                process_image(file, session_clone.clone(), &ui_weak_clone);
            }
        } else {
            process_image(PathBuf::from(path_str), session_clone.clone(), &ui_weak_clone);
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

fn process_image(path: PathBuf, session: Arc<std::sync::Mutex<Session>>, ui_weak: &slint::Weak<MainWindow>) {
    let ui = ui_weak.unwrap();
    ui.set_status_text("Processing...".into());
    
    // Slint UI updates need to be dispatched if done from another thread, 
    // but here we are blocking the main thread for simplicity first.
    // For a better UX, we should use std::thread::spawn and slint::invoke_from_event_loop.
    
    // 1. Load image
    let img = match image::open(&path) {
        Ok(img) => img,
        Err(e) => {
            ui.set_status_text(format!("Error loading image: {}", e).into());
            return;
        }
    };

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
    let outputs = match session_guard.run(ort::inputs![input_value]) {
        Ok(out) => out,
        Err(e) => {
            ui.set_status_text(format!("Inference Error: {}", e).into());
            return;
        }
    };

    // 4. Postprocess
    let (_shape, mask_data) = outputs[0].try_extract_tensor::<f32>().unwrap();
    
    // Min-Max Scaling (RMBG-2.0 officially uses Min-Max normalize on prediction)
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
    if let Err(e) = rgba_img.save(&out_path) {
        ui.set_status_text(format!("Error saving: {}", e).into());
        return;
    }

    // Load to Slint
    let slint_img = slint::Image::load_from_path(&out_path).unwrap();
    ui.set_preview_image(slint_img);
    ui.set_has_image(true);
    ui.set_status_text(format!("Saved to {}", out_path.display()).into());
}

