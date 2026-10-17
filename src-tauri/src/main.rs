#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::Command;

fn main() {
    // Запускаем Python сервер
    println!("Starting Python Backend...");
    
    Command::new("python")
        .args(["backend/main.py"])
        .spawn()
        .expect("Failed to start Python backend");
    
    // Запускаем Tauri приложение
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}