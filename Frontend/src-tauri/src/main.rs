#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::Command;

fn main() {
    // Запускаем Python бэкенд при старте приложения
    println!("Starting 3D Helper Core...");
    Command::new("C:/Python IDLE/python.exe")
        .args(["backend/main.py"])
        .spawn()
        .expect("Failed to start Python backend");

    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}