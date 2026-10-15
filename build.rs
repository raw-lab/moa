//! Capture build provenance for `moa info` / `moa version`.
use std::process::Command;

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let ver = Command::new(rustc)
        .arg("-V")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "rustc (unknown)".into());
    println!("cargo:rustc-env=MOA_RUSTC={ver}");
    println!("cargo:rustc-env=MOA_TARGET={}", std::env::var("TARGET").unwrap_or_default());
    println!("cargo:rustc-env=MOA_PROFILE={}", std::env::var("PROFILE").unwrap_or_default());
    println!("cargo:rerun-if-changed=build.rs");
}
