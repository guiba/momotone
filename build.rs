//! Link compiler-rt on Apple targets.
//!
//! ONNX Runtime's CoreML execution provider calls `___isPlatformVersionAtLeast`
//! (the clang availability-check builtin). rustc's *fat* LTO drops that symbol from
//! `std` because no Rust code references it, so the final link of the binary fails
//! with "Undefined symbols for architecture arm64" even though the profile's LTO
//! is worth keeping for the hot path. rustc tracks it as
//! <https://github.com/rust-lang/rust/issues/164915>.
//!
//! clang ships the symbol in its compiler-rt archive, so pointing the linker at the
//! clang resource directory that the active `cc` actually uses supplies it. The
//! lookup is dynamic rather than a hardcoded path so it survives a Homebrew LLVM
//! version bump and works on machines that use Apple's toolchain instead.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS")
        .map(|os| os != "macos")
        .unwrap_or(true)
    {
        return;
    }

    let output = match std::process::Command::new("cc")
        .arg("-print-search-dirs")
        .output()
    {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).into_owned()
        }
        _ => {
            println!("cargo:warning=cc not available; skipping the compiler-rt link");
            return;
        }
    };

    let Some(libs) = output
        .lines()
        .find_map(|line| line.strip_prefix("libraries: ="))
    else {
        println!("cargo:warning=cc did not report its search dirs; skipping the compiler-rt link");
        return;
    };

    for dir in libs.split('=') {
        let rt = std::path::Path::new(dir).join("lib/darwin/libclang_rt.osx.a");
        if rt.exists() {
            println!("cargo:rustc-link-arg=-L{dir}/lib/darwin");
            println!("cargo:rustc-link-arg=-lclang_rt.osx");
            return;
        }
    }

    println!("cargo:warning=no compiler-rt found in cc's search dirs; a release build with LTO may fail to link");
}
