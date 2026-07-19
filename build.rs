use std::{env, fs, path::PathBuf};

const DIAGNOSTIC_BEGIN: &str = "// DIAGNOSTIC_KERNEL_BEGIN";
const DIAGNOSTIC_END: &str = "// DIAGNOSTIC_KERNEL_END";

fn main() {
    println!("cargo:rerun-if-changed=src/solver/lean.wgsl");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_DIAGNOSTICS");

    let source = fs::read_to_string("src/solver/lean.wgsl").expect("read lean.wgsl");
    let source = if env::var_os("CARGO_FEATURE_DIAGNOSTICS").is_some() {
        source
    } else {
        without_diagnostic_kernels(source)
    };
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR")).join("lean.wgsl");
    fs::write(output, source).expect("write generated lean.wgsl");
}

fn without_diagnostic_kernels(mut source: String) -> String {
    while let Some(begin) = source.find(DIAGNOSTIC_BEGIN) {
        let tail = &source[begin..];
        let end = tail
            .find(DIAGNOSTIC_END)
            .map(|offset| begin + offset + DIAGNOSTIC_END.len())
            .expect("diagnostic kernel marker is not closed");
        source.replace_range(begin..end, "");
    }
    source
}
