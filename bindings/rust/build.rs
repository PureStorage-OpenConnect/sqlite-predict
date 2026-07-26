// Compile the amalgamated zero-dependency core into this crate. `make rust-src`
// stages it into csrc/; it is also shipped in the published crate.
fn main() {
    let src = std::path::Path::new("csrc/sqlite-predict.c");
    if src.exists() {
        cc::Build::new()
            .file(src)
            .include("csrc")
            .define("SQLITE_CORE", None) // statically linked: use sqlite3_* directly
            .warnings(false)
            .compile("sqlite_predict");
    } else {
        println!("cargo:warning=csrc/sqlite-predict.c missing; run `make rust-src`");
    }
}
