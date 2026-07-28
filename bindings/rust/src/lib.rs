//! Prediction as a SQL primitive for SQLite: `forecast()`, `detect_anomalies()`,
//! and `predict()`. The zero-dependency C core is compiled into this crate.
//!
//! Register it as an auto-extension once at startup, then every connection your
//! process opens (e.g. via `rusqlite`) has the functions available:
//!
//! ```no_run
//! unsafe { sqlite_predict::register().unwrap() };
//! let conn = rusqlite::Connection::open_in_memory().unwrap();
//! conn.prepare("SELECT forecast(ts, value, 24) FROM readings").unwrap();
//! ```
use std::os::raw::{c_char, c_int, c_void};

extern "C" {
    fn sqlite3_predict_init(
        db: *mut c_void,
        pz_err_msg: *mut *mut c_char,
        p_api: *const c_void,
    ) -> c_int;
    fn sqlite3_auto_extension(entry: Option<unsafe extern "C" fn()>) -> c_int;
}

/// Register sqlite-predict as an auto-extension: it loads into every SQLite
/// connection opened afterward in this process. Call once, before opening
/// connections. Returns SQLite's result code (0 = `SQLITE_OK`).
///
/// # Safety
/// Calls into the linked libsqlite3; the entry point is registered process-wide.
pub unsafe fn register() -> Result<(), c_int> {
    let entry: unsafe extern "C" fn() = std::mem::transmute(sqlite3_predict_init as *const ());
    match sqlite3_auto_extension(Some(entry)) {
        0 => Ok(()),
        code => Err(code),
    }
}
