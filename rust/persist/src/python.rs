//! Python bindings (`import swarmopt_persist`). Payloads are plain `bytes`;
//! encoding is left to the caller (SwarmOpt uses strict JSON, never pickle).

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::checkpoint::{self, CheckpointStore as RsCheckpointStore, Loaded};
use crate::error::Error;
use crate::log::{self, FullPolicy, LogConfig, LogWriter as RsLogWriter, ReadOptions, SyncPolicy};

create_exception!(
    swarmopt_persist,
    StoreError,
    PyException,
    "Base class for store errors."
);
create_exception!(
    swarmopt_persist,
    StorageFullError,
    StoreError,
    "The storage budget is used up."
);
create_exception!(
    swarmopt_persist,
    LockedError,
    StoreError,
    "Another writer holds the lock."
);
create_exception!(
    swarmopt_persist,
    CorruptError,
    StoreError,
    "Damaged data where it can't be skipped."
);
create_exception!(
    swarmopt_persist,
    UnsafePathError,
    StoreError,
    "A symlink where a real path is required."
);

fn to_py(e: Error) -> PyErr {
    let msg = e.to_string();
    match e {
        Error::Io(io) => PyErr::from(io),
        Error::StorageFull { .. } => StorageFullError::new_err(msg),
        Error::Locked(_) => LockedError::new_err(msg),
        Error::Corrupt(_) => CorruptError::new_err(msg),
        Error::UnsafePath(_) => UnsafePathError::new_err(msg),
        Error::TooLarge { .. } | Error::InvalidConfig(_) => PyValueError::new_err(msg),
    }
}

fn closed() -> PyErr {
    StoreError::new_err("store is closed")
}

fn parse_sync(obj: &Bound<'_, PyAny>) -> PyResult<SyncPolicy> {
    if let Ok(s) = obj.extract::<String>() {
        return match s.as_str() {
            "always" => Ok(SyncPolicy::Always),
            "manual" => Ok(SyncPolicy::Manual),
            _ => Err(PyValueError::new_err(
                "sync must be 'always', 'manual' or a positive int",
            )),
        };
    }
    let n: u32 = obj.extract()?;
    if n == 0 {
        return Err(PyValueError::new_err("sync=0 is not allowed"));
    }
    Ok(SyncPolicy::EveryN(n))
}

/// Append-only, replicated, CRC-checked record log.
#[pyclass(module = "swarmopt_persist")]
pub struct LogWriter {
    inner: Option<RsLogWriter>,
}

#[pymethods]
impl LogWriter {
    #[new]
    #[pyo3(signature = (path, *, stream_id=0, replicas=2, segment_bytes=4*1024*1024,
                        max_total_bytes=256*1024*1024, max_record_bytes=1024*1024,
                        sync=None, on_full="reject"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        path: std::path::PathBuf,
        stream_id: u64,
        replicas: usize,
        segment_bytes: u64,
        max_total_bytes: u64,
        max_record_bytes: usize,
        sync: Option<Bound<'_, PyAny>>,
        on_full: &str,
    ) -> PyResult<Self> {
        let sync = match sync {
            Some(s) => parse_sync(&s)?,
            None => SyncPolicy::Always,
        };
        let on_full = match on_full {
            "reject" => FullPolicy::Reject,
            "drop_oldest" => FullPolicy::DropOldest,
            _ => {
                return Err(PyValueError::new_err(
                    "on_full must be 'reject' or 'drop_oldest'",
                ))
            }
        };
        let cfg = LogConfig {
            stream_id,
            replicas,
            segment_bytes,
            max_total_bytes,
            max_record_bytes,
            sync,
            on_full,
        };
        let w = py.detach(|| RsLogWriter::open(&path, cfg)).map_err(to_py)?;
        Ok(LogWriter { inner: Some(w) })
    }

    /// Append ``payload`` under record type ``kind``; returns its sequence number.
    fn append(&mut self, py: Python<'_>, kind: u16, payload: Vec<u8>) -> PyResult<u64> {
        let w = self.inner.as_mut().ok_or_else(closed)?;
        py.detach(|| w.append(kind, &payload)).map_err(to_py)
    }

    fn sync(&mut self, py: Python<'_>) -> PyResult<()> {
        let w = self.inner.as_mut().ok_or_else(closed)?;
        py.detach(|| w.sync()).map_err(to_py)
    }

    /// Delete whole segments whose records are all ``<= seq``. Returns bytes freed.
    fn prune_through(&mut self, py: Python<'_>, seq: u64) -> PyResult<u64> {
        let w = self.inner.as_mut().ok_or_else(closed)?;
        py.detach(|| w.prune_through(seq)).map_err(to_py)
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if let Some(mut w) = self.inner.take() {
            py.detach(|| w.sync()).map_err(to_py)?;
        }
        Ok(())
    }

    #[getter]
    fn next_seq(&self) -> PyResult<u64> {
        Ok(self.inner.as_ref().ok_or_else(closed)?.next_seq())
    }

    #[getter]
    fn degraded_writes(&self) -> PyResult<u64> {
        Ok(self.inner.as_ref().ok_or_else(closed)?.degraded_writes())
    }

    #[getter]
    fn bytes_used(&self) -> PyResult<u64> {
        Ok(self.inner.as_ref().ok_or_else(closed)?.bytes_used())
    }

    /// What opening the log found and repaired.
    #[getter]
    fn recovery<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let r = self.inner.as_ref().ok_or_else(closed)?.recovery();
        let d = PyDict::new(py);
        d.set_item("truncated_bytes", r.truncated_bytes)?;
        d.set_item("quarantined_segments", r.quarantined_segments)?;
        d.set_item("removed_empty_segments", r.removed_empty_segments)?;
        Ok(d)
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        _t: Option<Bound<'_, PyAny>>,
        _v: Option<Bound<'_, PyAny>>,
        _tb: Option<Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close(py)?;
        Ok(false)
    }
}

type ReadResult<'py> = (Vec<(u64, u16, Bound<'py, PyBytes>)>, Bound<'py, PyDict>);

/// Read and merge every replica of the log at ``path``.
/// Returns ``(records, stats)`` with records as ``(seq, kind, payload)``.
#[pyfunction]
#[pyo3(signature = (path, *, max_record_bytes=log::HARD_MAX_RECORD, stream_id=None))]
fn read_log<'py>(
    py: Python<'py>,
    path: std::path::PathBuf,
    max_record_bytes: usize,
    stream_id: Option<u64>,
) -> PyResult<ReadResult<'py>> {
    let opts = ReadOptions {
        max_record_bytes,
        stream_id,
        ..Default::default()
    };
    let (records, s) = py.detach(|| log::read_log(&path, &opts)).map_err(to_py)?;
    let recs = records
        .into_iter()
        .map(|r| (r.seq, r.kind, PyBytes::new(py, &r.payload)))
        .collect();
    let d = PyDict::new(py);
    d.set_item("replicas", s.replicas)?;
    d.set_item("segments", s.segments)?;
    d.set_item("unreadable_segments", s.unreadable_segments)?;
    d.set_item("skipped_bytes", s.skipped_bytes)?;
    d.set_item("damaged_regions", s.damaged_regions)?;
    d.set_item("repaired_from_replica", s.repaired_from_replica)?;
    d.set_item("missing_seqs", s.missing_seqs)?;
    d.set_item("conflicts", s.conflicts)?;
    Ok((recs, d))
}

fn loaded_dict<'py>(py: Python<'py>, l: Loaded) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("generation", l.generation)?;
    d.set_item("payload", PyBytes::new(py, &l.payload))?;
    d.set_item("fell_back", l.fell_back)?;
    d.set_item("damaged_slots", l.damaged_slots)?;
    Ok(d)
}

/// Power-cut-safe A/B checkpoint slots.
#[pyclass(module = "swarmopt_persist")]
pub struct CheckpointStore {
    inner: Option<RsCheckpointStore>,
}

#[pymethods]
impl CheckpointStore {
    #[new]
    #[pyo3(signature = (path, *, max_bytes=16*1024*1024))]
    fn new(py: Python<'_>, path: std::path::PathBuf, max_bytes: usize) -> PyResult<Self> {
        let s = py
            .detach(|| RsCheckpointStore::open(&path, max_bytes))
            .map_err(to_py)?;
        Ok(CheckpointStore { inner: Some(s) })
    }

    /// Save ``payload``; returns the new generation number.
    fn save(&mut self, py: Python<'_>, payload: Vec<u8>) -> PyResult<u64> {
        let s = self.inner.as_mut().ok_or_else(closed)?;
        py.detach(|| s.save(&payload)).map_err(to_py)
    }

    /// The newest valid checkpoint as a dict, or ``None`` if there is none.
    fn load<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let s = self.inner.as_ref().ok_or_else(closed)?;
        match py.detach(|| s.load()).map_err(to_py)? {
            Some(l) => Ok(Some(loaded_dict(py, l)?)),
            None => Ok(None),
        }
    }

    fn close(&mut self) {
        self.inner = None;
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _t: Option<Bound<'_, PyAny>>,
        _v: Option<Bound<'_, PyAny>>,
        _tb: Option<Bound<'_, PyAny>>,
    ) -> bool {
        self.close();
        false
    }
}

/// Read the newest valid checkpoint without taking the lock.
#[pyfunction]
#[pyo3(signature = (path, *, max_bytes=16*1024*1024))]
fn read_checkpoint<'py>(
    py: Python<'py>,
    path: std::path::PathBuf,
    max_bytes: usize,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    match py
        .detach(|| checkpoint::read_checkpoint(&path, max_bytes))
        .map_err(to_py)?
    {
        Some(l) => Ok(Some(loaded_dict(py, l)?)),
        None => Ok(None),
    }
}

#[pymodule]
fn swarmopt_persist(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<LogWriter>()?;
    m.add_class::<CheckpointStore>()?;
    m.add_function(wrap_pyfunction!(read_log, m)?)?;
    m.add_function(wrap_pyfunction!(read_checkpoint, m)?)?;
    m.add("StoreError", py.get_type::<StoreError>())?;
    m.add("StorageFullError", py.get_type::<StorageFullError>())?;
    m.add("LockedError", py.get_type::<LockedError>())?;
    m.add("CorruptError", py.get_type::<CorruptError>())?;
    m.add("UnsafePathError", py.get_type::<UnsafePathError>())?;
    m.add("FORMAT_VERSION", log::FORMAT_VERSION)?;
    Ok(())
}
