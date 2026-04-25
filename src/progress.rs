use std::error::Error;
use std::fmt;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::engine::CodecChoice;

pub(crate) type VariantId = usize;

/// Фазы активной работы. Терминальное состояние отдельно — через `Finished`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SearchPhase {
    LoadingImages,
    Preprocessing,
    Jpeg,
    Jp2,
    Mrc,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FinishStatus {
    Success,
    Cancelled,
    Failed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogLevel {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VariantInfo {
    pub(crate) id: VariantId,
    pub(crate) filename: PathBuf,
    pub(crate) codec: CodecChoice,
    pub(crate) setting: String,
    pub(crate) scale_pct: u32,
    pub(crate) size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RecommendationKind {
    BestForText(VariantId),
    Balanced(VariantId),
    MaxDetail(VariantId),
    Smallest(VariantId),
}

#[derive(Debug, Clone)]
pub(crate) enum ProgressEvent {
    Phase(SearchPhase),
    ImagesFound {
        count: usize,
    },
    CurrentPage {
        path: String,
        index: usize,
        total: usize,
    },
    SettingStarted {
        codec: CodecChoice,
        setting: String,
        index: usize,
        total: usize,
    },
    Probe {
        setting: String,
        scale: u32,
        size_bytes: u64,
    },
    VariantReady(VariantInfo),
    SettingSkipped {
        codec: CodecChoice,
        setting: String,
        reason: String,
    },
    Recommendations(Vec<RecommendationKind>),
    Log {
        level: LogLevel,
        message: String,
    },
    Finished {
        status: FinishStatus,
    },
}

pub(crate) trait ProgressReporter: Send + Sync {
    fn report(&self, event: ProgressEvent);
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "useful for tests and non-CLI consumers")
)]
pub(crate) struct NoopReporter;

impl ProgressReporter for NoopReporter {
    fn report(&self, _event: ProgressEvent) {}
}

#[derive(Clone, Default)]
pub(crate) struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    #[cfg_attr(not(test), expect(dead_code, reason = "GUI cancel will call this"))]
    pub(crate) fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

#[derive(Debug)]
pub(crate) struct Cancelled;

impl fmt::Display for Cancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("операция отменена")
    }
}

impl Error for Cancelled {}

#[derive(Debug)]
pub(crate) struct EngineMessage(pub(crate) String);

impl fmt::Display for EngineMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for EngineMessage {}
