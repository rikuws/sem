//! Embedded API for host applications that want sem's semantic model without
//! going through the CLI or the on-disk `.sem` cache.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::git::types::{FileChange, FileStatus};
use crate::model::change::SemanticChange;
use crate::model::entity::SemanticEntity;
use crate::parser::context::{build_context, ContextEntry};
use crate::parser::differ::{compute_semantic_diff, DiffResult};
use crate::parser::graph::{EntityGraph, EntityInfo, EntityRef};
use crate::parser::plugins::create_default_registry;
use crate::parser::registry::ParserRegistry;
use crate::utils::hash::content_hash;

pub const SEM_EMBEDDED_API_VERSION: &str = "sem-embedded-v1";

pub type SemResult<T> = Result<T, SemError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SemErrorKind {
    Cancelled,
    InvalidInput,
    Io,
    NotFound,
    AmbiguousTarget,
    Unsupported,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemError {
    pub kind: SemErrorKind,
    pub message: String,
}

impl SemError {
    pub fn new(kind: SemErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for SemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", sem_error_kind_label(self.kind), self.message)
    }
}

impl std::error::Error for SemError {}

impl From<std::io::Error> for SemError {
    fn from(value: std::io::Error) -> Self {
        SemError::new(SemErrorKind::Io, value.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemProgressEvent {
    pub stage: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<usize>,
}

pub trait SemCancellationToken: Send + Sync {
    fn is_cancelled(&self) -> bool;
}

pub trait SemProgressReporter: Send + Sync {
    fn report(&self, event: SemProgressEvent);
}

#[derive(Default)]
pub struct SemExecutionContext<'a> {
    pub cancellation: Option<&'a dyn SemCancellationToken>,
    pub progress: Option<&'a dyn SemProgressReporter>,
}

/// Controls optional repository-local configuration for embedded calls.
///
/// The default intentionally does not load `.semrc` or `.gitattributes`, so
/// callers can get deterministic behavior from supplied content alone.
#[derive(Debug, Clone, Default)]
pub struct SemEmbeddedOptions {
    pub root: Option<PathBuf>,
    pub load_semrc: bool,
    pub load_gitattributes: bool,
}

impl SemEmbeddedOptions {
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self {
            root: Some(root.into()),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemRepoScanOptions {
    pub max_files: usize,
    pub max_file_bytes: u64,
    pub max_depth: usize,
    pub include_hidden: bool,
    #[serde(default)]
    pub extra_skip_dirs: Vec<String>,
}

impl Default for SemRepoScanOptions {
    fn default() -> Self {
        Self {
            max_files: 2_000,
            max_file_bytes: 512_000,
            max_depth: 12,
            include_hidden: false,
            extra_skip_dirs: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemDiscoveredFiles {
    pub api_version: String,
    pub cache_key: String,
    pub files: Vec<String>,
    #[serde(default)]
    pub skipped: Vec<SemSkippedFile>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemSkippedFile {
    pub path: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemFileInput {
    pub file_path: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemFileChange {
    pub file_path: String,
    pub status: FileStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_file_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hunks: Vec<SemHunk>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemLineRange {
    pub start_line: usize,
    pub end_line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemHunk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hunk_id: Option<String>,
    pub hunk_index: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hunk_header: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_range: Option<SemLineRange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_range: Option<SemLineRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemEntityRange {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemHunkOverlap {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hunk_id: Option<String>,
    pub hunk_index: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hunk_header: Option<String>,
    pub file_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_file_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_range: Option<SemLineRange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_range: Option<SemLineRange>,
    pub overlaps_before: bool,
    pub overlaps_after: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemDiffSummary {
    pub file_count: usize,
    pub added_count: usize,
    pub modified_count: usize,
    pub deleted_count: usize,
    pub moved_count: usize,
    pub renamed_count: usize,
    pub reordered_count: usize,
    pub orphan_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemEmbeddedChange {
    pub change: SemanticChange,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_range: Option<SemEntityRange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_range: Option<SemEntityRange>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hunk_overlaps: Vec<SemHunkOverlap>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemDiffAnalysis {
    pub api_version: String,
    pub cache_key: String,
    pub summary: SemDiffSummary,
    pub changes: Vec<SemEmbeddedChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemEntityIndex {
    pub api_version: String,
    pub cache_key: String,
    pub entities: Vec<SemanticEntity>,
}

#[derive(Debug)]
pub struct SemGraphSnapshot {
    pub api_version: String,
    pub cache_key: String,
    pub file_paths: Vec<String>,
    pub graph: EntityGraph,
    pub entities: Vec<SemanticEntity>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemGraphCacheRecord {
    pub api_version: String,
    pub cache_key: String,
    pub file_paths: Vec<String>,
    pub entities: Vec<SemanticEntity>,
    pub entity_infos: Vec<EntityInfo>,
    pub edges: Vec<EntityRef>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl SemGraphSnapshot {
    pub fn to_cache_record(&self) -> SemGraphCacheRecord {
        let mut entity_infos = self
            .graph
            .entities
            .values()
            .cloned()
            .collect::<Vec<EntityInfo>>();
        entity_infos.sort_by(|a, b| a.id.cmp(&b.id));
        SemGraphCacheRecord {
            api_version: self.api_version.clone(),
            cache_key: self.cache_key.clone(),
            file_paths: self.file_paths.clone(),
            entities: self.entities.clone(),
            entity_infos,
            edges: self.graph.edges.clone(),
            warnings: self.warnings.clone(),
        }
    }

    pub fn from_cache_record(record: SemGraphCacheRecord) -> SemResult<Self> {
        if record.api_version != SEM_EMBEDDED_API_VERSION {
            return Err(SemError::new(
                SemErrorKind::InvalidInput,
                format!(
                    "unsupported graph cache api version: {}",
                    record.api_version
                ),
            ));
        }
        let entity_map = record
            .entity_infos
            .iter()
            .cloned()
            .map(|entity| (entity.id.clone(), entity))
            .collect::<HashMap<_, _>>();
        Ok(Self {
            api_version: record.api_version,
            cache_key: record.cache_key,
            file_paths: record.file_paths,
            graph: EntityGraph::from_parts(entity_map, record.edges),
            entities: record.entities,
            warnings: record.warnings,
        })
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemEntityTarget {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
}

impl SemEntityTarget {
    pub fn by_id(entity_id: impl Into<String>) -> Self {
        Self {
            entity_id: Some(entity_id.into()),
            ..Self::default()
        }
    }

    pub fn by_name(entity_name: impl Into<String>) -> Self {
        Self {
            entity_name: Some(entity_name.into()),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SemSide {
    Before,
    After,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemLocationTarget {
    pub file_path: String,
    pub line: usize,
    pub side: SemSide,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemHunkTarget {
    pub file_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_file_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hunk_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hunk_header: Option<String>,
    pub side: SemSide,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_range: Option<SemLineRange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_range: Option<SemLineRange>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemFocusTarget {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity: Option<SemEntityTarget>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<SemLocationTarget>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hunk: Option<SemHunkTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemFocusedEntity {
    pub entity: SemanticEntity,
    pub range: SemEntityRange,
    pub side: SemSide,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemFocusResolution {
    pub api_version: String,
    pub cache_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_entity: Option<SemFocusedEntity>,
    #[serde(default)]
    pub overlapping_entities: Vec<SemFocusedEntity>,
    #[serde(default)]
    pub matching_changes: Vec<SemEmbeddedChange>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemImpactRequest {
    pub token_budget: usize,
    pub max_depth: usize,
}

impl Default for SemImpactRequest {
    fn default() -> Self {
        Self {
            token_budget: 4_096,
            max_depth: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemImpactEntity {
    pub entity: EntityInfo,
    pub depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemContextEntry {
    pub entity_id: String,
    pub entity_name: String,
    pub entity_type: String,
    pub file_path: String,
    pub role: String,
    pub content: String,
    pub estimated_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemImpactContext {
    pub api_version: String,
    pub cache_key: String,
    pub entity: EntityInfo,
    pub dependencies: Vec<EntityInfo>,
    pub dependents: Vec<EntityInfo>,
    pub impact: Vec<SemImpactEntity>,
    pub tests: Vec<EntityInfo>,
    pub context: Vec<SemContextEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemEntityReference {
    pub entity: EntityInfo,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemDiffImpactContext {
    pub api_version: String,
    pub cache_key: String,
    pub focus: SemFocusResolution,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo_context: Option<SemImpactContext>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deleted_entity: Option<SemFocusedEntity>,
    #[serde(default)]
    pub references: Vec<SemEntityReference>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemLayerGenerationOptions {
    pub max_layers: usize,
    pub max_changes_per_layer: usize,
}

impl Default for SemLayerGenerationOptions {
    fn default() -> Self {
        Self {
            max_layers: 24,
            max_changes_per_layer: 24,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemReviewLayerPlan {
    pub api_version: String,
    pub cache_key: String,
    pub layers: Vec<SemReviewLayer>,
    #[serde(default)]
    pub manual_review_change_indices: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub manual_review_atom_ids: Vec<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemReviewAtom {
    pub atom_id: String,
    pub file_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_file_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbol_name: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub defined_symbols: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub referenced_symbols: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_range: Option<SemLineRange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_range: Option<SemLineRange>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hunk_indices: Vec<usize>,
    #[serde(default)]
    pub changed_lines: usize,
    #[serde(default)]
    pub manual_review: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemReviewLayer {
    pub id: String,
    pub index: usize,
    pub title: String,
    pub summary: String,
    pub rationale: String,
    #[serde(default)]
    pub depends_on_layer_ids: Vec<String>,
    #[serde(default)]
    pub change_indices: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub atom_ids: Vec<String>,
    #[serde(default)]
    pub file_paths: Vec<String>,
    #[serde(default)]
    pub hunk_indices: Vec<usize>,
    #[serde(default)]
    pub entity_names: Vec<String>,
}

/// Analyze supplied before/after file contents at entity granularity.
///
/// This path never shells out, never reads git state, and never writes `.sem`.
pub fn analyze_file_changes(
    changes: &[SemFileChange],
    options: &SemEmbeddedOptions,
) -> SemDiffAnalysis {
    let registry = registry_for_options(options, None);
    let file_changes: Vec<FileChange> = changes.iter().map(to_file_change).collect();
    let result = compute_semantic_diff(&file_changes, &registry, None, None);
    let extracted = extract_change_entities(changes, &registry);
    let enriched = result
        .changes
        .iter()
        .cloned()
        .map(|change| enrich_change(change, changes, &extracted))
        .collect();

    SemDiffAnalysis {
        api_version: SEM_EMBEDDED_API_VERSION.to_string(),
        cache_key: diff_cache_key(changes, options),
        summary: diff_summary(&result),
        changes: enriched,
    }
}

/// Extract an entity index from caller-owned file contents.
pub fn extract_entity_index(
    files: &[SemFileInput],
    options: &SemEmbeddedOptions,
) -> SemEntityIndex {
    let registry = registry_for_options(options, None);
    let mut entities: Vec<SemanticEntity> = files
        .iter()
        .flat_map(|file| registry.extract_entities(&file.file_path, &file.content))
        .collect();
    entities.sort_by(|a, b| {
        a.file_path
            .cmp(&b.file_path)
            .then(a.start_line.cmp(&b.start_line))
            .then(a.id.cmp(&b.id))
    });

    SemEntityIndex {
        api_version: SEM_EMBEDDED_API_VERSION.to_string(),
        cache_key: entity_index_cache_key(files, options),
        entities,
    }
}

pub fn discover_repo_files(
    root: &Path,
    scan_options: &SemRepoScanOptions,
    options: &SemEmbeddedOptions,
) -> SemResult<SemDiscoveredFiles> {
    discover_repo_files_with_execution(root, scan_options, options, &SemExecutionContext::default())
}

pub fn discover_repo_files_with_execution(
    root: &Path,
    scan_options: &SemRepoScanOptions,
    options: &SemEmbeddedOptions,
    execution: &SemExecutionContext<'_>,
) -> SemResult<SemDiscoveredFiles> {
    ensure_not_cancelled(execution)?;
    if !root.is_dir() {
        return Err(SemError::new(
            SemErrorKind::InvalidInput,
            format!("root is not a directory: {}", root.display()),
        ));
    }

    let registry = registry_for_options(options, Some(root));
    let mut queue = VecDeque::from([(root.to_path_buf(), 0usize)]);
    let mut files = Vec::new();
    let mut skipped = Vec::new();
    let mut warnings = Vec::new();
    let skip_dirs = scan_skip_dirs(scan_options);

    report_progress(
        execution,
        "discover",
        format!("Scanning {}", root.display()),
        Some(0),
        None,
    );

    while let Some((dir, depth)) = queue.pop_front() {
        ensure_not_cancelled(execution)?;
        if depth > scan_options.max_depth {
            continue;
        }
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) => {
                warnings.push(format!("Could not read {}: {error}", dir.display()));
                continue;
            }
        };

        for entry in entries.flatten() {
            ensure_not_cancelled(execution)?;
            let path = entry.path();
            let Some(relative) = relative_repo_path(root, &path) else {
                continue;
            };
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();

            if path.is_dir() {
                if should_skip_scan_dir(name, scan_options.include_hidden, &skip_dirs) {
                    continue;
                }
                queue.push_back((path, depth + 1));
                continue;
            }

            if files.len() >= scan_options.max_files {
                skipped.push(SemSkippedFile {
                    path: relative,
                    reason: "max_files reached".to_string(),
                });
                continue;
            }
            if !scan_options.include_hidden && path_has_hidden_component(Path::new(&relative)) {
                continue;
            }

            let metadata = match entry.metadata() {
                Ok(metadata) => metadata,
                Err(error) => {
                    skipped.push(SemSkippedFile {
                        path: relative,
                        reason: format!("metadata unavailable: {error}"),
                    });
                    continue;
                }
            };
            if metadata.len() > scan_options.max_file_bytes {
                skipped.push(SemSkippedFile {
                    path: relative,
                    reason: "file too large".to_string(),
                });
                continue;
            }

            let content = match std::fs::read_to_string(&path) {
                Ok(content) => content,
                Err(_) => {
                    skipped.push(SemSkippedFile {
                        path: relative,
                        reason: "not utf-8 text".to_string(),
                    });
                    continue;
                }
            };
            let plugin = registry.get_plugin_with_content(&relative, &content);
            if plugin
                .map(|plugin| plugin.id() == "fallback")
                .unwrap_or(true)
            {
                skipped.push(SemSkippedFile {
                    path: relative,
                    reason: "no semantic parser".to_string(),
                });
                continue;
            }

            files.push(relative);
            report_progress(
                execution,
                "discover",
                "Found semantic file",
                Some(files.len()),
                Some(scan_options.max_files),
            );
        }
    }

    files.sort();
    Ok(SemDiscoveredFiles {
        api_version: SEM_EMBEDDED_API_VERSION.to_string(),
        cache_key: discovery_cache_key(root, scan_options, options, &files),
        files,
        skipped,
        warnings,
    })
}

pub fn build_repo_graph(
    root: &Path,
    file_paths: &[String],
    options: &SemEmbeddedOptions,
) -> SemResult<SemGraphSnapshot> {
    build_repo_graph_with_execution(root, file_paths, options, &SemExecutionContext::default())
}

pub fn build_repo_graph_with_execution(
    root: &Path,
    file_paths: &[String],
    options: &SemEmbeddedOptions,
    execution: &SemExecutionContext<'_>,
) -> SemResult<SemGraphSnapshot> {
    ensure_not_cancelled(execution)?;
    report_progress(
        execution,
        "graph",
        format!("Building semantic graph for {} files", file_paths.len()),
        Some(0),
        Some(file_paths.len()),
    );
    let registry = registry_for_options(options, Some(root));
    let (graph, entities) = EntityGraph::build(root, file_paths, &registry);
    ensure_not_cancelled(execution)?;
    report_progress(
        execution,
        "graph",
        "Semantic graph ready",
        Some(file_paths.len()),
        Some(file_paths.len()),
    );
    Ok(SemGraphSnapshot {
        api_version: SEM_EMBEDDED_API_VERSION.to_string(),
        cache_key: repo_graph_cache_key(root, file_paths, options, &entities),
        file_paths: sorted_strings(file_paths),
        graph,
        entities,
        warnings: Vec::new(),
    })
}

pub fn build_memory_graph(
    files: &[SemFileInput],
    options: &SemEmbeddedOptions,
) -> SemResult<SemGraphSnapshot> {
    build_memory_graph_with_execution(files, options, &SemExecutionContext::default())
}

pub fn build_memory_graph_with_execution(
    files: &[SemFileInput],
    options: &SemEmbeddedOptions,
    execution: &SemExecutionContext<'_>,
) -> SemResult<SemGraphSnapshot> {
    ensure_not_cancelled(execution)?;
    validate_memory_files(files)?;
    let temp_root = temp_graph_root(files, options);
    if temp_root.exists() {
        std::fs::remove_dir_all(&temp_root)?;
    }
    std::fs::create_dir_all(&temp_root)?;
    let build_result = (|| -> SemResult<SemGraphSnapshot> {
        for (index, file) in files.iter().enumerate() {
            ensure_not_cancelled(execution)?;
            let path = temp_root.join(&file.file_path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, &file.content)?;
            report_progress(
                execution,
                "graphOverlay",
                "Wrote in-memory graph overlay",
                Some(index + 1),
                Some(files.len()),
            );
        }
        let file_paths = files
            .iter()
            .map(|file| file.file_path.clone())
            .collect::<Vec<_>>();
        let mut snapshot =
            build_repo_graph_with_execution(&temp_root, &file_paths, options, execution)?;
        snapshot.cache_key = memory_graph_cache_key(files, options);
        Ok(snapshot)
    })();
    let _ = std::fs::remove_dir_all(&temp_root);
    build_result
}

pub fn build_impact_context_from_graph(
    snapshot: &SemGraphSnapshot,
    target: &SemEntityTarget,
    request: &SemImpactRequest,
) -> SemResult<SemImpactContext> {
    let entity = resolve_target_entity_result(&snapshot.graph, target)?;
    Ok(impact_context_from_parts(
        &snapshot.graph,
        &snapshot.entities,
        snapshot.file_paths.as_slice(),
        entity,
        target,
        request,
        snapshot.cache_key.as_str(),
    ))
}

/// Build impact and review context from a working tree without using the CLI cache.
pub fn build_repo_impact_context(
    root: &Path,
    file_paths: &[String],
    target: &SemEntityTarget,
    request: &SemImpactRequest,
    options: &SemEmbeddedOptions,
) -> Result<SemImpactContext, String> {
    build_repo_impact_context_result(root, file_paths, target, request, options)
        .map_err(|error| error.to_string())
}

pub fn build_repo_impact_context_result(
    root: &Path,
    file_paths: &[String],
    target: &SemEntityTarget,
    request: &SemImpactRequest,
    options: &SemEmbeddedOptions,
) -> SemResult<SemImpactContext> {
    build_repo_impact_context_with_execution(
        root,
        file_paths,
        target,
        request,
        options,
        &SemExecutionContext::default(),
    )
}

pub fn build_repo_impact_context_with_execution(
    root: &Path,
    file_paths: &[String],
    target: &SemEntityTarget,
    request: &SemImpactRequest,
    options: &SemEmbeddedOptions,
    execution: &SemExecutionContext<'_>,
) -> SemResult<SemImpactContext> {
    let snapshot = build_repo_graph_with_execution(root, file_paths, options, execution)?;
    build_impact_context_from_graph(&snapshot, target, request)
}

pub fn resolve_focus_target(
    changes: &[SemFileChange],
    target: &SemFocusTarget,
    options: &SemEmbeddedOptions,
) -> SemResult<SemFocusResolution> {
    resolve_focus_target_with_execution(changes, target, options, &SemExecutionContext::default())
}

pub fn resolve_focus_target_with_execution(
    changes: &[SemFileChange],
    target: &SemFocusTarget,
    options: &SemEmbeddedOptions,
    execution: &SemExecutionContext<'_>,
) -> SemResult<SemFocusResolution> {
    ensure_not_cancelled(execution)?;
    let registry = registry_for_options(options, None);
    let extracted = extract_change_entities(changes, &registry);
    let analysis = analyze_file_changes(changes, options);
    let overlapping_entities = focus_entities(target, &extracted);
    let target_entity = target
        .entity
        .as_ref()
        .and_then(|entity_target| focus_entity_by_target(entity_target, &extracted))
        .or_else(|| deepest_focused_entity(overlapping_entities.as_slice()));
    let matching_changes = focus_matching_changes(
        analysis.changes.as_slice(),
        target,
        target_entity.as_ref(),
        overlapping_entities.as_slice(),
    );

    Ok(SemFocusResolution {
        api_version: SEM_EMBEDDED_API_VERSION.to_string(),
        cache_key: focus_cache_key(changes, target, options),
        target_entity,
        overlapping_entities,
        matching_changes,
        warnings: Vec::new(),
    })
}

pub fn build_diff_impact_context(
    root: &Path,
    repo_file_paths: &[String],
    changes: &[SemFileChange],
    target: &SemFocusTarget,
    request: &SemImpactRequest,
    options: &SemEmbeddedOptions,
) -> SemResult<SemDiffImpactContext> {
    build_diff_impact_context_with_execution(
        root,
        repo_file_paths,
        changes,
        target,
        request,
        options,
        &SemExecutionContext::default(),
    )
}

pub fn build_diff_impact_context_with_execution(
    root: &Path,
    repo_file_paths: &[String],
    changes: &[SemFileChange],
    target: &SemFocusTarget,
    request: &SemImpactRequest,
    options: &SemEmbeddedOptions,
    execution: &SemExecutionContext<'_>,
) -> SemResult<SemDiffImpactContext> {
    let focus = resolve_focus_target_with_execution(changes, target, options, execution)?;
    let snapshot = build_repo_graph_with_execution(root, repo_file_paths, options, execution)?;
    let mut warnings = Vec::new();
    let mut repo_context = None;
    let mut deleted_entity = None;
    let mut references = Vec::new();

    if let Some(entity) = focus.target_entity.as_ref() {
        let entity_target = SemEntityTarget {
            entity_id: Some(entity.entity.id.clone()),
            entity_name: Some(entity.entity.name.clone()),
            file_path: Some(entity.entity.file_path.clone()),
        };
        match build_impact_context_from_graph(&snapshot, &entity_target, request) {
            Ok(context) => repo_context = Some(context),
            Err(error) if entity.side == SemSide::Before => {
                deleted_entity = Some(entity.clone());
                references = references_to_entity_name(&snapshot, &entity.entity.name, request);
                warnings.push(format!(
                    "Target entity was resolved on the before side; current graph lookup failed: {}",
                    error.message
                ));
            }
            Err(error) => return Err(error),
        }
    } else {
        warnings.push("No focus entity resolved for diff impact context.".to_string());
    }

    Ok(SemDiffImpactContext {
        api_version: SEM_EMBEDDED_API_VERSION.to_string(),
        cache_key: diff_impact_cache_key(
            repo_file_paths,
            changes,
            target,
            request,
            options,
            snapshot.cache_key.as_str(),
        ),
        focus,
        repo_context,
        deleted_entity,
        references,
        warnings,
    })
}

pub fn generate_review_layers(
    changes: &[SemFileChange],
    layer_options: &SemLayerGenerationOptions,
    options: &SemEmbeddedOptions,
) -> SemReviewLayerPlan {
    let analysis = analyze_file_changes(changes, options);
    review_layers_from_analysis(&analysis, layer_options, options)
}

pub fn generate_review_layers_for_atoms(
    changes: &[SemFileChange],
    atoms: &[SemReviewAtom],
    layer_options: &SemLayerGenerationOptions,
    options: &SemEmbeddedOptions,
) -> SemReviewLayerPlan {
    let analysis = analyze_file_changes(changes, options);
    let mut plan = review_layers_from_analysis(&analysis, layer_options, options);
    attach_atoms_to_review_layers(&mut plan, &analysis, atoms, layer_options);
    plan.cache_key = review_atom_layer_cache_key(&analysis, atoms, layer_options, options);
    plan
}

impl From<ContextEntry> for SemContextEntry {
    fn from(value: ContextEntry) -> Self {
        Self {
            entity_id: value.entity_id,
            entity_name: value.entity_name,
            entity_type: value.entity_type,
            file_path: value.file_path,
            role: value.role,
            content: value.content,
            estimated_tokens: value.estimated_tokens,
        }
    }
}

fn registry_for_options(
    options: &SemEmbeddedOptions,
    fallback_root: Option<&Path>,
) -> ParserRegistry {
    let mut registry = create_default_registry();
    let root = options.root.as_deref().or(fallback_root);
    if let Some(root) = root {
        if options.load_semrc {
            registry.load_semrc(root);
        }
        if options.load_gitattributes {
            registry.load_gitattributes(root);
        }
    }
    registry
}

fn to_file_change(change: &SemFileChange) -> FileChange {
    FileChange {
        file_path: change.file_path.clone(),
        status: change.status.clone(),
        old_file_path: change.old_file_path.clone(),
        before_content: change.before_content.clone(),
        after_content: change.after_content.clone(),
    }
}

fn diff_summary(result: &DiffResult) -> SemDiffSummary {
    SemDiffSummary {
        file_count: result.file_count,
        added_count: result.added_count,
        modified_count: result.modified_count,
        deleted_count: result.deleted_count,
        moved_count: result.moved_count,
        renamed_count: result.renamed_count,
        reordered_count: result.reordered_count,
        orphan_count: result.orphan_count,
    }
}

struct ExtractedChangeEntities {
    before: Vec<SemanticEntity>,
    after: Vec<SemanticEntity>,
    before_by_id: HashMap<String, usize>,
    after_by_id: HashMap<String, usize>,
}

fn extract_change_entities(
    changes: &[SemFileChange],
    registry: &ParserRegistry,
) -> ExtractedChangeEntities {
    let mut before = Vec::new();
    let mut after = Vec::new();

    for change in changes {
        if let Some(content) = change.before_content.as_deref() {
            let before_path = change
                .old_file_path
                .as_deref()
                .unwrap_or(change.file_path.as_str());
            before.extend(registry.extract_entities(before_path, content));
        }
        if let Some(content) = change.after_content.as_deref() {
            after.extend(registry.extract_entities(&change.file_path, content));
        }
    }

    let before_by_id = before
        .iter()
        .enumerate()
        .map(|(idx, entity)| (entity.id.clone(), idx))
        .collect();
    let after_by_id = after
        .iter()
        .enumerate()
        .map(|(idx, entity)| (entity.id.clone(), idx))
        .collect();

    ExtractedChangeEntities {
        before,
        after,
        before_by_id,
        after_by_id,
    }
}

fn enrich_change(
    change: SemanticChange,
    file_changes: &[SemFileChange],
    extracted: &ExtractedChangeEntities,
) -> SemEmbeddedChange {
    let before_name = change
        .old_entity_name
        .as_deref()
        .unwrap_or(change.entity_name.as_str());
    let before_path = change
        .old_file_path
        .as_deref()
        .unwrap_or(change.file_path.as_str());

    let before_entity = entity_by_id_or_content(
        &extracted.before,
        &extracted.before_by_id,
        &change.entity_id,
        before_path,
        &change.entity_type,
        before_name,
        change.before_content.as_deref(),
    );
    let after_entity = entity_by_id_or_content(
        &extracted.after,
        &extracted.after_by_id,
        &change.entity_id,
        &change.file_path,
        &change.entity_type,
        &change.entity_name,
        change.after_content.as_deref(),
    );

    let before_range = before_entity.map(entity_range);
    let after_range = after_entity.map(entity_range);
    let hunk_overlaps = hunk_overlaps_for_change(file_changes, &before_range, &after_range);

    SemEmbeddedChange {
        change,
        before_range,
        after_range,
        hunk_overlaps,
    }
}

fn entity_by_id_or_content<'a>(
    entities: &'a [SemanticEntity],
    by_id: &HashMap<String, usize>,
    entity_id: &str,
    file_path: &str,
    entity_type: &str,
    entity_name: &str,
    content: Option<&str>,
) -> Option<&'a SemanticEntity> {
    if let Some(entity) = by_id.get(entity_id).and_then(|idx| entities.get(*idx)) {
        return Some(entity);
    }

    let content = content?;
    entities
        .iter()
        .find(|entity| {
            entity.file_path == file_path
                && entity.entity_type == entity_type
                && entity.name == entity_name
                && entity.content == content
        })
        .or_else(|| {
            entities
                .iter()
                .find(|entity| entity.file_path == file_path && entity.content == content)
        })
        .or_else(|| entities.iter().find(|entity| entity.content == content))
}

fn entity_range(entity: &SemanticEntity) -> SemEntityRange {
    SemEntityRange {
        file_path: entity.file_path.clone(),
        start_line: entity.start_line,
        end_line: entity.end_line,
    }
}

fn hunk_overlaps_for_change(
    file_changes: &[SemFileChange],
    before_range: &Option<SemEntityRange>,
    after_range: &Option<SemEntityRange>,
) -> Vec<SemHunkOverlap> {
    let mut overlaps = Vec::new();
    for file_change in file_changes {
        let old_file_path = file_change
            .old_file_path
            .as_deref()
            .unwrap_or(file_change.file_path.as_str());

        let before_matches = before_range
            .as_ref()
            .is_some_and(|range| range.file_path == old_file_path);
        let after_matches = after_range
            .as_ref()
            .is_some_and(|range| range.file_path == file_change.file_path);

        if !before_matches && !after_matches {
            continue;
        }

        for hunk in &file_change.hunks {
            let overlaps_before = before_matches
                && before_range.as_ref().is_some_and(|range| {
                    hunk.old_range
                        .as_ref()
                        .is_some_and(|hunk_range| ranges_overlap(range, hunk_range))
                });
            let overlaps_after = after_matches
                && after_range.as_ref().is_some_and(|range| {
                    hunk.new_range
                        .as_ref()
                        .is_some_and(|hunk_range| ranges_overlap(range, hunk_range))
                });

            if overlaps_before || overlaps_after {
                overlaps.push(SemHunkOverlap {
                    hunk_id: hunk.hunk_id.clone(),
                    hunk_index: hunk.hunk_index,
                    hunk_header: hunk.hunk_header.clone(),
                    file_path: file_change.file_path.clone(),
                    old_file_path: file_change.old_file_path.clone(),
                    old_range: hunk.old_range,
                    new_range: hunk.new_range,
                    overlaps_before,
                    overlaps_after,
                });
            }
        }
    }
    overlaps
}

fn ranges_overlap(entity: &SemEntityRange, hunk: &SemLineRange) -> bool {
    entity.start_line > 0
        && entity.end_line > 0
        && hunk.start_line > 0
        && hunk.end_line > 0
        && entity.start_line <= hunk.end_line
        && hunk.start_line <= entity.end_line
}

fn resolve_target_entity_result(
    graph: &EntityGraph,
    target: &SemEntityTarget,
) -> SemResult<EntityInfo> {
    if let Some(entity_id) = target.entity_id.as_deref() {
        return graph.entities.get(entity_id).cloned().ok_or_else(|| {
            SemError::new(
                SemErrorKind::NotFound,
                format!("entity id not found: {entity_id}"),
            )
        });
    }

    let Some(entity_name) = target.entity_name.as_deref() else {
        return Err(SemError::new(
            SemErrorKind::InvalidInput,
            "target must include entity_id or entity_name",
        ));
    };

    let mut matches: Vec<&EntityInfo> = graph
        .entities
        .values()
        .filter(|entity| entity.name == entity_name)
        .filter(|entity| {
            target
                .file_path
                .as_deref()
                .is_none_or(|file_path| entity.file_path == file_path)
        })
        .collect();
    matches.sort_by(|a, b| a.file_path.cmp(&b.file_path).then(a.id.cmp(&b.id)));

    match matches.as_slice() {
        [entity] => Ok((*entity).clone()),
        [] => Err(SemError::new(
            SemErrorKind::NotFound,
            format!("entity name not found: {entity_name}"),
        )),
        many => Err(SemError::new(
            SemErrorKind::AmbiguousTarget,
            format!(
                "entity name is ambiguous: {entity_name} matched {} entities",
                many.len()
            ),
        )),
    }
}

fn impact_context_from_parts(
    graph: &EntityGraph,
    all_entities: &[SemanticEntity],
    file_paths: &[String],
    entity: EntityInfo,
    target: &SemEntityTarget,
    request: &SemImpactRequest,
    graph_cache_key: &str,
) -> SemImpactContext {
    let dependencies = graph
        .get_dependencies(&entity.id)
        .into_iter()
        .cloned()
        .collect();
    let dependents = graph
        .get_dependents(&entity.id)
        .into_iter()
        .cloned()
        .collect();
    let impact = graph
        .impact_analysis_bounded(&entity.id, request.max_depth)
        .into_iter()
        .map(|(entity, depth)| SemImpactEntity {
            entity: entity.clone(),
            depth,
        })
        .collect();
    let tests = graph
        .test_impact(&entity.id, all_entities)
        .into_iter()
        .cloned()
        .collect();
    let context = build_context(graph, &entity.id, all_entities, request.token_budget)
        .into_iter()
        .map(SemContextEntry::from)
        .collect();

    SemImpactContext {
        api_version: SEM_EMBEDDED_API_VERSION.to_string(),
        cache_key: stable_cache_key(
            "impact",
            &[
                graph_cache_key.to_string(),
                impact_request_fingerprint(file_paths, target, request),
            ],
        ),
        entity,
        dependencies,
        dependents,
        impact,
        tests,
        context,
    }
}

fn focus_entities(
    target: &SemFocusTarget,
    extracted: &ExtractedChangeEntities,
) -> Vec<SemFocusedEntity> {
    let mut entities = Vec::new();
    if let Some(location) = target.location.as_ref() {
        entities.extend(entities_at_location(extracted, location));
    }
    if let Some(hunk) = target.hunk.as_ref() {
        entities.extend(entities_for_hunk_target(extracted, hunk));
    }
    entities.sort_by(|a, b| {
        a.entity
            .file_path
            .cmp(&b.entity.file_path)
            .then(a.entity.start_line.cmp(&b.entity.start_line))
            .then(a.entity.end_line.cmp(&b.entity.end_line))
            .then(a.entity.id.cmp(&b.entity.id))
    });
    entities.dedup_by(|a, b| a.side == b.side && a.entity.id == b.entity.id);
    entities
}

fn entities_at_location(
    extracted: &ExtractedChangeEntities,
    location: &SemLocationTarget,
) -> Vec<SemFocusedEntity> {
    side_entities(extracted, location.side)
        .iter()
        .filter(|entity| {
            entity.file_path == location.file_path
                && entity.start_line <= location.line
                && location.line <= entity.end_line
        })
        .cloned()
        .map(|entity| focused_entity(entity, location.side))
        .collect()
}

fn entities_for_hunk_target(
    extracted: &ExtractedChangeEntities,
    hunk: &SemHunkTarget,
) -> Vec<SemFocusedEntity> {
    let (file_path, range) = match hunk.side {
        SemSide::Before => (
            hunk.old_file_path
                .as_deref()
                .unwrap_or(hunk.file_path.as_str()),
            hunk.old_range.as_ref(),
        ),
        SemSide::After => (hunk.file_path.as_str(), hunk.new_range.as_ref()),
    };
    let Some(range) = range else {
        return Vec::new();
    };
    side_entities(extracted, hunk.side)
        .iter()
        .filter(|entity| entity.file_path == file_path)
        .filter(|entity| entity_range_overlaps_line_range(entity, range))
        .cloned()
        .map(|entity| focused_entity(entity, hunk.side))
        .collect()
}

fn focus_entity_by_target(
    target: &SemEntityTarget,
    extracted: &ExtractedChangeEntities,
) -> Option<SemFocusedEntity> {
    [SemSide::After, SemSide::Before]
        .into_iter()
        .find_map(|side| {
            side_entities(extracted, side)
                .iter()
                .find(|entity| entity_matches_target(entity, target))
                .cloned()
                .map(|entity| focused_entity(entity, side))
        })
}

fn deepest_focused_entity(entities: &[SemFocusedEntity]) -> Option<SemFocusedEntity> {
    entities
        .iter()
        .max_by_key(|entity| {
            entity
                .entity
                .end_line
                .saturating_sub(entity.entity.start_line)
        })
        .cloned()
}

fn focus_matching_changes(
    changes: &[SemEmbeddedChange],
    target: &SemFocusTarget,
    target_entity: Option<&SemFocusedEntity>,
    overlapping_entities: &[SemFocusedEntity],
) -> Vec<SemEmbeddedChange> {
    let entity_ids = overlapping_entities
        .iter()
        .map(|entity| entity.entity.id.as_str())
        .chain(target_entity.iter().map(|entity| entity.entity.id.as_str()))
        .collect::<BTreeSet<_>>();
    changes
        .iter()
        .filter(|change| {
            if entity_ids.contains(change.change.entity_id.as_str()) {
                return true;
            }
            if let Some(entity) = target_entity {
                if change.change.entity_name == entity.entity.name {
                    return true;
                }
            }
            if let Some(hunk) = target.hunk.as_ref() {
                return change.hunk_overlaps.iter().any(|overlap| {
                    overlap.file_path == hunk.file_path
                        && hunk
                            .hunk_index
                            .map(|index| overlap.hunk_index == index)
                            .unwrap_or(true)
                        && hunk
                            .hunk_header
                            .as_deref()
                            .map(|header| overlap.hunk_header.as_deref() == Some(header))
                            .unwrap_or(true)
                });
            }
            false
        })
        .cloned()
        .collect()
}

fn references_to_entity_name(
    snapshot: &SemGraphSnapshot,
    entity_name: &str,
    request: &SemImpactRequest,
) -> Vec<SemEntityReference> {
    let mut refs = snapshot
        .entities
        .iter()
        .filter(|entity| entity.name != entity_name)
        .filter(|entity| contains_identifier(&entity.content, entity_name))
        .filter_map(|entity| {
            snapshot
                .graph
                .entities
                .get(&entity.id)
                .map(|info| SemEntityReference {
                    entity: info.clone(),
                    snippet: first_matching_line(&entity.content, entity_name),
                })
        })
        .collect::<Vec<_>>();
    refs.sort_by(|a, b| {
        a.entity
            .file_path
            .cmp(&b.entity.file_path)
            .then(a.entity.id.cmp(&b.entity.id))
    });
    refs.truncate(request.max_depth.max(1) * 32);
    refs
}

fn review_layers_from_analysis(
    analysis: &SemDiffAnalysis,
    layer_options: &SemLayerGenerationOptions,
    options: &SemEmbeddedOptions,
) -> SemReviewLayerPlan {
    let mut groups = BTreeMap::<String, Vec<usize>>::new();
    let mut manual = Vec::new();
    for (index, change) in analysis.changes.iter().enumerate() {
        if change.change.entity_type == "orphan" {
            manual.push(index);
            continue;
        }
        groups
            .entry(layer_key_for_change(change))
            .or_default()
            .push(index);
    }

    let mut keyed = groups.into_iter().collect::<Vec<_>>();
    keyed.sort_by(|(a, _), (b, _)| layer_order(a).cmp(&layer_order(b)).then_with(|| a.cmp(b)));

    let mut layers = Vec::new();
    for (key, mut indices) in keyed {
        while !indices.is_empty() && layers.len() < layer_options.max_layers {
            let take = layer_options
                .max_changes_per_layer
                .max(1)
                .min(indices.len());
            let chunk = indices.drain(0..take).collect::<Vec<_>>();
            layers.push(layer_from_change_indices(
                layers.len(),
                &key,
                chunk,
                analysis.changes.as_slice(),
            ));
        }
        manual.extend(indices);
    }

    let mut previous_id: Option<String> = None;
    for layer in &mut layers {
        if let Some(previous) = previous_id.clone() {
            if layer_order_key(&layer.title) >= 90 {
                layer.depends_on_layer_ids.push(previous);
            }
        }
        previous_id = Some(layer.id.clone());
    }

    SemReviewLayerPlan {
        api_version: SEM_EMBEDDED_API_VERSION.to_string(),
        cache_key: review_layer_cache_key(analysis, layer_options, options),
        layers,
        manual_review_change_indices: manual,
        manual_review_atom_ids: Vec::new(),
        warnings: Vec::new(),
    }
}

fn attach_atoms_to_review_layers(
    plan: &mut SemReviewLayerPlan,
    analysis: &SemDiffAnalysis,
    atoms: &[SemReviewAtom],
    layer_options: &SemLayerGenerationOptions,
) {
    let mut assigned_atom_ids = BTreeSet::<String>::new();
    let manual_atom_ids = atoms
        .iter()
        .filter(|atom| atom.manual_review)
        .map(|atom| atom.atom_id.clone())
        .collect::<Vec<_>>();
    plan.manual_review_atom_ids = manual_atom_ids.clone();

    for layer in &mut plan.layers {
        let layer_changes = layer
            .change_indices
            .iter()
            .filter_map(|index| analysis.changes.get(*index))
            .collect::<Vec<_>>();
        let mut atom_ids = atoms
            .iter()
            .filter(|atom| !atom.manual_review)
            .filter(|atom| review_layer_matches_atom(layer, &layer_changes, atom))
            .map(|atom| atom.atom_id.clone())
            .collect::<BTreeSet<_>>();
        for atom_id in &layer.atom_ids {
            atom_ids.insert(atom_id.clone());
        }
        layer.atom_ids = atom_ids.into_iter().collect();
        assigned_atom_ids.extend(layer.atom_ids.iter().cloned());
    }

    assign_file_matched_atoms(plan, atoms, &assigned_atom_ids);
    assigned_atom_ids.extend(
        plan.layers
            .iter()
            .flat_map(|layer| layer.atom_ids.iter().cloned()),
    );

    let remaining = atoms
        .iter()
        .enumerate()
        .filter(|(_, atom)| !atom.manual_review && !assigned_atom_ids.contains(&atom.atom_id))
        .map(|(index, atom)| (atom_layer_key(atom), index))
        .collect::<Vec<_>>();
    let mut grouped = BTreeMap::<String, Vec<usize>>::new();
    for (key, index) in remaining {
        grouped.entry(key).or_default().push(index);
    }

    for (key, mut atom_indices) in grouped {
        while !atom_indices.is_empty() && plan.layers.len() < layer_options.max_layers {
            let take = layer_options
                .max_changes_per_layer
                .max(1)
                .min(atom_indices.len());
            let chunk = atom_indices.drain(0..take).collect::<Vec<_>>();
            plan.layers.push(layer_from_atom_indices(
                plan.layers.len(),
                &key,
                chunk,
                atoms,
            ));
        }
        for index in atom_indices {
            plan.manual_review_atom_ids
                .push(atoms[index].atom_id.clone());
        }
    }

    for layer in &mut plan.layers {
        layer.atom_ids.sort();
        layer.atom_ids.dedup();
    }
    plan.manual_review_atom_ids.sort();
    plan.manual_review_atom_ids.dedup();
    add_atom_layer_dependencies(&mut plan.layers, atoms);
}

fn assign_file_matched_atoms(
    plan: &mut SemReviewLayerPlan,
    atoms: &[SemReviewAtom],
    assigned_atom_ids: &BTreeSet<String>,
) {
    for atom in atoms {
        if atom.manual_review || assigned_atom_ids.contains(&atom.atom_id) {
            continue;
        }
        if let Some(layer) = plan
            .layers
            .iter_mut()
            .find(|layer| layer_file_paths_match_atom(layer, atom))
        {
            layer.atom_ids.push(atom.atom_id.clone());
        }
    }
}

fn layer_from_change_indices(
    index: usize,
    key: &str,
    change_indices: Vec<usize>,
    changes: &[SemEmbeddedChange],
) -> SemReviewLayer {
    let mut file_paths = BTreeSet::new();
    let mut entity_names = BTreeSet::new();
    let mut hunk_indices = BTreeSet::new();
    let mut additions = 0usize;
    let mut deletions = 0usize;
    for change_index in &change_indices {
        let change = &changes[*change_index];
        file_paths.insert(change.change.file_path.clone());
        entity_names.insert(change.change.entity_name.clone());
        for overlap in &change.hunk_overlaps {
            hunk_indices.insert(overlap.hunk_index);
        }
        match change.change.change_type {
            crate::model::change::ChangeType::Added => additions += 1,
            crate::model::change::ChangeType::Deleted => deletions += 1,
            _ => {}
        }
    }
    let title = layer_title(key, &entity_names, &file_paths);
    let id = format!(
        "sem-layer-{}-{}",
        index,
        stable_cache_key("layer", &[title.clone()])
    );
    SemReviewLayer {
        id,
        index,
        title: title.clone(),
        summary: format!(
            "{} semantic change{} across {} file{}.",
            change_indices.len(),
            if change_indices.len() == 1 { "" } else { "s" },
            file_paths.len(),
            if file_paths.len() == 1 { "" } else { "s" }
        ),
        rationale: format!(
            "Grouped by semantic role `{key}` with {} added and {} deleted entities.",
            additions, deletions
        ),
        depends_on_layer_ids: Vec::new(),
        change_indices,
        atom_ids: Vec::new(),
        file_paths: file_paths.into_iter().collect(),
        hunk_indices: hunk_indices.into_iter().collect(),
        entity_names: entity_names.into_iter().collect(),
    }
}

fn layer_from_atom_indices(
    index: usize,
    key: &str,
    atom_indices: Vec<usize>,
    atoms: &[SemReviewAtom],
) -> SemReviewLayer {
    let mut file_paths = BTreeSet::new();
    let mut hunk_indices = BTreeSet::new();
    let mut entity_names = BTreeSet::new();
    let mut changed_lines = 0usize;
    let mut atom_ids = Vec::new();
    for atom_index in atom_indices {
        let atom = &atoms[atom_index];
        atom_ids.push(atom.atom_id.clone());
        file_paths.insert(atom.file_path.clone());
        if let Some(path) = atom.old_file_path.clone() {
            file_paths.insert(path);
        }
        hunk_indices.extend(atom.hunk_indices.iter().copied());
        if let Some(symbol) = atom.symbol_name.clone() {
            entity_names.insert(symbol);
        }
        entity_names.extend(atom.defined_symbols.iter().cloned());
        changed_lines += atom.changed_lines;
    }

    let title = atom_layer_title(key, &entity_names, &file_paths);
    let id = format!(
        "sem-atom-layer-{}-{}",
        index,
        stable_cache_key("atom-layer", &[title.clone(), atom_ids.join("\n")])
    );
    SemReviewLayer {
        id,
        index,
        title: title.clone(),
        summary: format!(
            "{} review atom{} across {} file{}.",
            atom_ids.len(),
            if atom_ids.len() == 1 { "" } else { "s" },
            file_paths.len(),
            if file_paths.len() == 1 { "" } else { "s" }
        ),
        rationale: format!(
            "Grouped by host-provided review atom role `{key}` with {changed_lines} changed line{}.",
            if changed_lines == 1 { "" } else { "s" }
        ),
        depends_on_layer_ids: Vec::new(),
        change_indices: Vec::new(),
        atom_ids,
        file_paths: file_paths.into_iter().collect(),
        hunk_indices: hunk_indices.into_iter().collect(),
        entity_names: entity_names.into_iter().collect(),
    }
}

fn review_layer_matches_atom(
    layer: &SemReviewLayer,
    changes: &[&SemEmbeddedChange],
    atom: &SemReviewAtom,
) -> bool {
    changes
        .iter()
        .any(|change| embedded_change_matches_atom(change, atom))
        || layer_hunks_match_atom(layer, atom)
        || layer_symbols_match_atom(layer, atom)
}

fn embedded_change_matches_atom(change: &SemEmbeddedChange, atom: &SemReviewAtom) -> bool {
    if !change_paths_match_atom(change, atom) {
        return false;
    }

    change_hunks_match_atom(change, atom)
        || change_ranges_match_atom(change, atom)
        || change_symbols_match_atom(change, atom)
        || atom.hunk_indices.is_empty() && atom.old_range.is_none() && atom.new_range.is_none()
}

fn change_paths_match_atom(change: &SemEmbeddedChange, atom: &SemReviewAtom) -> bool {
    let mut paths = BTreeSet::new();
    paths.insert(change.change.file_path.as_str());
    if let Some(path) = change.change.old_file_path.as_deref() {
        paths.insert(path);
    }
    if let Some(range) = change.before_range.as_ref() {
        paths.insert(range.file_path.as_str());
    }
    if let Some(range) = change.after_range.as_ref() {
        paths.insert(range.file_path.as_str());
    }
    paths.iter().any(|path| atom_path_matches(atom, path))
}

fn change_hunks_match_atom(change: &SemEmbeddedChange, atom: &SemReviewAtom) -> bool {
    !atom.hunk_indices.is_empty()
        && change
            .hunk_overlaps
            .iter()
            .any(|overlap| atom.hunk_indices.contains(&overlap.hunk_index))
}

fn change_ranges_match_atom(change: &SemEmbeddedChange, atom: &SemReviewAtom) -> bool {
    range_overlaps_entity(atom.old_range, change.before_range.as_ref())
        || range_overlaps_entity(atom.new_range, change.after_range.as_ref())
}

fn change_symbols_match_atom(change: &SemEmbeddedChange, atom: &SemReviewAtom) -> bool {
    let mut change_symbols = BTreeSet::new();
    change_symbols.insert(change.change.entity_name.as_str());
    if let Some(name) = change.change.old_entity_name.as_deref() {
        change_symbols.insert(name);
    }

    atom.symbol_name
        .as_deref()
        .is_some_and(|symbol| change_symbols.contains(symbol))
        || atom
            .defined_symbols
            .iter()
            .chain(atom.referenced_symbols.iter())
            .any(|symbol| change_symbols.contains(symbol.as_str()))
}

fn layer_hunks_match_atom(layer: &SemReviewLayer, atom: &SemReviewAtom) -> bool {
    layer_file_paths_match_atom(layer, atom)
        && !atom.hunk_indices.is_empty()
        && atom
            .hunk_indices
            .iter()
            .any(|hunk_index| layer.hunk_indices.contains(hunk_index))
}

fn layer_symbols_match_atom(layer: &SemReviewLayer, atom: &SemReviewAtom) -> bool {
    layer_file_paths_match_atom(layer, atom)
        && (atom
            .symbol_name
            .as_ref()
            .is_some_and(|symbol| layer.entity_names.contains(symbol))
            || atom
                .defined_symbols
                .iter()
                .chain(atom.referenced_symbols.iter())
                .any(|symbol| layer.entity_names.contains(symbol)))
}

fn layer_file_paths_match_atom(layer: &SemReviewLayer, atom: &SemReviewAtom) -> bool {
    layer
        .file_paths
        .iter()
        .any(|path| atom_path_matches(atom, path))
}

fn atom_path_matches(atom: &SemReviewAtom, path: &str) -> bool {
    atom.file_path == path || atom.old_file_path.as_deref() == Some(path)
}

fn range_overlaps_entity(
    atom_range: Option<SemLineRange>,
    entity_range: Option<&SemEntityRange>,
) -> bool {
    let Some(atom_range) = atom_range else {
        return false;
    };
    let Some(entity_range) = entity_range else {
        return false;
    };
    atom_range.start_line <= entity_range.end_line && entity_range.start_line <= atom_range.end_line
}

fn add_atom_layer_dependencies(layers: &mut [SemReviewLayer], atoms: &[SemReviewAtom]) {
    let mut atom_to_layer = HashMap::<String, String>::new();
    for layer in layers.iter() {
        for atom_id in &layer.atom_ids {
            atom_to_layer.insert(atom_id.clone(), layer.id.clone());
        }
    }

    let mut symbol_owner = BTreeMap::<String, String>::new();
    let mut atom_by_id = HashMap::<String, &SemReviewAtom>::new();
    for atom in atoms {
        atom_by_id.insert(atom.atom_id.clone(), atom);
        let Some(layer_id) = atom_to_layer.get(&atom.atom_id) else {
            continue;
        };
        for symbol in atom
            .symbol_name
            .iter()
            .chain(atom.defined_symbols.iter())
            .filter(|symbol| !symbol.is_empty())
        {
            symbol_owner
                .entry(symbol.clone())
                .or_insert_with(|| layer_id.clone());
        }
    }

    for layer in layers {
        let mut depends_on = BTreeSet::<String>::new();
        for atom_id in &layer.atom_ids {
            let Some(atom) = atom_by_id.get(atom_id) else {
                continue;
            };
            if atom.role.as_deref().is_some_and(is_test_role) {
                for owner_id in symbol_owner.values() {
                    if owner_id != &layer.id {
                        depends_on.insert(owner_id.clone());
                    }
                }
            }
            for symbol in &atom.referenced_symbols {
                if let Some(owner_id) = symbol_owner.get(symbol) {
                    if owner_id != &layer.id {
                        depends_on.insert(owner_id.clone());
                    }
                }
            }
        }
        layer.depends_on_layer_ids.extend(depends_on);
        layer.depends_on_layer_ids.sort();
        layer.depends_on_layer_ids.dedup();
    }
}

fn atom_layer_key(atom: &SemReviewAtom) -> String {
    if atom.role.as_deref().is_some_and(is_test_role) || path_is_test(&atom.file_path) {
        return "tests".to_string();
    }
    if atom.role.as_deref().is_some_and(is_config_role) || path_is_config(&atom.file_path) {
        return "config".to_string();
    }
    if atom.role.as_deref().is_some_and(is_foundation_role) {
        return "foundation".to_string();
    }
    let directory = atom
        .file_path
        .rsplit_once('/')
        .map(|(directory, _)| directory)
        .unwrap_or(".");
    format!("code:{directory}")
}

fn atom_layer_title(
    key: &str,
    entity_names: &BTreeSet<String>,
    file_paths: &BTreeSet<String>,
) -> String {
    layer_title(key, entity_names, file_paths)
}

fn is_test_role(role: &str) -> bool {
    role.eq_ignore_ascii_case("tests") || role.eq_ignore_ascii_case("test")
}

fn is_config_role(role: &str) -> bool {
    role.eq_ignore_ascii_case("config")
}

fn is_foundation_role(role: &str) -> bool {
    matches!(
        role.to_ascii_lowercase().as_str(),
        "foundation" | "model" | "schema" | "type"
    )
}

fn path_is_test(path: &str) -> bool {
    let path = path.to_ascii_lowercase();
    path.contains("/test/")
        || path.contains("/tests/")
        || path.contains(".test.")
        || path.contains(".spec.")
        || path.ends_with("_test.rs")
}

fn path_is_config(path: &str) -> bool {
    let path = path.to_ascii_lowercase();
    path.ends_with(".toml")
        || path.ends_with(".yaml")
        || path.ends_with(".yml")
        || path.ends_with(".json")
        || path.ends_with(".lock")
}

fn diff_cache_key(changes: &[SemFileChange], options: &SemEmbeddedOptions) -> String {
    let mut parts = vec![options_fingerprint(options)];
    for change in changes {
        parts.push(format!(
            "{}:{}:{}:{}:{}",
            change.file_path,
            file_status_label(&change.status),
            change.old_file_path.as_deref().unwrap_or(""),
            change
                .before_content
                .as_deref()
                .map(content_hash)
                .unwrap_or_default(),
            change
                .after_content
                .as_deref()
                .map(content_hash)
                .unwrap_or_default()
        ));
        for hunk in &change.hunks {
            parts.push(format!(
                "hunk:{}:{}:{:?}:{:?}:{:?}",
                hunk.hunk_index,
                hunk.hunk_id.as_deref().unwrap_or(""),
                hunk.hunk_header,
                hunk.old_range,
                hunk.new_range
            ));
        }
    }
    stable_cache_key("diff", &parts)
}

fn entity_index_cache_key(files: &[SemFileInput], options: &SemEmbeddedOptions) -> String {
    let mut parts = vec![options_fingerprint(options)];
    for file in files {
        parts.push(format!(
            "{}:{}",
            file.file_path,
            content_hash(&file.content)
        ));
    }
    stable_cache_key("entities", &parts)
}

fn options_fingerprint(options: &SemEmbeddedOptions) -> String {
    format!(
        "opts:semrc={}:gitattributes={}",
        options.load_semrc, options.load_gitattributes
    )
}

fn stable_cache_key(kind: &str, parts: &[String]) -> String {
    let mut key = String::from(SEM_EMBEDDED_API_VERSION);
    key.push('\0');
    key.push_str(kind);
    for part in parts {
        key.push('\0');
        key.push_str(part);
    }
    content_hash(&key)
}

fn file_status_label(status: &FileStatus) -> &'static str {
    match status {
        FileStatus::Added => "added",
        FileStatus::Modified => "modified",
        FileStatus::Deleted => "deleted",
        FileStatus::Renamed => "renamed",
    }
}

fn sem_error_kind_label(kind: SemErrorKind) -> &'static str {
    match kind {
        SemErrorKind::Cancelled => "cancelled",
        SemErrorKind::InvalidInput => "invalid input",
        SemErrorKind::Io => "io",
        SemErrorKind::NotFound => "not found",
        SemErrorKind::AmbiguousTarget => "ambiguous target",
        SemErrorKind::Unsupported => "unsupported",
        SemErrorKind::Internal => "internal",
    }
}

fn ensure_not_cancelled(execution: &SemExecutionContext<'_>) -> SemResult<()> {
    if execution
        .cancellation
        .map(|token| token.is_cancelled())
        .unwrap_or(false)
    {
        return Err(SemError::new(
            SemErrorKind::Cancelled,
            "operation cancelled",
        ));
    }
    Ok(())
}

fn report_progress(
    execution: &SemExecutionContext<'_>,
    stage: impl Into<String>,
    message: impl Into<String>,
    completed: Option<usize>,
    total: Option<usize>,
) {
    if let Some(reporter) = execution.progress {
        reporter.report(SemProgressEvent {
            stage: stage.into(),
            message: message.into(),
            completed,
            total,
        });
    }
}

fn side_entities(extracted: &ExtractedChangeEntities, side: SemSide) -> &[SemanticEntity] {
    match side {
        SemSide::Before => extracted.before.as_slice(),
        SemSide::After => extracted.after.as_slice(),
    }
}

fn focused_entity(entity: SemanticEntity, side: SemSide) -> SemFocusedEntity {
    SemFocusedEntity {
        range: entity_range(&entity),
        entity,
        side,
    }
}

fn entity_matches_target(entity: &SemanticEntity, target: &SemEntityTarget) -> bool {
    target
        .entity_id
        .as_deref()
        .map(|entity_id| entity.id == entity_id)
        .unwrap_or(true)
        && target
            .entity_name
            .as_deref()
            .map(|name| entity.name == name)
            .unwrap_or(true)
        && target
            .file_path
            .as_deref()
            .map(|path| entity.file_path == path)
            .unwrap_or(true)
}

fn entity_range_overlaps_line_range(entity: &SemanticEntity, range: &SemLineRange) -> bool {
    entity.start_line > 0
        && entity.end_line > 0
        && range.start_line > 0
        && range.end_line > 0
        && entity.start_line <= range.end_line
        && range.start_line <= entity.end_line
}

fn contains_identifier(content: &str, symbol: &str) -> bool {
    content
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .any(|token| token == symbol)
}

fn first_matching_line(content: &str, symbol: &str) -> Option<String> {
    content
        .lines()
        .find(|line| contains_identifier(line, symbol))
        .map(|line| line.trim().chars().take(260).collect())
}

fn relative_repo_path(root: &Path, path: &Path) -> Option<String> {
    path.strip_prefix(root)
        .ok()
        .and_then(|path| path.to_str())
        .map(|path| path.replace('\\', "/"))
}

fn scan_skip_dirs(options: &SemRepoScanOptions) -> BTreeSet<String> {
    [
        ".git",
        ".sem",
        "node_modules",
        "target",
        "dist",
        "build",
        "coverage",
        ".next",
        ".turbo",
        ".venv",
        "vendor",
    ]
    .into_iter()
    .map(str::to_string)
    .chain(options.extra_skip_dirs.iter().cloned())
    .collect()
}

fn should_skip_scan_dir(name: &str, include_hidden: bool, skip_dirs: &BTreeSet<String>) -> bool {
    skip_dirs.contains(name) || (!include_hidden && name.starts_with('.'))
}

fn path_has_hidden_component(path: &Path) -> bool {
    path.components().any(|component| {
        matches!(component, Component::Normal(value) if value.to_str().map(|name| name.starts_with('.')).unwrap_or(false))
    })
}

fn validate_memory_files(files: &[SemFileInput]) -> SemResult<()> {
    for file in files {
        let path = Path::new(&file.file_path);
        if path.is_absolute()
            || path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(SemError::new(
                SemErrorKind::InvalidInput,
                format!(
                    "memory graph file path must be relative: {}",
                    file.file_path
                ),
            ));
        }
    }
    Ok(())
}

fn temp_graph_root(files: &[SemFileInput], options: &SemEmbeddedOptions) -> PathBuf {
    let key = memory_graph_cache_key(files, options);
    std::env::temp_dir().join(format!("sem-embedded-{}-{key}", std::process::id()))
}

fn sorted_strings(values: &[String]) -> Vec<String> {
    let mut sorted = values.to_vec();
    sorted.sort();
    sorted
}

fn layer_key_for_change(change: &SemEmbeddedChange) -> String {
    let path = change.change.file_path.to_ascii_lowercase();
    if path_is_test(&path) {
        return "tests".to_string();
    }
    if path_is_config(&path) {
        return "config".to_string();
    }
    if matches!(
        change.change.entity_type.as_str(),
        "struct" | "class" | "interface" | "trait" | "type" | "enum"
    ) {
        return "foundation".to_string();
    }
    let directory = change
        .change
        .file_path
        .rsplit_once('/')
        .map(|(directory, _)| directory)
        .unwrap_or(".");
    format!("code:{directory}")
}

fn layer_order(key: &str) -> usize {
    match key {
        "config" => 0,
        "foundation" => 10,
        "tests" => 90,
        _ => 40,
    }
}

fn layer_order_key(title: &str) -> usize {
    let lower = title.to_ascii_lowercase();
    if lower.contains("test") {
        90
    } else if lower.contains("config") {
        0
    } else if lower.contains("foundation") {
        10
    } else {
        40
    }
}

fn layer_title(
    key: &str,
    entity_names: &BTreeSet<String>,
    file_paths: &BTreeSet<String>,
) -> String {
    match key {
        "config" => "Update configuration".to_string(),
        "foundation" => "Update foundation types".to_string(),
        "tests" => "Update tests".to_string(),
        _ => {
            if entity_names.len() == 1 {
                format!("Update {}", entity_names.iter().next().unwrap())
            } else if file_paths.len() == 1 {
                format!("Update {}", file_paths.iter().next().unwrap())
            } else {
                "Update related code".to_string()
            }
        }
    }
}

fn discovery_cache_key(
    root: &Path,
    scan_options: &SemRepoScanOptions,
    options: &SemEmbeddedOptions,
    files: &[String],
) -> String {
    let mut parts = vec![
        root.display().to_string(),
        options_fingerprint(options),
        format!(
            "scan:{}:{}:{}:{}",
            scan_options.max_files,
            scan_options.max_file_bytes,
            scan_options.max_depth,
            scan_options.include_hidden
        ),
    ];
    parts.extend(files.iter().cloned());
    stable_cache_key("discover", &parts)
}

fn repo_graph_cache_key(
    root: &Path,
    file_paths: &[String],
    options: &SemEmbeddedOptions,
    entities: &[SemanticEntity],
) -> String {
    let mut parts = vec![root.display().to_string(), options_fingerprint(options)];
    parts.extend(
        sorted_strings(file_paths)
            .into_iter()
            .map(|path| format!("path:{path}")),
    );
    let mut entity_parts = entities
        .iter()
        .map(|entity| format!("{}:{}", entity.id, entity.content_hash))
        .collect::<Vec<_>>();
    entity_parts.sort();
    parts.extend(entity_parts);
    stable_cache_key("repo-graph", &parts)
}

fn memory_graph_cache_key(files: &[SemFileInput], options: &SemEmbeddedOptions) -> String {
    let mut parts = vec![options_fingerprint(options)];
    for file in files {
        parts.push(format!(
            "{}:{}",
            file.file_path,
            content_hash(&file.content)
        ));
    }
    stable_cache_key("memory-graph", &parts)
}

fn focus_cache_key(
    changes: &[SemFileChange],
    target: &SemFocusTarget,
    options: &SemEmbeddedOptions,
) -> String {
    stable_cache_key(
        "focus",
        &[
            diff_cache_key(changes, options),
            serde_json::to_string(target).unwrap_or_default(),
        ],
    )
}

fn diff_impact_cache_key(
    file_paths: &[String],
    changes: &[SemFileChange],
    target: &SemFocusTarget,
    request: &SemImpactRequest,
    options: &SemEmbeddedOptions,
    graph_cache_key: &str,
) -> String {
    stable_cache_key(
        "diff-impact",
        &[
            graph_cache_key.to_string(),
            diff_cache_key(changes, options),
            serde_json::to_string(target).unwrap_or_default(),
            format!("request:{}:{}", request.token_budget, request.max_depth),
            sorted_strings(file_paths).join("\n"),
        ],
    )
}

fn review_layer_cache_key(
    analysis: &SemDiffAnalysis,
    layer_options: &SemLayerGenerationOptions,
    options: &SemEmbeddedOptions,
) -> String {
    stable_cache_key(
        "review-layers",
        &[
            analysis.cache_key.clone(),
            options_fingerprint(options),
            format!(
                "{}:{}",
                layer_options.max_layers, layer_options.max_changes_per_layer
            ),
        ],
    )
}

fn review_atom_layer_cache_key(
    analysis: &SemDiffAnalysis,
    atoms: &[SemReviewAtom],
    layer_options: &SemLayerGenerationOptions,
    options: &SemEmbeddedOptions,
) -> String {
    let mut parts = vec![
        review_layer_cache_key(analysis, layer_options, options),
        options_fingerprint(options),
        format!(
            "{}:{}",
            layer_options.max_layers, layer_options.max_changes_per_layer
        ),
    ];
    for atom in atoms {
        parts.push(format!(
            "{}:{}:{}:{}:{:?}:{:?}:{}:{}:{}:{}:{}",
            atom.atom_id,
            atom.file_path,
            atom.old_file_path.as_deref().unwrap_or(""),
            atom.role.as_deref().unwrap_or(""),
            atom.old_range,
            atom.new_range,
            sorted_strings(&atom.defined_symbols).join(","),
            sorted_strings(&atom.referenced_symbols).join(","),
            atom.hunk_indices
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
            atom.changed_lines,
            atom.manual_review
        ));
    }
    stable_cache_key("review-atom-layers", &parts)
}

fn impact_request_fingerprint(
    file_paths: &[String],
    target: &SemEntityTarget,
    request: &SemImpactRequest,
) -> String {
    format!(
        "files:{}:target:{}:{}:{}:request:{}:{}",
        sorted_strings(file_paths).join(","),
        target.entity_id.as_deref().unwrap_or(""),
        target.entity_name.as_deref().unwrap_or(""),
        target.file_path.as_deref().unwrap_or(""),
        request.token_budget,
        request.max_depth
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyze_file_changes_reports_ranges_and_hunks() {
        let before = r#"fn greet(name: &str) -> String {
    format!("hello {name}")
}

fn untouched() -> i32 {
    1
}
"#;
        let after = r#"fn greet(name: &str) -> String {
    format!("hello there {name}")
}

fn untouched() -> i32 {
    1
}
"#;

        let analysis = analyze_file_changes(
            &[SemFileChange {
                file_path: "src/lib.rs".to_string(),
                status: FileStatus::Modified,
                old_file_path: None,
                before_content: Some(before.to_string()),
                after_content: Some(after.to_string()),
                hunks: vec![SemHunk {
                    hunk_id: Some("hunk-1".to_string()),
                    hunk_index: 0,
                    hunk_header: Some("@@ -1,3 +1,3 @@".to_string()),
                    old_range: Some(SemLineRange {
                        start_line: 1,
                        end_line: 3,
                    }),
                    new_range: Some(SemLineRange {
                        start_line: 1,
                        end_line: 3,
                    }),
                }],
            }],
            &SemEmbeddedOptions::default(),
        );

        assert_eq!(analysis.summary.modified_count, 1);
        let change = analysis
            .changes
            .iter()
            .find(|change| change.change.entity_name == "greet")
            .expect("greet change");
        assert_eq!(
            change.before_range,
            Some(SemEntityRange {
                file_path: "src/lib.rs".to_string(),
                start_line: 1,
                end_line: 3,
            })
        );
        assert_eq!(
            change.after_range,
            Some(SemEntityRange {
                file_path: "src/lib.rs".to_string(),
                start_line: 1,
                end_line: 3,
            })
        );
        assert_eq!(change.hunk_overlaps.len(), 1);
        assert!(change.hunk_overlaps[0].overlaps_before);
        assert!(change.hunk_overlaps[0].overlaps_after);
    }

    #[test]
    fn extract_entity_index_uses_in_memory_files() {
        let index = extract_entity_index(
            &[SemFileInput {
                file_path: "src/lib.rs".to_string(),
                content: "fn compute() -> i32 { 42 }\n".to_string(),
            }],
            &SemEmbeddedOptions::default(),
        );

        assert_eq!(index.api_version, SEM_EMBEDDED_API_VERSION);
        assert!(index
            .entities
            .iter()
            .any(|entity| entity.file_path == "src/lib.rs" && entity.name == "compute"));
    }

    #[test]
    fn build_repo_impact_context_does_not_write_cache() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let src = tmp.path().join("src");
        std::fs::create_dir_all(&src).expect("create src");
        std::fs::write(
            src.join("lib.rs"),
            r#"pub fn helper() -> i32 {
    1
}

pub fn caller() -> i32 {
    helper()
}
"#,
        )
        .expect("write lib.rs");

        let file_paths = vec!["src/lib.rs".to_string()];
        let context = build_repo_impact_context(
            tmp.path(),
            &file_paths,
            &SemEntityTarget {
                entity_name: Some("helper".to_string()),
                file_path: Some("src/lib.rs".to_string()),
                ..SemEntityTarget::default()
            },
            &SemImpactRequest {
                token_budget: 512,
                max_depth: 1,
            },
            &SemEmbeddedOptions::default(),
        )
        .expect("impact context");

        assert_eq!(context.entity.name, "helper");
        assert!(context
            .impact
            .iter()
            .any(|impact| impact.entity.name == "caller" && impact.depth == 1));
        assert!(
            !tmp.path().join(".sem").exists(),
            "embedded impact path must not write .sem cache"
        );
    }

    #[test]
    fn resolve_focus_target_finds_entity_at_location() {
        let changes = vec![SemFileChange {
            file_path: "src/lib.rs".to_string(),
            status: FileStatus::Modified,
            old_file_path: None,
            before_content: Some("fn target() -> i32 {\n    1\n}\n".to_string()),
            after_content: Some("fn target() -> i32 {\n    2\n}\n".to_string()),
            hunks: vec![SemHunk {
                hunk_id: None,
                hunk_index: 0,
                hunk_header: Some("@@ -1,3 +1,3 @@".to_string()),
                old_range: Some(SemLineRange {
                    start_line: 1,
                    end_line: 3,
                }),
                new_range: Some(SemLineRange {
                    start_line: 1,
                    end_line: 3,
                }),
            }],
        }];

        let focus = resolve_focus_target(
            &changes,
            &SemFocusTarget {
                location: Some(SemLocationTarget {
                    file_path: "src/lib.rs".to_string(),
                    line: 2,
                    side: SemSide::After,
                }),
                ..SemFocusTarget::default()
            },
            &SemEmbeddedOptions::default(),
        )
        .expect("focus");

        assert_eq!(
            focus
                .target_entity
                .as_ref()
                .map(|entity| entity.entity.name.as_str()),
            Some("target")
        );
        assert_eq!(focus.matching_changes.len(), 1);
    }

    #[test]
    fn build_memory_graph_round_trips_cache_record() {
        let graph = build_memory_graph(
            &[SemFileInput {
                file_path: "src/lib.rs".to_string(),
                content: "fn helper() -> i32 { 1 }\nfn caller() -> i32 { helper() }\n".to_string(),
            }],
            &SemEmbeddedOptions::default(),
        )
        .expect("memory graph");

        assert!(graph.entities.iter().any(|entity| entity.name == "helper"));
        let restored =
            SemGraphSnapshot::from_cache_record(graph.to_cache_record()).expect("cache record");
        assert_eq!(restored.cache_key, graph.cache_key);
        assert!(restored
            .graph
            .entities
            .values()
            .any(|entity| entity.name == "caller"));
    }

    #[test]
    fn discover_repo_files_respects_bounds_and_skips_cache_dirs() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(tmp.path().join("src")).expect("src");
        std::fs::create_dir_all(tmp.path().join("target")).expect("target");
        std::fs::write(tmp.path().join("src/lib.rs"), "fn main() {}\n").expect("lib");
        std::fs::write(tmp.path().join("target/generated.rs"), "fn skip() {}\n").expect("skip");

        let files = discover_repo_files(
            tmp.path(),
            &SemRepoScanOptions::default(),
            &SemEmbeddedOptions::default(),
        )
        .expect("discover");

        assert_eq!(files.files, vec!["src/lib.rs".to_string()]);
    }

    #[test]
    fn diff_impact_context_reports_deleted_symbol_references() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(tmp.path().join("src")).expect("src");
        std::fs::write(
            tmp.path().join("src/lib.rs"),
            "pub fn caller() -> i32 {\n    helper()\n}\n",
        )
        .expect("lib");
        let changes = vec![SemFileChange {
            file_path: "src/lib.rs".to_string(),
            status: FileStatus::Modified,
            old_file_path: None,
            before_content: Some(
                "pub fn helper() -> i32 {\n    1\n}\n\npub fn caller() -> i32 {\n    helper()\n}\n"
                    .to_string(),
            ),
            after_content: Some("pub fn caller() -> i32 {\n    helper()\n}\n".to_string()),
            hunks: vec![SemHunk {
                hunk_id: None,
                hunk_index: 0,
                hunk_header: Some("@@ -1,7 +1,3 @@".to_string()),
                old_range: Some(SemLineRange {
                    start_line: 1,
                    end_line: 7,
                }),
                new_range: Some(SemLineRange {
                    start_line: 1,
                    end_line: 3,
                }),
            }],
        }];

        let context = build_diff_impact_context(
            tmp.path(),
            &["src/lib.rs".to_string()],
            &changes,
            &SemFocusTarget {
                entity: Some(SemEntityTarget {
                    entity_name: Some("helper".to_string()),
                    file_path: Some("src/lib.rs".to_string()),
                    ..SemEntityTarget::default()
                }),
                ..SemFocusTarget::default()
            },
            &SemImpactRequest {
                token_budget: 512,
                max_depth: 1,
            },
            &SemEmbeddedOptions::default(),
        )
        .expect("diff impact");

        assert_eq!(
            context
                .deleted_entity
                .as_ref()
                .map(|entity| entity.entity.name.as_str()),
            Some("helper")
        );
        assert!(context
            .references
            .iter()
            .any(|reference| reference.entity.name == "caller"));
    }

    #[test]
    fn generate_review_layers_groups_code_and_tests() {
        let changes = vec![
            SemFileChange {
                file_path: "src/lib.rs".to_string(),
                status: FileStatus::Modified,
                old_file_path: None,
                before_content: Some("fn helper() -> i32 { 1 }\n".to_string()),
                after_content: Some("fn helper() -> i32 { 2 }\n".to_string()),
                hunks: Vec::new(),
            },
            SemFileChange {
                file_path: "tests/helper_test.rs".to_string(),
                status: FileStatus::Added,
                old_file_path: None,
                before_content: None,
                after_content: Some(
                    "#[test]\nfn helper_test() { assert_eq!(2, 2); }\n".to_string(),
                ),
                hunks: Vec::new(),
            },
        ];

        let plan = generate_review_layers(
            &changes,
            &SemLayerGenerationOptions::default(),
            &SemEmbeddedOptions::default(),
        );

        assert!(plan
            .layers
            .iter()
            .any(|layer| layer.title == "Update helper"));
        assert!(plan
            .layers
            .iter()
            .any(|layer| layer.title == "Update tests"));
    }

    #[test]
    fn generate_review_layers_for_atoms_preserves_atom_ids() {
        let changes = vec![SemFileChange {
            file_path: "src/lib.rs".to_string(),
            status: FileStatus::Modified,
            old_file_path: None,
            before_content: Some("fn helper() -> i32 { 1 }\n".to_string()),
            after_content: Some("fn helper() -> i32 { 2 }\n".to_string()),
            hunks: vec![SemHunk {
                hunk_id: Some("hunk-0".to_string()),
                hunk_index: 0,
                hunk_header: None,
                old_range: Some(SemLineRange {
                    start_line: 1,
                    end_line: 1,
                }),
                new_range: Some(SemLineRange {
                    start_line: 1,
                    end_line: 1,
                }),
            }],
        }];
        let atoms = vec![SemReviewAtom {
            atom_id: "atom-helper".to_string(),
            file_path: "src/lib.rs".to_string(),
            old_file_path: None,
            role: Some("coreLogic".to_string()),
            semantic_kind: Some("function".to_string()),
            symbol_name: Some("helper".to_string()),
            defined_symbols: vec!["helper".to_string()],
            referenced_symbols: Vec::new(),
            old_range: Some(SemLineRange {
                start_line: 1,
                end_line: 1,
            }),
            new_range: Some(SemLineRange {
                start_line: 1,
                end_line: 1,
            }),
            hunk_indices: vec![0],
            changed_lines: 2,
            manual_review: false,
        }];

        let plan = generate_review_layers_for_atoms(
            &changes,
            &atoms,
            &SemLayerGenerationOptions::default(),
            &SemEmbeddedOptions::default(),
        );

        assert!(plan
            .layers
            .iter()
            .any(|layer| layer.atom_ids == vec!["atom-helper"]));
        assert!(plan.manual_review_atom_ids.is_empty());
        assert!(!plan.cache_key.is_empty());
    }

    #[test]
    fn atom_layer_dependencies_link_tests_to_changed_symbols() {
        let changes = vec![
            SemFileChange {
                file_path: "src/lib.rs".to_string(),
                status: FileStatus::Modified,
                old_file_path: None,
                before_content: Some("fn helper() -> i32 { 1 }\n".to_string()),
                after_content: Some("fn helper() -> i32 { 2 }\n".to_string()),
                hunks: Vec::new(),
            },
            SemFileChange {
                file_path: "tests/helper_test.rs".to_string(),
                status: FileStatus::Modified,
                old_file_path: None,
                before_content: Some(
                    "#[test]\nfn helper_test() { assert_eq!(1, helper()); }\n".to_string(),
                ),
                after_content: Some(
                    "#[test]\nfn helper_test() { assert_eq!(2, helper()); }\n".to_string(),
                ),
                hunks: Vec::new(),
            },
        ];
        let atoms = vec![
            SemReviewAtom {
                atom_id: "atom-helper".to_string(),
                file_path: "src/lib.rs".to_string(),
                old_file_path: None,
                role: Some("coreLogic".to_string()),
                semantic_kind: Some("function".to_string()),
                symbol_name: Some("helper".to_string()),
                defined_symbols: vec!["helper".to_string()],
                referenced_symbols: Vec::new(),
                old_range: None,
                new_range: None,
                hunk_indices: Vec::new(),
                changed_lines: 1,
                manual_review: false,
            },
            SemReviewAtom {
                atom_id: "atom-test".to_string(),
                file_path: "tests/helper_test.rs".to_string(),
                old_file_path: None,
                role: Some("tests".to_string()),
                semantic_kind: Some("function".to_string()),
                symbol_name: Some("helper_test".to_string()),
                defined_symbols: vec!["helper_test".to_string()],
                referenced_symbols: vec!["helper".to_string()],
                old_range: None,
                new_range: None,
                hunk_indices: Vec::new(),
                changed_lines: 1,
                manual_review: false,
            },
        ];

        let plan = generate_review_layers_for_atoms(
            &changes,
            &atoms,
            &SemLayerGenerationOptions::default(),
            &SemEmbeddedOptions::default(),
        );
        let code_layer = plan
            .layers
            .iter()
            .find(|layer| layer.atom_ids == vec!["atom-helper"])
            .expect("code layer");
        let test_layer = plan
            .layers
            .iter()
            .find(|layer| layer.atom_ids == vec!["atom-test"])
            .expect("test layer");

        assert!(test_layer.depends_on_layer_ids.contains(&code_layer.id));
    }

    #[test]
    fn manual_review_atoms_are_reported_without_layer_assignment() {
        let plan = generate_review_layers_for_atoms(
            &[],
            &[SemReviewAtom {
                atom_id: "atom-manual".to_string(),
                file_path: "generated/output.rs".to_string(),
                old_file_path: None,
                role: Some("generated".to_string()),
                semantic_kind: None,
                symbol_name: None,
                defined_symbols: Vec::new(),
                referenced_symbols: Vec::new(),
                old_range: None,
                new_range: None,
                hunk_indices: Vec::new(),
                changed_lines: 10,
                manual_review: true,
            }],
            &SemLayerGenerationOptions::default(),
            &SemEmbeddedOptions::default(),
        );

        assert!(plan.layers.is_empty());
        assert_eq!(plan.manual_review_atom_ids, vec!["atom-manual"]);
    }

    #[test]
    fn execution_context_can_cancel_work() {
        struct Cancelled;
        impl SemCancellationToken for Cancelled {
            fn is_cancelled(&self) -> bool {
                true
            }
        }

        let tmp = tempfile::tempdir().expect("tempdir");
        let error = discover_repo_files_with_execution(
            tmp.path(),
            &SemRepoScanOptions::default(),
            &SemEmbeddedOptions::default(),
            &SemExecutionContext {
                cancellation: Some(&Cancelled),
                progress: None,
            },
        )
        .expect_err("cancelled");

        assert_eq!(error.kind, SemErrorKind::Cancelled);
    }
}
