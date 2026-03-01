use crate::ai::{self, AiConfig};
use crate::db::{Layer, Memory};
use crate::prompts;
use crate::SharedDB;
use crate::util::truncate_chars;
use tracing::{info, warn};

/// Distill session notes into project context memories.
///
/// Groups session notes by namespace, then for each group with 3+ undistilled
/// notes, LLM synthesizes a project status snapshot stored in Buffer as episodic.
/// Tagged `auto-distilled` to block Core promotion (project status is ephemeral).
/// Source notes get `distilled` tag to prevent reprocessing.
pub(super) async fn distill_sessions(db: &SharedDB, cfg: &AiConfig) -> usize {
    let db2 = db.clone();
    let by_ns: std::collections::HashMap<String, Vec<Memory>> = match tokio::task::spawn_blocking(move || {
        let mut groups: std::collections::HashMap<String, Vec<Memory>> = std::collections::HashMap::new();
        for mem in db2.list_by_layer_meta(Layer::Buffer, 500, 0)
            .unwrap_or_default()
            .into_iter()
            .chain(db2.list_by_layer_meta(Layer::Working, 500, 0).unwrap_or_default())
        {
            let is_session = mem.source == "session"
                || mem.tags.iter().any(|t| t == "session");
            let already_done = mem.tags.iter().any(|t| t == "distilled");
            if is_session && !already_done {
                let ns = mem.namespace.clone();
                groups.entry(ns).or_default().push(mem);
            }
        }
        for v in groups.values_mut() {
            v.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        }
        groups
    }).await {
        Ok(g) => g,
        Err(_) => return 0,
    };

    let mut total = 0;
    for (ns, sessions) in &by_ns {
        if sessions.len() < 3 { continue; }
        total += distill_one_ns(db, cfg, ns, sessions).await;
    }
    total
}

async fn distill_one_ns(
    db: &SharedDB, cfg: &AiConfig, ns: &str, sessions: &[Memory],
) -> usize {
    let to_distill: Vec<&Memory> = sessions.iter().rev().take(10).collect();
    let text: String = to_distill.iter().rev().map(|m| {
        format!("- {}", truncate_chars(&m.content, 300))
    }).collect::<Vec<_>>().join("\n");

    let user = format!(
        "Distill these {} session notes into a project status snapshot:\n\n{}",
        to_distill.len(), text
    );

    let summary = match ai::llm_chat_as(cfg, "gate", prompts::DISTILL_SYSTEM_PROMPT, &user).await {
        Ok(r) => {
            super::log_llm_usage(db, "distill", &r.usage, &r.model, r.duration_ms);
            r.content.trim().to_string()
        }
        Err(e) => {
            warn!("session distillation failed: {e}");
            return 0;
        }
    };
    if summary.is_empty() || summary.len() > 500 { return 0; }

    let db3 = db.clone();
    let sc = summary.clone();
    let is_dup = tokio::task::spawn_blocking(move || db3.is_near_duplicate_with(&sc, crate::thresholds::TRIAGE_DEDUP_SIM))
        .await.unwrap_or(false);

    if is_dup {
        info!("session distillation: skipped (near-duplicate exists)");
    } else {
        let ns_val = if ns == "default" { None } else { Some(ns.to_string()) };
        let input = crate::db::MemoryInput {
            content: summary.clone(),
            tags: Some(vec!["project-status".into(), "auto-distilled".into()]),
            source: Some("consolidation".into()),
            importance: Some(0.7),
            kind: Some("episodic".into()),
            namespace: ns_val,
            ..Default::default()
        };
        let db4 = db.clone();
        // spawn_blocking returns Result<Result<Memory, EngramError>, JoinError>.
        // Must check both layers to avoid deleting sources on a silent insert failure.
        let insert_ok = match tokio::task::spawn_blocking(move || db4.insert(input)).await {
            Err(e) => {
                warn!("session distillation insert task failed: {e}");
                false
            }
            Ok(Err(e)) => {
                warn!("session distillation insert failed: {e}");
                false
            }
            Ok(Ok(_)) => true,
        };
        if !insert_ok {
            return 0;
        }
        info!(len = summary.len(), sessions = to_distill.len(), ns = ns,
              "distilled sessions into project status");
    }

    // Only delete source notes after successful insert.
    // When is_dup, tag them as "distilled" to prevent reprocessing, but keep the data.
    let db5 = db.clone();
    let sources: Vec<(String, Vec<String>)> = to_distill.iter()
        .map(|m| (m.id.clone(), m.tags.clone()))
        .collect();
    let count = sources.len();
    let delete_sources = !is_dup;
    let _ = tokio::task::spawn_blocking(move || {
        for (id, existing_tags) in &sources {
            if delete_sources {
                let _ = db5.delete(id);
            } else {
                // Near-duplicate case: mark as processed without deleting
                let mut tags = existing_tags.clone();
                if !tags.iter().any(|t| t == "distilled") {
                    tags.push("distilled".into());
                }
                let _ = db5.update_fields(id, None, None, None, Some(&tags));
            }
        }
    }).await;
    count
}
