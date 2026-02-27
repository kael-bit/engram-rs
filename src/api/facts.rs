//! Facts handlers.

use axum::extract::{Path, Query, State};
use axum::Json;
use serde::Deserialize;

use crate::error::EngramError;
use crate::extract::LenientJson;
use crate::{db, AppState};
use super::{blocking, get_namespace};

#[derive(Deserialize)]
pub(super) struct CreateFactsBody {
    facts: Vec<db::FactInput>,
}

#[derive(serde::Serialize)]
pub(super) struct CreateFactsResponse {
    facts: Vec<db::Fact>,
    conflicts: Vec<db::Fact>,
    resolved: usize,
}

pub(super) async fn create_facts(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    LenientJson(body): LenientJson<CreateFactsBody>,
) -> Result<Json<CreateFactsResponse>, EngramError> {
    let ns = get_namespace(&headers).unwrap_or_else(|| "default".into());

    if body.facts.is_empty() {
        return Ok(Json(CreateFactsResponse { facts: vec![], conflicts: vec![], resolved: 0 }));
    }

    let db = state.db.clone();
    let (inserted, superseded) = blocking(move || db.insert_facts(body.facts, &ns)).await??;
    let resolved = superseded.len();

    Ok(Json(CreateFactsResponse {
        facts: inserted,
        conflicts: superseded,
        resolved,
    }))
}

#[derive(Deserialize)]
pub(super) struct FactQuery {
    entity: Option<String>,
    ns: Option<String>,
    #[serde(default)]
    include_superseded: Option<bool>,
}

pub(super) async fn query_facts(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(q): Query<FactQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let ns = q.ns.or_else(|| get_namespace(&headers)).unwrap_or_else(|| "default".into());
    let entity = q.entity.unwrap_or_default();
    if entity.is_empty() {
        return Err(EngramError::Validation("entity parameter is required".into()));
    }
    let include_superseded = q.include_superseded.unwrap_or(false);
    let db = state.db.clone();
    let facts = blocking(move || db.query_facts(&entity, &ns, include_superseded)).await??;
    Ok(Json(serde_json::json!({ "facts": facts, "count": facts.len() })))
}

#[derive(Deserialize)]
pub(super) struct ConflictQuery {
    subject: Option<String>,
    predicate: Option<String>,
    ns: Option<String>,
}

pub(super) async fn get_fact_conflicts(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(q): Query<ConflictQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let ns = q.ns.or_else(|| get_namespace(&headers)).unwrap_or_else(|| "default".into());
    let subject = q.subject.ok_or_else(|| EngramError::Validation("subject is required".into()))?;
    let predicate = q.predicate.ok_or_else(|| EngramError::Validation("predicate is required".into()))?;
    let db = state.db.clone();
    let all = blocking(move || db.get_conflicts(&subject, &predicate, &ns)).await??;
    let conflicts: Vec<_> = all.into_iter().filter(|f| f.valid_until.is_none()).collect();
    Ok(Json(serde_json::json!({ "conflicts": conflicts, "count": conflicts.len() })))
}

#[derive(Deserialize)]
pub(super) struct HistoryQuery {
    subject: Option<String>,
    predicate: Option<String>,
    ns: Option<String>,
}

pub(super) async fn get_fact_history(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(q): Query<HistoryQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let ns = q.ns.or_else(|| get_namespace(&headers)).unwrap_or_else(|| "default".into());
    let subject = q.subject.ok_or_else(|| EngramError::Validation("subject is required".into()))?;
    let predicate = q.predicate.ok_or_else(|| EngramError::Validation("predicate is required".into()))?;
    let db = state.db.clone();
    let history = blocking(move || db.get_fact_history(&subject, &predicate, &ns)).await??;
    Ok(Json(serde_json::json!({ "history": history, "count": history.len() })))
}

pub(super) async fn delete_fact(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();
    let deleted = blocking(move || db.delete_fact(&id)).await??;
    if !deleted {
        return Err(EngramError::NotFound);
    }
    Ok(Json(serde_json::json!({ "deleted": true })))
}

#[derive(Deserialize)]
pub(super) struct ListFactsQuery {
    ns: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

pub(super) async fn list_all_facts(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(q): Query<ListFactsQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let ns = q.ns.or_else(|| get_namespace(&headers)).unwrap_or_else(|| "default".into());
    let limit = q.limit.unwrap_or(100).min(1000);
    let offset = q.offset.unwrap_or(0);
    let db = state.db.clone();
    let facts = blocking(move || db.list_facts(&ns, limit, offset)).await??;
    Ok(Json(serde_json::json!({ "facts": facts, "count": facts.len() })))
}

#[derive(Deserialize)]
pub(super) struct GraphQuery {
    entity: Option<String>,
    hops: Option<usize>,
    ns: Option<String>,
}

pub(super) async fn query_graph(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(q): Query<GraphQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let ns = q.ns.or_else(|| get_namespace(&headers)).unwrap_or_else(|| "default".into());
    let entity = q.entity.ok_or_else(|| EngramError::Validation("entity parameter is required".into()))?;
    let hops = q.hops.unwrap_or(2).clamp(1, 3);

    let db = state.db.clone();
    let ent = entity.clone();
    let chains = blocking(move || db.query_multihop(&ent, hops, &ns)).await??;

    Ok(Json(serde_json::json!({
        "entity": entity,
        "hops": hops,
        "chains": chains,
        "count": chains.len(),
    })))
}
