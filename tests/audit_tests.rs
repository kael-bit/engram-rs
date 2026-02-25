use engram::consolidate::{AuditOp, RawAuditOp, AuditToolResponse, audit_tool_schema, resolve_audit_ops};
use engram::db::{Layer, Memory};
use std::collections::HashMap;

/// Helper: create a Memory with the given id and layer, everything else zeroed/defaults.
fn mem(id: &str, layer: Layer) -> Memory {
    Memory {
        id: id.to_string(),
        content: String::new(),
        layer,
        importance: 0.5,
        created_at: 0,
        last_accessed: 0,
        access_count: 0,
        repetition_count: 0,
        decay_rate: 0.1,
        source: String::new(),
        tags: vec![],
        namespace: "default".into(),
        embedding: None,
        kind: "episodic".into(),
        modified_at: 0,
    }
}

/// Helper: create a RawAuditOp for testing resolve_audit_ops.
fn raw_op(op: &str) -> RawAuditOp {
    RawAuditOp {
        op: op.to_string(),
        id: None,
        to: None,
        importance: None,
        ids: None,
        content: None,
        layer: None,
        tags: None,
    }
}

/// Helper: build an alias map from a list of memories (A, B, C, ...).
fn build_alias_map(memories: &[Memory]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for (i, m) in memories.iter().enumerate() {
        let alias = if i < 26 {
            String::from((b'A' + i as u8) as char)
        } else {
            let first = (b'A' + ((i - 26) / 26) as u8) as char;
            let second = (b'A' + ((i - 26) % 26) as u8) as char;
            format!("{}{}", first, second)
        };
        map.insert(alias, m.id.clone());
    }
    map
}

// ── 1. Valid ops with all 4 op types ───────────────────────────────

#[test]
fn all_four_op_types() {
    let core = vec![
        mem("aaaaaaaa-1111-2222-3333-444444444444", Layer::Core),   // A
        mem("bbbbbbbb-1111-2222-3333-444444444444", Layer::Core),   // B
    ];
    let working = vec![
        mem("cccccccc-1111-2222-3333-444444444444", Layer::Working), // C
        mem("dddddddd-1111-2222-3333-444444444444", Layer::Working), // D
        mem("eeeeeeee-1111-2222-3333-444444444444", Layer::Working), // E
    ];

    let all: Vec<Memory> = core.iter().chain(working.iter()).cloned().collect();
    let alias_map = build_alias_map(&all);

    let raw_ops = vec![
        RawAuditOp { op: "promote".into(), id: Some("C".into()), to: Some(3), ..raw_op("promote") },
        RawAuditOp { op: "demote".into(), id: Some("A".into()), to: Some(2), ..raw_op("demote") },
        RawAuditOp { op: "delete".into(), id: Some("D".into()), ..raw_op("delete") },
        RawAuditOp {
            op: "merge".into(),
            ids: Some(vec!["B".into(), "E".into()]),
            content: Some("merged content".into()),
            layer: Some(3),
            tags: Some(vec!["t1".into(), "t2".into()]),
            ..raw_op("merge")
        },
    ];

    let ops = resolve_audit_ops(raw_ops, &alias_map);
    assert_eq!(ops.len(), 4);

    match &ops[0] {
        AuditOp::Promote { id, to } => {
            assert_eq!(id, "cccccccc-1111-2222-3333-444444444444");
            assert_eq!(*to, 3);
        }
        other => panic!("expected Promote, got {:?}", other),
    }
    match &ops[1] {
        AuditOp::Demote { id, to } => {
            assert_eq!(id, "aaaaaaaa-1111-2222-3333-444444444444");
            assert_eq!(*to, 2);
        }
        other => panic!("expected Demote, got {:?}", other),
    }
    match &ops[2] {
        AuditOp::Delete { id } => {
            assert_eq!(id, "dddddddd-1111-2222-3333-444444444444");
        }
        other => panic!("expected Delete, got {:?}", other),
    }
    match &ops[3] {
        AuditOp::Merge { ids, content, layer, tags } => {
            assert_eq!(ids.len(), 2);
            assert_eq!(ids[0], "bbbbbbbb-1111-2222-3333-444444444444");
            assert_eq!(ids[1], "eeeeeeee-1111-2222-3333-444444444444");
            assert_eq!(content, "merged content");
            assert_eq!(*layer, 3);
            assert_eq!(tags, &["t1", "t2"]);
        }
        other => panic!("expected Merge, got {:?}", other),
    }
}

// ── 2. Letter aliases resolved against alias map ───────────────────

#[test]
fn aliases_resolved() {
    let core = vec![mem("abcd1234-aaaa-bbbb-cccc-dddddddddddd", Layer::Core)];   // A
    let working = vec![mem("ef567890-aaaa-bbbb-cccc-dddddddddddd", Layer::Working)]; // B

    let all: Vec<Memory> = core.iter().chain(working.iter()).cloned().collect();
    let alias_map = build_alias_map(&all);

    let raw_ops = vec![
        RawAuditOp { op: "delete".into(), id: Some("A".into()), ..raw_op("delete") },
        RawAuditOp { op: "promote".into(), id: Some("B".into()), to: Some(3), ..raw_op("promote") },
    ];
    let ops = resolve_audit_ops(raw_ops, &alias_map);

    assert_eq!(ops.len(), 2);
    match &ops[0] {
        AuditOp::Delete { id } => assert_eq!(id, "abcd1234-aaaa-bbbb-cccc-dddddddddddd"),
        other => panic!("expected Delete, got {:?}", other),
    }
    match &ops[1] {
        AuditOp::Promote { id, to } => {
            assert_eq!(id, "ef567890-aaaa-bbbb-cccc-dddddddddddd");
            assert_eq!(*to, 3);
        }
        other => panic!("expected Promote, got {:?}", other),
    }
}

// ── 3. Full UUIDs passed through directly ──────────────────────────

#[test]
fn full_uuids_passed_through() {
    let full_id = "12345678-abcd-ef01-2345-6789abcdef01";
    let raw_ops = vec![
        RawAuditOp { op: "delete".into(), id: Some(full_id.into()), ..raw_op("delete") },
    ];
    let alias_map = HashMap::new();
    let ops = resolve_audit_ops(raw_ops, &alias_map);

    assert_eq!(ops.len(), 1);
    match &ops[0] {
        AuditOp::Delete { id } => assert_eq!(id, full_id),
        other => panic!("expected Delete, got {:?}", other),
    }
}

// ── 4. Empty ops array → empty result ──────────────────────────────

#[test]
fn empty_ops_array() {
    let alias_map = HashMap::new();
    let ops = resolve_audit_ops(vec![], &alias_map);
    assert!(ops.is_empty());
}

// ── 5. Tool response deserialization ───────────────────────────────

#[test]
fn tool_response_deserializes_from_json() {
    let json = r#"{"operations":[
        {"op":"delete","id":"A"},
        {"op":"promote","id":"B","to":3}
    ]}"#;
    let resp: AuditToolResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.operations.len(), 2);
    assert_eq!(resp.operations[0].op, "delete");
    assert_eq!(resp.operations[0].id.as_deref(), Some("A"));
    assert_eq!(resp.operations[1].op, "promote");
    assert_eq!(resp.operations[1].to, Some(3));
}

#[test]
fn tool_response_empty_operations() {
    let json = r#"{"operations":[]}"#;
    let resp: AuditToolResponse = serde_json::from_str(json).unwrap();
    assert!(resp.operations.is_empty());
}

#[test]
fn tool_response_merge_op_deserializes() {
    let json = r#"{"operations":[
        {"op":"merge","ids":["A","B"],"content":"merged","layer":2,"tags":["x","y"]}
    ]}"#;
    let resp: AuditToolResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.operations.len(), 1);
    assert_eq!(resp.operations[0].op, "merge");
    assert_eq!(resp.operations[0].ids.as_ref().unwrap(), &["A", "B"]);
    assert_eq!(resp.operations[0].content.as_deref(), Some("merged"));
    assert_eq!(resp.operations[0].layer, Some(2));
    assert_eq!(resp.operations[0].tags.as_ref().unwrap(), &["x", "y"]);
}

// ── 6. Missing required fields → op skipped ────────────────────────

#[test]
fn merge_without_source_ids_skipped() {
    let raw_ops = vec![
        RawAuditOp {
            op: "merge".into(),
            content: Some("combined".into()),
            layer: Some(2),
            tags: Some(vec!["x".into()]),
            ..raw_op("merge")
        },
    ];
    let alias_map = HashMap::new();
    let ops = resolve_audit_ops(raw_ops, &alias_map);
    assert!(ops.is_empty(), "merge without ids should be skipped");
}

#[test]
fn merge_with_single_id_skipped() {
    let core = vec![mem("aaaaaaaa-1111-2222-3333-444444444444", Layer::Core)]; // A
    let alias_map = build_alias_map(&core);
    let raw_ops = vec![
        RawAuditOp {
            op: "merge".into(),
            ids: Some(vec!["A".into()]),
            content: Some("combined".into()),
            layer: Some(2),
            tags: Some(vec![]),
            ..raw_op("merge")
        },
    ];
    let ops = resolve_audit_ops(raw_ops, &alias_map);
    assert!(ops.is_empty(), "merge with only 1 id should be skipped");
}

#[test]
fn merge_with_empty_content_skipped() {
    let core = vec![
        mem("aaaaaaaa-1111-2222-3333-444444444444", Layer::Core), // A
        mem("bbbbbbbb-1111-2222-3333-444444444444", Layer::Core), // B
    ];
    let alias_map = build_alias_map(&core);
    let raw_ops = vec![
        RawAuditOp {
            op: "merge".into(),
            ids: Some(vec!["A".into(), "B".into()]),
            content: Some("".into()),
            layer: Some(2),
            tags: Some(vec![]),
            ..raw_op("merge")
        },
    ];
    let ops = resolve_audit_ops(raw_ops, &alias_map);
    assert!(ops.is_empty(), "merge with empty content should be skipped");
}

#[test]
fn promote_without_to_skipped() {
    let core = vec![mem("aaaaaaaa-1111-2222-3333-444444444444", Layer::Core)]; // A
    let alias_map = build_alias_map(&core);
    let raw_ops = vec![
        RawAuditOp { op: "promote".into(), id: Some("A".into()), ..raw_op("promote") },
    ];
    let ops = resolve_audit_ops(raw_ops, &alias_map);
    assert!(ops.is_empty(), "promote without 'to' should be skipped");
}

#[test]
fn delete_with_unresolvable_alias_skipped() {
    let alias_map = HashMap::new();
    let raw_ops = vec![
        RawAuditOp { op: "delete".into(), id: Some("ZZ".into()), ..raw_op("delete") },
    ];
    let ops = resolve_audit_ops(raw_ops, &alias_map);
    assert!(ops.is_empty(), "unresolvable alias should be skipped");
}

// ── 7. Unknown op type → skipped ───────────────────────────────────

#[test]
fn unknown_op_type_skipped() {
    let alias_map = HashMap::new();
    let raw_ops = vec![
        RawAuditOp { op: "explode".into(), id: Some("A".into()), ..raw_op("explode") },
    ];
    let ops = resolve_audit_ops(raw_ops, &alias_map);
    assert!(ops.is_empty());
}

// ── 8. Mix of valid and invalid ops → only valid returned ──────────

#[test]
fn mix_valid_and_invalid() {
    let core = vec![
        mem("aaaaaaaa-1111-2222-3333-444444444444", Layer::Core),   // A
    ];
    let working = vec![
        mem("bbbbbbbb-1111-2222-3333-444444444444", Layer::Working), // B
    ];

    let all: Vec<Memory> = core.iter().chain(working.iter()).cloned().collect();
    let alias_map = build_alias_map(&all);

    let raw_ops = vec![
        RawAuditOp { op: "delete".into(), id: Some("A".into()), ..raw_op("delete") },
        RawAuditOp { op: "unknown_type".into(), id: Some("B".into()), ..raw_op("unknown_type") },
        RawAuditOp { op: "promote".into(), id: Some("B".into()), to: Some(3), ..raw_op("promote") },
        RawAuditOp { op: "promote".into(), id: Some("ZZ".into()), ..raw_op("promote") },
        RawAuditOp { op: "demote".into(), id: Some("A".into()), to: Some(2), ..raw_op("demote") },
    ];

    let ops = resolve_audit_ops(raw_ops, &alias_map);
    // Valid: delete A, promote B to 3, demote A to 2
    // Invalid: unknown_type (skipped), promote ZZ (unresolvable + no 'to')
    assert_eq!(ops.len(), 3);
    assert!(matches!(&ops[0], AuditOp::Delete { .. }));
    assert!(matches!(&ops[1], AuditOp::Promote { to: 3, .. }));
    assert!(matches!(&ops[2], AuditOp::Demote { to: 2, .. }));
}

// ── 9. `to` out of range (0 or 4) → skipped ───────────────────────

#[test]
fn to_out_of_range_skipped() {
    let core = vec![mem("aaaaaaaa-1111-2222-3333-444444444444", Layer::Core)]; // A
    let alias_map = build_alias_map(&core);
    let raw_ops = vec![
        RawAuditOp { op: "promote".into(), id: Some("A".into()), to: Some(0), ..raw_op("promote") },
        RawAuditOp { op: "demote".into(), id: Some("A".into()), to: Some(4), ..raw_op("demote") },
    ];
    let ops = resolve_audit_ops(raw_ops, &alias_map);
    assert!(ops.is_empty(), "to=0 and to=4 should both be skipped");
}

// ── 10. Schema structure validation ────────────────────────────────

#[test]
fn audit_tool_schema_is_valid_json() {
    let schema = audit_tool_schema();
    assert!(schema.is_object());
    assert!(schema.get("properties").is_some());
    let ops = schema["properties"]["operations"].clone();
    assert_eq!(ops["type"], "array");
    let items = ops["items"].clone();
    assert!(items["properties"]["op"]["enum"].is_array());
}

// ── 11. Merge default layer ────────────────────────────────────────

#[test]
fn merge_defaults_to_layer_2() {
    let core = vec![
        mem("aaaaaaaa-1111-2222-3333-444444444444", Layer::Core), // A
        mem("bbbbbbbb-1111-2222-3333-444444444444", Layer::Core), // B
    ];
    let alias_map = build_alias_map(&core);
    let raw_ops = vec![
        RawAuditOp {
            op: "merge".into(),
            ids: Some(vec!["A".into(), "B".into()]),
            content: Some("merged".into()),
            // layer omitted — should default to 2
            ..raw_op("merge")
        },
    ];
    let ops = resolve_audit_ops(raw_ops, &alias_map);
    assert_eq!(ops.len(), 1);
    match &ops[0] {
        AuditOp::Merge { layer, .. } => assert_eq!(*layer, 2),
        other => panic!("expected Merge, got {:?}", other),
    }
}
