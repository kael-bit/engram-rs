use engram::ai::{cosine_similarity, embedding_to_bytes, bytes_to_embedding, unwrap_json};

#[test]
fn cosine_same_vec() {
    let v: Vec<f32> = vec![1.0, 2.0, 3.0];
    let sim = cosine_similarity(&v, &v);
    assert!((sim - 1.0).abs() < 1e-10);
}

#[test]
fn cosine_perpendicular() {
    let a: Vec<f32> = vec![1.0, 0.0];
    let b: Vec<f32> = vec![0.0, 1.0];
    let sim = cosine_similarity(&a, &b);
    assert!(sim.abs() < 1e-10);
}

#[test]
fn cosine_empty() {
    assert_eq!(cosine_similarity(&[], &[]), 0.0);
}

#[test]
fn embedding_roundtrip() {
    let original: Vec<f32> = vec![1.0, -2.5, 3.125, 0.0, f32::MAX];
    let bytes = embedding_to_bytes(&original);
    let decoded = bytes_to_embedding(&bytes);
    assert_eq!(original, decoded);
}

#[test]
fn unwrap_json_from_markdown() {
    let raw = "```json\n[{\"content\": \"test\"}]\n```";
    let result = unwrap_json(raw);
    assert_eq!(result, "[{\"content\": \"test\"}]");
}

#[test]
fn unwrap_json_bare() {
    let raw = "[{\"content\": \"test\"}]";
    let result = unwrap_json(raw);
    assert_eq!(result, raw);
}

#[test]
fn unwrap_json_with_leading_text() {
    let raw = "Here are the results:\n```json\n[1, 2, 3]\n```\nDone!";
    let result = unwrap_json(raw);
    assert_eq!(result, "[1, 2, 3]");
}

#[test]
fn unwrap_json_no_array() {
    // Without brackets, returns trimmed input
    let raw = "  {\"key\": \"value\"}  ";
    let result = unwrap_json(raw);
    assert_eq!(result, "{\"key\": \"value\"}");
}

#[test]
fn expand_output_parsing() {
    // Simulates what expand_query does to LLM output
    let parse = |raw: &str| -> Vec<String> {
        raw.lines()
            .map(|l| l.trim().trim_start_matches("- ").trim().to_string())
            .filter(|l| {
                !l.is_empty() && l.len() > 2
                    && !l.contains("过于简洁")
                    && !l.contains("缺乏上下文")
                    && !l.contains("无法生成")
                    && !l.starts_with("这个查询")
                    && !l.starts_with("该查询")
                    && !l.starts_with("注意")
                    && !l.starts_with("Note:")
            })
            .take(5)
            .collect()
    };

    // Normal case
    let parsed = parse("连接池实现方案\nr2d2 SQLite pool\n\nab\n数据库并发访问\n");
    assert_eq!(parsed, vec![
        "连接池实现方案",
        "r2d2 SQLite pool",
        "数据库并发访问",
    ]);

    // Meta-commentary filtered out
    let parsed = parse("这个查询过于简洁，无法生成有意义的替代搜索短语。\n\"alice是谁\" 缺乏上下文，可能指：\n- 某个具体的人名\n- 项目/产品代号\n- 团队成员昵称");
    assert_eq!(parsed, vec![
        "某个具体的人名",
        "项目/产品代号",
        "团队成员昵称",
    ]);

    // Dash-prefixed lines cleaned
    let parsed = parse("- alice的身份信息\n- alice是什么角色\n- 关于alice的描述");
    assert_eq!(parsed, vec![
        "alice的身份信息",
        "alice是什么角色",
        "关于alice的描述",
    ]);
}
