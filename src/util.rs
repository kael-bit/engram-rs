/// Safe ID prefix — never panics on non-ASCII or short strings.
#[inline]
pub fn short_id(id: &str) -> &str {
    let end = id.floor_char_boundary(8.min(id.len()));
    &id[..end]
}

/// Truncate a string to `max` characters, appending "…" if truncated.
/// Handles multi-byte (CJK) correctly via char boundary.
pub fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}…")
    }
}

