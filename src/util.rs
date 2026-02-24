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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_no_truncate() {
        assert_eq!(truncate_chars("hello", 10), "hello");
    }

    #[test]
    fn ascii_truncate() {
        assert_eq!(truncate_chars("hello world", 5), "hello…");
    }

    #[test]
    fn cjk_truncate() {
        assert_eq!(truncate_chars("你好世界测试", 4), "你好世界…");
    }

    #[test]
    fn empty_string() {
        assert_eq!(truncate_chars("", 5), "");
    }
}
