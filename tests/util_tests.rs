use engram::util::truncate_chars;

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
