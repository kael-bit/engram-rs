use engram::db::MemoryDB;

#[test]
fn meta_get_set() {
    let db = MemoryDB::open(":memory:").unwrap();
    assert_eq!(db.get_meta("nonexistent"), None);
    db.set_meta("last_audit_ms", "1234567890").unwrap();
    assert_eq!(db.get_meta("last_audit_ms"), Some("1234567890".to_string()));
    db.set_meta("last_audit_ms", "9999999999").unwrap();
    assert_eq!(db.get_meta("last_audit_ms"), Some("9999999999".to_string()));
}
