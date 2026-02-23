//! Proxy turn persistence.

use rusqlite::params;

use super::*;

impl MemoryDB {
    pub fn save_proxy_turn(&self, session_key: &str, content: &str) -> Result<(), EngramError> {
        let conn = self.conn()?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        conn.execute(
            "INSERT INTO proxy_turns (session_key, content, created_at) VALUES (?1, ?2, ?3)",
            params![session_key, content, now],
        )?;
        Ok(())
    }

    /// Drain all turns for a session, returning concatenated content.
    pub fn drain_proxy_session(&self, session_key: &str) -> Result<String, EngramError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT content FROM proxy_turns WHERE session_key = ?1 ORDER BY id ASC"
        )?;
        let turns: Vec<String> = stmt
            .query_map(params![session_key], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        if !turns.is_empty() {
            conn.execute("DELETE FROM proxy_turns WHERE session_key = ?1", params![session_key])?;
        }
        Ok(turns.join("\n---\n"))
    }

    /// Drain all sessions, returning (session_key, context) pairs.
    pub fn drain_all_proxy_turns(&self) -> Result<Vec<(String, String)>, EngramError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT DISTINCT session_key FROM proxy_turns"
        )?;
        let keys: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        let mut results = Vec::new();
        for key in keys {
            let mut turn_stmt = conn.prepare(
                "SELECT content FROM proxy_turns WHERE session_key = ?1 ORDER BY id ASC"
            )?;
            let turns: Vec<String> = turn_stmt
                .query_map(params![&key], |row| row.get(0))?
                .filter_map(|r| r.ok())
                .collect();
            if !turns.is_empty() {
                results.push((key, turns.join("\n---\n")));
            }
        }
        conn.execute("DELETE FROM proxy_turns", [])?;
        Ok(results)
    }

    /// Count buffered turns across all sessions.
    pub fn proxy_turn_count(&self) -> usize {
        let conn = match self.conn() {
            Ok(c) => c,
            Err(_) => return 0,
        };
        conn.query_row("SELECT COUNT(*) FROM proxy_turns", [], |row| row.get::<_, i64>(0))
            .unwrap_or(0) as usize
    }

    pub fn proxy_session_should_flush(&self, session_key: &str, max_turns: usize, max_chars: usize) -> bool {
        let conn = match self.conn() {
            Ok(c) => c,
            Err(_) => return false,
        };
        let (count, chars): (i64, i64) = conn
            .query_row(
                "SELECT COUNT(*), COALESCE(SUM(LENGTH(content)), 0) FROM proxy_turns WHERE session_key = ?1",
                params![session_key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap_or((0, 0));
        count as usize >= max_turns || chars as usize >= max_chars
    }

}
