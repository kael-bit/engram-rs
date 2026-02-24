//! Injection detection and output sanitization for memory content.
//!
//! Memories get injected into LLM context windows via /recall and /resume.
//! Malicious content stored as a memory could act as a prompt injection.
//! This module scores content for injection risk and sanitizes output.

use regex::Regex;
use std::sync::LazyLock;

struct Pattern {
    re: Regex,
    score: f64,
}

static PATTERNS: LazyLock<Vec<Pattern>> = LazyLock::new(|| {
    vec![
        Pattern {
            re: Regex::new(r"(?i)ignore\s+(all\s+)?previous\s+instructions").unwrap(),
            score: 0.9,
        },
        Pattern {
            re: Regex::new(r"(?i)\bdisregard\b").unwrap(),
            score: 0.9,
        },
        Pattern {
            re: Regex::new(r"(?i)\byou\s+are\s+now\b").unwrap(),
            score: 0.7,
        },
        Pattern {
            re: Regex::new(r"(?i)\bact\s+as\b").unwrap(),
            score: 0.7,
        },
        Pattern {
            re: Regex::new(r"(?i)\bpretend\s+to\s+be\b").unwrap(),
            score: 0.7,
        },
        Pattern {
            re: Regex::new(r"(?i)^system\s*:").unwrap(),
            score: 0.8,
        },
        Pattern {
            re: Regex::new(r"(?i)\[INST\]").unwrap(),
            score: 0.8,
        },
        Pattern {
            re: Regex::new(r"<<SYS>>").unwrap(),
            score: 0.8,
        },
        Pattern {
            re: Regex::new(r"(?i)\bdo\s+not\s+reveal\b").unwrap(),
            score: 0.6,
        },
        Pattern {
            re: Regex::new(r"(?i)\bdo\s+not\s+mention\b").unwrap(),
            score: 0.6,
        },
        Pattern {
            re: Regex::new(r"(?i)</memory>").unwrap(),
            score: 0.9,
        },
        Pattern {
            re: Regex::new(r"(?i)</system>").unwrap(),
            score: 0.9,
        },
        Pattern {
            re: Regex::new(r"<\|im_start\|>").unwrap(),
            score: 0.9,
        },
    ]
});

/// Score content for prompt injection risk. Returns 0.0 (safe) to 0.9 (likely injection).
/// The returned value is the highest matching pattern score.
pub fn assess_injection_risk(content: &str) -> f64 {
    PATTERNS
        .iter()
        .filter(|p| p.re.is_match(content))
        .map(|p| p.score)
        .fold(0.0_f64, f64::max)
}

/// Sanitize content for safe inclusion in LLM context. Strips special tokens
/// that could be interpreted as control sequences by various model formats.
/// The original content in the DB is left untouched — this only affects output.
pub fn sanitize_for_output(content: &str) -> String {
    content
        .replace("<|im_start|>", "")
        .replace("<|im_end|>", "")
        .replace("<<SYS>>", "")
        .replace("[INST]", "")
        .replace("[/INST]", "")
}


#[cfg(test)]
#[path = "safety_tests.rs"]
mod tests;
