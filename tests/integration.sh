#!/usr/bin/env bash
# Integration tests for engram API
# Uses only fictional data — no PII, no real names/locations/keys.
set -euo pipefail

AUTH="${ENGRAM_API_KEY:-test-key}"
HOST="${ENGRAM_HOST:-http://127.0.0.1:3917}"
NS="integration-test-$$"  # unique namespace per run
PASS=0
FAIL=0

# Colors (if terminal supports them)
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ─── Helpers ────────────────────────────────────────────────────────

# req METHOD PATH [BODY]
# Sets $STATUS (http code) and $BODY (response body)
req() {
  local method="$1" path="$2" data="${3:-}"
  local curl_args=(
    --noproxy '*'
    -s -w '\n%{http_code}'
    -H "Authorization: Bearer $AUTH"
    -H "Content-Type: application/json"
    -X "$method"
  )
  if [[ -n "$data" ]]; then
    curl_args+=(-d "$data")
  fi
  local raw
  raw=$(curl "${curl_args[@]}" "${HOST}${path}")
  STATUS=$(echo "$raw" | tail -1)
  BODY=$(echo "$raw" | sed '$d')
}

# req_noauth METHOD PATH [BODY]  — same but without auth header
req_noauth() {
  local method="$1" path="$2" data="${3:-}"
  local curl_args=(
    --noproxy '*'
    -s -w '\n%{http_code}'
    -H "Content-Type: application/json"
    -X "$method"
  )
  if [[ -n "$data" ]]; then
    curl_args+=(-d "$data")
  fi
  local raw
  raw=$(curl "${curl_args[@]}" "${HOST}${path}")
  STATUS=$(echo "$raw" | tail -1)
  BODY=$(echo "$raw" | sed '$d')
}

# req_badauth METHOD PATH — with wrong auth
req_badauth() {
  local method="$1" path="$2"
  local raw
  raw=$(curl --noproxy '*' -s -w '\n%{http_code}' \
    -H "Authorization: Bearer wrong-key-abc123" \
    -H "Content-Type: application/json" \
    -X "$method" "${HOST}${path}")
  STATUS=$(echo "$raw" | tail -1)
  BODY=$(echo "$raw" | sed '$d')
}

assert_status() {
  local test_name="$1" expected="$2"
  if [[ "$STATUS" == "$expected" ]]; then
    echo -e "  ${GREEN}✓${NC} $test_name (HTTP $STATUS)"
    PASS=$((PASS + 1))
  else
    echo -e "  ${RED}✗${NC} $test_name — expected HTTP $expected, got $STATUS"
    echo "    Response: $(echo "$BODY" | head -c 300)"
    FAIL=$((FAIL + 1))
  fi
}

# assert_json TESTNAME JQFILTER EXPECTED
assert_json() {
  local test_name="$1" filter="$2" expected="$3"
  local actual
  actual=$(echo "$BODY" | jq -r "$filter" 2>/dev/null || echo "JQ_ERROR")
  if [[ "$actual" == "$expected" ]]; then
    echo -e "  ${GREEN}✓${NC} $test_name ($filter = $expected)"
    PASS=$((PASS + 1))
  else
    echo -e "  ${RED}✗${NC} $test_name — $filter: expected '$expected', got '$actual'"
    FAIL=$((FAIL + 1))
  fi
}

# assert_json_gte TESTNAME JQFILTER MIN_VALUE
assert_json_gte() {
  local test_name="$1" filter="$2" min_val="$3"
  local actual
  actual=$(echo "$BODY" | jq -r "$filter" 2>/dev/null || echo "0")
  if [[ "$actual" -ge "$min_val" ]] 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} $test_name ($filter = $actual >= $min_val)"
    PASS=$((PASS + 1))
  else
    echo -e "  ${RED}✗${NC} $test_name — $filter: expected >= $min_val, got '$actual'"
    FAIL=$((FAIL + 1))
  fi
}

assert_contains() {
  local test_name="$1" needle="$2"
  if echo "$BODY" | grep -q "$needle"; then
    echo -e "  ${GREEN}✓${NC} $test_name (contains '$needle')"
    PASS=$((PASS + 1))
  else
    echo -e "  ${RED}✗${NC} $test_name — body does not contain '$needle'"
    echo "    Body: $(echo "$BODY" | head -c 300)"
    FAIL=$((FAIL + 1))
  fi
}

echo "═══════════════════════════════════════════════════"
echo " engram integration tests"
echo " host: $HOST  namespace: $NS"
echo "═══════════════════════════════════════════════════"
echo ""

# ─── 1. Health endpoint ────────────────────────────────────────────

echo "1. Health endpoint"
req GET /health
assert_status "GET /health returns 200" 200
assert_json "response has name=engram" '.name' 'engram'
assert_contains "response has version field" '"version"'
echo ""

# ─── 2. Store a memory ─────────────────────────────────────────────

echo "2. Store a memory"
req POST "/memories" \
  "{\"content\": \"the team decided to use Redis for caching\", \"namespace\": \"$NS\"}"
assert_status "POST /memories returns 201" 201
MEM_ID1=$(echo "$BODY" | jq -r '.id')
assert_json "memory has correct content" '.content' 'the team decided to use Redis for caching'
assert_json "memory has correct namespace" '.namespace' "$NS"
echo "   stored id: $MEM_ID1"
echo ""

# ─── 3. Store with tags ────────────────────────────────────────────

echo "3. Store with tags"
req POST "/memories" \
  "{\"content\": \"alice prefers dark mode in the IDE\", \"tags\": [\"preference\", \"alice\"], \"namespace\": \"$NS\"}"
assert_status "POST /memories with tags returns 201" 201
MEM_ID2=$(echo "$BODY" | jq -r '.id')
assert_json "first tag is preference" '.tags[0]' 'preference'
echo ""

# ─── 4. Store procedural memory ────────────────────────────────────

echo "4. Store procedural memory (kind=procedural)"
req POST "/memories" \
  "{\"content\": \"deploy procedure: run tests, build release, stop service, copy binary, start service\", \"kind\": \"procedural\", \"tags\": [\"deploy\", \"procedure\"], \"namespace\": \"$NS\"}"
assert_status "POST procedural memory returns 201" 201
MEM_ID3=$(echo "$BODY" | jq -r '.id')
assert_json "kind is procedural" '.kind' 'procedural'
echo ""

# ─── 5. Store lesson ───────────────────────────────────────────────

echo "5. Store lesson with trigger"
req POST "/memories" \
  "{\"content\": \"never force-push to main without code review\", \"tags\": [\"lesson\", \"trigger:git-push\"], \"namespace\": \"$NS\"}"
assert_status "POST lesson memory returns 201" 201
MEM_ID4=$(echo "$BODY" | jq -r '.id')
assert_contains "has lesson tag" '"lesson"'
echo ""

# ─── 6. List memories ──────────────────────────────────────────────

echo "6. List memories"
req GET "/memories?ns=$NS"
assert_status "GET /memories returns 200" 200
assert_json_gte "at least 4 memories in namespace" '.count' 4
echo ""

# ─── 7. Get single memory by ID ────────────────────────────────────

echo "7. Get single memory by ID"
req GET "/memories/$MEM_ID1"
assert_status "GET /memories/:id returns 200" 200
assert_json "returns correct memory" '.id' "$MEM_ID1"
assert_contains "content matches" 'Redis for caching'
echo ""

# ─── 8. Recall (semantic search / keyword fallback) ─────────────────

echo "8. Recall (semantic search / keyword fallback)"
req POST "/recall" \
  "{\"query\": \"caching strategy\", \"namespace\": \"$NS\"}"
assert_status "POST /recall returns 200" 200
assert_contains "recall response has memories array" '"memories"'
echo ""

# ─── 9. FTS search ─────────────────────────────────────────────────

echo "9. FTS search"
req GET "/search?q=Redis&ns=$NS"
assert_status "GET /search returns 200" 200
assert_json_gte "finds at least 1 result" '.count' 1
assert_contains "result mentions Redis" 'Redis'
echo ""

# ─── 10. Recent memories ───────────────────────────────────────────

echo "10. Recent memories"
req GET "/recent?hours=1&ns=$NS"
assert_status "GET /recent returns 200" 200
assert_json_gte "recent has at least 1 memory" '.count' 1
echo ""

# ─── 11. Resume ────────────────────────────────────────────────────

echo "11. Resume"
req GET "/resume?hours=1&ns=$NS&budget=0"
assert_status "GET /resume returns 200" 200
assert_contains "resume has core section" '"core"'
assert_contains "resume has buffer section" '"buffer"'
assert_contains "resume has recent section" '"recent"'
assert_contains "resume has sessions section" '"sessions"'
assert_contains "resume has next_actions section" '"next_actions"'
echo ""

# ─── 12. Triggers ──────────────────────────────────────────────────

echo "12. Triggers"
req GET "/triggers/git-push"
assert_status "GET /triggers/git-push returns 200" 200
assert_json "trigger action matches" '.action' 'git-push'
assert_json_gte "at least 1 trigger memory" '.count' 1
assert_contains "trigger content about force-push" 'force-push'
echo ""

# ─── 13. Update memory (PATCH) ─────────────────────────────────────

echo "13. Update memory (PATCH)"
req PATCH "/memories/$MEM_ID1" \
  "{\"content\": \"the team decided to use Memcached instead of Redis\", \"tags\": [\"architecture\", \"caching\"]}"
assert_status "PATCH /memories/:id returns 200" 200
assert_contains "content updated" 'Memcached instead of Redis'
assert_contains "tags updated" '"architecture"'
echo ""

# ─── 14. Delete memory ─────────────────────────────────────────────

echo "14. Delete memory"
# Create a throwaway memory to delete
req POST "/memories" \
  "{\"content\": \"bob tested the login flow on staging\", \"namespace\": \"$NS\"}"
assert_status "create throwaway memory" 201
DEL_ID=$(echo "$BODY" | jq -r '.id')

req DELETE "/memories/$DEL_ID"
assert_status "DELETE /memories/:id returns 200" 200
assert_json "delete confirms ok" '.ok' 'true'

# Verify it's gone (should be 404)
req GET "/memories/$DEL_ID"
assert_status "deleted memory returns 404" 404
echo ""

# ─── 15. Namespace isolation ───────────────────────────────────────

echo "15. Namespace isolation"
OTHER_NS="other-ns-$$"
req POST "/memories" \
  "{\"content\": \"charlie wrote the API docs for the payment module\", \"namespace\": \"$OTHER_NS\"}"
assert_status "store in other namespace" 201

# Search in our test NS — should NOT find charlie's memory
req GET "/search?q=payment+module&ns=$NS"
CHARLIE_COUNT=$(echo "$BODY" | jq -r '.count')
if [[ "$CHARLIE_COUNT" == "0" ]]; then
  echo -e "  ${GREEN}✓${NC} namespace isolation: memory not visible in $NS"
  PASS=$((PASS + 1))
else
  echo -e "  ${RED}✗${NC} namespace isolation: memory leaked across namespaces"
  FAIL=$((FAIL + 1))
fi

# But it IS visible in its own namespace
req GET "/search?q=payment+module&ns=$OTHER_NS"
assert_json_gte "memory visible in own namespace" '.count' 1

# Cleanup other namespace
req DELETE "/memories" "{\"namespace\": \"$OTHER_NS\"}"
echo ""

# ─── 16. Auth rejection ────────────────────────────────────────────

echo "16. Auth rejection"
req_badauth GET "/memories"
assert_status "wrong key returns 401" 401

req_noauth GET "/memories"
assert_status "no auth returns 401" 401

# Public endpoints should work without auth
req_noauth GET "/health"
assert_status "health works without auth" 200

req_noauth GET "/stats"
assert_status "stats works without auth" 200
echo ""

# ─── 17. Stats endpoint ────────────────────────────────────────────

echo "17. Stats endpoint"
req GET "/stats?ns=$NS"
assert_status "GET /stats returns 200" 200
assert_json_gte "stats shows memories" '.total' 1
echo ""

# ─── 18. Export/Import round-trip ──────────────────────────────────

echo "18. Export/Import round-trip"
req GET "/export"
assert_status "GET /export returns 200" 200
EXPORT_COUNT=$(echo "$BODY" | jq -r '.count')
assert_json_gte "export has memories" '.count' 1
EXPORT_DATA="$BODY"

# Import into a fresh namespace
IMPORT_NS="import-test-$$"
# Modify the export data: inject namespace and give new IDs so they don't conflict
IMPORT_BODY=$(echo "$EXPORT_DATA" | jq --arg ns "$IMPORT_NS" \
  '.memories = [.memories[] | .namespace = $ns | .id = ("import-" + .id)]')
req POST "/import" "$IMPORT_BODY"
assert_status "POST /import returns 200" 200
IMPORTED=$(echo "$BODY" | jq -r '.imported')
if [[ "$IMPORTED" -ge 1 ]]; then
  echo -e "  ${GREEN}✓${NC} imported $IMPORTED memories"
  PASS=$((PASS + 1))
else
  echo -e "  ${RED}✗${NC} import count too low: $IMPORTED"
  FAIL=$((FAIL + 1))
fi

# Verify imported data exists
req GET "/memories?ns=$IMPORT_NS"
assert_json_gte "imported memories visible" '.count' 1

# Cleanup import namespace
req DELETE "/memories" "{\"namespace\": \"$IMPORT_NS\"}"
echo ""

# ─── Cleanup ────────────────────────────────────────────────────────

echo "Cleanup: deleting test namespace $NS"
req DELETE "/memories" "{\"namespace\": \"$NS\"}"
assert_status "batch delete test namespace" 200
DELETED=$(echo "$BODY" | jq -r '.deleted')
echo "   deleted $DELETED memories"
echo ""

# ─── Summary ────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════"
echo -e " Results: ${GREEN}PASS: $PASS${NC}  ${RED}FAIL: $FAIL${NC}"
echo "═══════════════════════════════════════════════════"

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
