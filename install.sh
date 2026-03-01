#!/usr/bin/env bash
set -euo pipefail

# engram installer — interactive setup wizard
# Usage: curl -fsSL https://raw.githubusercontent.com/kael-bit/engram-rs/main/install.sh | bash

REPO="kael-bit/engram-rs"

# Fetch latest version from GitHub
fetch_version() {
  local latest
  if has curl; then
    latest=$(curl -fsSL "https://api.github.com/repos/${REPO}/tags" 2>/dev/null | grep -o '"name": "v[^"]*"' | head -1 | cut -d'"' -f4 | sed 's/^v//')
  elif has wget; then
    latest=$(wget -qO- "https://api.github.com/repos/${REPO}/tags" 2>/dev/null | grep -o '"name": "v[^"]*"' | head -1 | cut -d'"' -f4 | sed 's/^v//')
  fi
  if [[ -z "$latest" ]]; then
    err "Failed to fetch latest version from GitHub"
    exit 1
  fi
  echo "$latest"
}
DEFAULT_PORT=3917
DEFAULT_DB="$HOME/.engram/engram.db"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}→${NC} $*"; }
ok()    { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}!${NC} $*"; }
err()   { echo -e "${RED}✗${NC} $*"; }
ask()   { echo -en "${BOLD}$*${NC} "; }

# --- Detect platform ---
detect_platform() {
  local os arch
  os=$(uname -s | tr '[:upper:]' '[:lower:]')
  arch=$(uname -m)

  case "$os" in
    linux)  OS="linux" ;;
    darwin) OS="macos" ;;
    *)      err "Unsupported OS: $os"; exit 1 ;;
  esac

  case "$arch" in
    x86_64|amd64) ARCH="x86_64" ;;
    aarch64|arm64) ARCH="aarch64" ;;
    *)             err "Unsupported architecture: $arch"; exit 1 ;;
  esac

  # Binary name from CI
  if [[ "$OS" == "macos" ]]; then
    BINARY_NAME="engram-macos-${ARCH}"
  else
    BINARY_NAME="engram-linux-${ARCH}"
  fi
}

# --- Check available tools ---
has() { command -v "$1" &>/dev/null; }

detect_tools() {
  HAVE_CARGO=false; HAVE_NODE=false
  has cargo  && HAVE_CARGO=true
  has node   && HAVE_NODE=true
}

# --- Install binary ---
install_binary() {
  local url="https://github.com/${REPO}/releases/download/v${VERSION}/${BINARY_NAME}"
  local install_dir="$HOME/.engram/bin"
  local target="$install_dir/engram"
  mkdir -p "$install_dir"

  if [[ -x "$target" ]]; then
    local existing_ver
    existing_ver=$("$target" --version 2>/dev/null || echo "unknown")
    warn "engram binary already exists at $target ($existing_ver)"
    ask "Re-download? [y/N]:"
    read -r redownload
    if [[ "$redownload" != "y" && "$redownload" != "Y" ]]; then
      ok "Keeping existing binary"
      # Ensure PATH
      if [[ ":$PATH:" != *":$install_dir:"* ]]; then
        export PATH="$install_dir:$PATH"
      fi
      return
    fi
  fi

  info "Downloading engram v${VERSION} for ${OS}/${ARCH}..."
  if has curl; then
    curl -fsSL "$url" -o "$target"
  elif has wget; then
    wget -qO "$target" "$url"
  else
    err "Need curl or wget to download binary"
    exit 1
  fi

  chmod +x "$target"

  # Add to PATH if not already there
  if [[ ":$PATH:" != *":$install_dir:"* ]]; then
    export PATH="$install_dir:$PATH"
    local shell_rc=""
    if [[ -f "$HOME/.bashrc" ]]; then shell_rc="$HOME/.bashrc"
    elif [[ -f "$HOME/.zshrc" ]]; then shell_rc="$HOME/.zshrc"
    fi
    if [[ -n "$shell_rc" ]] && ! grep -q '.engram/bin' "$shell_rc" 2>/dev/null; then
      echo 'export PATH="$HOME/.engram/bin:$PATH"' >> "$shell_rc"
      info "Added ~/.engram/bin to PATH in $(basename "$shell_rc")"
    fi
  fi

  ok "Installed engram to $install_dir/engram"
}

install_cargo() {
  info "Building engram from source (this may take a few minutes)..."
  cargo install --git "https://github.com/${REPO}.git" --tag "v${VERSION}"
  ok "Built and installed via cargo"
}

# --- Choose install method ---
choose_method() {
  echo ""
  echo -e "${BOLD}How would you like to install engram?${NC}"
  echo ""

  local options=()
  local n=1

  echo "  ${n}) Download pre-built binary (fastest)"
  options+=("binary")
  n=$((n+1))

  if $HAVE_CARGO; then
    echo "  ${n}) Build from source with cargo"
    options+=("cargo")
    n=$((n+1))
  fi

  echo ""
  ask "Choose [1]:"
  read -r choice
  choice=${choice:-1}

  if [[ "$choice" -lt 1 || "$choice" -gt ${#options[@]} ]]; then
    choice=1
  fi

  METHOD="${options[$((choice-1))]}"
}

# --- Configure ---
configure() {
  local env_file="$HOME/.engram/env"

  # Check for existing config
  if [[ -f "$env_file" ]]; then
    echo ""
    info "Existing configuration found:"
    echo ""
    grep -v '^#' "$env_file" | grep -v '^$' | while read -r line; do
      local key="${line%%=*}"
      local val="${line#*=}"
      # Mask API keys
      if [[ "$key" == *KEY* && ${#val} -gt 8 ]]; then
        val="${val:0:4}...${val: -4}"
      fi
      echo "    $key = $val"
    done
    echo ""
    ask "Use existing configuration? [Y/n]:"
    read -r use_existing
    if [[ "$use_existing" != "n" && "$use_existing" != "N" ]]; then
      # Load existing config
      set -a; source "$env_file"; set +a
      PORT="${ENGRAM_PORT:-$DEFAULT_PORT}"
      DB="${ENGRAM_DB:-$DEFAULT_DB}"
      LLM_URL="${ENGRAM_LLM_URL:-}"
      LLM_KEY="${ENGRAM_LLM_KEY:-}"
      LLM_MODEL="${ENGRAM_LLM_MODEL:-}"
      EMBED_URL="${ENGRAM_EMBED_URL:-}"
      EMBED_KEY="${ENGRAM_EMBED_KEY:-}"
      EMBED_MODEL="${ENGRAM_EMBED_MODEL:-}"
      SKIP_ENV_WRITE=true
      ok "Using existing config"
      return
    fi
  fi

  SKIP_ENV_WRITE=false
  echo ""
  echo -e "${BOLD}Configuration${NC}"
  echo ""

  # Port
  ask "Port [${DEFAULT_PORT}]:"
  read -r port
  PORT=${port:-$DEFAULT_PORT}

  # Database
  ask "Database path [${DEFAULT_DB}]:"
  read -r db
  DB=${db:-$DEFAULT_DB}

  # Ensure DB directory exists
  mkdir -p "$(dirname "$DB")"

  if [[ -f "$DB" ]]; then
    local db_size
    db_size=$(du -h "$DB" | cut -f1)
    warn "Database already exists at $DB ($db_size)"
    ask "Use existing database? [Y/n]:"
    read -r use_existing
    if [[ "$use_existing" == "n" || "$use_existing" == "N" ]]; then
      ask "New path:"
      read -r new_db
      if [[ -n "$new_db" ]]; then
        DB="$new_db"
        mkdir -p "$(dirname "$DB")"
      fi
    fi
  fi

  # LLM setup
  echo ""
  echo -e "${BOLD}LLM Configuration${NC} (optional — enables smart consolidation)"
  echo ""
  echo "  engram works without an LLM (embedding is already configured above)."
  echo "  With an LLM, it can automatically triage, merge, and audit memories."
  echo ""

  ask "LLM API URL (e.g. https://api.openai.com/v1, or leave empty to skip):"
  read -r llm_url
  LLM_URL=${llm_url:-}

  if [[ -n "$LLM_URL" ]]; then
    ask "API Key:"
    read -rs api_key
    echo ""
    LLM_KEY=${api_key:-}

    ask "Model [gpt-4o-mini]:"
    read -r model
    LLM_MODEL=${model:-gpt-4o-mini}
  else
    LLM_KEY=""
    LLM_MODEL=""
  fi

  # Embedding setup
  echo ""
  if [[ -n "$LLM_URL" ]]; then
    ask "Embedding API URL [same as LLM URL]:"
    read -r embed_url
    EMBED_URL=${embed_url:-${LLM_URL}}
    EMBED_KEY=${LLM_KEY}
  else
    echo "  engram needs an embedding API for semantic search."
    echo "  Any OpenAI-compatible endpoint works."
    echo ""
    while true; do
      ask "Embedding API URL (required):"
      read -r embed_url
      if [[ -n "$embed_url" ]]; then
        EMBED_URL="$embed_url"
        break
      fi
      warn "Embedding URL is required for engram to work."
    done
    ask "Embedding API Key:"
    read -rs embed_key
    echo ""
    EMBED_KEY=${embed_key:-}
  fi

  ask "Embedding model [text-embedding-3-small]:"
  read -r embed_model
  EMBED_MODEL=${embed_model:-text-embedding-3-small}
}

# --- Generate env file ---
generate_env() {
  if [[ "$SKIP_ENV_WRITE" == "true" ]]; then
    return
  fi

  local env_file="$HOME/.engram/env"
  mkdir -p "$HOME/.engram"

  # Fix ownership if previously created by root
  if [[ -d "$HOME/.engram" && ! -w "$HOME/.engram" ]]; then
    warn "~/.engram owned by another user, fixing with sudo..."
    sudo chown -R "$(id -u):$(id -g)" "$HOME/.engram"
  fi

  if [[ -f "$env_file" ]]; then
    echo ""
    warn "Existing config found at $env_file:"
    echo -e "  ${YELLOW}$(cat "$env_file" | grep -v '^#' | grep -v '^$')${NC}"
    echo ""
    ask "Overwrite? [y/N]:"
    read -r overwrite
    if [[ "$overwrite" != "y" && "$overwrite" != "Y" ]]; then
      info "Keeping existing config"
      return
    fi
  fi

  cat > "$env_file" <<EOF
# engram configuration — generated by install.sh
ENGRAM_PORT=${PORT}
ENGRAM_DB=${DB}
EOF

  if [[ -n "$LLM_URL" ]]; then
    cat >> "$env_file" <<EOF
ENGRAM_LLM_URL=${LLM_URL}
ENGRAM_LLM_KEY=${LLM_KEY}
ENGRAM_LLM_MODEL=${LLM_MODEL}
EOF
  fi

  if [[ -n "$EMBED_URL" ]]; then
    cat >> "$env_file" <<EOF
ENGRAM_EMBED_URL=${EMBED_URL}
ENGRAM_EMBED_KEY=${EMBED_KEY}
ENGRAM_EMBED_MODEL=${EMBED_MODEL}
EOF
  fi

  ok "Config saved to $env_file"
}

# --- MCP client config ---
setup_mcp() {
  echo ""
  ask "Configure MCP for a client? [y/N]:"
  read -r do_mcp
  [[ "$do_mcp" != "y" && "$do_mcp" != "Y" ]] && return

  echo ""
  echo "  1) Claude Code"
  echo "  2) Claude Desktop"
  echo "  3) Cursor"
  echo "  4) Skip"
  echo ""
  ask "Choose [1]:"
  read -r mcp_choice
  mcp_choice=${mcp_choice:-1}

  local mcp_json
  mcp_json=$(cat <<'MCPJSON'
{
  "mcpServers": {
    "engram": {
      "command": "npx",
      "args": ["-y", "engram-rs-mcp"],
      "env": {
        "ENGRAM_URL": "http://localhost:PORT"
      }
    }
  }
}
MCPJSON
)
  mcp_json="${mcp_json//PORT/$PORT}"

  case "$mcp_choice" in
    1)
      local cc_dir="$HOME/.claude"
      mkdir -p "$cc_dir"
      if [[ -f "$cc_dir/claude_desktop_config.json" ]]; then
        warn "Config exists at $cc_dir/claude_desktop_config.json — printing MCP config for manual merge:"
        echo "$mcp_json"
      else
        echo "$mcp_json" > "$cc_dir/claude_desktop_config.json"
        ok "Claude Code config written to $cc_dir/claude_desktop_config.json"
      fi
      ;;
    2)
      local cd_dir
      if [[ "$OS" == "macos" ]]; then
        cd_dir="$HOME/Library/Application Support/Claude"
      else
        cd_dir="$HOME/.config/claude"
      fi
      mkdir -p "$cd_dir"
      if [[ -f "$cd_dir/claude_desktop_config.json" ]]; then
        warn "Config exists — printing MCP config for manual merge:"
        echo "$mcp_json"
      else
        echo "$mcp_json" > "$cd_dir/claude_desktop_config.json"
        ok "Claude Desktop config written"
      fi
      ;;
    3)
      local cursor_dir="$HOME/.cursor"
      mkdir -p "$cursor_dir"
      if [[ -f "$cursor_dir/mcp.json" ]]; then
        warn "Config exists at $cursor_dir/mcp.json — printing MCP config for manual merge:"
        echo "$mcp_json"
      else
        echo "$mcp_json" > "$cursor_dir/mcp.json"
        ok "Cursor MCP config written"
      fi
      ;;
    *)
      return
      ;;
  esac
}

# --- Systemd service (Linux only) ---
setup_systemd() {
  [[ "$OS" != "linux" ]] && return

  echo ""
  ask "Create systemd service? [y/N]:"
  read -r do_systemd
  [[ "$do_systemd" != "y" && "$do_systemd" != "Y" ]] && return

  local service_file="/etc/systemd/system/engram.service"
  local engram_bin
  engram_bin=$(which engram 2>/dev/null || echo "/usr/local/bin/engram")

  sudo tee "$service_file" > /dev/null <<EOF
[Unit]
Description=engram memory server
After=network.target

[Service]
Type=simple
ExecStart=${engram_bin}
EnvironmentFile=$HOME/.engram/env
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

  sudo systemctl daemon-reload
  sudo systemctl enable engram
  sudo systemctl start engram
  ok "engram.service created and started"
}

# --- Start directly ---
start_engram() {
  echo ""
  echo -e "${BOLD}Ready to start!${NC}"
  echo ""
  echo "  Start now:  source ~/.engram/env && engram"
  echo "  Or use:     systemctl start engram (if you set up the service)"
  echo ""
  ask "Start engram now? [Y/n]:"
  read -r do_start
  if [[ "$do_start" != "n" && "$do_start" != "N" ]]; then
    info "Starting engram on port $PORT..."
    set -a; source "$HOME/.engram/env"; set +a
    nohup engram > "$HOME/.engram/engram.log" 2>&1 &
    disown
    sleep 1
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
      ok "engram is running at http://localhost:${PORT}"
    else
      warn "engram started but health check failed — check logs"
    fi
  fi
}

# --- Summary ---
summary() {
  echo ""
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${GREEN}${BOLD}  engram v${VERSION} installed!${NC}"
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo ""
  echo "  API:    http://localhost:${PORT}"
  echo "  DB:     ${DB}"
  echo "  Config: ~/.engram/env"
  echo ""
  echo "  Quick test:"
  echo "    curl localhost:${PORT}/health"
  echo "    curl localhost:${PORT}/memories -d '{\"content\":\"hello world\"}'"
  echo "    curl localhost:${PORT}/recall -d '{\"query\":\"hello\"}'"
  echo ""
  echo "  Stop:  kill \$(lsof -ti:${PORT})"
  echo "  Logs:  ~/.engram/engram.log"
  echo "  Docs:  https://github.com/${REPO}"
  echo ""
}

# --- Main ---
main() {
  echo ""
  echo -e "${BOLD}  ┌─────────────────────────────────┐${NC}"
  echo -e "${BOLD}  │   engram — memory for agents    │${NC}"
  echo -e "${BOLD}  └─────────────────────────────────┘${NC}"
  echo ""

  detect_platform
  detect_tools

  info "Checking latest version..."
  VERSION=$(fetch_version)
  ok "Latest version: v${VERSION}"
  choose_method

  case "$METHOD" in
    binary) install_binary ;;
    cargo)  install_cargo ;;
  esac

  configure
  generate_env

  setup_systemd

  setup_mcp
  start_engram
  summary
}

main "$@"
