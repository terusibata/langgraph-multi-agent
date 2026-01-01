# LangGraph Multi-Agent Backend

階層型マルチエージェントアーキテクチャを実現する内部バックエンドAPIシステム。
Next.jsアプリケーションからのみアクセスされ、LangGraphによるエージェント処理をSSE（Server-Sent Events）でストリーミング返却します。

## システム概要

### 特徴

- **階層型マルチエージェント**: MainAgent（Supervisor）が複数のSubAgentを制御
- **並列実行**: 複数のSubAgentを同時実行可能
- **SSEストリーミング**: リアルタイムで処理状況を返却
- **自律的リトライ**: SubAgentが検索結果を評価し、必要に応じて再検索
- **トークン管理**: 二層トークン構造（アクセスキー + サービストークン）

### アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│                         MainAgent（Supervisor）                  │
│  - ユーザー意図の解析                                            │
│  - タスク分解・実行計画策定                                      │
│  - SubAgentへのルーティング（並列/逐次）                         │
│  - 中間結果の評価・追加アクション判断                            │
│  - 最終結果の統合・応答生成                                      │
└─────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   SubAgent A    │  │   SubAgent B    │  │   SubAgent C    │
│  ナレッジ検索    │  │  ベクトル検索    │  │  カタログ調査    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     Tools       │  │     Tools       │  │     Tools       │
│ - SNOW検索API   │  │ - ベクトルDB    │  │ - カタログ一覧  │
│ - SNOW詳細API   │  │   検索API       │  │ - カタログ詳細  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## クイックスタート

### 前提条件

- Python 3.11+
- Docker & Docker Compose
- AWS credentials（Bedrock アクセス用）

### セットアップ

1. リポジトリをクローン:

```bash
git clone <repository-url>
cd langgraph-multi-agent-backend
```

2. 環境変数を設定:

```bash
cp .env.example .env
# .env ファイルを編集して必要な値を設定
```

3. Docker Compose で起動:

```bash
cd docker
docker-compose up -d
```

4. ヘルスチェック:

```bash
curl http://localhost:8000/health
```

### ローカル開発

```bash
# 仮想環境を作成
python -m venv .venv
source .venv/bin/activate

# 依存関係をインストール
pip install -e ".[dev]"

# 開発サーバーを起動
python -m src.main
```

## API エンドポイント

### エージェント実行

| メソッド | パス | 説明 |
|---------|------|------|
| POST | `/api/v1/agent/stream` | エージェント実行（SSEストリーミング） |
| POST | `/api/v1/agent/invoke` | エージェント実行（同期） |
| GET | `/api/v1/models` | 利用可能モデル一覧 |
| GET | `/api/v1/agents` | 利用可能SubAgent一覧 |
| GET | `/api/v1/tools` | 利用可能Tool一覧 |

### スレッド管理

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/api/v1/threads/{thread_id}` | スレッド情報取得 |
| GET | `/api/v1/threads/{thread_id}/status` | ステータスのみ取得 |
| DELETE | `/api/v1/threads/{thread_id}` | スレッド削除 |

### ヘルスチェック

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/health` | ヘルスチェック |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe |
| GET | `/metrics` | Prometheusメトリクス |

## 認証

### リクエストヘッダー

| ヘッダー | 必須 | 説明 |
|---------|------|------|
| X-Access-Key | ✓ | バックエンドアクセス用キー（JWT） |
| X-Service-Tokens | - | 外部サービストークン（Base64エンコードJSON） |

### X-Service-Tokens 形式

```json
{
  "servicenow": {
    "token": "eyJhbGciOiJSUzI1NiIs...",
    "instance": "company.service-now.com",
    "expires_at": "2025-12-31T12:00:00Z"
  },
  "vector_db": {
    "api_key": "vdb_xxxxxxxxxxxx"
  }
}
```

## SSE イベント種別

| イベント種別 | タイミング | 含まれるデータ |
|-------------|-----------|---------------|
| `session_start` | 処理開始時 | session_id, thread_id |
| `plan_created` | 実行計画策定時 | execution_plan概要 |
| `agent_start` | SubAgent開始時 | agent_name |
| `agent_retry` | SubAgentリトライ時 | agent_name, attempt, modified_query |
| `agent_end` | SubAgent終了時 | agent_name, status, duration_ms |
| `tool_call` | ツール呼び出し時 | tool_name, agent_name |
| `tool_result` | ツール結果時 | tool_name, success |
| `evaluation` | 中間評価時 | has_sufficient_info, next_action |
| `token` | トークン生成時 | content |
| `llm_metrics` | LLM完了時 | input_tokens, output_tokens, cost_usd |
| `session_complete` | 正常完了時 | 全メトリクス情報 |
| `error` | エラー発生時 | error詳細 + partial_metrics |

## プロジェクト構成

```
langgraph-multi-agent-backend/
├── docker/                      # Docker設定
├── src/
│   ├── main.py                  # アプリケーションエントリーポイント
│   ├── config/                  # 設定
│   │   ├── settings.py          # 環境変数設定
│   │   ├── models.py            # モデル設定
│   │   └── agents.yaml          # SubAgent/Tool設定
│   ├── api/                     # API層
│   │   ├── routes/              # ルート定義
│   │   ├── middleware/          # ミドルウェア
│   │   └── schemas/             # リクエスト/レスポンススキーマ
│   ├── agents/                  # エージェント
│   │   ├── graph.py             # LangGraph定義
│   │   ├── state.py             # AgentState定義
│   │   ├── registry.py          # Agent/Toolレジストリ
│   │   ├── main_agent/          # MainAgent
│   │   ├── sub_agents/          # SubAgents
│   │   └── tools/               # Tools
│   ├── services/                # サービス層
│   │   ├── llm/                 # LLMサービス
│   │   ├── execution/           # 実行制御
│   │   ├── streaming/           # SSEストリーミング
│   │   ├── metrics/             # メトリクス収集
│   │   ├── thread/              # スレッド管理
│   │   └── error/               # エラーハンドリング
│   └── utils/                   # ユーティリティ
├── tests/                       # テスト
├── alembic/                     # DBマイグレーション
├── pyproject.toml               # プロジェクト設定
└── requirements.txt             # 依存関係
```

## 環境変数

| 変数名 | 必須 | デフォルト | 説明 |
|--------|------|-----------|------|
| `DATABASE_URL` | ✓ | - | PostgreSQL接続URL |
| `REDIS_URL` | ✓ | - | Redis接続URL |
| `AWS_REGION` | ✓ | `us-east-1` | AWSリージョン |
| `ACCESS_KEY_SECRET` | ✓ | - | アクセスキー署名用 |
| `DEFAULT_MODEL_ID` | - | `claude-3-5-sonnet` | MainAgentデフォルト |
| `SUB_AGENT_MODEL_ID` | - | `claude-3-5-haiku` | SubAgentモデル |
| `CONTEXT_WARNING_THRESHOLD` | - | `80` | 警告閾値（%） |
| `CONTEXT_LOCK_THRESHOLD` | - | `95` | ロック閾値（%） |

## ライセンス

MIT License
