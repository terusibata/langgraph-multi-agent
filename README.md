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
- **動的エージェント/ツール管理**: API経由でエージェントやツールを動的に追加・変更・削除
- **OpenAPIサポート**: OpenAPI仕様からツールを自動生成・登録

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
│   (静的/動的)   │  │   (静的/動的)   │  │   (静的/動的)   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     Tools       │  │     Tools       │  │     Tools       │
│ - SNOW検索API   │  │ - ベクトルDB    │  │ - カタログ一覧  │
│ - SNOW詳細API   │  │   検索API       │  │ - カタログ詳細  │
│   (静的/動的)   │  │   (静的/動的)   │  │   (静的/動的)   │
└─────────────────┘  └─────────────────┘  └─────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        Admin API                                │
│  - 動的ツール管理（CRUD）                                        │
│  - 動的エージェント管理（CRUD）                                  │
│  - OpenAPI仕様からのツール自動生成                               │
│  - 一括インポート/エクスポート                                   │
└─────────────────────────────────────────────────────────────────┘
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

### 動的ツール管理（Admin API）

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/api/v1/admin/tools` | ツール定義一覧取得 |
| POST | `/api/v1/admin/tools` | ツール定義作成 |
| GET | `/api/v1/admin/tools/{name}` | ツール定義取得 |
| PUT | `/api/v1/admin/tools/{name}` | ツール定義更新 |
| DELETE | `/api/v1/admin/tools/{name}` | ツール定義削除 |
| POST | `/api/v1/admin/tools/{name}/test` | ツール実行テスト |
| POST | `/api/v1/admin/tools/bulk` | 一括操作（有効化/無効化/削除） |

### 動的エージェント管理（Admin API）

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/api/v1/admin/agents` | エージェント定義一覧取得 |
| POST | `/api/v1/admin/agents` | エージェント定義作成 |
| GET | `/api/v1/admin/agents/{name}` | エージェント定義取得 |
| PUT | `/api/v1/admin/agents/{name}` | エージェント定義更新 |
| DELETE | `/api/v1/admin/agents/{name}` | エージェント定義削除 |
| POST | `/api/v1/admin/agents/{name}/test` | エージェント実行テスト |
| POST | `/api/v1/admin/agents/bulk` | 一括操作（有効化/無効化/削除） |

### OpenAPIインポート（Admin API）

| メソッド | パス | 説明 |
|---------|------|------|
| POST | `/api/v1/admin/tools/openapi` | OpenAPI仕様からツール生成・登録 |
| POST | `/api/v1/admin/tools/openapi/url` | URLからOpenAPI仕様を取得してツール生成 |
| POST | `/api/v1/admin/tools/openapi/preview` | ツール生成プレビュー（登録なし） |

### インポート/エクスポート（Admin API）

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/api/v1/admin/export` | 全定義をエクスポート |
| POST | `/api/v1/admin/import` | 定義をインポート |

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

## 動的ツール/エージェント管理

### 概要

静的なPython実装に加えて、API経由で動的にツールやエージェントを追加・変更・削除できます。

- **静的ツール/エージェント**: Pythonコードで実装、デプロイ時に登録
- **動的ツール/エージェント**: API経由で定義、ランタイムで登録・変更可能

### ツール定義の例

```json
{
  "name": "weather_lookup",
  "description": "指定した都市の現在の天気を取得します",
  "category": "external_api",
  "parameters": [
    {
      "name": "city",
      "type": "string",
      "description": "都市名",
      "required": true
    }
  ],
  "executor": {
    "type": "http",
    "url": "https://api.weather.com/v1/current?city={{params.city}}",
    "method": "GET",
    "auth_type": "api_key",
    "auth_config": {
      "header": "X-API-Key",
      "key_env": "WEATHER_API_KEY"
    },
    "response_path": "$.data"
  },
  "timeout_seconds": 10,
  "enabled": true
}
```

### エージェント定義の例

```json
{
  "name": "weather_agent",
  "description": "天気情報を取得・分析するエージェント",
  "capabilities": ["weather_lookup", "weather_forecast"],
  "tools": ["weather_lookup", "forecast_api"],
  "executor": {
    "type": "llm",
    "system_prompt": "あなたは天気情報の専門家です。ユーザーの質問に基づいて適切な天気情報を提供してください。",
    "temperature": 0.0,
    "max_tokens": 2048
  },
  "retry_strategy": {
    "max_attempts": 3,
    "retry_conditions": ["no_results", "api_error"],
    "query_modification": "synonym",
    "backoff_seconds": 1.0
  },
  "priority": 10,
  "enabled": true
}
```

### エクゼキュータータイプ

#### ツールエクゼキューター

| タイプ | 説明 | 主な設定 |
|--------|------|----------|
| `http` | HTTP APIを呼び出し | `url`, `method`, `headers`, `auth_type` |
| `python` | Pythonモジュールを実行 | `module_path`, `function_name`, `class_name` |
| `mock` | モックレスポンスを返却（テスト用） | `mock_response`, `mock_delay_ms` |

#### エージェントエクゼキューター

| タイプ | 説明 | 主な設定 |
|--------|------|----------|
| `llm` | LLMによる推論・ツール呼び出し | `model_id`, `system_prompt`, `temperature` |
| `rule_based` | ルールベースで処理 | `rules` (条件とアクションの配列) |
| `hybrid` | ルール優先、LLMフォールバック | 両方の設定を併用 |

### OpenAPIからのツールインポート

OpenAPI 3.x仕様からツールを自動生成できます。

```bash
# JSON仕様を直接送信
curl -X POST http://localhost:8000/api/v1/admin/tools/openapi \
  -H "X-Access-Key: ${ACCESS_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "spec": { ... OpenAPI仕様 ... },
    "options": {
      "prefix": "petstore",
      "include_tags": ["pets"],
      "auth_config": {
        "type": "api_key",
        "header_name": "X-API-Key",
        "token_env": "PETSTORE_API_KEY"
      }
    }
  }'

# URLから取得
curl -X POST http://localhost:8000/api/v1/admin/tools/openapi/url \
  -H "X-Access-Key: ${ACCESS_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://petstore.swagger.io/v2/swagger.json",
    "options": {
      "prefix": "petstore"
    }
  }'
```

### 必要な権限

Admin APIは以下の権限が必要です（アクセスキーのJWTに含める）:

| 権限 | 説明 |
|------|------|
| `admin:tools:read` | ツール定義の参照 |
| `admin:tools:write` | ツール定義の作成・更新・削除 |
| `admin:agents:read` | エージェント定義の参照 |
| `admin:agents:write` | エージェント定義の作成・更新・削除 |
| `admin:import` | 定義のインポート |
| `admin:export` | 定義のエクスポート |

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
│   │   │   ├── agent.py         # エージェント実行API
│   │   │   ├── threads.py       # スレッド管理API
│   │   │   ├── health.py        # ヘルスチェックAPI
│   │   │   └── admin.py         # 動的管理Admin API
│   │   ├── middleware/          # ミドルウェア
│   │   └── schemas/             # リクエスト/レスポンススキーマ
│   │       ├── request.py       # リクエストスキーマ
│   │       ├── response.py      # レスポンススキーマ
│   │       └── admin.py         # Admin APIスキーマ
│   ├── agents/                  # エージェント
│   │   ├── graph.py             # LangGraph定義
│   │   ├── state.py             # AgentState定義
│   │   ├── registry.py          # Agent/Toolレジストリ（静的/動的）
│   │   ├── main_agent/          # MainAgent
│   │   │   └── planner.py       # 実行計画策定（動的エージェント対応）
│   │   ├── sub_agents/          # SubAgents
│   │   │   ├── base.py          # 基底クラス
│   │   │   ├── dynamic.py       # 動的エージェント実行
│   │   │   └── ...              # 静的エージェント実装
│   │   └── tools/               # Tools
│   │       ├── base.py          # 基底クラス
│   │       ├── dynamic.py       # 動的ツール実行
│   │       ├── openapi.py       # OpenAPIパーサー/ジェネレーター
│   │       └── ...              # 静的ツール実装
│   ├── services/                # サービス層
│   │   ├── llm/                 # LLMサービス
│   │   ├── execution/           # 実行制御
│   │   │   └── parallel.py      # 並列実行（動的エージェント対応）
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
