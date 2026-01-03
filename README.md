# LangGraph Multi-Agent

階層型マルチエージェントアーキテクチャを実現する内部バックエンドAPIシステム。
Next.jsアプリケーションからのみアクセスされ、LangGraphによるエージェント処理をSSE（Server-Sent Events）でストリーミング返却します。

## システム概要

### 本番運用を想定したマルチテナント構造

このシステムは、個人利用ではなく**本番運用での企業向けマルチテナント構造**を前提に設計されています。

- **各企業が契約**: 複数の企業（テナント）が契約し、それぞれ独立したエージェント環境を利用
- **ユーザー管理はフロントエンド**: フロントエンド（Next.jsなど）でユーザー管理・認証を実施
- **バックエンド通信**: 完全にバックエンド側の通信で、このエージェントシステムが動作
- **会社コンテキストの送信**: 各企業のビジョン、社内用語、参考情報などをリクエスト時に送信可能
- **動的なエージェント/ツール定義**: Admin API経由で、各企業ごとにエージェントやツールを動的に登録・管理

#### 会社コンテキストの活用

各リクエストに会社固有の情報を含めることで、エージェントの応答を企業文化に合わせてカスタマイズできます：

```json
{
  "message": "システムにアクセスできません",
  "company_context": {
    "company_id": "acme_corp",
    "company_name": "Acme Corporation",
    "vision": "革新的なソリューションで顧客の課題を解決する",
    "terminology": {
      "システム": "プラットフォーム",
      "ユーザー": "メンバー"
    },
    "reference_info": {
      "support_hours": "平日9:00-18:00",
      "escalation_policy": "重大な問題は即座にチームリーダーに報告"
    }
  }
}
```

エージェントは、会社のビジョンや用語を理解し、その企業に最適化された回答を生成します。

### 特徴

- **マルチテナント対応**: 企業ごとに独立したエージェント環境を提供
- **会社コンテキスト**: 各企業のビジョン、用語、参考情報を考慮した応答生成
- **階層型マルチエージェント**: MainAgent（Supervisor）が複数のSubAgentを制御
- **動的Ad-hocエージェント生成**: MainAgentがツールを分析し、最適なエージェントをその場で生成
- **並列実行**: 複数のSubAgentを同時実行可能
- **SSEストリーミング**: リアルタイムで処理状況を返却
- **自律的リトライ**: SubAgentが検索結果を評価し、必要に応じて再検索
- **テンプレートエージェント**: よく使うパターンを事前定義して再利用
- **トークン管理**: 二層トークン構造（アクセスキー + サービストークン）
- **動的エージェント/ツール管理**: API経由でエージェントやツールを動的に追加・変更・削除
- **OpenAPIサポート**: OpenAPI仕様からツールを自動生成・登録
- **プロンプトキャッシュ**: AWS Bedrock Prompt Cachingによるコスト削減とレイテンシ改善

### アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MainAgent（Supervisor）                            │
│                                                                             │
│  1. ユーザー意図の解析                                                       │
│  2. 利用可能なツールを分析                                                   │
│  3. 最適なエージェント構成を決定（テンプレート or Ad-hoc生成）                 │
│  4. 並列実行計画の策定                                                       │
│  5. 結果の評価・統合・応答生成                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                │                          │                          │
                ▼                          ▼                          ▼
┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
│   Template Agent      │  │    Ad-hoc Agent       │  │    Ad-hoc Agent       │
│   (事前定義済み)       │  │   (動的生成)          │  │   (動的生成)          │
│                       │  │                       │  │                       │
│  knowledge_search     │  │  目的: ○○を調査       │  │  目的: △△を分析       │
│  Tools:               │  │  Tools:               │  │  Tools:               │
│  - snow_kb_search     │  │  - tool_A             │  │  - tool_C             │
│  - snow_case_search   │  │  - tool_B             │  │  - tool_D             │
└───────────────────────┘  └───────────────────────┘  └───────────────────────┘
          │                          │                          │
          └──────────────────────────┼──────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Tool Registry                                     │
│                                                                             │
│  静的ツール (Python実装)         │  動的ツール (API経由で定義)              │
│  - servicenow_knowledge_search  │  - weather_lookup                        │
│  - servicenow_case_search       │  - external_api_call                     │
│  - vector_db_search             │  - custom_http_tool                      │
│  - catalog_list                 │  - ...                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## エージェント生成モード

### Dynamic Mode（デフォルト）

MainAgentが利用可能なツールを分析し、タスクに最適なエージェントをその場で生成します。

```yaml
# config/agents.yaml
planning:
  dynamic_mode: true
  prefer_templates: true  # テンプレートがあれば優先
```

**動作フロー:**
1. ユーザーの質問を分析
2. 必要な能力（capabilities）を特定
3. 関連するツールを選択
4. テンプレートエージェントが適切か判断
5. 適切なテンプレートがなければ、Ad-hocエージェントを生成
6. 並列実行可能なタスクをグループ化
7. 実行・結果統合

**例: Ad-hocエージェントの生成**
```
ユーザー: 「ServiceNowで最近のインシデントと関連するナレッジを調べて」

MainAgentの思考:
- 必要な能力: incident_search, knowledge_search
- 関連ツール: servicenow_case_search, servicenow_knowledge_search, vector_db_search
- テンプレート "comprehensive_search" が適切
- 並列実行可能

生成される実行計画:
- Task 1 (Template): knowledge_search (servicenow_knowledge_search, servicenow_case_search)
- Task 2 (Template): vector_search (vector_db_search)
- Parallel Group: [Task 1, Task 2]
```

### Simple Mode

事前定義されたエージェントのみを使用します。

```yaml
planning:
  dynamic_mode: false
```

## テンプレートエージェント

よく使うエージェントパターンを事前定義できます。

```yaml
# config/agents.yaml
template_agents:
  comprehensive_search:
    enabled: true
    description: "包括的な検索を実行（ナレッジベース + ベクトル検索）"
    purpose: "ユーザーの質問に対して、複数のソースから情報を収集"
    capabilities:
      - knowledge_search
      - semantic_search
    tools:
      - servicenow_knowledge_search
      - vector_db_search
    parallel_execution: true
    expected_output: "複数ソースからの検索結果とその関連度"

  it_support:
    enabled: true
    description: "IT問題のトラブルシューティング"
    purpose: "IT関連の問題を診断し、解決策を提示"
    capabilities:
      - troubleshooting
      - case_search
    tools:
      - servicenow_knowledge_search
      - servicenow_case_search
    parallel_execution: false
    expected_output: "問題の診断結果と推奨される解決手順"
```

## Ad-hocエージェント

Plannerがタスクに応じて動的に生成するエージェントです。

**AdHocAgentSpec（仕様）:**
```python
{
    "id": "adhoc_abc123",
    "name": "custom_search_agent",
    "purpose": "ServiceNowとベクトルDBを横断して情報を収集",
    "tools": ["servicenow_knowledge_search", "vector_db_search"],
    "expected_output": "関連する情報のリストと要約",
    "reasoning": "ユーザーが包括的な検索を求めているため、両方のツールを使用"
}
```

**特徴:**
- 実行時に動的に生成・破棄
- 必要なツールのみを持つ軽量なエージェント
- LLMがシステムプロンプトを自動生成
- リトライロジックも備える

## 実行計画の構造

```python
ExecutionPlan(
    tasks=[
        Task(
            id="task_0",
            agent_name="knowledge_search",  # テンプレート使用
            parameters={"query": "VPN接続エラー"}
        ),
        Task(
            id="task_1",
            adhoc_spec=AdHocAgentSpec(  # Ad-hoc生成
                name="adhoc_incident_analyzer",
                purpose="関連インシデントを分析",
                tools=["servicenow_case_search"],
            ),
            parameters={"query": "VPN インシデント"}
        ),
    ],
    parallel_groups=[
        ParallelGroup(
            group_id="search_phase",
            task_ids=["task_0", "task_1"],
            timeout_seconds=30
        )
    ],
    execution_order=["search_phase"]
)
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
| `plan_created` | 実行計画策定時 | tasks（テンプレート/ad-hoc情報含む）, parallel_groups |
| `agent_start` | SubAgent開始時 | agent_name, type (template/adhoc), tools |
| `agent_retry` | SubAgentリトライ時 | agent_name, attempt, modified_query |
| `agent_end` | SubAgent終了時 | agent_name, status, duration_ms, type |
| `tool_call` | ツール呼び出し時 | tool_name, agent_name |
| `tool_result` | ツール結果時 | tool_name, success |
| `evaluation` | 中間評価時 | has_sufficient_info, next_action |
| `token` | トークン生成時 | content |
| `llm_metrics` | LLM完了時 | input_tokens, output_tokens, cost_usd |
| `session_complete` | 正常完了時 | 全メトリクス情報（ad-hoc情報含む） |
| `error` | エラー発生時 | error詳細 + partial_metrics |

## 動的ツール/エージェント管理

### 概要

静的なPython実装に加えて、API経由で動的にツールやエージェントを追加・変更・削除できます。

- **静的ツール/エージェント**: Pythonコードで実装、デプロイ時に登録
- **動的ツール/エージェント**: API経由で定義、ランタイムで登録・変更可能
- **テンプレートエージェント**: YAML設定で定義、よく使うパターンを再利用
- **Ad-hocエージェント**: 実行時にPlannerが動的に生成

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
langgraph-multi-agent/
├── docker/                      # Docker設定
├── src/
│   ├── main.py                  # アプリケーションエントリーポイント
│   ├── config/                  # 設定
│   │   ├── settings.py          # 環境変数設定
│   │   ├── models.py            # モデル設定
│   │   ├── agents.yaml          # エージェント/ツール/テンプレート設定（マルチテナント用テンプレート）
│   │   └── agents.yaml.example  # 設定例（サンプル定義）
│   ├── api/                     # API層
│   │   ├── routes/              # ルート定義
│   │   │   ├── agent.py         # エージェント実行API
│   │   │   ├── threads.py       # スレッド管理API
│   │   │   ├── health.py        # ヘルスチェックAPI
│   │   │   └── admin.py         # 動的管理Admin API
│   │   ├── middleware/          # ミドルウェア
│   │   └── schemas/             # リクエスト/レスポンススキーマ
│   ├── agents/                  # エージェント
│   │   ├── graph.py             # LangGraph定義（dynamic mode対応）
│   │   ├── state.py             # AgentState, AdHocAgentSpec定義
│   │   ├── registry.py          # Agent/Tool/Templateレジストリ
│   │   ├── main_agent/          # MainAgent
│   │   │   ├── agent.py         # MainAgent（dynamic mode対応）
│   │   │   ├── planner.py       # 実行計画策定（Ad-hoc生成対応）
│   │   │   ├── router.py        # ルーティング
│   │   │   ├── evaluator.py     # 中間評価
│   │   │   └── synthesizer.py   # 応答生成
│   │   ├── sub_agents/          # SubAgents
│   │   │   ├── base.py          # 基底クラス
│   │   │   ├── adhoc.py         # Ad-hocエージェント実行
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
│   │   │   └── parallel.py      # 並列実行（Ad-hoc対応）
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
| `AWS_ACCESS_KEY_ID` | ✓ | - | AWS認証用アクセスキーID |
| `AWS_SECRET_ACCESS_KEY` | ✓ | - | AWS認証用シークレットアクセスキー |
| `HTTP_PROXY` | - | - | HTTPプロキシURL（例: `http://proxy.example.com:8080`） |
| `HTTPS_PROXY` | - | - | HTTPSプロキシURL（例: `https://proxy.example.com:8080`） |
| `PROXY_CA_BUNDLE` | - | - | プロキシ用カスタムCA証明書バンドルのパス |
| `PROXY_CLIENT_CERT` | - | - | プロキシ認証用クライアント証明書のパス |
| `PROXY_USE_FORWARDING_FOR_HTTPS` | - | `true` | HTTPSプロキシにCONNECTメソッドを使用 |
| `ACCESS_KEY_SECRET` | ✓ | - | アクセスキー署名用 |
| `DEFAULT_MODEL_ID` | - | `anthropic.claude-3-5-sonnet-20241022-v2:0` | MainAgentデフォルト |
| `SUB_AGENT_MODEL_ID` | - | `anthropic.claude-3-5-haiku-20241022-v1:0` | SubAgentモデル |
| `CONTEXT_WARNING_THRESHOLD` | - | `80` | 警告閾値（%） |
| `CONTEXT_LOCK_THRESHOLD` | - | `95` | ロック閾値（%） |

## プロキシ設定

このシステムは、企業プロキシ環境下での動作をサポートしています。HTTP/HTTPSプロキシ経由でAWS Bedrockへのアクセスが可能です。

### 基本設定

環境変数で簡単にプロキシを設定できます：

```bash
# .env ファイルに追加
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=https://proxy.example.com:8080
```

### 高度な設定

企業環境で証明書認証が必要な場合：

```bash
# カスタムCA証明書を使用
PROXY_CA_BUNDLE=/path/to/corporate-ca-bundle.crt

# クライアント証明書認証
PROXY_CLIENT_CERT=/path/to/client-certificate.pem

# HTTPS接続にCONNECTメソッドを使用（デフォルト: true）
PROXY_USE_FORWARDING_FOR_HTTPS=true
```

### プロキシ設定の動作

- プロキシ設定は、AWS Bedrock APIへの全てのHTTP/HTTPS接続に適用されます
- boto3の`Config`オブジェクトを通じて設定され、LangChain経由のBedrock呼び出しにも適用されます
- プロキシが設定されている場合、起動時にログに記録されます
- プロキシ設定が不要な場合は、環境変数を設定しなければデフォルトの直接接続が使用されます

### 動作確認

プロキシ経由で正しく動作しているか確認する方法：

1. アプリケーション起動時のログを確認：
   ```
   {"event": "boto3_proxy_configured", "http_proxy": "http://proxy.example.com:8080", ...}
   ```

2. ヘルスチェックエンドポイントで確認：
   ```bash
   curl http://localhost:8000/health
   ```

3. 簡単なエージェント呼び出しをテスト：
   ```bash
   curl -X POST http://localhost:8000/api/v1/agent/invoke \
     -H "X-Access-Key: ${ACCESS_KEY}" \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello"}'
   ```

## プロンプトキャッシュ

このシステムは、AWS Bedrock Prompt Cachingを活用してコストとレイテンシを最適化しています。

### キャッシュ対象

以下のコンポーネントでプロンプトがキャッシュされます：

1. **Planner（実行計画策定）**
   - システムプロンプト + 利用可能なツールとテンプレートエージェントのリスト
   - 全リクエストで最初に実行されるため、最も効果が高い
   - ツール定義が変更されない限り、ほぼ100%のキャッシュヒット率

2. **Evaluator（中間評価）**
   - システムプロンプト: 固定の評価基準プロンプト
   - 会話履歴: 最新6メッセージ（文脈考慮のため）
   - SubAgent実行後の評価で毎回使用

3. **Synthesizer（最終応答生成）**
   - システムプロンプト: 固定の応答生成ガイドライン
   - 会話履歴: 最新10メッセージ（マルチターン対話で重要）
   - 最終回答の生成時に使用（通常とストリーミング両方）
   - 会話が続くほど、キャッシュ効果が大きくなる

### 効果

- **コスト削減**: キャッシュヒット時、入力トークンコストが**90%削減**
- **レイテンシ改善**: キャッシュされたプロンプトの処理が高速化
- **キャッシュTTL**: 5分間（ヒットするたびにリセット）
- **最小サイズ**: 1024トークン以上のプロンプトで効果を発揮

### 実装詳細

AWS Bedrockの`cache_control`機能を使用し、SystemMessageに以下の構造でキャッシュポイントを設定：

```python
SystemMessage(
    content=[
        {"type": "text", "text": "システムプロンプト内容"},
        {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
    ]
)
```

クロスリージョン推論プロファイル（`us.anthropic.*`など）でも正常に動作します。

## 会話履歴とマルチターン対話

このシステムはスレッド管理により会話履歴を保持し、マルチターン対話をサポートします。

### 会話履歴の活用

- **Synthesizer**: 最新10メッセージを参照し、文脈を考慮した回答を生成
- **Evaluator**: 最新6メッセージを参照し、会話の流れに沿った評価を実施
- **キャッシュ最適化**: 会話履歴もキャッシュされ、長い対話ほどコスト効率が向上

### スレッド管理

各スレッドでトークン使用量を追跡し、コンテキストウィンドウの限界に達する前に警告：

- **警告閾値**: 80% でwarning状態
- **ロック閾値**: 95% でlocked状態（新規メッセージ拒否）

## ライセンス

MIT License
