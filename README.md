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

### データベースストレージ

**全てのデータはPostgreSQLデータベースに永続化されます：**

- **エージェント定義**: 動的エージェントの設定はデータベースに保存
- **ツール定義**: 動的ツールの設定はデータベースに保存
- **スレッド管理**: 会話スレッドとコンテキスト情報をデータベースで管理
- **実行結果**: セッション、実行計画、SubAgent実行結果を完全に保存
- **設定**: プランニングモード、実行設定などのシステム設定もデータベースで管理

メモリ管理は完全に廃止され、全ての状態はデータベースで管理されます。

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
- **データベース永続化**: 全ての状態・定義・実行結果をPostgreSQLに保存
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
│  静的ツール (Python実装)         │  動的ツール (DB定義)                      │
│  - servicenow_knowledge_search  │  - weather_lookup                        │
│  - servicenow_case_search       │  - external_api_call                     │
│  - vector_db_search             │  - custom_http_tool                      │
│  - catalog_list                 │  - ...                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PostgreSQL Database                               │
│                                                                             │
│  agents        │  tools         │  threads      │  execution_sessions      │
│  template_agents                 │  execution_results  │  system_config   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## エージェント生成モード

### Dynamic Mode（デフォルト）

MainAgentが利用可能なツールを分析し、タスクに最適なエージェントをその場で生成します。

設定はデータベースの `system_config` テーブルで管理されます：

```json
{
  "key": "planning",
  "value": {
    "dynamic_mode": true,
    "prefer_templates": true
  }
}
```

**動作フロー:**
1. ユーザーの質問を分析
2. 必要な能力（capabilities）を特定
3. 関連するツールを選択
4. テンプレートエージェントが適切か判断
5. 適切なテンプレートがなければ、Ad-hocエージェントを生成
6. 並列実行可能なタスクをグループ化
7. 実行・結果統合
8. 実行結果をデータベースに保存

### Simple Mode

事前定義されたエージェントのみを使用します。

```json
{
  "key": "planning",
  "value": {
    "dynamic_mode": false
  }
}
```

## データベーススキーマ

### テーブル一覧

| テーブル名 | 説明 |
|-----------|------|
| `agents` | 動的エージェント定義 |
| `template_agents` | テンプレートエージェント定義 |
| `tools` | 動的ツール定義 |
| `threads` | 会話スレッド |
| `execution_sessions` | 実行セッション |
| `execution_results` | SubAgent/ツール実行結果 |
| `system_config` | システム設定 |

### マイグレーション

```bash
# マイグレーションを実行
alembic upgrade head

# 新しいマイグレーションを作成
alembic revision --autogenerate -m "description"
```

## クイックスタート

### 前提条件

- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 16+
- AWS credentials（Bedrock アクセス用）

### セットアップ

1. リポジトリをクローン:

```bash
git clone <repository-url>
cd langgraph-multi-agent
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

4. データベースマイグレーションを実行:

```bash
alembic upgrade head
```

5. ヘルスチェック:

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

# PostgreSQLを起動（Dockerで）
cd docker && docker-compose up -d postgres

# マイグレーションを実行
alembic upgrade head

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
| GET | `/api/v1/admin/tools` | ツール定義一覧取得（DBから） |
| POST | `/api/v1/admin/tools` | ツール定義作成（DBに保存） |
| GET | `/api/v1/admin/tools/{name}` | ツール定義取得 |
| PUT | `/api/v1/admin/tools/{name}` | ツール定義更新 |
| DELETE | `/api/v1/admin/tools/{name}` | ツール定義削除 |
| POST | `/api/v1/admin/tools/{name}/test` | ツール実行テスト |

### 動的エージェント管理（Admin API）

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/api/v1/admin/agents` | エージェント定義一覧取得（DBから） |
| POST | `/api/v1/admin/agents` | エージェント定義作成（DBに保存） |
| GET | `/api/v1/admin/agents/{name}` | エージェント定義取得 |
| PUT | `/api/v1/admin/agents/{name}` | エージェント定義更新 |
| DELETE | `/api/v1/admin/agents/{name}` | エージェント定義削除 |
| PATCH | `/api/v1/admin/agents/{name}/toggle` | エージェント有効/無効切り替え |

### OpenAPIインポート（Admin API）

| メソッド | パス | 説明 |
|---------|------|------|
| POST | `/api/v1/admin/tools/openapi` | OpenAPI仕様からツール生成・登録 |
| POST | `/api/v1/admin/tools/openapi/url` | URLからOpenAPI仕様を取得してツール生成 |
| POST | `/api/v1/admin/tools/openapi/preview` | ツール生成プレビュー（登録なし） |

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

## 実行結果の永続化

全ての実行結果はデータベースに保存され、後から参照可能です：

### execution_sessions テーブル

- セッションID、スレッドID、テナントID
- ユーザー入力と最終レスポンス
- 実行計画（JSON）
- 開始/完了時刻、所要時間
- トークン使用量、コスト
- LLM呼び出し詳細
- エラー情報（発生時）

### execution_results テーブル

- SubAgent/ツール実行結果
- Ad-hocエージェント仕様（該当する場合）
- 実行ステータス、データ、エラー
- リトライ回数、検索バリエーション
- 実行時間

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

## プロジェクト構成

```
langgraph-multi-agent/
├── docker/                      # Docker設定
├── alembic/                     # DBマイグレーション
│   ├── versions/                # マイグレーションスクリプト
│   └── env.py                   # Alembic環境設定
├── src/
│   ├── main.py                  # アプリケーションエントリーポイント
│   ├── config/                  # 設定
│   │   ├── settings.py          # 環境変数設定
│   │   └── models.py            # モデル設定
│   ├── models/                  # データベースモデル
│   │   ├── base.py              # ベースモデル・セッション管理
│   │   ├── agent.py             # エージェントモデル
│   │   ├── tool.py              # ツールモデル
│   │   ├── thread.py            # スレッドモデル
│   │   ├── execution.py         # 実行結果モデル
│   │   └── config.py            # 設定モデル
│   ├── repositories/            # リポジトリ層
│   │   ├── agent.py             # エージェントCRUD
│   │   ├── tool.py              # ツールCRUD
│   │   ├── thread.py            # スレッドCRUD
│   │   ├── execution.py         # 実行結果CRUD
│   │   └── config.py            # 設定CRUD
│   ├── api/                     # API層
│   │   ├── routes/              # ルート定義
│   │   │   ├── agent.py         # エージェント実行API
│   │   │   ├── threads.py       # スレッド管理API
│   │   │   ├── health.py        # ヘルスチェックAPI
│   │   │   ├── admin_agents.py  # エージェント管理Admin API
│   │   │   └── admin_tools.py   # ツール管理Admin API
│   │   ├── middleware/          # ミドルウェア
│   │   └── schemas/             # リクエスト/レスポンススキーマ
│   ├── agents/                  # エージェント
│   │   ├── graph.py             # LangGraph定義（DB永続化対応）
│   │   ├── state.py             # AgentState, AdHocAgentSpec定義
│   │   ├── registry.py          # Agent/Toolレジストリ（DB対応）
│   │   ├── main_agent/          # MainAgent
│   │   │   ├── agent.py         # MainAgent
│   │   │   ├── planner.py       # 実行計画策定
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
│   │       ├── openapi.py       # OpenAPIパーサー
│   │       └── ...              # 静的ツール実装
│   ├── services/                # サービス層
│   │   ├── llm/                 # LLMサービス
│   │   ├── execution/           # 実行制御
│   │   ├── streaming/           # SSEストリーミング
│   │   ├── metrics/             # メトリクス収集
│   │   ├── thread/              # スレッド管理（DB対応）
│   │   └── error/               # エラーハンドリング
│   └── utils/                   # ユーティリティ
├── tests/                       # テスト
├── alembic.ini                  # Alembic設定
├── pyproject.toml               # プロジェクト設定
└── requirements.txt             # 依存関係
```

## 環境変数

| 変数名 | 必須 | デフォルト | 説明 |
|--------|------|-----------|------|
| `DATABASE_URL` | ✓ | - | PostgreSQL接続URL（async: `postgresql+asyncpg://...`） |
| `DATABASE_URL_SYNC` | ✓ | - | PostgreSQL接続URL（sync: `postgresql://...`） |
| `REDIS_URL` | - | - | Redis接続URL（キャッシュ用） |
| `AWS_REGION` | ✓ | `us-east-1` | AWSリージョン |
| `AWS_ACCESS_KEY_ID` | ✓ | - | AWS認証用アクセスキーID |
| `AWS_SECRET_ACCESS_KEY` | ✓ | - | AWS認証用シークレットアクセスキー |
| `HTTP_PROXY` | - | - | HTTPプロキシURL |
| `HTTPS_PROXY` | - | - | HTTPSプロキシURL |
| `PROXY_CA_BUNDLE` | - | - | プロキシ用カスタムCA証明書バンドルのパス |
| `PROXY_CLIENT_CERT` | - | - | プロキシ認証用クライアント証明書のパス |
| `ACCESS_KEY_SECRET` | ✓ | - | アクセスキー署名用 |
| `DEFAULT_MODEL_ID` | - | `anthropic.claude-3-5-sonnet-20241022-v2:0` | MainAgentデフォルト |
| `SUB_AGENT_MODEL_ID` | - | `anthropic.claude-3-5-haiku-20241022-v1:0` | SubAgentモデル |
| `CONTEXT_WARNING_THRESHOLD` | - | `80` | 警告閾値（%） |
| `CONTEXT_LOCK_THRESHOLD` | - | `95` | ロック閾値（%） |

## 動的ツール/エージェント管理

### 概要

静的なPython実装に加えて、API経由で動的にツールやエージェントを追加・変更・削除できます。
全ての定義はPostgreSQLデータベースに保存され、サーバー再起動後も永続化されます。

- **静的ツール/エージェント**: Pythonコードで実装、デプロイ時に登録（メモリ内）
- **動的ツール/エージェント**: API経由で定義、データベースに保存
- **テンプレートエージェント**: データベースで定義、よく使うパターンを再利用
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
    "system_prompt": "あなたは天気情報の専門家です。",
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

## プロンプトキャッシュ

このシステムは、AWS Bedrock Prompt Cachingを活用してコストとレイテンシを最適化しています。

### キャッシュ対象

1. **Planner（実行計画策定）**: システムプロンプト + ツールリスト
2. **Evaluator（中間評価）**: 評価基準プロンプト + 会話履歴
3. **Synthesizer（最終応答生成）**: 応答生成ガイドライン + 会話履歴

### 効果

- **コスト削減**: キャッシュヒット時、入力トークンコストが**90%削減**
- **レイテンシ改善**: キャッシュされたプロンプトの処理が高速化
- **キャッシュTTL**: 5分間（ヒットするたびにリセット）

## 会話履歴とマルチターン対話

このシステムはスレッド管理により会話履歴を保持し、マルチターン対話をサポートします。

### スレッド管理

各スレッドでトークン使用量をデータベースで追跡し、コンテキストウィンドウの限界に達する前に警告：

- **警告閾値**: 80% でwarning状態
- **ロック閾値**: 95% でlocked状態（新規メッセージ拒否）

## ライセンス

MIT License
