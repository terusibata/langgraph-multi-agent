# LangGraph Multi-Agent

本番運用向けマルチテナント対応のLangGraphベースマルチエージェントシステム。

## 概要

- **マルチテナント**: 企業ごとに独立したエージェント環境
- **動的エージェント生成**: 実行時にタスク最適なAd-hocエージェントを自動生成
- **データベース永続化**: 全ての状態・定義・実行履歴をPostgreSQLに保存
- **SSEストリーミング**: リアルタイムで処理状況を返却
- **会社コンテキスト**: 企業のビジョン・用語・参考情報を考慮した応答生成
- **Admin API**: API経由でエージェント・ツールを動的に管理
- **エージェントテスト**: サンドボックス環境でエージェントをテスト実行可能
- **高速応答モード**: 用途に応じて最適な実行モードを選択可能（通常/高速/ダイレクトツール）

## アーキテクチャ

```
MainAgent (Supervisor)
  ↓
  ├─ Template Agents (事前定義)
  ├─ Dynamic Agents (DB定義)
  └─ Ad-hoc Agents (実行時生成)
      ↓
      Tools (Static + Dynamic)
          ↓
          PostgreSQL Database
```

**エージェントの種類:**
- **動的エージェント**: Admin APIで作成、DBに保存（推奨）
- **Ad-hocエージェント**: Plannerが実行時に動的生成（仕様はDB保存）

> **Note:** 静的エージェント・ツールのサンプル実装は`examples/`ディレクトリにあります。本番環境では動的ツール・エージェントの使用を推奨します。

## クイックスタート

### 前提条件

- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 16+
- AWS credentials（Bedrock用）

### セットアップ

```bash
# 1. 環境変数設定
cp .env.example .env
# .env を編集

# 2. Docker起動
cd docker
docker-compose up -d

# 3. マイグレーション
alembic upgrade head

# 4. ヘルスチェック
curl http://localhost:8000/health
```

### ローカル開発

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cd docker && docker-compose up -d postgres
alembic upgrade head
python -m src.main
```

## API エンドポイント

### エージェント実行

- `POST /api/v1/agent/stream` - SSEストリーミング実行
- `POST /api/v1/agent/invoke` - 同期実行
- `GET /api/v1/agents` - 利用可能エージェント一覧
- `GET /api/v1/tools` - 利用可能ツール一覧

#### 実行モードオプション

リクエストボディに以下のフラグを指定することで、実行モードを切り替えられます：

| フラグ | 説明 | 用途 | レスポンス速度 |
|--------|------|------|----------------|
| `fast_response: true` | 高速回答モード（sub agent/tools不使用） | 一般的な知識で回答可能な質問 | 最速 |
| `direct_tool_mode: true` | ダイレクトツールモード（MainAgentが直接tools使用） | ツールが必要だが高速に回答したい場合 | 高速 |
| 両方false（デフォルト） | 通常モード（sub agentを使用） | 詳細な情報収集が必要な場合 | 標準 |

**使用例:**

```bash
# 高速回答モード（最速）
curl -X POST http://localhost:8000/api/v1/agent/stream \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: your-access-key" \
  -d '{
    "message": "プリンターに接続できない問題の一般的な対処法は？",
    "fast_response": true
  }'

# ダイレクトツールモード（高速 + ツール使用）
curl -X POST http://localhost:8000/api/v1/agent/stream \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: your-access-key" \
  -d '{
    "message": "エラーコードE500について調べて",
    "direct_tool_mode": true
  }'

# 通常モード（詳細な情報収集）
curl -X POST http://localhost:8000/api/v1/agent/stream \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: your-access-key" \
  -d '{
    "message": "システムにアクセスできない問題を解決したい"
  }'
```

### スレッド管理

- `GET /api/v1/threads/{thread_id}` - スレッド情報
- `DELETE /api/v1/threads/{thread_id}` - スレッド削除

### Admin API - エージェント管理

- `GET /api/v1/admin/agents` - エージェント一覧
- `POST /api/v1/admin/agents` - エージェント作成
- `PUT /api/v1/admin/agents/{name}` - エージェント更新
- `DELETE /api/v1/admin/agents/{name}` - エージェント削除
- `POST /api/v1/admin/agents/static/{name}/test` - 静的エージェントテスト
- `POST /api/v1/admin/agents/dynamic/{name}/test` - 動的エージェントテスト
- `POST /api/v1/admin/agents/adhoc/test` - Ad-hocエージェントテスト

### Admin API - ツール管理

- `GET /api/v1/admin/tools` - ツール一覧
- `POST /api/v1/admin/tools` - ツール作成
- `PUT /api/v1/admin/tools/{name}` - ツール更新
- `DELETE /api/v1/admin/tools/{name}` - ツール削除
- `POST /api/v1/admin/tools/{name}/test` - ツールテスト
- `POST /api/v1/admin/tools/openapi` - OpenAPI仕様からツール生成

### ヘルスチェック

- `GET /health` - ヘルスチェック
- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe
- `GET /metrics` - Prometheusメトリクス

## 認証

### リクエストヘッダー

| ヘッダー | 必須 | 説明 |
|---------|------|------|
| X-Access-Key | ✓ | バックエンドアクセスキー（JWT） |
| X-Service-Tokens | - | 外部サービストークン（Base64 JSON） |

### 会社コンテキスト

リクエストボディに会社情報を含めることで、企業文化に合わせた応答を生成：

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
      "support_hours": "平日9:00-18:00"
    }
  }
}
```

## データベース

### テーブル

| テーブル名 | 説明 |
|-----------|------|
| `agents` | 動的エージェント定義 |
| `template_agents` | テンプレートエージェント |
| `tools` | 動的ツール定義 |
| `threads` | 会話スレッド |
| `execution_sessions` | 実行セッション |
| `execution_results` | 実行結果（Ad-hoc仕様含む） |
| `system_config` | システム設定 |

### マイグレーション

```bash
# 実行
alembic upgrade head

# 新規作成
alembic revision --autogenerate -m "description"
```

## 環境変数

| 変数名 | 必須 | デフォルト | 説明 |
|--------|------|-----------|------|
| `DATABASE_URL` | ✓ | - | PostgreSQL接続URL（async） |
| `DATABASE_URL_SYNC` | ✓ | - | PostgreSQL接続URL（sync） |
| `AWS_REGION` | ✓ | `us-east-1` | AWSリージョン |
| `AWS_ACCESS_KEY_ID` | ✓ | - | AWS認証 |
| `AWS_SECRET_ACCESS_KEY` | ✓ | - | AWS認証 |
| `ACCESS_KEY_SECRET` | ✓ | - | アクセスキー署名用 |
| `DEFAULT_MODEL_ID` | - | `claude-3-5-sonnet-20241022-v2:0` | MainAgent |
| `SUB_AGENT_MODEL_ID` | - | `claude-3-5-haiku-20241022-v1:0` | SubAgent |

## 動的ツールの使用方法

### ツール定義の登録

フロントエンドAPI経由でServiceNow検索を行うツールの例:

```bash
curl -X POST http://localhost:8000/api/v1/admin/tools \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: your-access-key" \
  -d '{
    "name": "frontend_servicenow_search",
    "description": "フロントエンド経由でServiceNow検索",
    "category": "frontend_api",
    "parameters": [
      {
        "name": "query",
        "type": "string",
        "description": "検索クエリ",
        "required": true
      }
    ],
    "executor": {
      "type": "http",
      "url": "https://your-frontend.com/api/servicenow/search",
      "method": "POST",
      "auth_type": "custom_header",
      "auth_config": {
        "headers": {
          "X-Session-Token": "{{ context.service_tokens.frontend_session.token }}"
        }
      },
      "body_template": {
        "query": "{{ params.query }}",
        "count": 30
      }
    },
    "required_service_token": "frontend_session",
    "timeout_seconds": 30,
    "enabled": true
  }'
```

### セッショントークンの送信

フロントエンドからエージェントを呼び出す際:

```typescript
const serviceTokens = {
  frontend_session: {
    token: sessionToken,  // フロントエンドで生成
    expires_at: new Date(Date.now() + 3600000).toISOString()
  }
};

const response = await fetch('/api/agent/stream', {
  headers: {
    'X-Access-Key': accessKey,
    'X-Service-Tokens': btoa(JSON.stringify(serviceTokens)),
  },
  body: JSON.stringify({ message: 'エラーについて調べて' })
});
```

詳細は[examples/tool_definitions/README.md](examples/tool_definitions/README.md)を参照。

## エージェントテスト機能

サンドボックス環境で安全にエージェントをテスト実行できます。

```bash
# 動的エージェントのテスト
curl -X POST http://localhost:8000/api/v1/admin/agents/dynamic/search_agent/test \
  -H "Content-Type: application/json" \
  -d '{
    "test_input": "テストクエリ",
    "task_params": {"query": "ServiceNowの使い方"}
  }'

# Ad-hocエージェントのテスト
curl -X POST http://localhost:8000/api/v1/admin/agents/adhoc/test \
  -H "Content-Type: application/json" \
  -d '{
    "spec": {
      "name": "test_agent",
      "purpose": "情報検索",
      "tools": ["frontend_servicenow_search"],
      "expected_output": "検索結果"
    },
    "test_input": "エラー500について"
  }'
```

## プロジェクト構成

```
src/
├── main.py                  # エントリーポイント
├── config/                  # 設定
├── models/                  # DBモデル
├── repositories/            # CRUD操作
├── api/
│   ├── routes/              # API エンドポイント
│   ├── middleware/          # 認証・権限
│   └── schemas/             # リクエスト/レスポンス
├── agents/
│   ├── graph.py             # LangGraph定義
│   ├── state.py             # State定義
│   ├── registry.py          # レジストリ（DB対応）
│   ├── main_agent/          # Supervisor
│   │   ├── planner.py       # 実行計画
│   │   ├── evaluator.py     # 評価
│   │   └── synthesizer.py   # 応答生成
│   ├── sub_agents/
│   │   ├── adhoc.py         # Ad-hoc実行
│   │   └── dynamic.py       # 動的エージェント
│   └── tools/
│       ├── dynamic.py       # 動的ツール
│       └── openapi.py       # OpenAPIパーサー
├── services/
│   ├── agent_testing.py     # エージェントテスト
│   ├── execution/           # 実行制御
│   ├── streaming/           # SSE
│   └── thread/              # スレッド管理
└── examples/                # サンプル実装
    ├── agents/              # 静的エージェント例
    ├── tools/               # 静的ツール例
    └── tool_definitions/    # 動的ツール定義例
```

## プロンプトキャッシング

AWS Bedrock Prompt Cachingでコスト削減（キャッシュヒット時90%削減）：

- システムプロンプト + ツール/エージェントリスト
- 会話履歴
- TTL: 5分

## ライセンス

MIT License
