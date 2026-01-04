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
- **JSON形式レスポンス**: スキーマ指定でJSON形式の構造化された応答を取得可能
- **統一リソースフォーマット**: 全てのツールの検索結果を統一形式で取得
- **スレッドタイトル自動生成**: 新規会話から日本語で簡潔なタイトルを自動生成

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

#### JSON形式レスポンス

Main AgentおよびSub Agentの応答を、指定したJSON schemaに従った構造化データとして取得できます。

**Main Agentの例（翻訳アプリ）:**

```bash
curl -X POST http://localhost:8000/api/v1/agent/stream \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: your-access-key" \
  -d '{
    "message": "Translate to English: こんにちは、世界",
    "response_format": "json",
    "response_schema": {
      "type": "object",
      "properties": {
        "original_text": {"type": "string"},
        "translated_text": {"type": "string"},
        "detected_language": {"type": "string"},
        "confidence": {"type": "number"}
      },
      "required": ["original_text", "translated_text"]
    },
    "fast_response": true
  }'
```

**レスポンス例:**
```json
{
  "original_text": "こんにちは、世界",
  "translated_text": "Hello, world",
  "detected_language": "ja",
  "confidence": 0.99
}
```

**Sub Agentへの設定:**

Admin APIでSub Agentを登録・更新する際に、`response_format`と`response_schema`を指定できます：

```bash
curl -X POST http://localhost:8000/api/v1/admin/agents \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: your-access-key" \
  -d '{
    "name": "structured_search_agent",
    "description": "構造化された検索結果を返すエージェント",
    "capabilities": ["search", "structured_output"],
    "tools": ["frontend_servicenow_search"],
    "response_format": "json",
    "response_schema": {
      "type": "object",
      "properties": {
        "results": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "title": {"type": "string"},
              "summary": {"type": "string"},
              "relevance": {"type": "number"}
            }
          }
        },
        "total_count": {"type": "integer"}
      }
    },
    "enabled": true
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
| `agents` | 動的エージェント定義（response_format, response_schema含む） |
| `template_agents` | テンプレートエージェント |
| `tools` | 動的ツール定義 |
| `threads` | 会話スレッド（title含む） |
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

**最新のマイグレーション (002):**
- `agents`テーブルに`response_format`と`response_schema`カラムを追加（JSON形式レスポンス対応）
- `threads`テーブルに`title`カラムを追加（スレッドタイトル自動生成対応）

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

## レスポンス形式

### SSEイベントストリーム

`/api/v1/agent/stream`エンドポイントは、以下のSSE（Server-Sent Events）イベントを返します：

| イベント | タイミング | 説明 |
|---------|----------|------|
| `session_start` | セッション開始時 | セッションIDとスレッドIDを通知 |
| `plan_created` | 実行計画作成後 | 実行予定のエージェントとタスクを通知 |
| `agent_start` | Sub Agent開始時 | エージェント名とタスク内容を通知 |
| `agent_retry` | リトライ時 | リトライ回数と修正クエリを通知 |
| `agent_end` | Sub Agent完了時 | ステータスと実行時間を通知 |
| `tool_call` | ツール呼び出し時 | ツール名とパラメータを通知 |
| `tool_result` | ツール結果受信時 | 成功/失敗と結果サマリを通知 |
| `evaluation` | 中間評価時 | 評価結果と次のアクションを通知 |
| `token` | 応答生成中 | 生成されたトークン（ストリーミング） |
| `llm_metrics` | LLM呼び出し後 | トークン使用量とコストを通知 |
| `session_complete` | セッション完了時 | 最終応答と全メトリクスを返却 |
| `error` | エラー発生時 | エラー情報とリカバリ手順を通知 |

### session_complete イベント構造

セッション完了時に返される最も重要なイベントです：

```json
{
  "event": "session_complete",
  "data": {
    "session_id": "sess_abc123",
    "thread_id": "thread_xyz789",
    "title": "プリンター接続エラーの解決",
    "response": {
      "content": "プリンターに接続できない問題について...",
      "finish_reason": "stop"
    },
    "execution_summary": {
      "plan": {
        "initial_agents": ["knowledge_search", "catalog_search"],
        "parallel_groups": [["knowledge_search", "catalog_search"]],
        "estimated_steps": 2
      },
      "agents_executed": [
        {
          "name": "knowledge_search",
          "type": "template",
          "status": "success",
          "retries": 0,
          "search_variations": [],
          "duration_ms": 1500
        }
      ],
      "tools_executed": [
        {
          "tool": "frontend_servicenow_search",
          "agent": "knowledge_search",
          "success": true
        }
      ]
    },
    "metrics": {
      "duration_ms": 5000,
      "llm_calls": [
        {
          "call_id": "llm_abc123",
          "model_id": "claude-3-5-sonnet-20241022-v2:0",
          "agent": "MainAgent",
          "phase": "plan",
          "input_tokens": 1000,
          "output_tokens": 200,
          "cost_usd": 0.005
        }
      ],
      "totals": {
        "input_tokens": 1500,
        "output_tokens": 500,
        "total_tokens": 2000,
        "total_cost_usd": 0.01,
        "llm_call_count": 3,
        "tool_call_count": 2
      }
    },
    "thread_state": {
      "status": "active",
      "context_tokens_used": 2000,
      "context_max_tokens": 200000,
      "context_usage_percent": 1.0,
      "message_count": 1,
      "thread_total_tokens": 2000,
      "thread_total_cost_usd": 0.01
    },
    "resources": [
      {
        "id": "KB001",
        "type": "knowledge_base",
        "title": "プリンター接続トラブルシューティング",
        "content": "プリンターに接続できない場合の対処法...",
        "score": 0.95,
        "tool_name": "frontend_servicenow_search",
        "metadata": {
          "category": "IT Support",
          "last_updated": "2024-01-15"
        }
      },
      {
        "id": "DOC456",
        "type": "document",
        "title": "ネットワーク設定ガイド",
        "content": null,
        "score": 0.82,
        "tool_name": "vector_search",
        "metadata": {
          "author": "IT Team",
          "version": "2.0"
        }
      }
    ]
  }
}
```

### 統一リソースフォーマット

`resources`配列に含まれる各リソースは以下の統一形式に従います：

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| `id` | string | ✓ | リソースID（sys_id、KB番号など） |
| `type` | string | ✓ | リソース種別（後述） |
| `title` | string | ✓ | リソースタイトル |
| `content` | string \| null | - | リソース内容またはスニペット |
| `score` | number \| null | - | 関連度スコア（0.0-1.0） |
| `tool_name` | string | ✓ | このリソースを取得したツール名 |
| `metadata` | object | ✓ | ツール固有の追加情報 |

**リソース種別 (`type`):**

- `knowledge_base`: ナレッジベース記事
- `document`: ドキュメント・ファイル
- `catalog`: サービスカタログ項目
- `search_result`: 一般的な検索結果
- `general`: その他

### スレッドタイトル自動生成

新規スレッド（初回メッセージ）の場合、会話内容から日本語で簡潔なタイトル（最大30文字）が自動生成されます：

- `session_complete`イベントの`title`フィールドに含まれます
- スレッド情報（`GET /api/v1/threads/{thread_id}`）でも取得可能
- 会話のトピックを表す、わかりやすいタイトルが生成されます

**タイトル例:**
- "プリンター接続エラーの解決"
- "パスワードリセット手順"
- "新規ユーザー登録方法"
- "VPN接続トラブルシューティング"

### JSON形式レスポンス時の構造

`response_format: "json"`を指定した場合、`response.content`には指定したスキーマに従ったJSON文字列が返されます：

```json
{
  "response": {
    "content": "{\"original_text\":\"こんにちは、世界\",\"translated_text\":\"Hello, world\",\"detected_language\":\"ja\",\"confidence\":0.99}",
    "finish_reason": "stop"
  }
}
```

パース後:
```javascript
const response = JSON.parse(data.response.content);
// {
//   original_text: "こんにちは、世界",
//   translated_text: "Hello, world",
//   detected_language: "ja",
//   confidence: 0.99
// }
```

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
