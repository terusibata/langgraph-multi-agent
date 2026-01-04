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
- **静的エージェント**: Pythonで実装、起動時に登録（例: knowledge_search, vector_search, catalog）
- **動的エージェント**: Admin APIで作成、DBに保存
- **Ad-hocエージェント**: Plannerが実行時に動的生成（仕様はDB保存）

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

## エージェントテスト機能

サンドボックス環境で安全にエージェントをテスト実行できます。

### テスト例

```bash
# 静的エージェントのテスト
curl -X POST http://localhost:8000/api/v1/admin/agents/static/knowledge_search/test \
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
      "purpose": "天気情報を取得",
      "tools": ["weather_api"],
      "expected_output": "現在の天気情報"
    },
    "test_input": "東京の天気は?"
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
│   │   ├── dynamic.py       # 動的エージェント
│   │   └── *.py             # 静的エージェント
│   └── tools/
│       ├── dynamic.py       # 動的ツール
│       ├── openapi.py       # OpenAPIパーサー
│       └── */               # 静的ツール
└── services/
    ├── agent_testing.py     # エージェントテスト
    ├── execution/           # 実行制御
    ├── streaming/           # SSE
    └── thread/              # スレッド管理
```

## プロンプトキャッシング

AWS Bedrock Prompt Cachingでコスト削減（キャッシュヒット時90%削減）：

- システムプロンプト + ツール/エージェントリスト
- 会話履歴
- TTL: 5分

## ライセンス

MIT License
