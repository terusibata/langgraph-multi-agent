# Dynamic Tool Definitions

このディレクトリには、Admin API経由で登録可能な動的ツール定義のサンプルが含まれています。

## ツール定義の登録方法

### 1. Admin APIで登録

```bash
curl -X POST http://localhost:8000/api/v1/admin/tools \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: your-access-key" \
  -d @frontend_servicenow_search.json
```

### 2. 登録されたツールの確認

```bash
curl http://localhost:8000/api/v1/admin/tools \
  -H "X-Access-Key: your-access-key"
```

### 3. ツールのテスト

```bash
curl -X POST http://localhost:8000/api/v1/admin/tools/frontend_servicenow_search/test \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: your-access-key" \
  -H "X-Service-Tokens: $(echo '{"frontend_session":{"token":"test_session_token_abc123"}}' | base64)" \
  -d '{
    "params": {
      "query": "エラーについて",
      "count": 10
    }
  }'
```

## ツール定義の構造

### 基本構造

```json
{
  "name": "tool_name",
  "description": "ツールの説明",
  "category": "カテゴリ",
  "parameters": [
    {
      "name": "param_name",
      "type": "string|integer|boolean|array|object",
      "description": "パラメータの説明",
      "required": true|false,
      "default": "デフォルト値（オプション）"
    }
  ],
  "executor": {
    "type": "http|python|mock",
    "url": "APIエンドポイント",
    "method": "GET|POST|PUT|DELETE",
    "auth_type": "none|bearer|api_key|custom_header",
    "auth_config": {},
    "headers": {},
    "body_template": {}
  },
  "required_service_token": "トークン名（オプション）",
  "timeout_seconds": 30,
  "enabled": true
}
```

### 認証タイプ

#### 1. カスタムヘッダー認証 (推奨)

フロントエンドAPIなど、独自のヘッダーを使用する場合:

```json
{
  "auth_type": "custom_header",
  "auth_config": {
    "headers": {
      "X-Session-Token": "{{ context.service_tokens.frontend_session.token }}",
      "X-Tenant-ID": "{{ context.tenant_id }}"
    }
  }
}
```

#### 2. Bearer認証

```json
{
  "auth_type": "bearer",
  "required_service_token": "api_service"
}
```

これにより、`Authorization: Bearer <token>`ヘッダーが自動的に追加されます。

#### 3. APIキー認証

```json
{
  "auth_type": "api_key",
  "auth_config": {
    "header": "X-API-Key",
    "key_env": "EXTERNAL_API_KEY"
  }
}
```

環境変数からAPIキーを取得してヘッダーに設定します。

### テンプレート変数

ツール定義では、以下のテンプレート変数が使用できます:

#### パラメータ

```jinja2
{{ params.query }}
{{ params.count }}
```

#### コンテキスト

```jinja2
{{ context.tenant_id }}
{{ context.user_id }}
{{ context.request_id }}
{{ context.service_tokens.token_name.token }}
{{ context.service_tokens.token_name.instance }}
```

#### 使用例

```json
{
  "url": "https://api.example.com/search?tenant={{ context.tenant_id }}",
  "headers": {
    "X-Session-Token": "{{ context.service_tokens.frontend_session.token }}",
    "X-User-ID": "{{ context.user_id }}"
  },
  "body_template": {
    "query": "{{ params.query }}",
    "limit": "{{ params.count }}",
    "tenant": "{{ context.tenant_id }}"
  }
}
```

## フロントエンドとの連携

### セッショントークンの送信

フロントエンドからバックエンドへのリクエスト時、`X-Service-Tokens`ヘッダーにセッショントークンを含めます:

```typescript
// フロントエンド (Next.js)
const serviceTokens = {
  frontend_session: {
    token: sessionToken,  // フロントエンドで生成したセッショントークン
    expires_at: new Date(Date.now() + 3600000).toISOString()
  }
};

const response = await fetch('/api/agent/stream', {
  method: 'POST',
  headers: {
    'X-Access-Key': accessKey,
    'X-Service-Tokens': btoa(JSON.stringify(serviceTokens)),
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: 'ServiceNowでエラーについて調べて'
  })
});
```

### フロントエンドAPIの実装

```typescript
// pages/api/servicenow/search.ts
export default async function handler(req: Request) {
  const sessionToken = req.headers.get('X-Session-Token');

  // セッショントークンを検証
  const session = await validateSessionToken(sessionToken);
  if (!session) {
    return new Response('Unauthorized', { status: 401 });
  }

  // セッションからServiceNowトークンを取得
  const serviceNowToken = session.serviceNowToken;

  // ServiceNow APIを呼び出し
  const { query, count } = await req.json();
  const results = await searchServiceNow(serviceNowToken, query, count);

  return new Response(JSON.stringify(results), {
    headers: { 'Content-Type': 'application/json' }
  });
}
```

## OpenAPIからの自動生成

OpenAPI仕様からツールを自動生成することもできます:

```bash
curl -X POST http://localhost:8000/api/v1/admin/tools/openapi \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: your-access-key" \
  -d '{
    "spec": {
      "openapi": "3.0.0",
      "paths": {
        "/api/servicenow/search": {
          "post": {
            "operationId": "searchServiceNow",
            "requestBody": {...},
            "responses": {...}
          }
        }
      }
    },
    "base_url": "https://your-frontend.com",
    "selected_operations": ["searchServiceNow"]
  }'
```

## トラブルシューティング

### 認証エラー

```json
{
  "success": false,
  "error": "HTTP 401: Unauthorized"
}
```

→ セッショントークンが正しく送信されているか確認してください。

### テンプレートエラー

```json
{
  "success": false,
  "error": "Template render failed"
}
```

→ テンプレート変数の構文を確認してください（`{{ }}`の二重括弧）。

### タイムアウト

```json
{
  "success": false,
  "error": "Request timeout"
}
```

→ `timeout_seconds`を増やすか、フロントエンドAPIのパフォーマンスを確認してください。
