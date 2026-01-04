# Examples

このディレクトリには、LangGraph Multi-Agentシステムのサンプル実装が含まれています。

## 📁 ディレクトリ構成

```
examples/
├── agents/                          # 静的エージェントのサンプル実装
│   ├── knowledge_search.py          # ServiceNowナレッジ検索エージェント
│   ├── vector_search.py             # ベクトル検索エージェント
│   └── catalog.py                   # カタログ検索エージェント
├── tools/                           # 静的ツールのサンプル実装
│   ├── servicenow/                  # ServiceNow関連ツール
│   ├── catalog/                     # カタログツール
│   └── vector_db/                   # ベクトルDBツール
└── tool_definitions/                # 動的ツール定義のサンプル
    ├── frontend_servicenow_search.json
    └── README.md
```

## 🎯 使用方法

### 静的実装からの学習

`agents/`と`tools/`ディレクトリには、Pythonで実装された静的エージェント・ツールのサンプルがあります。

これらは以下の目的で使用できます:
- 実装パターンの参考
- カスタムエージェント・ツールの開発テンプレート
- テスト・デバッグ用

**注意:** これらの静的実装は、本番環境では使用されません。すべてのエージェント・ツールは、Admin API経由で動的に登録することを推奨します。

### 動的ツール定義の使用

`tool_definitions/`ディレクトリには、Admin API経由で登録可能なツール定義のサンプルがあります。

詳細は[tool_definitions/README.md](tool_definitions/README.md)を参照してください。

## 🔄 本番環境での推奨構成

### すべて動的ツールで管理

```bash
# 1. ツール定義を登録
curl -X POST http://localhost:8000/api/v1/admin/tools \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: your-key" \
  -d @tool_definitions/frontend_servicenow_search.json

# 2. 動的エージェントを登録
curl -X POST http://localhost:8000/api/v1/admin/agents \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: your-key" \
  -d '{
    "name": "servicenow_helper",
    "description": "ServiceNow情報検索アシスタント",
    "capabilities": ["servicenow_search", "knowledge_lookup"],
    "tools": ["frontend_servicenow_search"],
    "executor": {
      "type": "llm",
      "temperature": 0.0
    },
    "enabled": true
  }'
```

## 📚 参考

- [メインREADME](../README.md) - システム全体の概要
- [動的ツール定義ガイド](tool_definitions/README.md) - ツール定義の詳細
