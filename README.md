# RAG w/ langchain

1. Setup Qdrant database:

- `docker pull qdrant/qdrant`
- `docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant`

2. Create new qdrant collection:

- ```
  curl -X PUT 'http://localhost:6333/collections/rag' \
    -H 'Content-Type: application/json' \
    --data-raw '{
      "vectors": {
        "size": 4096,
        "distance": "Dot"
      }
    }'
  ```

3. Install dependencies:

- `npm install`

4. Run import:

- `npm run start langchain-qdrant.ts import`

5. Ask question:

- `npm run start langchain-qdrant.ts ask`
