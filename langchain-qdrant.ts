import { Ollama } from "@langchain/community/llms/ollama";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { QdrantVectorStore } from "@langchain/community/vectorstores/qdrant";

const Q = {
  url: "http://localhost:6333",
  collectionName: "rag",
};
const LLM = {
  url: "http://localhost:11434",
  model: "llama2",
};

const action = process.argv[2];

switch (action) {
  case "import":
    console.log("Embedding story into database...");

    save("./data/zenon-story.txt");
    break;

  case "ask":
    const question = "What Crystal of Aether was?";

    console.log("Asking question: " + question);

    answer(question).then(console.log);
    break;
}

async function save(filePath: string) {
  const embeddings = new OllamaEmbeddings();
  const textLoader = new TextLoader(filePath);
  const docs = await textLoader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const splits = await textSplitter.splitDocuments(docs);

  await QdrantVectorStore.fromDocuments(splits, embeddings, {
    collectionName: Q.collectionName,
    url: Q.url,
  });
}

async function answer(question: string) {
  const embeddings = new OllamaEmbeddings();
  const qdrant = await QdrantVectorStore.fromExistingCollection(embeddings, {
    collectionName: Q.collectionName,
    url: Q.url,
  });

  const retrievedDocs = await qdrant.similaritySearch(question, 2);

  const llm = new Ollama({
    baseUrl: LLM.url,
    model: LLM.model,
  });

  const prompt = await pull<ChatPromptTemplate>("rlm/rag-prompt");
  /* Prompt:
      You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
      If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
      Question: {question}
      Context: {context}
      Answer:
    */

  const ragChain = await createStuffDocumentsChain({
    llm,
    prompt,
    outputParser: new StringOutputParser(),
  });

  return await ragChain.invoke({
    question,
    context: retrievedDocs,
  });
}
