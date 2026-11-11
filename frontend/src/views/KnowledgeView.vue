<script setup lang="ts">
import { onMounted, ref } from "vue";
import { ElMessage } from "element-plus";
import { getApiErrorMessage } from "../api/client";
import {
  addKnowledge,
  getKnowledgeStats,
  searchKnowledge,
  uploadKnowledge,
} from "../api/knowledge";
import type { SearchResultItem } from "../types/api";

const totalChunks = ref<number | null>(null);
const title = ref("");
const content = ref("");
const selectedFile = ref<File | null>(null);
const searchQuery = ref("");
const searchResults = ref<SearchResultItem[]>([]);
const reranked = ref(false);
const adding = ref(false);
const uploading = ref(false);
const searching = ref(false);

async function refreshStats(): Promise<void> {
  try {
    totalChunks.value = (await getKnowledgeStats()).total_chunks;
  } catch (error) {
    ElMessage.error(getApiErrorMessage(error));
  }
}

async function submitDocument(): Promise<void> {
  if (!title.value.trim() || !content.value.trim()) {
    ElMessage.warning("请填写资料标题和正文");
    return;
  }

  adding.value = true;
  try {
    const result = await addKnowledge([{ title: title.value.trim(), content: content.value.trim() }]);
    totalChunks.value = result.total_chunks;
    title.value = "";
    content.value = "";
    ElMessage.success(result.message);
  } catch (error) {
    ElMessage.error(getApiErrorMessage(error));
  } finally {
    adding.value = false;
  }
}

function selectFile(event: Event): void {
  const input = event.target as HTMLInputElement;
  selectedFile.value = input.files?.[0] ?? null;
}

async function submitFile(): Promise<void> {
  if (!selectedFile.value) {
    ElMessage.warning("请先选择文件");
    return;
  }

  uploading.value = true;
  try {
    const result = await uploadKnowledge(selectedFile.value);
    totalChunks.value = result.total_chunks;
    selectedFile.value = null;
    const quality = result.quality_score == null
      ? ""
      : `，解析质量 ${(result.quality_score * 100).toFixed(1)}%`;
    ElMessage.success(`${result.message}${quality}`);
  } catch (error) {
    ElMessage.error(getApiErrorMessage(error));
  } finally {
    uploading.value = false;
  }
}

async function runSearch(): Promise<void> {
  if (!searchQuery.value.trim()) return;

  searching.value = true;
  try {
    const result = await searchKnowledge(searchQuery.value.trim());
    searchResults.value = result.results || [];
    reranked.value = result.reranked;
  } catch (error) {
    ElMessage.error(getApiErrorMessage(error));
  } finally {
    searching.value = false;
  }
}

onMounted(refreshStats);
</script>

<template>
  <section class="content-page">
    <header class="page-title">
      <div>
        <span class="eyebrow">COURSE LIBRARY</span>
        <h1>课程资料</h1>
        <p>上传课程文档，ProgMind 会在对话中优先检索这些内容。</p>
      </div>
      <div class="stat-card">
        <strong>{{ totalChunks ?? "—" }}</strong>
        <span>知识片段</span>
      </div>
    </header>

    <div class="two-column">
      <article class="panel">
        <div class="panel-heading">
          <div><span>01</span><h2>手动录入</h2></div>
          <small>适合短篇笔记</small>
        </div>
        <label>资料标题</label>
        <input v-model="title" placeholder="例如：Python 函数基础" />
        <label>课程正文</label>
        <textarea v-model="content" rows="9" placeholder="粘贴课程内容……" />
        <button class="primary-button" type="button" :disabled="adding" @click="submitDocument">
          {{ adding ? "正在导入…" : "导入知识库" }}
        </button>
      </article>

      <article class="panel upload-panel">
        <div class="panel-heading">
          <div><span>02</span><h2>文件上传</h2></div>
          <small>最大 30 MB</small>
        </div>
        <div class="drop-zone">
          <div class="file-icon">↑</div>
          <strong>{{ selectedFile?.name || "选择课程文件" }}</strong>
          <p>支持 TXT、Markdown、JSON 和 PDF 文档；扫描 PDF 自动 OCR</p>
          <input type="file" accept=".txt,.md,.json,.pdf,application/pdf" @change="selectFile" />
        </div>
        <button class="primary-button" type="button" :disabled="uploading || !selectedFile" @click="submitFile">
          {{ uploading ? "正在上传…" : "上传并解析" }}
        </button>
      </article>
    </div>

    <article class="panel search-panel">
      <div class="panel-heading">
        <div><span>03</span><h2>检索测试</h2></div>
        <small v-if="searchResults.length">{{ reranked ? "已重排" : "原始排序" }}</small>
      </div>
      <div class="search-row">
        <input v-model="searchQuery" placeholder="输入需要检索的知识点" @keyup.enter="runSearch" />
        <button class="primary-button" type="button" :disabled="searching" @click="runSearch">
          {{ searching ? "检索中…" : "开始检索" }}
        </button>
      </div>
      <div v-if="searchResults.length" class="result-list">
        <article v-for="(item, index) in searchResults" :key="`${item.title}-${index}`" class="result-item">
          <div class="result-title">
            <strong>{{ item.title || "未命名资料" }}</strong>
            <span>{{ Number(item.score || 0).toFixed(3) }}</span>
          </div>
          <p>{{ item.content }}</p>
        </article>
      </div>
      <div v-else class="empty-state">检索结果会显示在这里</div>
    </article>
  </section>
</template>

<style scoped>
.content-page { width: min(1120px, 100%); margin: 0 auto; }
.page-title { display: flex; align-items: end; justify-content: space-between; gap: 24px; margin-bottom: 28px; }
.eyebrow { color: #3b8e71; font-size: 11px; font-weight: 800; letter-spacing: .18em; }
h1 { margin: 7px 0; color: #173b32; font-family: Georgia, "Noto Serif SC", serif; font-size: 38px; }
.page-title p { margin: 0; color: #73817d; }
.stat-card { display: flex; min-width: 130px; padding: 18px; flex-direction: column; border: 1px solid #dce6e2; border-radius: 18px; background: #fff; }
.stat-card strong { color: #19694f; font-size: 30px; }
.stat-card span { color: #84908c; font-size: 12px; }
.two-column { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.panel { padding: 24px; border: 1px solid #dfe6e3; border-radius: 22px; background: #fff; box-shadow: 0 12px 36px rgb(44 70 62 / 5%); }
.panel-heading { display: flex; align-items: center; justify-content: space-between; margin-bottom: 22px; }
.panel-heading div { display: flex; align-items: center; gap: 10px; }
.panel-heading span { display: grid; width: 30px; height: 30px; place-items: center; border-radius: 10px; background: #edf6f3; color: #24765c; font-size: 11px; font-weight: 800; }
.panel-heading h2 { margin: 0; color: #29463e; font-size: 18px; }
.panel-heading small { color: #929d99; }
label { display: block; margin: 16px 0 7px; color: #53625d; font-size: 13px; font-weight: 700; }
input, textarea { box-sizing: border-box; width: 100%; padding: 12px 14px; border: 1px solid #dbe4e1; border-radius: 12px; outline: none; color: #253630; font: inherit; }
input:focus, textarea:focus { border-color: #4e9b80; box-shadow: 0 0 0 3px rgb(78 155 128 / 12%); }
textarea { resize: vertical; line-height: 1.6; }
.primary-button { padding: 12px 17px; border: 0; border-radius: 12px; background: #17694f; color: #fff; cursor: pointer; font-weight: 700; }
.panel > .primary-button { width: 100%; margin-top: 18px; }
.primary-button:disabled { background: #aabbb5; cursor: not-allowed; }
.drop-zone { display: grid; min-height: 205px; place-items: center; align-content: center; border: 1px dashed #9fc7b9; border-radius: 18px; background: #f5fbf8; text-align: center; }
.drop-zone p { margin: 5px 0 16px; color: #8a9692; font-size: 13px; }
.file-icon { display: grid; width: 44px; height: 44px; margin-bottom: 12px; place-items: center; border-radius: 14px; background: #dff1ea; color: #24775c; font-size: 24px; }
.drop-zone input { width: auto; max-width: 85%; padding: 8px; background: #fff; font-size: 12px; }
.search-panel { margin-top: 18px; }
.search-row { display: grid; grid-template-columns: 1fr auto; gap: 10px; }
.result-list { display: grid; gap: 10px; margin-top: 20px; }
.result-item { padding: 16px; border: 1px solid #e3e9e7; border-radius: 14px; background: #fbfcfc; }
.result-title { display: flex; justify-content: space-between; gap: 15px; color: #2f5146; }
.result-title span { color: #2f8065; font-family: monospace; }
.result-item p { margin: 10px 0 0; color: #64716d; line-height: 1.65; white-space: pre-wrap; }
.empty-state { padding: 34px; color: #a0aaa7; text-align: center; }
@media (max-width: 760px) { .page-title { align-items: flex-start; flex-direction: column; } .two-column { grid-template-columns: 1fr; } .search-row { grid-template-columns: 1fr; } }
</style>
