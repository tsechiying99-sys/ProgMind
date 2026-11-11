# PDF 原生解析与 OCR 摄取改进日志

日期：2026-10-21
范围：知识库文件上传、PDF 解析、OCR、文本清洗、知识切片与 ChromaDB 入库
状态：已实现并通过自动化测试

## 一、改进目标

在不改变现有 TXT、Markdown、JSON 导入流程的前提下，为知识库增加 PDF 摄取能力：

1. 原生 PDF 优先使用 PyMuPDF 直接提取文字。
2. 扫描 PDF 或文本层质量不足的页面使用 RapidOCR。
3. 同一份 PDF 可以逐页混合处理，不以整份文档为单位简单二选一。
4. 完成文本清洗、质量门控和分页感知切片后，再使用现有 BGE 编码器向量化并写入 ChromaDB。
5. 保留页码、解析策略、OCR 置信度和质量分，便于后续检索解释与问题排查。

## 二、处理流程

```text
用户上传 PDF
    ↓
扩展名、大小、PDF 文件头、加密状态、页数校验
    ↓
逐页提取 PyMuPDF 原生文本
    ↓
文本长度 + 文本质量 + 图片覆盖率联合判断
    ├─ 原生文本可靠：直接采用原生文本
    ├─ 无可靠文本：渲染页面并调用 RapidOCR
    └─ 原生文本质量存疑：比较原生文本与 OCR 结果，择优保留
    ↓
Unicode/空白/断词清洗 + 重复页眉页脚清理
    ↓
文档质量门控
    ↓
按页、按句切片并生成稳定 SHA-256 ID
    ↓
BGE 向量化 → ChromaDB upsert → 重建 BM25 索引
```

重要约束：混合页不会把原生文本和 OCR 文本直接拼接，而是依据质量择优，避免重复内容污染知识库。

## 三、新增模块

### `ingestion/models.py`

- 定义解析页、知识片段、摄取报告和最终结果的数据结构。
- 区分 PDF 格式校验失败与文档质量不足，供 API 返回不同状态码。

### `ingestion/pdf_parser.py`

- 使用 PyMuPDF 打开 PDF、提取原生文本和计算图片覆盖率。
- 使用文本长度、文本质量和图片覆盖率联合判断页面类型，不使用“存在图片即扫描件”的错误规则。
- RapidOCR 采用延迟初始化，纯原生 PDF 不加载 OCR 模型。
- OCR 推理串行加锁，避免同一模型实例被并发调用。
- 支持 RapidOCR 3.x 的 NumPy 数组返回值，也兼容旧式三元组返回值。
- 单页渲染默认限制为 2500 万像素，异常大画布会自动降低 DPI，降低内存耗尽风险。
- 加密 PDF、空 PDF、伪造 PDF、超页数 PDF 会在 OCR 和入库前拒绝。

### `ingestion/text_cleaner.py`

- 执行 Unicode NFKC 归一化。
- 清除不可见控制字符、压缩异常空白、修复英文换行断词。
- 至少三页时检测高频短页眉/页脚，并从各页移除。
- 根据有效字符比例、乱码比例、字符多样性和有效长度计算文本质量分。

### `ingestion/chunker.py`

- 按页切片，优先在句号、问号、感叹号、分号和换行处断开。
- 默认目标长度 700 字，重叠 80 字；超长单句使用滑动窗口。
- 使用“文档哈希 + 页码 + 页内序号 + 内容”生成稳定 SHA-256 ID，重复上传相同文件时由 ChromaDB 覆盖相同记录，不产生随机重复 ID。
- 为每个片段保留文档 ID、文件名、页码、解析策略、OCR 标记、OCR 置信度、质量分和片段序号。

### `ingestion/pipeline.py`

- 统一编排解析、清洗、质量门控和切片。
- 完整文档解析并通过质量门控后才返回待入库片段。
- 生成页面数量、原生页数量、OCR 页数量、混合页数量、质量均分和警告信息。

## 四、现有代码调整

### `mcp/knowledge_base.py`

- 新增 `add_chunks()`，用于写入已完成切片并带有详细元数据的记录。
- `add_documents()` 保持原有调用接口和 500 字切片行为，但内部统一转交给 `add_chunks()`。
- PDF 片段不会再次被旧逻辑二次切片。
- 先完成整批 BGE 向量计算，再统一执行 ChromaDB `upsert`，随后刷新 BM25 索引。
- 写入前过滤 ChromaDB 不接受的嵌套或空元数据值。

### `api/main.py`

- 应用启动时初始化 PDF 摄取流水线。
- `/knowledge/upload` 新增 `.pdf`，扩展名大小写不敏感。
- 上传读取限制改为可配置，默认 30MB，并只读取“上限 + 1 字节”用于判断超限。
- PDF 解析、OCR、向量化和入库通过工作线程执行，避免直接阻塞 FastAPI 事件循环。
- PDF 格式问题返回 HTTP 400；质量不足返回 HTTP 422；不支持的格式返回 HTTP 415。
- 成功响应新增：`document_id`、`pages`、`native_pages`、`ocr_pages`、`hybrid_pages`、`quality_score`、`warnings`。

### 前端

- 文件选择器新增 PDF MIME 与扩展名。
- 上传说明增加扫描 PDF 自动 OCR，并将默认大小提示改为 30MB。
- PDF 导入成功后显示解析质量百分比。
- TypeScript 响应类型增加 PDF 摄取报告字段，同时保持这些字段可选，兼容非 PDF 导入。

### 依赖

- `PyMuPDF==2.0`
- `rapidocr==3.10.0`
- `onnxruntime==1.29.0`

不依赖系统级 Poppler 或 Tesseract；PyMuPDF 负责页面解析和渲染，RapidOCR 使用 ONNX Runtime 在 CPU 上推理。

## 五、新增配置

| 环境变量 | 默认值 | 作用 |
|---|---:|---|
| `KNOWLEDGE_MAX_UPLOAD_MB` | 30 | 上传文件大小上限 |
| `KNOWLEDGE_MAX_PDF_PAGES` | 200 | PDF 最大页数 |
| `PDF_OCR_DPI` | 220 | OCR 页面渲染 DPI |
| `PDF_NATIVE_MIN_CHARS` | 40 | 可靠原生文本的最少字符数 |
| `PDF_NATIVE_MIN_QUALITY` | 0.75 | 原生文本质量阈值 |
| `PDF_OCR_MIN_CONFIDENCE` | 0.55 | OCR 行级最低置信度 |
| `PDF_IMAGE_COVERAGE_THRESHOLD` | 0.35 | 辅助判断扫描/混合页的图片覆盖率 |
| `PDF_MAX_RENDER_PIXELS` | 25000000 | OCR 单页最大渲染像素数 |
| `PDF_DOCUMENT_MIN_QUALITY` | 0.55 | 文档入库最低质量分 |
| `KNOWLEDGE_CHUNK_SIZE` | 700 | PDF 片段目标长度 |
| `KNOWLEDGE_CHUNK_OVERLAP` | 80 | 相邻片段重叠长度 |

## 六、测试与验证

新增 `tests/test_pdf_ingestion.py`，覆盖：

- 原生 PDF 直接提取且不初始化 OCR。
- 原生页与扫描页混合时，仅扫描页执行 OCR。
- RapidOCR 列表和 NumPy 数组结果兼容。
- 非 PDF 内容拒绝。
- 加密 PDF 拒绝。
- 低质量 OCR 文本被质量门控拦截。
- 相同文件生成稳定片段 ID。
- 页码、解析策略和文档 ID 元数据保留。
- 非标量元数据在 ChromaDB 写入前过滤。
- PDF 上传接口返回完整解析报告。
- 不支持的上传格式返回 HTTP 415。

验证结果：

- PDF/OCR 专项测试：10 项通过。
- 项目完整 Python 测试：40 项通过。
- Python `compileall`：通过。
- 前端 `npm run build`（含 TypeScript 检查）：通过。
- 真实 RapidOCR 烟雾测试：扫描 PDF 成功识别 `Java class method variable loop`，解析策略为 `ocr`，置信度约 0.982。

## 七、当前边界与后续可选项

本次已完成 PDF 原生解析、扫描 OCR、清洗、质量门控、切片和入库闭环，但以下能力不在本次范围：

- 表格结构还原、公式语义识别和图片说明生成。
- 手写体、复杂竖排、多栏阅读顺序的专项模型优化。
- 后台异步任务队列、上传进度条和 OCR 任务取消。
- 同一课程不同文件之间的语义级重复片段合并。
- 在检索回答中向用户展示 PDF 页码引用。

这些限制不会破坏当前流程；遇到低质量文档时系统会拒绝入库或在响应的 `warnings` 中明确提示，而不是静默写入不可用内容。
