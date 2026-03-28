-- =============================================================
-- 评测领域数据库初始化脚本
-- 为AI导购评测系统创建必要的表结构
-- =============================================================

-- 1. 测试数据集表
-- 存储测试数据集的元数据和版本信息，实际数据存储在本地文件
CREATE TABLE IF NOT EXISTS eval_datasets (
    id SERIAL PRIMARY KEY,
    dataset_id VARCHAR(64) NOT NULL,       -- 数据集业务ID（同一逻辑数据集不同版本共享）
    dataset_name VARCHAR(255) NOT NULL,
    version VARCHAR(16) NOT NULL,          -- 版本格式: vX.Y
    file_path TEXT NOT NULL,               -- 本地文件路径
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    creator VARCHAR(64) NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'active', -- active/deprecated
    metadata JSONB,
    task_count INTEGER NOT NULL DEFAULT 0,

    -- 唯一约束：同一数据集同一版本只能存在一个
    CONSTRAINT unique_dataset_version UNIQUE(dataset_id, version)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_eval_datasets_dataset_id ON eval_datasets(dataset_id);
CREATE INDEX IF NOT EXISTS idx_eval_datasets_status ON eval_datasets(status);

COMMENT ON TABLE eval_datasets IS '测试数据集表，存储版本化管理的测试数据集元数据';
COMMENT ON COLUMN eval_datasets.dataset_id IS '数据集业务ID，同一逻辑数据集不同版本共享';
COMMENT ON COLUMN eval_datasets.status IS '状态: active-生效, deprecated-已废弃';

-- =============================================================

-- 2. 评测任务表
-- 存储评测任务的生命周期信息
CREATE TABLE IF NOT EXISTS eval_tasks (
    id SERIAL PRIMARY KEY,
    task_id VARCHAR(64) NOT NULL UNIQUE,
    task_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    dataset_id VARCHAR(64) NOT NULL,
    dataset_version VARCHAR(16) NOT NULL,
    status VARCHAR(16) NOT NULL, -- pending/running/completed/failed
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    parameters JSONB,
    creator VARCHAR(64) NOT NULL,
    error_message TEXT,

    INDEX(idx_task_id) (task_id),
    INDEX(idx_dataset_version) (dataset_id, dataset_version),
    INDEX(idx_status) (status)
);

COMMENT ON TABLE eval_tasks IS '评测任务表，管理单个评测任务的生命周期';
COMMENT ON COLUMN eval_tasks.status IS '任务状态: pending-等待, running-执行中, completed-完成, failed-失败';

-- =============================================================

-- 3. 评测结果表
-- 存储评测指标结果，一经插入不允许修改删除，保证可信度
CREATE TABLE IF NOT EXISTS eval_results (
    id SERIAL PRIMARY KEY,
    result_id VARCHAR(64) NOT NULL UNIQUE,
    task_id VARCHAR(64) NOT NULL,
    dataset_id VARCHAR(64) NOT NULL,
    dataset_version VARCHAR(16) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    metric_name VARCHAR(64) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    confidence_lower DOUBLE PRECISION,
    confidence_upper DOUBLE PRECISION,
    details JSONB,               -- 详细结果数据（JSON格式）
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    INDEX(idx_task_id) (task_id),
    INDEX(idx_dataset_model_version) (dataset_id, dataset_version, model_version),
    INDEX(idx_metric_name) (metric_name)
);

COMMENT ON TABLE eval_results IS '评测结果表，存储各个指标的评测结果，只允许插入不允许修改删除';
COMMENT ON COLUMN eval_results.metric_name IS '指标名称，如 recall@10, mrr, ndcg 等';
COMMENT ON COLUMN eval_results.metric_value IS '指标数值';
COMMENT ON COLUMN eval_results.confidence_lower IS '置信区间下限';
COMMENT ON COLUMN eval_results.confidence_upper IS '置信区间上限';

-- =============================================================

-- 4. 向量元数据表
-- 存储向量元数据，实际向量存储在Milvus
CREATE TABLE IF NOT EXISTS eval_vectors (
    id SERIAL PRIMARY KEY,
    vector_id VARCHAR(64) NOT NULL UNIQUE,
    milvus_id VARCHAR(64),            -- Milvus中的主键ID
    task_id VARCHAR(64) NOT NULL,
    dataset_id VARCHAR(64) NOT NULL,
    dataset_version VARCHAR(16) NOT NULL,
    record_id VARCHAR(64) NOT NULL,   -- 原始数据集中的记录ID
    content TEXT,                     -- 原始内容
    meta_json JSONB,                  -- 附加元数据JSON
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    INDEX(idx_task_id) (task_id),
    INDEX(idx_dataset_version) (dataset_id, dataset_version)
);

COMMENT ON TABLE eval_vectors IS '向量元数据表，存储评测向量的元数据信息';
COMMENT ON COLUMN eval_vectors.milvus_id IS 'Milvus中的向量ID';
COMMENT ON COLUMN eval_vectors.content IS '原始文本内容';

-- =============================================================

-- 完成
SELECT '评测领域数据库初始化完成' as message;
