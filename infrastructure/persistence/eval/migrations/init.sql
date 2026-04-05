-- =============================================================
-- 评测领域数据库初始化脚本 (MySQL)
-- 为AI导购评测系统创建必要的表结构
-- =============================================================

-- 1. 测试数据集表
-- 存储测试数据集的元数据和版本信息，实际数据存储在本地文件
CREATE TABLE IF NOT EXISTS eval_datasets (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    dataset_id VARCHAR(64) NOT NULL COMMENT '数据集业务ID（同一逻辑数据集不同版本共享）',
    dataset_name VARCHAR(255) NOT NULL COMMENT '数据集名称',
    version VARCHAR(16) NOT NULL COMMENT '版本格式: vX.Y',
    file_path TEXT NOT NULL COMMENT '本地文件路径',
    create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    update_time DATETIME DEFAULT NULL COMMENT '更新时间',
    creator VARCHAR(64) NOT NULL COMMENT '创建人',
    updater VARCHAR(64) DEFAULT NULL COMMENT '更新人',
    status VARCHAR(16) NOT NULL DEFAULT 'active' COMMENT '状态: active-生效, deprecated-已废弃',
    metadata JSON COMMENT '扩展元数据',
    task_count INT NOT NULL DEFAULT 0 COMMENT '关联任务数',

    -- 索引
    INDEX idx_eval_datasets_dataset_id (dataset_id),
    INDEX idx_eval_datasets_status (status),

    -- 唯一约束：同一数据集同一版本只能存在一个
    CONSTRAINT uk_dataset_version UNIQUE (dataset_id, version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='测试数据集表，存储版本化管理的测试数据集元数据';

-- =============================================================

-- 2. 评测任务表
-- 存储评测任务的生命周期信息
CREATE TABLE IF NOT EXISTS eval_tasks (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    task_id VARCHAR(64) NOT NULL UNIQUE COMMENT '任务业务ID',
    task_name VARCHAR(255) NOT NULL COMMENT '任务名称',
    model_version VARCHAR(64) NOT NULL COMMENT '模型版本',
    dataset_id VARCHAR(64) NOT NULL COMMENT '数据集ID',
    dataset_version VARCHAR(16) NOT NULL COMMENT '数据集版本',
    status VARCHAR(16) NOT NULL COMMENT '任务状态: pending-等待, running-执行中, completed-完成, failed-失败',
    create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    start_time DATETIME COMMENT '开始时间',
    end_time DATETIME COMMENT '结束时间',
    parameters JSON COMMENT '评测参数（JSON格式）',
    creator VARCHAR(64) NOT NULL COMMENT '创建人',
    error_message TEXT COMMENT '错误信息',

    INDEX idx_task_id (task_id),
    INDEX idx_dataset_version (dataset_id, dataset_version),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='评测任务表，管理单个评测任务的生命周期';

-- =============================================================

-- 3. 评测结果表
-- 存储评测指标结果，一经插入不允许修改删除，保证可信度
CREATE TABLE IF NOT EXISTS eval_results (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    result_id VARCHAR(64) NOT NULL UNIQUE COMMENT '结果业务ID',
    task_id VARCHAR(64) NOT NULL COMMENT '任务ID',
    dataset_id VARCHAR(64) NOT NULL COMMENT '数据集ID',
    dataset_version VARCHAR(16) NOT NULL COMMENT '数据集版本',
    model_version VARCHAR(64) NOT NULL COMMENT '模型版本',
    metric_name VARCHAR(64) NOT NULL COMMENT '指标名称，如 recall@10, mrr, ndcg 等',
    metric_value DOUBLE NOT NULL COMMENT '指标数值',
    confidence_lower DOUBLE COMMENT '置信区间下限',
    confidence_upper DOUBLE COMMENT '置信区间上限',
    details JSON COMMENT '详细结果数据（JSON格式）',
    create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',

    INDEX idx_task_id (task_id),
    INDEX idx_dataset_model_version (dataset_id, dataset_version, model_version),
    INDEX idx_metric_name (metric_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='评测结果表，存储各个指标的评测结果，只允许插入不允许修改删除';

-- =============================================================

-- 4. 向量元数据表
-- 存储向量元数据，实际向量存储在Milvus
CREATE TABLE IF NOT EXISTS eval_vectors (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    vector_id VARCHAR(64) NOT NULL UNIQUE COMMENT '向量业务ID',
    milvus_id VARCHAR(64) COMMENT 'Milvus中的向量ID',
    task_id VARCHAR(64) NOT NULL COMMENT '任务ID',
    dataset_id VARCHAR(64) NOT NULL COMMENT '数据集ID',
    dataset_version VARCHAR(16) NOT NULL COMMENT '数据集版本',
    record_id VARCHAR(64) NOT NULL COMMENT '原始数据集中的记录ID',
    content TEXT COMMENT '原始文本内容',
    meta_json JSON COMMENT '附加元数据JSON',
    create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',

    INDEX idx_task_id (task_id),
    INDEX idx_dataset_version (dataset_id, dataset_version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='向量元数据表，存储评测向量的元数据信息';

-- =============================================================

-- 完成
SELECT '评测领域数据库初始化完成' as message;
