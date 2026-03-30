"""
测试数据集生成CLI命令
生成单跳测试数据集
"""
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from interface.container import container
from infrastructure.core.log import app_logger


def main():
    parser = argparse.ArgumentParser(description="生成单跳测试数据集")
    parser.add_argument(
        "--config",
        default="config/eval/test_dataset_config.yaml",
        help="配置文件路径 (默认: config/eval/test_dataset_config.yaml)"
    )
    parser.add_argument("--dataset-id", required=True, help="数据集业务ID")
    parser.add_argument("--dataset-name", required=True, help="数据集名称")
    parser.add_argument("--creator", default="cli", help="创建者")
    parser.add_argument("--test-size", type=int, default=10, help="总问题数量限制")
    parser.add_argument("--major-change", action="store_true", help="是否为重大变更（主版本号+1）")
    parser.add_argument("--format", default=None, help="输出格式 (parquet/csv/json)")

    args = parser.parse_args()

    try:
        # 从容器获取应用服务
        generation_service = container.getTestDatasetGenerationService()

        # 生成并保存
        # 如果传入自定义配置，需要重新创建生成器，否则使用容器中默认配置
        if args.config != "config/eval/test_dataset_config.yaml":
            from infrastructure.external.eval.adapters.ragas_single_hop_adapter import RagasSingleHopAdapter
            from application.services.eval.test_dataset_generation_service import TestDatasetGenerationService
            test_generator = RagasSingleHopAdapter(args.config)
            generation_service = TestDatasetGenerationService(
                dataset_management_service=container.getDatasetManagementService(),
                test_generator=test_generator,
            )

        # 生成并保存
        dataset, version = generation_service.generate_and_save(
            dataset_id=args.dataset_id,
            dataset_name=args.dataset_name,
            creator=args.creator,
            documents=[],  # 空列表表示从配置路径加载
            test_size=args.test_size,
            distribution="single_hop",
            is_major_change=args.major_change,
            format=args.format if args.format else "parquet",
        )

        app_logger.info(
            f"测试数据集生成完成: dataset_id={dataset.dataset_id}, version={version}, "
            f"task_count={dataset.task_count}, file_path={dataset.file_path}"
        )
        print(f"✓ 生成完成")
        print(f"  数据集ID: {dataset.dataset_id}")
        print(f"  版本: {version}")
        print(f"  问题数量: {dataset.task_count}")
        print(f"  保存路径: {dataset.file_path}")

    except Exception as e:
        app_logger.error(f"生成失败: {str(e)}", exc_info=True)
        print(f"✗ 生成失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
