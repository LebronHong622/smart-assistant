"""
评测结果查询命令行接口
支持查询结果和统计比较
"""
import argparse
from interface.container import container


def query_by_task(args):
    """按任务ID查询结果"""
    service = container.getResultQueryService()
    results = service.get_by_task(args.task_id)

    if not results:
        print(f"未找到评测结果: task_id={args.task_id}")
        return

    print(f"\n📊 评测结果: task_id={args.task_id}\n")
    print(f"{'Metric':<20} {'Value':<12} {'Confidence'}")
    print("-" * 55)
    for result in results:
        val = result.metric_value
        if val.confidence_lower is not None and val.confidence_upper is not None:
            conf = f"[{val.confidence_lower:.4f}, {val.confidence_upper:.4f}]"
        else:
            conf = "-"
        print(f"{result.metric_name:<20} {val.value:<12.6f} {conf}")

    # 聚合输出
    print("\n📈 聚合结果:")
    aggregated = service.aggregate_by_metric(results)
    for metric, data in aggregated.items():
        print(f"  {metric}: {data['value']:.6f}")


def query_by_version(args):
    """按版本查询结果"""
    service = container.getResultQueryService()
    results = service.get_by_version(args.dataset_id, args.dataset_version, args.model_version)

    if not results:
        print(f"未找到评测结果: {args.dataset_id}@{args.dataset_version}, model={args.model_version}")
        return

    print(f"\n📊 评测结果: {args.dataset_id}@{args.dataset_version}, model={args.model_version}\n")
    print(f"{'Metric':<20} {'Value':<12}")
    print("-" * 35)
    for result in results:
        print(f"{result.metric_name:<20} {result.metric_value.value:<12.6f}")


def compare_models(args):
    """比较多个模型版本"""
    service = container.getResultQueryService()
    model_versions = args.model_versions.split(',')

    comparison = service.compare_models(
        args.dataset_id,
        args.dataset_version,
        model_versions
    )

    print(f"\n🔍 模型版本比较: {args.dataset_id}@{args.dataset_version}\n")

    # 计算宽度
    max_metric_len = max(len(m) for m in comparison.keys()) if comparison else 20

    header = f"{'Metric':<{max_metric_len}} " + " ".join(f"{mv:<12}" for mv in model_versions)
    print(header)
    print("-" * (max_metric_len + 1 + sum(13 for _ in model_versions)))

    for metric, values in comparison.items():
        line = f"{metric:<{max_metric_len}} "
        line += " ".join(f"{values.get(mv, 0):<12.6f}" for mv in model_versions)
        print(line)


def main():
    parser = argparse.ArgumentParser(description='评测结果查询命令行工具')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # query task 命令
    query_task_parser = subparsers.add_parser('task', help='按任务ID查询结果')
    query_task_parser.add_argument('--task-id', required=True, help='任务ID')

    # query version 命令
    query_version_parser = subparsers.add_parser('version', help='按数据集版本和模型版本查询结果')
    query_version_parser.add_argument('--dataset-id', required=True, help='数据集ID')
    query_version_parser.add_argument('--dataset-version', required=True, help='数据集版本')
    query_version_parser.add_argument('--model-version', required=True, help='模型版本')

    # compare 命令
    compare_parser = subparsers.add_parser('compare', help='比较多个模型版本')
    compare_parser.add_argument('--dataset-id', required=True, help='数据集ID')
    compare_parser.add_argument('--dataset-version', required=True, help='数据集版本')
    compare_parser.add_argument('--model-versions', required=True, help='模型版本列表，逗号分隔')

    args = parser.parse_args()

    if args.command == 'task':
        query_by_task(args)
    elif args.command == 'version':
        query_by_version(args)
    elif args.command == 'compare':
        compare_models(args)


if __name__ == '__main__':
    main()
