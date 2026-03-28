"""
评测执行命令行接口
支持运行评测任务，查看状态
"""
import argparse
import json
from interface.container import container


def run_eval(args):
    """运行评测任务"""
    service = container.getEvalExecutionService()

    # 创建任务
    parameters = {}
    if args.parameters:
        with open(args.parameters, 'r') as f:
            parameters = json.load(f)

    task = service.create_task(
        task_name=args.name,
        model_version=args.model_version,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        creator=args.creator,
        parameters=parameters
    )

    print(f"\n🚀 评测任务创建成功!")
    print(f"   Task ID: {task.task_id}")
    print(f"   Name: {task.task_name}")
    print(f"   Model: {task.model_version}")
    print(f"   Dataset: {task.dataset_id}@{task.dataset_version}")
    print(f"\n下一步: 运行评测计算并保存结果")


def main():
    parser = argparse.ArgumentParser(description='评测执行命令行工具')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # run 命令
    run_parser = subparsers.add_parser('run', help='创建并运行评测任务')
    run_parser.add_argument('--name', required=True, help='任务名称')
    run_parser.add_argument('--model-version', required=True, help='模型版本')
    run_parser.add_argument('--dataset-id', required=True, help='数据集ID')
    run_parser.add_argument('--dataset-version', required=True, help='数据集版本')
    run_parser.add_argument('--creator', default='cli', help='创建者')
    run_parser.add_argument('--parameters', help='参数JSON文件路径')

    args = parser.parse_args()

    if args.command == 'run':
        run_eval(args)


if __name__ == '__main__':
    main()
