"""
数据集管理命令行接口
支持创建、列表、查询数据集版本
"""
import argparse
import pandas as pd
from interface.container import container


def create_dataset(args):
    """创建新版本数据集"""
    service = container.getDatasetManagementService()

    # 读取源文件
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    elif args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    else:
        print(f"不支持的文件格式: {args.input}，支持 .parquet 或 .csv")
        return

    # 创建新版本
    dataset, version = service.create_from_dataframe(
        dataset_id=args.dataset_id,
        dataset_name=args.name,
        creator=args.creator,
        df=df,
        is_major_change=args.major
    )

    print(f"\n✅ 数据集版本创建成功!")
    print(f"   Dataset ID: {dataset.dataset_id}")
    print(f"   Name: {dataset.dataset_name}")
    print(f"   Version: {version}")
    print(f"   ID: {dataset.id}")
    print(f"   Tasks: {dataset.task_count}")
    print(f"   File: {dataset.file_path}")


def list_versions(args):
    """列出数据集所有版本"""
    service = container.getDatasetManagementService()
    versions = service.list_versions(args.dataset_id)

    if not versions:
        print(f"未找到数据集: {args.dataset_id}")
        return

    print(f"\n📋 数据集版本列表: {args.dataset_id}\n")
    print(f"{'ID':<5} {'Version':<8} {'Status':<10} {'Created':<20} {'Tasks':<5} {'Name'}")
    print("-" * 70)
    for ds in versions:
        status_mark = "✅" if ds.is_active else "⚠️ "
        print(f"{ds.id:<5} {ds.version.to_string():<8} {status_mark}{ds.status.value:<8} "
              f"{ds.create_time.strftime('%Y-%m-%d %H:%M'):<20} {ds.task_count:<5} {ds.dataset_name}")


def get_version(args):
    """获取特定版本信息"""
    service = container.getDatasetManagementService()

    if args.version == 'latest':
        dataset = service.get_latest(args.dataset_id)
    else:
        dataset = service.get_by_version(args.dataset_id, args.version)

    if not dataset:
        print(f"未找到数据集版本: {args.dataset_id}@{args.version}")
        return

    print(f"\n📄 数据集信息:\n")
    print(f"  ID: {dataset.id}")
    print(f"  Dataset ID: {dataset.dataset_id}")
    print(f"  Name: {dataset.dataset_name}")
    print(f"  Version: {dataset.version.to_string()}")
    print(f"  Status: {dataset.status.value} {'(active)' if dataset.is_active else '(deprecated)'}")
    print(f"  Creator: {dataset.creator}")
    print(f"  Created: {dataset.create_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Tasks: {dataset.task_count}")
    print(f"  File: {dataset.file_path}")
    if dataset.metadata:
        print(f"  Metadata: {dataset.metadata}")


def load_dataset(args):
    """加载数据集到DataFrame并显示预览"""
    service = container.getDatasetManagementService()

    if args.version == 'latest':
        dataset = service.get_latest(args.dataset_id)
    else:
        dataset = service.get_by_version(args.dataset_id, args.version)

    if not dataset:
        print(f"未找到数据集版本: {args.dataset_id}@{args.version}")
        return

    df = service.load_dataframe(dataset)
    print(f"\n📊 数据集预览: {dataset.dataset_id}@{dataset.version.to_string()}\n")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    print(df.head(args.rows).to_string())
    print("\n...")


def main():
    parser = argparse.ArgumentParser(description='评测数据集管理命令行工具')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # create 命令
    create_parser = subparsers.add_parser('create', help='创建新版本数据集')
    create_parser.add_argument('--dataset-id', required=True, help='数据集业务ID')
    create_parser.add_argument('--name', required=True, help='数据集名称')
    create_parser.add_argument('--input', required=True, help='输入数据文件 (.parquet 或 .csv)')
    create_parser.add_argument('--creator', default='cli', help='创建者')
    create_parser.add_argument('--major', action='store_true', help='是否为主版本升级')

    # list 命令
    list_parser = subparsers.add_parser('list', help='列出数据集所有版本')
    list_parser.add_argument('--dataset-id', required=True, help='数据集业务ID')

    # get 命令
    get_parser = subparsers.add_parser('get', help='获取特定版本信息')
    get_parser.add_argument('--dataset-id', required=True, help='数据集业务ID')
    get_parser.add_argument('--version', required=True, help='版本 (vX.Y 或 "latest")')

    # load 命令
    load_parser = subparsers.add_parser('load', help='加载数据集并显示预览')
    load_parser.add_argument('--dataset-id', required=True, help='数据集业务ID')
    load_parser.add_argument('--version', required=True, help='版本 (vX.Y 或 "latest")')
    load_parser.add_argument('--rows', type=int, default=5, help='显示行数')

    args = parser.parse_args()

    if args.command == 'create':
        create_dataset(args)
    elif args.command == 'list':
        list_versions(args)
    elif args.command == 'get':
        get_version(args)
    elif args.command == 'load':
        load_dataset(args)


if __name__ == '__main__':
    main()
