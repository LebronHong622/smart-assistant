from interface.container import container
from application.services.qa.qa_service import create_qa_agent
from application.common.app_initializer import AppInitializer


def main():
    """主函数，创建QA代理并处理用户输入"""
    logger = container.get_logger()
    logger.info("🤖 多任务问答助手启动中...")
    print("🤖 多任务问答助手启动中...")

    try:
        # 初始化底层组件
        app_initializer = AppInitializer.get_instance()
        app_initializer.initialize()

        # 使用容器获取 QA 服务并创建代理
        qa_service = container.get_qa_service()
        agent = create_qa_agent(qa_service=qa_service)
        logger.info("✅ 代理初始化完成，开始对话")
        print("✅ 代理初始化完成，开始对话")
        print("💡 提示：输入 'exit' 或 'quit' 退出对话")
        print("=" * 50)

        # 命令行交互循环
        while True:
            try:
                # 获取用户输入
                user_input = input("👤 用户: ")

                # 检查退出条件
                if user_input.lower() in ["exit", "quit"]:
                    logger.info("👋 对话结束，再见！")
                    print("👋 对话结束，再见！")
                    break

                # 处理空输入
                if not user_input.strip():
                    logger.warning("💬 请输入您的问题")
                    print("💬 请输入您的问题")
                    continue

                # 调用代理的chat方法处理用户输入
                logger.info(f"处理用户输入: {user_input}")
                print("🤖 助手: 正在思考...")
                response = agent.chat(user_input)
                logger.info(f"助手回复: {response}")
                print(f"🤖 助手: {response}")
                print("=" * 50)

            except KeyboardInterrupt:
                # 处理Ctrl+C中断
                logger.info("👋 对话被中断，再见！")
                print("\n👋 对话被中断，再见！")
                break
            except Exception as e:
                logger.error(f"❌ 发生错误: {str(e)}")
                print(f"❌ 发生错误: {str(e)}")
                print("=" * 50)
    except Exception as e:
        logger.error(f"❌ 应用程序初始化失败: {str(e)}")
        print(f"❌ 应用程序初始化失败: {str(e)}")


if __name__ == "__main__":
    main()
