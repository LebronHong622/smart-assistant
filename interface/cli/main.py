from application.agent import create_qa_agent
from infrastructure.log import app_logger


def main():
    """主函数，创建QA代理并处理用户输入"""
    app_logger.info("🤖 多任务问答助手启动中...")
    print("🤖 多任务问答助手启动中...")

    try:
        # 创建QA代理实例
        agent = create_qa_agent()
        app_logger.info("✅ 代理初始化完成，开始对话")
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
                    app_logger.info("👋 对话结束，再见！")
                    print("👋 对话结束，再见！")
                    break

                # 处理空输入
                if not user_input.strip():
                    app_logger.warning("💬 请输入您的问题")
                    print("💬 请输入您的问题")
                    continue

                # 调用代理的chat方法处理用户输入
                app_logger.info(f"处理用户输入: {user_input}")
                print("🤖 助手: 正在思考...")
                response = agent.chat(user_input)
                app_logger.info(f"助手回复: {response}")
                print(f"🤖 助手: {response}")
                print("=" * 50)

            except KeyboardInterrupt:
                # 处理Ctrl+C中断
                app_logger.info("👋 对话被中断，再见！")
                print("\n👋 对话被中断，再见！")
                break
            except Exception as e:
                app_logger.error(f"❌ 发生错误: {str(e)}")
                print(f"❌ 发生错误: {str(e)}")
                print("=" * 50)
    except Exception as e:
        app_logger.error(f"❌ 应用程序初始化失败: {str(e)}")
        print(f"❌ 应用程序初始化失败: {str(e)}")


if __name__ == "__main__":
    main()
