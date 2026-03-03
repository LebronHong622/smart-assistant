from application.agent import create_qa_agent


def main():
    """主函数，创建QA代理并处理用户输入"""
    print("🤖 多任务问答助手启动中...")

    # 创建QA代理实例
    agent = create_qa_agent()
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
                print("👋 对话结束，再见！")
                break

            # 处理空输入
            if not user_input.strip():
                print("💬 请输入您的问题")
                continue

            # 调用代理的chat方法处理用户输入
            print("🤖 助手: 正在思考...")
            response = agent.chat(user_input)
            print(f"🤖 助手: {response}")
            print("=" * 50)

        except KeyboardInterrupt:
            # 处理Ctrl+C中断
            print("\n👋 对话被中断，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")
            print("=" * 50)


if __name__ == "__main__":
    main()
