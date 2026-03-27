from interface.container import container
from application.agent.agentic_rag_agent import AgenticRagAgent
from application.common.app_initializer import AppInitializer


def main():
    """主函数，创建Agentic RAG代理并处理用户输入"""
    logger = container.get_logger()
    logger.info("🤖 多任务问答助手启动中...")
    print("🤖 多任务问答助手启动中...")

    try:
        # 初始化底层组件
        app_initializer = AppInitializer.get_instance()
        app_initializer.initialize()

        # 创建Agentic RAG代理实例
        agent = container.get_agentic_rag_agent(session_id="cli_session")
        logger.info("✅ Agentic RAG代理初始化完成，开始对话")
        print("✅ Agentic RAG代理初始化完成，开始对话")
        print("💡 提示：输入 'exit' 或 'quit' 退出对话")
        print("💡 新特性：智能工具选择、多轮对话记忆、动态工作流编排")
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

                # 调用代理的chat_with_documents方法处理用户输入
                logger.info(f"处理用户输入: {user_input}")
                print("🤖 助手: 正在思考（使用Agentic RAG架构）...")
                answer, documents = agent.chat_with_documents(user_input)
                logger.info(f"助手回复: {answer[:100]}..." if len(answer) > 100 else f"助手回复: {answer}")
                
                print(f"🤖 助手: {answer}")
                
                # 显示检索到的文档信息
                if documents:
                    print(f"\n📚 参考文档数量: {len(documents)}")
                    for i, doc in enumerate(documents[:2], 1):  # 只显示前2个文档
                        source = doc.get('metadata', {}).get('source', '未知来源')
                        print(f"   📄 文档{i}: {source}")
                
                print("=" * 50)

            except KeyboardInterrupt:
                # 处理Ctrl+C中断
                logger.info("👋 对话被中断，再见！")
                print("\n👋 对话被中断，再见！")
                break
            except Exception as e:
                logger.error(f"❌ 发生错误: {str(e)}", exc_info=True)
                print(f"❌ 发生错误: {str(e)}")
                print("=" * 50)
    except Exception as e:
        logger.error(f"❌ 应用程序初始化失败: {str(e)}", exc_info=True)
        print(f"❌ 应用程序初始化失败: {str(e)}")


if __name__ == "__main__":
    main()