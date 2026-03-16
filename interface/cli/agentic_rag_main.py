#!/usr/bin/env python3
"""
Agentic RAG 命令行交互入口
独立于现有QA CLI，提供Agentic RAG功能的交互界面
"""
import sys
from typing import Optional
from uuid import uuid4

from interface.container import container
from infrastructure.core.log import app_logger
from infrastructure.config.settings import AppSettings


def main():
    """主函数"""
    print("=" * 60)
    print("🤖 Agentic RAG 智能问答助手")
    print("=" * 60)
    print("输入 /help 查看帮助，输入 /exit 退出程序")
    print("=" * 60)

    # 初始化代理
    session_id = str(uuid4())
    agent = container.get_agentic_rag_agent(session_id=session_id)
    app_logger.info(f"Agentic RAG CLI 启动，session_id={session_id}")

    while True:
        try:
            user_input = input("\n🧑 请输入问题: ").strip()

            if not user_input:
                continue

            # 处理命令
            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                print("👋 再见！")
                break
            elif user_input.lower() == "/help":
                print("\n📖 帮助信息:")
                print("/help    - 显示帮助信息")
                print("/clear   - 清空当前会话")
                print("/history - 查看会话历史")
                print("/exit    - 退出程序")
                continue
            elif user_input.lower() == "/clear":
                agent.clear_session()
                print("✅ 会话已清空")
                continue
            elif user_input.lower() == "/history":
                history = agent.get_session_history()
                print("\n📜 会话历史:")
                for msg in history:
                    role = "🧑 用户" if msg["role"] == "user" else "🤖 助手"
                    print(f"{role}: {msg['content']}")
                continue

            # 处理用户问题
            print("\n🤔 思考中...")
            response = agent.chat(user_input)
            print(f"\n🤖 回答: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            app_logger.error(f"CLI 处理错误: {str(e)}", exc_info=True)
            print(f"\n❌ 出错了: {str(e)}")


if __name__ == "__main__":
    main()
