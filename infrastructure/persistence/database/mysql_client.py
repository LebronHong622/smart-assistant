"""
MySQL 客户端实现
采用单例模式设计，全局唯一实例
支持连接池、自动重连、SQL执行等功能
"""
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from config.settings import settings
from infrastructure.core.log import app_logger


class MySQLClient:
    """MySQL 客户端单例类"""
    _instance = None
    _initialized = False
    _engine = None
    _session_factory = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._config = settings.mysql
            self._connect()
            self._initialized = True

    def _connect(self) -> None:
        """建立数据库连接并初始化连接池"""
        try:
            app_logger.info("正在初始化MySQL连接池")

            # 创建SQLAlchemy引擎
            self._engine = create_engine(
                self._config.database_url,
                pool_size=10,
                max_overflow=20,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=False
            )

            # 创建会话工厂
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False
            )

            app_logger.info("MySQL连接池初始化成功")
        except OperationalError as e:
            app_logger.error(f"MySQL连接失败: {str(e)}")
            raise
        except Exception as e:
            app_logger.error(f"MySQL初始化失败: {str(e)}")
            raise

    def _reconnect_if_needed(self) -> None:
        """检查连接是否有效，无效则重新连接"""
        try:
            if self._engine is None:
                self._connect()
                return

            # 测试连接
            with self._engine.connect():
                pass
        except OperationalError:
            app_logger.warning("MySQL连接已断开，尝试重新连接")
            self._connect()

    @contextmanager
    def get_session(self) -> Session:
        """获取数据库会话上下文管理器"""
        self._reconnect_if_needed()
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            app_logger.error(f"数据库操作失败，事务回滚: {str(e)}")
            raise
        finally:
            session.close()

    def execute(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """执行SQL语句并返回结果

        Args:
            sql: SQL语句
            params: SQL参数

        Returns:
            查询结果列表，每个元素为字段名到值的映射
        """
        if params is None:
            params = {}

        try:
            app_logger.debug(f"执行SQL: {sql}, 参数: {params}")

            with self.get_session() as session:
                result = session.execute(text(sql), params)

                # 如果是查询语句，返回结果
                if result.returns_rows:
                    columns = result.keys()
                    return [dict(zip(columns, row)) for row in result.fetchall()]

                return []
        except SQLAlchemyError as e:
            app_logger.error(f"SQL执行失败: {str(e)}, SQL: {sql}, 参数: {params}")
            raise

    def execute_many(self, sql: str, params_list: List[Dict[str, Any]]) -> None:
        """批量执行SQL语句

        Args:
            sql: SQL语句
            params_list: 参数列表
        """
        try:
            app_logger.debug(f"批量执行SQL: {sql}, 参数数量: {len(params_list)}")

            with self.get_session() as session:
                session.execute(text(sql), params_list)
        except SQLAlchemyError as e:
            app_logger.error(f"批量SQL执行失败: {str(e)}, SQL: {sql}")
            raise

    def ping(self) -> bool:
        """测试数据库连接是否正常

        Returns:
            连接正常返回True，否则返回False
        """
        try:
            self.execute("SELECT 1")
            return True
        except Exception as e:
            app_logger.error(f"MySQL ping失败: {str(e)}")
            return False

    def close(self) -> None:
        """关闭数据库连接池"""
        if self._engine is not None:
            app_logger.info("正在关闭MySQL连接池")
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._initialized = False
            app_logger.info("MySQL连接池已关闭")


# 全局单例实例
mysql_client = MySQLClient()
