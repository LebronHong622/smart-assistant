"""
PostgreSQL 客户端单元测试
"""
import pytest
from sqlalchemy.exc import OperationalError

from infrastructure.database.postgres_client import PostgreSQLClient, postgres_client


class TestPostgreSQLClient:
    """PostgreSQL客户端测试类"""

    def test_singleton_pattern(self):
        """测试单例模式是否生效"""
        client1 = PostgreSQLClient()
        client2 = PostgreSQLClient()
        assert client1 is client2
        assert id(client1) == id(client2)
        assert postgres_client is client1

    def test_client_initialization(self):
        """测试客户端初始化"""
        client = PostgreSQLClient()
        assert client is not None
        assert hasattr(client, '_config')
        assert hasattr(client, '_engine')
        assert hasattr(client, '_session_factory')

    @pytest.mark.skip(reason="需要实际PostgreSQL服务才能运行")
    def test_ping(self):
        """测试数据库连接ping功能"""
        assert postgres_client.ping() is True

    @pytest.mark.skip(reason="需要实际PostgreSQL服务才能运行")
    def test_execute_query(self):
        """测试SQL查询执行"""
        result = postgres_client.execute("SELECT 1 AS test_value")
        assert len(result) == 1
        assert result[0]['test_value'] == 1

    @pytest.mark.skip(reason="需要实际PostgreSQL服务才能运行")
    def test_get_session_context(self):
        """测试会话上下文管理器"""
        with postgres_client.get_session() as session:
            result = session.execute("SELECT 1 AS test_value")
            row = result.fetchone()
            assert row.test_value == 1

    @pytest.mark.skip(reason="需要实际PostgreSQL服务才能运行")
    def test_execute_with_parameters(self):
        """测试带参数的SQL执行"""
        result = postgres_client.execute(
            "SELECT :param1 + :param2 AS sum_result",
            {"param1": 10, "param2": 20}
        )
        assert len(result) == 1
        assert result[0]['sum_result'] == 30

    def test_close_connection(self):
        """测试关闭连接池"""
        client = PostgreSQLClient()
        client.close()
        assert client._engine is None
        assert client._session_factory is None
        assert client._initialized is False

        # 重新连接
        client._initialized = False
        client.__init__()
        assert client._engine is not None
        assert client._session_factory is not None
