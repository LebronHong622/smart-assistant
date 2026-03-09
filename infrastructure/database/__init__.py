"""
PostgreSQL 数据库模块
"""
from .postgres_client import PostgreSQLClient, postgres_client

__all__ = ["PostgreSQLClient", "postgres_client"]
