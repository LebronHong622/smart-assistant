import pytest
from domain.shared.model_enums import ModelType, RoutingStrategy
from infrastructure.external.model.routing.model_router import ModelRouter
from infrastructure.external.model.routing.strategy_factory import StrategyFactory
from domain.shared.ports.model_router_port import BaseRoutingStrategy
from domain.shared.ports.model_capability_port import BaseModel


class TestModelRouter:
    """模型路由测试类"""

    def setup_method(self):
        """每个测试前清空缓存和重置单例"""
        router = ModelRouter()
        router.clear_cache()
        # 重置单例
        ModelRouter._instance = None
        ModelRouter._initialized = False

    def test_singleton_pattern(self):
        """测试单例模式"""
        router1 = ModelRouter()
        router2 = ModelRouter()
        assert router1 is router2

    def test_get_chat_model(self):
        """测试获取普通聊天模型"""
        router = ModelRouter()
        model = router.get_model(ModelType.CHAT)
        assert model is not None
        assert isinstance(model, BaseModel)

    def test_get_tool_enabled_chat_model(self):
        """测试获取带工具调用的聊天模型"""
        router = ModelRouter()
        model = router.get_model(ModelType.TOOL_ENABLED_CHAT)
        assert model is not None
        assert isinstance(model, BaseModel)

    def test_get_embedding_model(self):
        """测试获取嵌入模型"""
        router = ModelRouter()
        model = router.get_model(ModelType.EMBEDDING)
        assert model is not None
        assert isinstance(model, BaseModel)

    def test_model_caching(self):
        """测试模型缓存功能"""
        router = ModelRouter()
        model1 = router.get_model(ModelType.CHAT)
        model2 = router.get_model(ModelType.CHAT)
        assert model1 is model2  # 同一个实例

    def test_different_model_types_different_instances(self):
        """测试不同模型类型返回不同实例"""
        router = ModelRouter()
        chat_model = router.get_model(ModelType.CHAT)
        tool_model = router.get_model(ModelType.TOOL_ENABLED_CHAT)
        embedding_model = router.get_model(ModelType.EMBEDDING)

        assert chat_model is not tool_model
        assert chat_model is not embedding_model
        assert tool_model is not embedding_model

    def test_different_framework_parameter(self):
        """测试指定framework参数"""
        router = ModelRouter()
        # 默认langchain框架
        model = router.get_model(ModelType.CHAT, framework="langchain")
        assert model is not None

    def test_invalid_model_type(self):
        """测试不支持的模型类型"""
        router = ModelRouter()
        with pytest.raises(ValueError):
            # 传入无效的模型类型
            router.get_model("invalid_type")

    def test_custom_model_name(self):
        """测试指定自定义模型名称"""
        router = ModelRouter()
        model1 = router.get_model(ModelType.CHAT, model_name="custom-model-1")
        model2 = router.get_model(ModelType.CHAT, model_name="custom-model-2")
        model3 = router.get_model(ModelType.CHAT, model_name="custom-model-1")

        assert model1 is not model2
        assert model1 is model3  # 相同模型名称返回同一个实例

    def test_clear_cache(self):
        """测试清空缓存功能"""
        router = ModelRouter()
        model1 = router.get_model(ModelType.CHAT)
        router.clear_cache()
        model2 = router.get_model(ModelType.CHAT)
        assert model1 is not model2  # 缓存清空后返回新实例

    def test_strategy_factory_registration(self):
        """测试策略工厂注册功能"""
        # 创建一个测试策略
        class TestStrategy(BaseRoutingStrategy):
            def select_model(self, model_type, **kwargs):
                return None

        # 注册测试策略
        StrategyFactory.register_strategy(
            "test_framework",
            ModelType.CHAT,
            RoutingStrategy.DEFAULT,
            TestStrategy
        )

        # 验证可以获取到注册的策略
        strategy = StrategyFactory.get_strategy(
            "test_framework",
            ModelType.CHAT,
            RoutingStrategy.DEFAULT
        )
        assert isinstance(strategy, TestStrategy)

    def test_invalid_strategy_combination(self):
        """测试不存在的策略组合"""
        with pytest.raises(ValueError):
            StrategyFactory.get_strategy(
                "non_existent_framework",
                ModelType.CHAT,
                RoutingStrategy.DEFAULT
            )