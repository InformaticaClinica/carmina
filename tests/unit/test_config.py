"""
Unit tests for configuration module.
"""
import pytest
import os
from unittest.mock import patch
from src.carmina.config import *


@pytest.mark.unit
class TestConfig:
    """Test configuration management."""
    
    @pytest.fixture(autouse=False)
    def no_test_env(self):
        """Fixture to disable test environment setup for specific tests."""
        pass
    
    def test_debug_enabled_default(self):
        """Test DEBUG_ENABLED default value."""
        with patch.dict(os.environ, {}, clear=True):
            from importlib import reload
            import src.carmina.config as config_module
            reload(config_module)
            assert config_module.DEBUG_ENABLED == False
    
    def test_debug_enabled_true(self):
        """Test DEBUG_ENABLED when set to true."""
        with patch.dict(os.environ, {'DEBUG_ENABLED': 'true'}):
            from importlib import reload
            import src.carmina.config as config_module
            reload(config_module)
            assert config_module.DEBUG_ENABLED == True
    
    def test_log_level_default(self):
        """Test LOG_LEVEL default value."""
        # Test the default value directly from the config logic
        with patch.dict(os.environ, {}, clear=True):
            # Import inline to avoid caching issues
            from importlib import import_module
            import sys
            # Remove from cache if exists
            if 'src.carmina.config' in sys.modules:
                del sys.modules['src.carmina.config']
            config_module = import_module('src.carmina.config')
            assert config_module.LOG_LEVEL == 'INFO'
    
    def test_log_level_custom(self):
        """Test LOG_LEVEL custom value."""
        with patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG'}):
            from importlib import reload
            import src.carmina.config as config_module
            reload(config_module)
            assert config_module.LOG_LEVEL == 'DEBUG'
    
    def test_model_configuration_defaults(self):
        """Test model configuration defaults."""
        with patch.dict(os.environ, {}, clear=True):
            from importlib import reload
            import src.carmina.config as config_module
            reload(config_module)
            assert config_module.DEFAULT_MODEL == 'claude-3.7-sonnet'
            assert config_module.DEFAULT_TEMPERATURE == 0.7
            assert config_module.DEFAULT_MAX_TOKENS == 1000
    
    def test_model_configuration_custom(self):
        """Test model configuration with custom values."""
        with patch.dict(os.environ, {
            'DEFAULT_MODEL': 'gpt-4-turbo',
            'DEFAULT_TEMPERATURE': '0.5',
            'DEFAULT_MAX_TOKENS': '2000'
        }):
            from importlib import reload
            import src.carmina.config as config_module
            reload(config_module)
            assert config_module.DEFAULT_MODEL == 'gpt-4-turbo'
            assert config_module.DEFAULT_TEMPERATURE == 0.5
            assert config_module.DEFAULT_MAX_TOKENS == 2000
    
    def test_directory_configuration(self):
        """Test directory configuration."""
        with patch.dict(os.environ, {
            'INPUT_DIR': '/custom/input',
            'OUTPUT_DIR': '/custom/output',
            'METRICS_DIR': '/custom/metrics',
            'DEBUG_DIR': '/custom/debug'
        }):
            from importlib import reload
            import src.carmina.config as config_module
            reload(config_module)
            assert config_module.INPUT_DIR == '/custom/input'
            assert config_module.OUTPUT_DIR == '/custom/output'
            assert config_module.METRICS_DIR == '/custom/metrics'
            assert config_module.DEBUG_DIR == '/custom/debug'