"""
MCP 警告屏蔽工具

这个模块通过猴子补丁的方式来屏蔽 MCP 服务器产生的验证警告。
在导入 MCP 相关模块之前先导入这个模块。
"""

import logging
import warnings
from typing import Any


def suppress_mcp_warnings():
    """屏蔽 MCP 相关的警告信息"""
    
    # 1. 屏蔽所有 Python warnings
    warnings.filterwarnings("ignore")
    
    # 2. 设置相关日志器的级别
    loggers_to_suppress = [
        'mcp',
        'mcp.client',
        'mcp.client.sse',
        'mcp.client.stdio',
        'mcp.client.streamable_http',
        'mcp.shared',
        'mcp.shared.session',
        'pydantic',
        'httpx',
        'httpcore',
        'qwen_agent',
    ]
    
    for logger_name in loggers_to_suppress:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
        # 同时也屏蔽所有handler
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
    
    # 3. 猴子补丁 logging.warning 方法
    original_warning = logging.warning
    original_logger_warning = logging.Logger.warning
    
    def patched_warning(msg, *args, **kwargs):
        # 检查消息内容，如果包含MCP相关的警告则忽略
        msg_str = str(msg)
        if any(keyword in msg_str.lower() for keyword in [
            'failed to validate notification',
            'validation errors',
            'pydantic',
            'servernotification',
            'cancellednotification',
            'progressnotification',
            'loggingmessagenotification',
            'resourceupdatednotification',
            'resourcelistchangednotification',
            'toollistchangednotification',
            'promptlistchangednotification',
            'literal_error',
            'input_value=\'ping\'',
            'field required'
        ]):
            return
        # 对于其他警告，正常处理
        return original_warning(msg, *args, **kwargs)
    
    def patched_logger_warning(self, msg, *args, **kwargs):
        # 检查logger名称和消息内容
        if hasattr(self, 'name') and self.name and any(name in self.name.lower() for name in [
            'mcp', 'pydantic', 'httpx', 'httpcore', 'qwen_agent'
        ]):
            return
        
        msg_str = str(msg)
        if any(keyword in msg_str.lower() for keyword in [
            'failed to validate notification',
            'validation errors',
            'pydantic',
            'servernotification',
            'cancellednotification',
            'progressnotification',
            'loggingmessagenotification',
            'resourceupdatednotification',
            'resourcelistchangednotification',
            'toollistchangednotification',
            'promptlistchangednotification',
            'literal_error',
            'input_value=\'ping\'',
            'field required'
        ]):
            return
        # 对于其他警告，正常处理
        return original_logger_warning(self, msg, *args, **kwargs)
    
    # 应用补丁
    logging.warning = patched_warning
    logging.Logger.warning = patched_logger_warning
    
    # 4. 尝试导入并补丁 MCP session 模块
    try:
        import mcp.shared.session
        
        # 替换session中的logging.warning调用
        if hasattr(mcp.shared.session, 'logging'):
            original_session_warning = mcp.shared.session.logging.warning
            
            def patched_session_warning(msg, *args, **kwargs):
                msg_str = str(msg)
                if 'failed to validate' in msg_str.lower() or 'validation errors' in msg_str.lower():
                    return
                return original_session_warning(msg, *args, **kwargs)
            
            mcp.shared.session.logging.warning = patched_session_warning
    except ImportError:
        pass
    
    print("✅ MCP 警告屏蔽已启用")


def patch_ray_worker_logging():
    """在Ray worker中也应用日志屏蔽"""
    import os
    os.environ['PYTHONWARNINGS'] = 'ignore'
    suppress_mcp_warnings()


# 自动执行屏蔽
suppress_mcp_warnings()
