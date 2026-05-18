import os
from pathlib import Path
import gradio as gr
from gradio.events import Dependency

class FlowEditor(gr.HTML):
    """基于Drawflow的流程图编辑器组件"""
    
    def __init__(
        self,
        value=None,
        *,
        label=None,
        every=None,
        show_label=True,
        visible=True,
        elem_id=None,
        **kwargs,
    ):
        """初始化流程图编辑器"""
        html_content = """
        <div class="drawflow-container">
            <div id="drawflow" style="width: 100%; height: 600px; background: #f6f6f6;"></div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/drawflow@0.0.59/dist/drawflow.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/drawflow@0.0.59/dist/drawflow.min.css" rel="stylesheet">
        <style>
            .drawflow-node {
                background: white;
                border: 1px solid #ccc;
                padding: 10px;
                border-radius: 4px;
                min-width: 150px;
            }
            .drawflow-node.rule {
                border-color: #2196f3;
            }
            .drawflow-node.rule .title {
                color: #2196f3;
                font-weight: bold;
            }
            .drawflow-node.logic {
                border-color: #4caf50;
            }
            .drawflow-node.logic .title {
                color: #4caf50;
                font-weight: bold;
            }
            .drawflow .connection {
                stroke: #999;
            }
            .drawflow .connection.selected {
                stroke: #4caf50;
            }
        </style>
        <script>
            // 等待DOM加载完成
            window.addEventListener('load', function() {
                // 初始化Drawflow
                var container = document.getElementById("drawflow");
                var editor = new Drawflow(container);
                editor.start();
                
                // 注册节点类型
                editor.registerNode('rule', {
                    html: `
                        <div class="rule-node">
                            <div class="title"></div>
                            <div class="type"></div>
                            <div class="form"></div>
                        </div>
                    `,
                    props: {
                        description: { type: 'string' },
                        label: { type: 'string' },
                        form: { type: 'string' }
                    }
                });
                
                editor.registerNode('logic', {
                    html: `
                        <div class="logic-node">
                            <div class="title">逻辑节点</div>
                            <div class="expression" onclick="editLogicExpression(this)"></div>
                        </div>
                    `,
                    props: {
                        expression: { type: 'string' }
                    }
                });
                
                // 监听连接事件
                editor.on('connectionCreated', function(connection) {
                    var fieldExtractor = prompt('请输入字段提取表达式：');
                    if (fieldExtractor) {
                        connection.data = { fieldExtractor: fieldExtractor };
                    }
                });
                
                // 导出数据到Gradio
                function exportToGradio() {
                    var data = editor.export();
                    // 触发gradio的自定义事件
                    document.dispatchEvent(new CustomEvent('gradio-flow-update', {
                        detail: { data: data }
                    }));
                }
                
                // 监听节点和连接的变化
                editor.on('nodeCreated', exportToGradio);
                editor.on('nodeRemoved', exportToGradio);
                editor.on('connectionCreated', exportToGradio);
                editor.on('connectionRemoved', exportToGradio);
                
                // 暴露editor实例到全局
                window.flowEditor = editor;
            });
            
            // 添加逻辑节点的函数
            function addLogicNode() {
                if (window.flowEditor) {
                    window.flowEditor.addNode(
                        'logic',
                        1,  // 输入连接数
                        1,  // 输出连接数
                        300,  // x坐标
                        200,  // y坐标
                        'logic',  // 类名
                        { expression: '' },  // 数据
                        `logic_${Date.now()}`  // id
                    );
                }
            }
            
            // 重置画布的函数
            function resetFlow() {
                if (window.flowEditor) {
                    window.flowEditor.clear();
                }
            }

            // 添加逻辑表达式编辑功能
            window.editLogicExpression = function(element) {
                var nodeId = element.closest('.drawflow-node').id;
                var node = editor.getNodeFromId(nodeId.replace('node-', ''));
                var currentExpression = node.data.expression || '';
                
                // 获取输入节点的ID列表
                var inputNodes = [];
                for (var i = 0; i < node.inputs.input_1.connections.length; i++) {
                    var conn = node.inputs.input_1.connections[i];
                    var sourceNode = editor.getNodeFromId(conn.node);
                    inputNodes.push({
                        id: sourceNode.id,
                        description: sourceNode.data.description || '逻辑节点'
                    });
                }
                
                // 构建表达式编辑对话框
                var dialog = document.createElement('div');
                dialog.className = 'logic-expression-dialog';
                dialog.style.cssText = `
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    z-index: 1000;
                `;
                
                dialog.innerHTML = `
                    <h3>编辑逻辑表达式</h3>
                    <p>可用节点：</p>
                    <ul>
                        ${inputNodes.map(n => `<li>${n.description} (${n.id})</li>`).join('')}
                    </ul>
                    <p>支持的运算符：AND, OR, NOT</p>
                    <textarea style="width: 100%; height: 100px; margin: 10px 0;">${currentExpression}</textarea>
                    <div style="text-align: right; margin-top: 10px;">
                        <button onclick="this.parentElement.parentElement.remove()">取消</button>
                        <button onclick="saveLogicExpression('${nodeId}', this.parentElement.previousElementSibling.value)">保存</button>
                    </div>
                `;
                
                document.body.appendChild(dialog);
            };
            
            window.saveLogicExpression = function(nodeId, expression) {
                var node = editor.getNodeFromId(nodeId.replace('node-', ''));
                node.data.expression = expression;
                
                // 更新显示
                var expressionElement = document.querySelector(`#${nodeId} .expression`);
                if (expressionElement) {
                    expressionElement.textContent = expression || '请设置逻辑表达式';
                }
                
                // 关闭对话框
                var dialog = document.querySelector('.logic-expression-dialog');
                if (dialog) {
                    dialog.remove();
                }
                
                // 触发更新事件
                exportToGradio();
            };
        </script>
        """
        super().__init__(
            value=html_content,
            label=label,
            every=every,
            show_label=show_label,
            visible=visible,
            elem_id=elem_id,
            **kwargs,
        )
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer
        from gradio.components.base import Component 