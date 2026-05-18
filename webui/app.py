import gradio as gr
from tabs import (
    create_data_processing_tab,
    create_tool_definition_tab,
    create_reward_definition_tab,
    create_training_deployment_tab,
    create_project_management_tab
)

def create_app():
    """创建主应用
    
    整合所有标签页模块，创建完整的 WebUI 应用。
    每个标签页都是独立的模块，便于维护和扩展。
    """
    with gr.Blocks(title="SEARL WebUI") as app:
        gr.Markdown("# SEARL WebUI")
        gr.Markdown("""
        欢迎使用 SEARL WebUI，这是一个用于强化学习实验管理的工具。
        """)
        with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 当前项目")
                    current_project_name = gr.Markdown("**项目名称**: 未选择", elem_id="current_project_name")
                    current_project_path = gr.Markdown("**项目路径**: -", elem_id="current_project_path")
                    current_project_desc = gr.Markdown("**项目描述**: -", elem_id="current_project_desc")
        
        with gr.Tabs() as tabs:
            with gr.TabItem("项目管理"):
                project_tab, get_project_info, project_dropdown = create_project_management_tab()
                project_dropdown.change(
                    fn=lambda x: get_project_info(x, is_global=True),
                    inputs=[project_dropdown],
                    outputs=[current_project_name, current_project_path, current_project_desc])
                
            with gr.TabItem("奖赏定义"):
                create_reward_definition_tab()
            with gr.TabItem("工具定义"):
                create_tool_definition_tab()
            with gr.TabItem("数据处理"):
                create_data_processing_tab(current_project_path)
            with gr.TabItem("训练部署"):
                create_training_deployment_tab()
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    ) 
