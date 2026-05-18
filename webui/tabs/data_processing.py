"""
数据处理模块
"""

import re
import os
import json
import gradio as gr
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

DEFAULT_SYSTEM_PREFIX = """You are a helpful AI assistant. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."""

DEFAULT_USER_PREFIX = """Question: {question}\n"""

def get_project_path(current_project_path):
    """从显示文本中提取项目路径，并返回数据目录路径
    
    Args:
        current_project_path: 当前项目路径显示组件
    
    Returns:
        str: 项目数据目录路径，如果未选择项目则返回空字符串
    """
    path_text = current_project_path

    if path_text == "**项目路径**: -":
        return ""
    
    # 获取项目根路径
    project_path = path_text.replace("**项目路径**: ", "")
    
    # 返回数据目录路径
    data_path = os.path.join(project_path, "data")
    
    # 确保数据目录存在
    os.makedirs(data_path, exist_ok=True)
    #print(f"Debug - Data path: {data_path}")  # 添加打印调试
    
    return data_path

def make_prefix(question: str, template_type: str, custom_system_prefix: str = None, custom_user_prefix: str = None) -> str:
    if template_type == 'base':
        system_prefix = custom_system_prefix if custom_system_prefix else DEFAULT_SYSTEM_PREFIX
        user_prefix = custom_user_prefix if custom_user_prefix else DEFAULT_USER_PREFIX
        return f"{system_prefix}\n\n{user_prefix.format(question=question)}"
    elif template_type == 'multiturn':
        system_prefix = custom_system_prefix if custom_system_prefix else DEFAULT_SYSTEM_PREFIX
        user_prefix = custom_user_prefix if custom_user_prefix else DEFAULT_USER_PREFIX
        return f"{system_prefix}\n\n{user_prefix.format(question=question)}"
    else:
        raise NotImplementedError(f"Template type {template_type} not implemented")

def process_data(raw_data: str, template_type: str, split: str, custom_system_prefix: str = None, custom_user_prefix: str = None) -> tuple[List[Dict[str, Any]], str]:
    # Parse raw data (assuming it's in a format where each line is a JSON object)
    data_list = []
    status_messages = []
    total_lines = len(raw_data.strip().split('\n'))
    processed_lines = 0
    error_lines = 0
    
    for idx, line in enumerate(raw_data.strip().split('\n')):
        try:
            if not line.strip():  # Skip empty lines
                continue
            example = json.loads(line)  # Parse JSON instead of using eval
            question = example['question'].strip()
            if question[-1] != '?':
                question += '?'
            
            prefix = make_prefix(question, template_type, custom_system_prefix, custom_user_prefix)
            
            data_item = {
                "data_source": "nq",
                "prompt": [{
                    "role": "user",
                    "content": prefix,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "target": example['golden_answers'],
                    }
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            data_list.append(data_item)
            processed_lines += 1
        except json.JSONDecodeError as e:
            error_lines += 1
            status_messages.append(f"Error in line {idx + 1}: Invalid JSON format - {str(e)}")
        except KeyError as e:
            error_lines += 1
            status_messages.append(f"Error in line {idx + 1}: Missing required field {str(e)}")
        except Exception as e:
            error_lines += 1
            status_messages.append(f"Error in line {idx + 1}: {str(e)}")
    
    status_summary = [
        f"处理完成:",
        f"- 总行数: {total_lines}",
        f"- 成功处理: {processed_lines}",
        f"- 错误行数: {error_lines}",
        f"- 成功率: {(processed_lines/total_lines)*100:.1f}%"
    ]
    
    if status_messages:
        status_summary.append("\n详细错误信息:")
        status_summary.extend(status_messages)
    
    return data_list, "\n".join(status_summary)

def save_to_parquet(data_list: List[Dict[str, Any]], output_path: str, current_project_path: gr.Markdown, split: str = "train", raw_data_path: str = None) -> str:
    """保存数据到parquet文件，并生成元数据JSON文件
    
    Args:
        data_list: 要保存的数据列表
        output_path: 用户指定的输出路径（可选）
        current_project_path: 当前项目路径组件
        split: 数据集分割类型（train/test）
        raw_data_path: 原始数据文件路径
    
    Returns:
        str: 保存状态信息
    """
    try:
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path:
            print(output_path)
            df = pd.DataFrame(data_list)
            df.to_parquet(output_path)
            parquet_path = output_path
        else:
            # 获取项目数据目录
            project_data_path = get_project_path(current_project_path)
            if not project_data_path:
                return "错误：未选择项目"
            
            # 创建processed目录
            processed_dir = os.path.join(project_data_path, "processed")
            os.makedirs(processed_dir, exist_ok=True)
            
            # 生成文件名：split_YYYYMMDD_HHMMSS.parquet
            filename = f"{split}_{timestamp}.parquet"
            parquet_path = os.path.join(processed_dir, filename)
            
            print(f"保存到: {parquet_path}")
            df = pd.DataFrame(data_list)
            df.to_parquet(parquet_path)

        # 获取原始数据的字段信息
        raw_fields = []
        if raw_data_path and os.path.exists(raw_data_path):
            raw_fields = get_available_keys(raw_data_path)

        # 生成元数据JSON文件
        metadata = {
            "raw_data_path": raw_data_path,
            "processed_data_path": parquet_path,
            "data_description": {
                "total_samples": len(data_list),
                "split": split,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "raw_data_fields": raw_fields,
                "processed_data_fields": {
                    "prompt": "用户输入的问题",
                    "ability": "任务类型",
                    "reward_model": {
                        "style": "奖励模型类型",
                        "ground_truth": "标准答案"
                    },
                    "extra_info": "额外信息，包含split和index"
                }
            }
        }

        # 保存元数据JSON文件到data目录
        project_data_path = get_project_path(current_project_path)
        
        # 从parquet路径中提取文件名（不包含processed目录）
        filename = os.path.basename(parquet_path)
        base_name = os.path.splitext(filename)[0]  # 移除.parquet后缀
        metadata_path = os.path.join(project_data_path, f"{base_name}_metadata.json")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return f"成功保存 {len(data_list)} 条记录到 {parquet_path}\n元数据文件已保存到: {metadata_path}"
    except Exception as e:
        return f"保存文件时出错: {str(e)}"

def process_data_from_file(file_path: str, template_type: str, split: str, custom_system_prefix: str = None, custom_user_prefix: str = None, ground_truth_keys: List[str] = None, question_key: str = 'question') -> tuple[List[Dict[str, Any]], str]:
    """从文件流式处理数据
    
    Args:
        file_path: 输入文件路径
        template_type: 模板类型
        split: 数据集分割
        custom_system_prefix: 自定义系统前缀
        custom_user_prefix: 自定义用户前缀
        ground_truth_keys: 作为ground_truth的字段列表
        question_key: 作为question的字段名
    
    Returns:
        tuple: (处理后的数据列表, 状态信息)
    """
    data_list = []
    status_messages = []
    processed_lines = 0
    error_lines = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    if not line.strip():  # Skip empty lines
                        continue
                    example = json.loads(line)
                    
                    # 使用选定的字段作为question
                    if question_key not in example:
                        raise KeyError(f"Question field '{question_key}' not found")
                    question = example[question_key].strip()
                    if question[-1] != '?':
                        question += '?'
                    
                    prefix = make_prefix(question, template_type, custom_system_prefix, custom_user_prefix)
                    
                    # 获取ground_truth值
                    ground_truth_values = []
                    if ground_truth_keys:
                        for key in ground_truth_keys:
                            if key in example:
                                ground_truth_values.append(example[key])
                            else:
                                status_messages.append(f"Warning in line {idx + 1}: Key '{key}' not found")
                    else:
                        # 如果没有指定keys，使用默认的golden_answers
                        ground_truth_values = example.get('golden_answers', [])
                    
                    data_item = {
                        "data_source": "nq",
                        "prompt": [{
                            "role": "user",
                            "content": prefix,
                        }],
                        "ability": "fact-reasoning",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": {
                                "target": ground_truth_values,
                            }
                        },
                        "extra_info": {
                            'split': split,
                            'index': idx,
                        }
                    }
                    data_list.append(data_item)
                    processed_lines += 1
                    
                    # 每处理1000行打印一次进度
                    if processed_lines % 1000 == 0:
                        print(f"已处理 {processed_lines} 行数据")
                        
                except json.JSONDecodeError as e:
                    error_lines += 1
                    status_messages.append(f"Error in line {idx + 1}: Invalid JSON format - {str(e)}")
                except KeyError as e:
                    error_lines += 1
                    status_messages.append(f"Error in line {idx + 1}: Missing required field {str(e)}")
                except Exception as e:
                    error_lines += 1
                    status_messages.append(f"Error in line {idx + 1}: {str(e)}")
    
    except Exception as e:
        return [], f"读取文件时出错: {str(e)}"
    
    status_summary = [
        f"处理完成:",
        f"- 总行数: {processed_lines + error_lines}",
        f"- 成功处理: {processed_lines}",
        f"- 错误行数: {error_lines}",
        f"- 成功率: {(processed_lines/(processed_lines + error_lines))*100:.1f}%"
    ]
    
    if status_messages:
        status_summary.append("\n详细错误信息:")
        status_summary.extend(status_messages)
    
    return data_list, "\n".join(status_summary)

def read_parquet_file(file_path: str) -> pd.DataFrame:
    """读取parquet文件并返回DataFrame
    
    Args:
        file_path: parquet文件路径
    
    Returns:
        pd.DataFrame: 读取的数据
    """
    try:
        if not file_path:
            return pd.DataFrame({"Error": ["请输入文件路径"]})
        if not os.path.exists(file_path):
            return pd.DataFrame({"Error": [f"文件不存在: {file_path}"]})
            
        # 使用保守的参数读取parquet文件
        df = pd.read_parquet(
            file_path,
            engine='pyarrow',
            use_threads=False,
            memory_map=True,
            coerce_int96_timestamp_unit='ms'
        )
        
        # 只返回前5行数据
        return df.head(5)
    except Exception as e:
        return pd.DataFrame({"Error": [f"读取文件时出错: {str(e)}"]})

def get_available_keys(file_path: str) -> List[str]:
    """从输入文件中获取可用的字段名
    
    Args:
        file_path: 输入文件路径
    
    Returns:
        List[str]: 可用的字段名列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取前几行来获取字段名
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    return list(data.keys())
    except Exception as e:
        print(f"获取字段名时出错: {str(e)}")
    return []

def get_dataset_metadata(project_path: str) -> List[Dict[str, Any]]:
    """获取项目下所有数据集的元数据信息
    
    Args:
        project_path: 项目路径
    
    Returns:
        List[Dict[str, Any]]: 元数据信息列表
    """
    metadata_list = []
    if not project_path or project_path == "**项目路径**: -":
        return metadata_list
        
    # 获取项目数据目录
    data_dir = os.path.join(project_path.replace("**项目路径**: ", ""), "data")
    if not os.path.exists(data_dir):
        return metadata_list
        
    # 遍历目录查找所有metadata文件
    for file in os.listdir(data_dir):
        if file.endswith("_metadata.json"):
            try:
                with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    metadata_list.append(metadata)
            except Exception as e:
                print(f"读取元数据文件 {file} 时出错: {str(e)}")
                
    return metadata_list

def format_metadata_display(metadata: Dict[str, Any]) -> str:
    """格式化元数据信息用于显示
    
    Args:
        metadata: 元数据字典
    
    Returns:
        str: 格式化后的显示文本
    """
    if not metadata:
        return "未选择数据集"
        
    display = []
    display.append("## 数据集信息")
    display.append(f"- 原始数据路径: {metadata['raw_data_path']}")
    display.append(f"- 处理后数据路径: {metadata['processed_data_path']}")
    
    desc = metadata['data_description']
    display.append("\n## 数据描述")
    display.append(f"- 样本数量: {desc['total_samples']}")
    display.append(f"- 数据集类型: {desc['split']}")
    display.append(f"- 处理时间: {desc['timestamp']}")
    
    display.append("\n## 原始数据字段")
    for field in desc['raw_data_fields']:
        display.append(f"- {field}")
    
    display.append("\n## 处理后数据字段")
    for field, desc in desc['processed_data_fields'].items():
        if isinstance(desc, dict):
            display.append(f"- {field}:")
            for subfield, subdesc in desc.items():
                display.append(f"  - {subfield}: {subdesc}")
        else:
            display.append(f"- {field}: {desc}")
            
    return "\n".join(display)

def create_data_processing_tab(current_project_path):
    """数据处理标签页
    
    该标签页用于管理和处理实验数据，包括：
    - 数据导入导出
    - 数据预处理
    - 数据可视化
    - 数据集管理
    """
    
    with gr.Blocks() as tab:
        gr.Markdown("# 数据处理")
        gr.Markdown("""
        ## 功能说明
        在此标签页中，您可以：
        - 导入和导出实验数据
        - 进行数据预处理和转换
        - 查看数据可视化结果
        - 管理实验数据集
        """)
        
        with gr.Tabs() as tabs:
            with gr.TabItem("数据预处理"):
                gr.Markdown("# 数据处理")
                gr.Markdown("""
                ## 使用说明
                1. 您可以直接输入JSON文件的路径
                2. 每行应该是一个JSON对象，包含问题字段和答案字段
                3. 示例格式:
                ```json
                {"query": "What is the capital of France?", "golden_answers": ["Paris"]}
                {"query": "Who wrote Romeo and Juliet?", "golden_answers": ["William Shakespeare"]}
                ```
                4. 您可以通过勾选"使用自定义前缀"并输入模板来自定义前缀
                5. 在模板中使用 {question} 作为问题的占位符
                6. 您可以从可用字段中选择一个作为问题字段，以及一个或多个作为ground_truth的值
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 输入数据")
                            file_path = gr.Textbox(
                                label="输入文件路径",
                                placeholder="请输入JSON文件的完整路径",
                            )
                            load_keys_btn = gr.Button("加载字段", variant="secondary")
                        
                        with gr.Group():
                            gr.Markdown("### 处理设置")
                            
                            template_type = gr.Dropdown(
                                choices=["base", "multiturn"],
                                value="base",
                                label="模板类型"
                            )
                            question_key = gr.Dropdown(
                                label="问题字段",
                                choices=[],
                                interactive=True,
                                visible=False
                            )
                            available_keys = gr.Dropdown(
                                label="Ground Truth字段",
                                choices=[],
                                multiselect=True,
                                interactive=True,
                                visible=False
                            )
                            use_custom_system_prefix = gr.Checkbox(
                                label="使用自定义系统前缀",
                                value=False
                            )
                            use_custom_user_prefix = gr.Checkbox(
                                label="使用自定义用户前缀",
                                value=False
                            )
                            custom_system_prefix = gr.Textbox(
                                label="自定义系统前缀模板",
                                placeholder="在此输入自定义系统前缀模板。",
                                lines=5,
                                value=DEFAULT_SYSTEM_PREFIX,
                                visible=False
                            )
                            custom_user_prefix = gr.Textbox(
                                label="自定义用户前缀模板",
                                placeholder="在此输入自定义用户前缀模板。使用 {question} 作为问题的占位符。",
                                lines=5,
                                value=DEFAULT_USER_PREFIX,
                                visible=False
                            )
                            split = gr.Dropdown(
                                choices=["train", "test"],
                                value="train",
                                label="数据集分割"
                            )
                            
                            output_path = gr.Textbox(
                                label="输出路径",
                                placeholder="输入parquet文件的保存路径，默认保存到项目目录下的data目录中",
                            )
                            process_btn = gr.Button("处理并保存", variant="primary")
                    
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 处理状态")
                            output = gr.Textbox(
                                label="处理结果",
                                lines=20,
                                show_copy_button=True
                            )
            
            with gr.TabItem("数据可视化"):
                gr.Markdown("""
                ## 数据可视化功能
                在此标签页中，您可以：
                - 读取并预览Parquet格式的数据文件
                - 查看数据的表格形式展示
                - 直观地了解数据结构和内容
                
                ### 使用说明
                1. 输入Parquet文件的完整路径
                2. 点击"预览数据"按钮查看数据内容
                3. 在数据预览区域可以查看表格数据
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 读取文件")
                            parquet_file_path = gr.Textbox(
                                label="Parquet文件路径",
                                placeholder="请输入Parquet文件的完整路径",
                            )
                            preview_btn = gr.Button("预览数据", variant="primary")
                    
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 数据预览")
                            preview_output = gr.Dataframe(
                                label="Parquet文件内容",
                                wrap=True,
                            )
                            preview_status = gr.Textbox(
                                label="预览状态",
                                interactive=False,
                                lines=2
                            )
            
            with gr.TabItem("数据集管理"):
                gr.Markdown("""
                ## 数据集管理
                在此标签页中，您可以：
                - 查看所有已处理的数据集
                - 查看数据集的详细信息
                - 管理数据集
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 数据集列表")
                            refresh_btn = gr.Button("刷新数据集列表", variant="primary")
                            dataset_dropdown = gr.Dropdown(
                                label="选择数据集",
                                choices=[],
                                interactive=True
                            )
                    
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 数据集详情")
                            metadata_display = gr.Markdown()
                
                def refresh_dataset_list(project_path):
                    metadata_list = get_dataset_metadata(project_path)
                    choices = []
                    for metadata in metadata_list:
                        # 从processed_data_path中提取文件名作为显示名称
                        filename = os.path.basename(metadata['processed_data_path'])
                        choices.append(filename)
                    return gr.update(choices=choices, value=None)
                
                def display_metadata(project_path, selected_dataset):
                    if not selected_dataset:
                        return "请选择数据集"
                    
                    metadata_list = get_dataset_metadata(project_path)
                    for metadata in metadata_list:
                        if os.path.basename(metadata['processed_data_path']) == selected_dataset:
                            return format_metadata_display(metadata)
                    return "未找到选中的数据集信息"
                
                # 绑定事件处理
                refresh_btn.click(
                    fn=refresh_dataset_list,
                    inputs=[current_project_path],
                    outputs=[dataset_dropdown]
                )
                
                dataset_dropdown.change(
                    fn=display_metadata,
                    inputs=[current_project_path, dataset_dropdown],
                    outputs=[metadata_display]
                )
                
                # 初始加载数据集列表
                refresh_btn.click()
        
        def load_available_keys(file_path: str):
            if not file_path:
                return gr.update(choices=[], visible=False), gr.update(choices=[], visible=False), "请输入文件路径"
            if not os.path.exists(file_path):
                return gr.update(choices=[], visible=False), gr.update(choices=[], visible=False), f"文件不存在: {file_path}"
            
            keys = get_available_keys(file_path)
            if not keys:
                return gr.update(choices=[], visible=False), gr.update(choices=[], visible=False), "无法从文件中获取字段名"
            
            return gr.update(choices=keys, visible=True), gr.update(choices=keys, visible=True), f"成功加载字段: {', '.join(keys)}"
        
        def process_and_save_from_path(file_path: str, template_type: str, split: str, output_path: str, 
                                     use_custom_system_prefix: bool, use_custom_user_prefix: bool,
                                     custom_system_prefix: str, custom_user_prefix: str,
                                     current_project_path: gr.Markdown, selected_keys: List[str], question_key: str):
            system_prefix = custom_system_prefix if use_custom_system_prefix else None
            user_prefix = custom_user_prefix if use_custom_user_prefix else None
            processed_data, status = process_data_from_file(file_path, template_type, split, system_prefix, user_prefix, selected_keys, question_key)
            save_status = save_to_parquet(processed_data, output_path, current_project_path, split, raw_data_path=file_path)
            return f"{status}\n\n{save_status}"
        
        def preview_parquet_file(file_path: str):
            try:
                if not file_path:
                    return pd.DataFrame(), "请输入文件路径"
                if not os.path.exists(file_path):
                    return pd.DataFrame(), f"文件不存在: {file_path}"
                df = read_parquet_file(file_path)
                return df, f"成功读取文件: {file_path}\n显示行数: {len(df)}"
            except Exception as e:
                return pd.DataFrame(), f"读取文件时出错: {str(e)}"
        
        # Handle loading available keys
        load_keys_btn.click(
            fn=load_available_keys,
            inputs=[file_path],
            outputs=[question_key, available_keys, output]
        )
        
        # Handle file processing
        process_btn.click(
            fn=process_and_save_from_path,
            inputs=[file_path, template_type, split, output_path, 
                   use_custom_system_prefix, use_custom_user_prefix,
                   custom_system_prefix, custom_user_prefix,
                   current_project_path, available_keys, question_key],
            outputs=output
        )
        
        # Handle parquet preview
        preview_btn.click(
            fn=preview_parquet_file,
            inputs=[parquet_file_path],
            outputs=[preview_output, preview_status]
        )
        
        # Handle custom prefix visibility
        use_custom_system_prefix.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_custom_system_prefix],
            outputs=[custom_system_prefix]
        )
        
        use_custom_user_prefix.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_custom_user_prefix],
            outputs=[custom_user_prefix]
        )
    
    return tab 