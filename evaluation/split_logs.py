import os
import argparse

def split_log_into_chunks(log_file_path, output_dir=None, start_pattern='actor/kl_loss:'):
    """
    根据起始模式将一个大日志文件分割成多个小文件。

    :param log_file_path: 输入的日志文件路径。
    :param start_pattern: 标志着一个新 chunk 开始的字符串。
    :param output_dir: 存放切分后文件的目录。
    """
    # 如果未指定输出目录，默认使用日志文件去除后缀的版本作为目录
    if output_dir is None:
        output_dir = os.path.splitext(log_file_path)[0]
    
    chunks_dir = os.path.join(output_dir, "chunks")
    
    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir)
        print(f"创建目录: {chunks_dir}")

    chunk_count = 0
    current_chunk_file = None

    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if start_pattern in line.strip():
                    # 如果已经有一个文件在写入，先关闭它
                    if current_chunk_file:
                        current_chunk_file.close()
                    
                    # 准备开始写入下一个新文件
                    chunk_count += 1
                    # 使用日志文件名作为前缀，避免混淆
                    log_basename = os.path.splitext(os.path.basename(log_file_path))[0]
                    chunk_filename = os.path.join(chunks_dir, f'{log_basename}_chunk_{chunk_count}.log')
                    current_chunk_file = open(chunk_filename, 'w', encoding='utf-8')
                    print(f"检测到新 chunk，开始写入文件: {chunk_filename}")
                
                # 如果已经开始了一个 chunk，就把当前行写入
                if current_chunk_file:
                    current_chunk_file.write(line)
        # 循环结束后，确保最后一个文件被关闭
        if current_chunk_file:
            current_chunk_file.close()

        print(f"\n处理完成！总共分割出 {chunk_count} 个文件。")
        print(f"输出目录: {chunks_dir}")

    except FileNotFoundError:
        print(f"错误: 文件 '{log_file_path}' 未找到。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split log file into chunks based on a pattern.")
    parser.add_argument("--log_path", type=str, 
                        default="logs/train.log",
                        help="Path to the input log file.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the chunks. Defaults to the log file's directory.")
    parser.add_argument("--pattern", type=str, default="actor/kl_loss:",
                        help="The string pattern that signals the start of a new chunk.")

    args = parser.parse_args()

    split_log_into_chunks(args.log_path, args.output_dir, args.pattern)
