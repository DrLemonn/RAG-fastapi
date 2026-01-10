import re
import os
from bs4 import BeautifulSoup



def clean_fastapi_markdown(text, root_dir="."):

    # 用来去重重复的代码文件
    seen_files = set()

    def load_file_content(match):
        """
        解析标签内容，查找对应的代码文件（优先查找 annotated 版本），并返回 markdown 代码块。
        """

        tag_content = match.group(0)

        # 1. 清理标签包裹符号 {* 和 *}
        content_inner = tag_content.replace('{*', '').replace('*}', '').strip()
        
        # 2. 分离文件路径和额外参数 (如 hl[1])
        # 假设以空格分隔，第一部分是路径
        parts = content_inner.split()
        if not parts:
            return ""
        
        raw_path = parts[0] # 例如: ../../docs_src/request_files/tutorial001.py
        
        # --- [关键修改点 Start] ---
        # 既然脚本和 docs_src 同级，我们需要去掉路径前面的 ../ 
        # 策略：找到 "docs_src" 在字符串中的位置，切片截取其后的部分
        if "docs_src" in raw_path:
            start_index = raw_path.find("docs_src")
            # 截取后变成: docs_src/request_files/tutorial001.py
            rel_path = raw_path[start_index:] 
        else:
            # 如果路径里没有 docs_src，尝试简单去除开头的 ./ 或 ../
            # 这种是为了兼容可能引用了非 docs_src 目录的情况
            rel_path = raw_path.lstrip('./').replace('../', '')
        
        # 拼接当前脚本所在的根目录 (root_dir)
        full_path_raw = os.path.join(root_dir, rel_path)
        
        # 获取目录路径、文件名（不含扩展名）、扩展名
        dir_name = os.path.dirname(full_path_raw)
        base_name = os.path.basename(full_path_raw)
        file_name_no_ext, ext = os.path.splitext(base_name)
        
        # 4. 定义查找优先级列表
        # FastAPI 文档通常的优先级：Annotated Py3.10+ > Annotated > Py3.10+ > 原文件
        # 根据你的描述，我们重点关注 _an (annotated) 版本
        candidate_filenames = [
            f"{file_name_no_ext}_an_py39{ext}",  # 优先级 1: Annotated + Python 3.9+
            f"{file_name_no_ext}_py39{ext}",     # 优先级 3: Python 3.9+ (无 Annotated)
            f"{file_name_no_ext}{ext}"           # 优先级 4: 原文件名
        ]

        target_content = None
        found_key = None

        # 5. 遍历查找文件是否存在
        for candidate in candidate_filenames:
            check_path = os.path.join(dir_name, candidate)
            if os.path.exists(check_path):
                found_key = os.path.abspath(check_path) # 获取绝对路径作为 Key
                # 【核心修改 2】检查是否已经引用过
                if found_key in seen_files:
                    # RAG 优化：如果已经引用过，返回一个简短的引用说明
                    # 这样保留了语义（这里有代码），但节省了 Token
                    return f"\n> *[Ref: Code file `{candidate}` is already included above]*\n"
                
                try:
                    with open(check_path, 'r', encoding='utf-8') as f:
                        target_content = f.read()
                    seen_files.add(found_key)
                    print(f"this set{seen_files}")
                    
                    break # 找到高优先级文件后立即停止
                except Exception as e:
                    return f"\n[Error reading file {candidate}: {e}]\n"

        # 6. 返回结果
        if target_content is not None:
            # 去掉开头的 . 用于 markdown 标记 (如 .py -> py)
            lang_tag = ext.lstrip('.')
            # 可选：如果你想在代码块上方注释实际引用的文件，可以取消下面这行的注释
            # return f"\n\n```{lang_tag}\n{target_content}\n```\n"
            return f"\n```{lang_tag}\n{target_content}\n```\n"
        else:
            return f"\n[Warning: Code file not found. Looked for: {', '.join(candidate_filenames)} at {dir_name}]\n"
    # 1. 处理文件引入 {* ... *}
    # 正则匹配 {* ... *} 模式
    pattern_include = r'\{\*.*?\*\}'
    text = re.sub(pattern_include, load_file_content, text)

    # 2. 处理 <div class="termy"> 及其内部的 HTML 标签
    # 这一步比较复杂，因为是混合 markdown。
    # 策略：先找到 termy 块，然后只清理块内的 html 标签
    
    def clean_termy(match):
        content = match.group(1)
        # 去除所有 HTML 标签，只保留文本
        clean_content = re.sub(r'<[^>]+>', '', content)
        # 去除多余的空行或奇怪的转义字符
        return f"```console\n{clean_content.strip()}\n```"

    # 匹配 <div class="termy">...</div> 的大概范围 (非贪婪匹配)
    # 注意：如果 div 嵌套可能会有问题，简单的正则处理这一层通常足够
    text = re.sub(r'<div class="termy">\s*(.*?)\s*</div>', clean_termy, text, flags=re.DOTALL)

    # 3. 处理图片 (保留 alt 文本)
    # 处理 HTML <img> 标签
    text = re.sub(r'<img[^>]*alt="([^"]*)"[^>]*>', r'[Image: \1]', text)
    # 处理 Markdown ![]() 标签
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'[Image: \1]', text)

    # 4. 清理通用 HTML 标签 (但在保留 Markdown 结构的前提下)
    # 移除 <style> 块
    text = re.sub(r'<style>.*?</style>', '', text, flags=re.DOTALL)
    
    # 移除特定的容器标签但保留内容，如 <p align="center">
    # 简单粗暴的方法是利用 BeautifulSoup 提取 text，但那样会破坏 Markdown 的标题格式 (#)
    # 所以建议用正则针对性去除干扰标签
    
    # 去除 <a ...> 和 </a>，只保留链接文字 (或者你可以选择保留 href)
    # text = re.sub(r'<a [^>]*href="([^"]+)"[^>]*>(.*?)</a>', r'\2 (\1)', text) # 保留 url
    text = re.sub(r'<a [^>]*>(.*?)</a>', r'\1', text) # 不保留 url
    
    # 去除其他常见标签 <p>, <div>, <br>, <small>, <abbr> 等，保留内容
    text = re.sub(r'</?(p|div|small|abbr|span|font|u|b|em)[^>]*>', '', text)
    
    # 5. 去除多余的空行
    text = re.sub(r'\n\s*\n', '\n\n', text)

    return text

# 使用示例
# 假设你已经下载了 fastapi 的源码，root_dir 指向源码根目录
# processed_text = clean_fastapi_markdown(original_content, root_dir="/path/to/fastapi/repo")
# print(processed_text)

if __name__ == "__main__":
    # --- 配置区域 ---
    
    # 1. 设置根目录 (root_dir)
    # 假设你的目录结构是这样的：
    # /project_root
    #   ├── your_script.py (当前脚本)
    #   ├── docs_src/      (存放代码文件的目录)
    #   └── docs/          (存放 markdown 文档的目录)
    
    # 获取当前脚本所在的目录作为根目录
    current_root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 指定一个要测试的 Markdown 文件路径
    # 请修改为你实际存在的 md 文件路径，例如:
    # input_md_file = "docs/en/docs/tutorial/first-steps.md" 
    # 如果你手头暂时没有 md 文件，可以使用下面的 "模拟测试模式"
    
    input_md_file = "test_doc.md" # 假设你要读取的文件名

    # --- 模拟数据测试 (如果你还没准备好 md 文件，用这个) ---
    # 为了演示，我们先创建一个包含引入标签的模拟字符串
    # 假设 docs_src 下有一个 main.py

# ... (上面是你提供的 import 和 函数定义) ...

if __name__ == "__main__":
    # --- 配置区域 ---
    
    # 1. 设置根目录 (root_dir)
    # 假设你的目录结构是这样的：
    # /project_root
    #   ├── your_script.py (当前脚本)
    #   ├── docs_src/      (存放代码文件的目录)
    #   └── docs/          (存放 markdown 文档的目录)
    
    # 获取当前脚本所在的目录作为根目录
    current_root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 指定一个要测试的 Markdown 文件路径
    # 请修改为你实际存在的 md 文件路径，例如:
    # input_md_file = "docs/en/docs/tutorial/first-steps.md" 
    # 如果你手头暂时没有 md 文件，可以使用下面的 "模拟测试模式"
    
    input_md_file = "test_doc.md" # 假设你要读取的文件名

    # --- 模拟数据测试 (如果你还没准备好 md 文件，用这个) ---
    # 为了演示，我们先创建一个包含引入标签的模拟字符串
    # 假设 docs_src 下有一个 main.py
    fake_md_content = """
        # Hello FastAPI

        Here is a code example:

        {* ../../docs_src/main.py *}

        <div class="termy">
        ```console
        $ uvicorn main:app --reload
        </div>
        """
        
    # --- 实际文件读取逻辑 ---
    target_content = ""

    if os.path.exists(input_md_file):
        print(f"正在读取文件: {input_md_file} ...")
        with open(input_md_file, "r", encoding="utf-8") as f:
            target_content = f.read()
    else:
        print(f"⚠️ 未找到文件 '{input_md_file}'，使用模拟字符串进行测试...")
        target_content = fake_md_content

    # 3. 执行核心转换函数
    # 注意：root_dir 必须能让 os.path.join(root_dir, 'docs_src/...') 找到正确路径
    processed_result = clean_fastapi_markdown(target_content, root_dir=current_root_dir)

    # 4. 输出结果
    print("\n" + "="*20 + " 处理结果预览 " + "="*20)
    print(processed_result[:1000]) # 打印前1000个字符预览
    print("="*50)

    # 5. 保存到新文件以便查看
    output_filename = "output_result.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(processed_result)

    print(f"\n✅ 处理完成！完整结果已保存至: {os.path.join(current_root_dir, output_filename)}")