import re
import os
from bs4 import BeautifulSoup

def clean_fastapi_markdown(text, root_dir="."):
    """
    混合清洗策略：
    1. Regex: 处理文件引入 (宏替换)
    2. Regex: 处理特殊 Termy 格式 (HTML -> Markdown 代码块)
    3. BeautifulSoup: 结构化清理 HTML (移除标签、提取图片/链接文字)
    4. Regex: 文本降噪 (去除多余符号、空行)
    """

    # --- 2. 处理 <div class="termy"> (保持原有逻辑) ---
    # 必须在 BS4 清理之前做，因为它要把 HTML 里的内容提取为 console 代码块
    def clean_termy(match):
        content = match.group(1)
        clean_content = re.sub(r'<[^>]+>', '', content)
        return f"{clean_content.strip()}"

    text = re.sub(r'<div class="termy">\s*(.*?)\s*</div>', clean_termy, text, flags=re.DOTALL)

    # --- 3. [关键修改] 使用 BeautifulSoup 进行结构化清理 ---
    # 这里开始处理图片、链接和其他 HTML 标签
    soup = BeautifulSoup(text, "html.parser")

    # 3.1 暴力删除无用的标签 (包括内容一起删掉)
    for tag in soup.find_all(['script', 'style', 'svg']):
        tag.decompose()

    # 3.2 处理图片 (img)
    # 逻辑：如果有 alt，保留为 [Image: alt]；否则直接删除
    for img in soup.find_all('img'):
        alt_text = img.get('alt', '').strip()
        if alt_text:
            # 替换为文本占位符
            img.replace_with(f" [Image: {alt_text}] ")
        else:
            # 无 alt 属性，视为装饰性图片，直接删除
            img.decompose()

    # 3.3 处理超链接 (a)
    # 逻辑：
    # A. 先获取链接内的文本 (注意：如果在上一步 img 被替换成了文本，这里也能获取到)
    # B. 如果文本包含 "[Image:" 或者 链接本身包含 "badge" 字样，视为装饰性链接，整行删除
    # C. 否则，剥离 <a> 标签，只保留里面的文字
    for a in soup.find_all('a'):
        # 获取标签内的文本（包含我们刚才替换进去的 [Image: ...]）
        inner_text = a.get_text(strip=True)
        
        # 这里的判断可以根据需求调整
        # 如果链接里全是图片占位符，或者是徽章链接，直接删除
        is_badge_link = "badge" in str(a) or "shield" in str(a)
        is_image_wrapper = "[Image:" in inner_text
        
        if is_badge_link or is_image_wrapper:
            a.decompose() # 删除整个 <a> 节点
        else:
            a.replace_with(inner_text) # 只保留文字，去掉链接外壳

    # ================= [新增步骤] 处理段落和格式标签 =================
    
    # 定义需要“去皮留肉”的标签列表
    # p: 段落, div: 块, span: 行内, em/i: 斜体, strong/b: 加粗, center: 居中
    formatting_tags = ['p', 'div', 'span', 'em', 'i', 'strong', 'b', 'u', 'center', 'font', 'small', 'attr']

    for tag_name in formatting_tags:
        for tag in soup.find_all(tag_name):
            # 1. 检查是否是“空心”标签 (只有空格或换行)
            # 你的例子里有 <p>  </p>，这种应该直接删除，不要留空行
            if not tag.get_text(strip=True):
                tag.decompose()
            else:
                # 2. 如果有内容，则剥掉标签，保留内容 (Unwrap)
                # 例如: <p align="center">Text</p>  ==>  Text
                tag.unwrap()

    

    # 将处理后的 DOM 树转回字符串
    text = str(soup)

    # --- 4. 正则扫尾 (Text Noise Cleaning) ---
    
    # 4.1 去除 Markdown 分隔符 --- (对于 RAG 来说通常是噪音)
    text = re.sub(r'-{3,}', '', text)
    
    # 4.2 清理残留的 HTML 实体 (可选，BS4 转换回 str 时可能会产生 &gt;)
    # text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")

    # 去除markdown加粗
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text) 
    text = re.sub(r'__([^_]+)__', r'\1', text)

    # 4.3 清理多余的空行 (非常重要，合并 2 个以上的换行为 2 个)
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # --- 1. 最后处理文件引入 {* ... *}  ---
    seen_files = set()

    def load_file_content(match):
        tag_content = match.group(0)
        content_inner = tag_content.replace('{*', '').replace('*}', '').strip()
        parts = content_inner.split()
        if not parts: return ""
        
        raw_path = parts[0]
        
        if "docs_src" in raw_path:
            start_index = raw_path.find("docs_src")
            rel_path = raw_path[start_index:] 
        else:
            rel_path = raw_path.lstrip('./').replace('../', '')
        
        full_path_raw = os.path.join(root_dir, rel_path)
        dir_name = os.path.dirname(full_path_raw)
        base_name = os.path.basename(full_path_raw)
        file_name_no_ext, ext = os.path.splitext(base_name)
        
        candidate_filenames = [
            f"{file_name_no_ext}_an_py39{ext}",
            f"{file_name_no_ext}_py39{ext}",
            f"{file_name_no_ext}{ext}"
        ]

        target_content = None
        found_key = None

        for candidate in candidate_filenames:
            check_path = os.path.join(dir_name, candidate)
            if os.path.exists(check_path):
                found_key = os.path.abspath(check_path)
                if found_key in seen_files:
                    return f"\n> *[Ref: Code file `{candidate}` is already included above]*\n"
                
                try:
                    with open(check_path, 'r', encoding='utf-8') as f:
                        target_content = f.read()
                    seen_files.add(found_key)                    
                    break
                except Exception as e:
                    return f"\n[Error reading file {candidate}: {e}]\n"

        if target_content is not None:
            lang_tag = ext.lstrip('.')
            return f"\n```{lang_tag}\n{target_content}\n```\n"
        else:
            return f"\n[Warning: Code file not found. Looked for: {', '.join(candidate_filenames)} at {dir_name}]\n"

    pattern_include = r'\{\*.*?\*\}'
    text = re.sub(pattern_include, load_file_content, text)
    
    # 4.4 去除首尾空白
    return text.strip()


if __name__ == "__main__":
    # --- 配置区域 ---
    current_root_dir = os.path.dirname(os.path.abspath(__file__))
    input_md_file = "rag_test_doc.md" 

    # --- 模拟数据测试 ---
    fake_md_content = """
    # FastAPI [Image: FastAPI]

    FastAPI 框架，高性能，易于学习。

    <a href="https://github.com/fastapi/fastapi/actions?query=workflow%3ATest+event%3Apush+branch%3Amaster" target="_blank">
    <img src="https://github.com/fastapi/fastapi/workflows/Test/badge.svg" alt="Test">
    </a>
    <a href="https://pypi.org/project/fastapi" target="_blank">
        <img src="https://img.shields.io/pypi/v/fastapi?color=%2334D058&label=pypi%20package" alt="Package version">
    </a>

    Here is a code example:
    {* ../../docs_src/main.py *}

    <div class="termy">
    ```console
    $ uvicorn main:app --reload
    </div> """

    target_content = ""

    if os.path.exists(input_md_file):
        print(f"正在读取文件: {input_md_file} ...")
        with open(input_md_file, "r", encoding="utf-8") as f:
            target_content = f.read()
    else:
        print(f"⚠️ 未找到文件 '{input_md_file}'，使用模拟字符串进行测试...")
        target_content = fake_md_content

    # 执行
    processed_result = clean_fastapi_markdown(target_content, root_dir=current_root_dir)

    print("\n" + "="*20 + " 处理结果预览 " + "="*20)
    print(processed_result[:1000]) 
    print("="*50)

    output_filename = "test_clean_result.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(processed_result)

    print(f"\n✅ 处理完成！完整结果已保存至: {os.path.join(current_root_dir, output_filename)}")