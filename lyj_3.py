import os
import shutil

def extract_alternate_images(source_folder, dest_folder):
    """
    从源文件夹中隔一张提取一张图片，并保存到新文件夹
    """
    # 1. 如果目标文件夹不存在，则自动创建
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"已创建新文件夹: {dest_folder}")

    # 2. 获取源文件夹中的所有图片文件
    # 定义常见的图片后缀名，防止把非图片文件（如 .txt, .DS_Store）混入
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    
    # 过滤出图片文件并按名称排序（保证每次提取的顺序一致）
    files = [f for f in os.listdir(source_folder) 
             if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(source_folder, f))]
    files.sort()

    if not files:
        print("源文件夹中没有找到图片！")
        return

    # 3. 核心逻辑：利用 Python 切片 [::2] 隔一个取一个
    # 如果你想取第2、4、6...张，可以改成 files[1::2]
    selected_files = files[::2] 

    print(f"源文件夹共找到图片: {len(files)} 张")
    print(f"准备提取并复制: {len(selected_files)} 张")

    # 4. 遍历提取的列表，复制到新文件夹
    for i, file_name in enumerate(selected_files):
        src_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)
        
        # 使用 shutil.copy2 可以保留文件的原始创建时间和修改时间等元数据
        shutil.copy2(src_path, dest_path)
        
        # 打印进度（每处理 100 张提示一次）
        if (i + 1) % 100 == 0:
            print(f"已复制 {i + 1} 张图片...")

    print(f"✅ 提取完成！所有筛选出的图片已成功保存至: {dest_folder}")

# ==========================================
# 使用方法：在这里修改你的文件夹路径
# ==========================================
if __name__ == "__main__":
    # 注意：Windows 路径中的反斜杠 \ 需要写成双反斜杠 \\，或者在字符串前面加 r
    # 例如：r"D:\images\source"
    
    SOURCE_DIR = r"/root/autodl-tmp/image/images"   # 替换为你存放 1000 多张照片的文件夹路径
    DEST_DIR = r"/root/autodl-tmp/image"     # 替换为你要保存新照片的文件夹路径

    extract_alternate_images(SOURCE_DIR, DEST_DIR)