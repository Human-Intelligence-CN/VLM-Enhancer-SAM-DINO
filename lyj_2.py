import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import glob
import torch
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm 
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# [配置中心 - 危险驾驶行为监测 精确升级版]
# =====================================================================
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dino_dir": "/root/autodl-tmp/GroundingDINO",
    "sam2_dir": "/root/autodl-tmp/sam2",
    "lavis_dir": "/root/autodl-tmp/LAVIS", 
    
    "image_path": "/root/autodl-tmp/data/driver_images/img-1.jpg",        
    "output_path": "/root/autodl-tmp/output_driver/output_res.png", 
    
    "batch_input_dir": "/root/autodl-tmp/image", 
    "batch_output_dir": "/root/autodl-tmp/output_driver",              
    
    "gsd": 0.003,  # 地面采样距离: 0.003 米 (3毫米)
    
    # [核心优化 1] 提示词工程升级：采用“具体物体+方位/状态”的高显性描述，提升 DINO 锚定置信度
    "dino_classes": [
        'mobile phone in driver hand',              # 强化“手机”这一物理实体的抓取
        'driver chest without a seatbelt',          # 强化胸前无安全带特征
        'driver face with closed eyes sleeping',    # 强化面部闭眼特征
        'cigarette in driver mouth or hand',        # 强化香烟实体
        'driver hands off the steering wheel'       # 动作特征保留
    ], 
    
    # [核心优化 2] 阈值调优：微升至 0.28 以减少环境背景产生的低置信度误报
    "box_threshold": 0.28,  
    "text_threshold": 0.28, 
    
    "prompt_step1": "Analyze the driver's state in this cabin image. Please describe any dangerous or distracted driving behaviors such as using a mobile phone, sleeping, smoking, or not wearing a seatbelt. Be highly specific about objects in their hands.",
}

sys.path.append(CONFIG["dino_dir"])
sys.path.append(CONFIG["sam2_dir"])
sys.path.append(CONFIG["lavis_dir"])

from groundingdino.util.inference import Model as DINOModel
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from lavis.models import load_model_and_preprocess 

# =====================================================================
# [核心系统类]
# =====================================================================
class DangerousDrivingPerceptionSystem:
    def __init__(self, config):
        self.cfg = config
        self.device = self.cfg["device"]
        self.load_models()

    def load_models(self):
        print(f"\n[🔧] 正在初始化危险驾驶行为监测系统 (设备: {self.device})...")
        print(" -> 加载 InstructBLIP (LAVIS Vicuna-7B 宏观认知中枢)...")
        self.ib_model, self.ib_vis_processors, _ = load_model_and_preprocess(
            name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=self.device
        )
        print(" -> 加载 Grounding DINO (开放词汇定位桥梁)...")
        dino_config_path = os.path.join(self.cfg["dino_dir"], "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        dino_weight_path = os.path.join(self.cfg["dino_dir"], "weights/groundingdino_swint_ogc.pth")
        self.dino_model = DINOModel(model_config_path=dino_config_path, model_checkpoint_path=dino_weight_path, device=self.device)
        
        print(" -> 加载 SAM 2 (微观测量中枢)...")
        sam2_checkpoint = os.path.join(self.cfg["sam2_dir"], "checkpoints/sam2_hiera_large.pt")
        self.sam2_base = build_sam2("sam2_hiera_l.yaml", sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_base)
        print("[✅] 所有模型加载完毕！系统就绪。\n")

    def step1_macro_understanding(self, image_pil, verbose=True):
        if verbose: print(">>> [步骤 1] InstructBLIP 执行零样本宏观定性...")
        image_tensor = self.ib_vis_processors["eval"](image_pil).unsqueeze(0).to(self.device)
        initial_desc = self.ib_model.generate({"image": image_tensor, "prompt": self.cfg["prompt_step1"]})[0].strip()
        return initial_desc

    def step2_open_vocabulary_localization(self, image_path, verbose=True):
        if verbose: print(">>> [步骤 2] Grounding DINO 提取空间坐标...")
        image_cv = cv2.imread(image_path)
        image_area = image_cv.shape[0] * image_cv.shape[1]
        
        detections = self.dino_model.predict_with_classes(
            image=image_cv, classes=self.cfg["dino_classes"],
            box_threshold=self.cfg["box_threshold"], text_threshold=self.cfg["text_threshold"]
        )
        
        if len(detections.xyxy) == 0: return None, None, None
            
        valid_detections = []
        for i in range(len(detections.xyxy)):
            box = detections.xyxy[i]
            conf = detections.confidence[i]
            cls_id = detections.class_id[i]
            phrase = self.cfg["dino_classes"][cls_id] if cls_id is not None else "Unknown Behavior"
            
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            if box_area < (image_area * 0.8):
                valid_detections.append((box, conf, phrase))
                
        if len(valid_detections) == 0: return None, None, None
        
        # 按置信度排序，取最好的一个
        valid_detections.sort(key=lambda x: x[1], reverse=True)
        best_box, best_conf, best_phrase = valid_detections[0]
        return best_box, best_conf, best_phrase

    def step3_micro_measurement(self, image_pil, bbox, verbose=True):
        if verbose: print(">>> [步骤 3] SAM 2 执行微观解算...")
        self.sam2_predictor.set_image(np.array(image_pil))
        masks, _, _ = self.sam2_predictor.predict(box=np.array(bbox), multimask_output=False)
        best_mask = masks[0] 
        
        gsd = self.cfg["gsd"]
        physical_area = np.sum(best_mask) * (gsd ** 2)
        
        y_idx, x_idx = np.where(best_mask > 0)
        physical_length = max(np.max(y_idx) - np.min(y_idx), np.max(x_idx) - np.min(x_idx)) * gsd if len(y_idx) > 0 else 0
        return best_mask, physical_area, physical_length

    def step4_final_report(self, image_pil, initial_desc, area, length, verbose=True):
        prompt = (f"Based on visual observation: '{initial_desc}', and SAM 2 measurements: Area = {area:.4f} m², "
                  f"Max Length = {length:.4f} m. Write a professional assessment report explicitly stating these numbers regarding the driver's state.")
        image_tensor = self.ib_vis_processors["eval"](image_pil).unsqueeze(0).to(self.device)
        return self.ib_model.generate({"image": image_tensor, "prompt": prompt}, max_length=512, min_length=50, repetition_penalty=1.5)[0]

    def visualize_and_save(self, image_pil, bbox, mask, area, length, phrase, out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(np.array(image_pil))
        
        mask_img = mask.reshape(mask.shape[-2], mask.shape[-1], 1) * np.array([30/255, 144/255, 255/255, 0.6]).reshape(1, 1, -1)
        plt.gca().imshow(mask_img)
        
        plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor='red', linewidth=2))
        
        label_text = f"Type: {phrase}\nArea: {area:.4f} m²\nLength: {length:.4f} m"
        plt.gca().text(bbox[0], bbox[1] - 10, label_text, color='white', weight='bold', bbox=dict(facecolor='red', alpha=0.6))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def generate_research_dashboard(self, df, output_dir):
        """生成顶刊级科研数据仪表盘"""
        print("\n[📊] 正在生成顶级科研级数据仪表盘...")
        sns.set_theme(style="ticks", context="paper", font_scale=1.2)
        npg_palette = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4"]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 1. 行为类型占比饼图
        type_counts = df['Behavior Type'].value_counts()
        axes[0].pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', colors=npg_palette, startangle=140, explode=[0.05]*len(type_counts))
        axes[0].set_title('Distribution of Dangerous Driving Behaviors', fontweight='bold')
        
        # 2. 物理面积分布直方图
        sns.histplot(data=df, x='Area (m²)', kde=True, ax=axes[1], color=npg_palette[1])
        axes[1].set_title('Distribution of Object/Feature Area', fontweight='bold')
        axes[1].set_xlabel('Area (Square Meters)')
        axes[1].set_ylabel('Image Count')
        
        # 3. 置信度 vs 面积散点图
        sns.scatterplot(data=df, x='Area (m²)', y='Confidence', hue='Behavior Type', palette=npg_palette[:len(type_counts)], ax=axes[2], s=100, alpha=0.8)
        axes[2].set_title('Detection Confidence vs Area', fontweight='bold')
        
        sns.despine()
        plt.tight_layout()
        dashboard_path = os.path.join(output_dir, "Dashboard_Scientific_Analysis.png")
        plt.savefig(dashboard_path, dpi=300)
        plt.close()

    def generate_top_cases_grid(self, df, output_dir):
        """生成严重行为对比矩阵图"""
        print("[🖼️] 正在生成高危驾驶行为对比矩阵图...")
        top_cases = df.sort_values(by='Area (m²)', ascending=False).head(4)
        
        fig, axes = plt.subplots(len(top_cases), 2, figsize=(12, 5 * len(top_cases)))
        fig.suptitle('Critical Driving Behaviors Analysis', fontsize=20, fontweight='bold', y=1.02)
        
        if len(top_cases) == 1:
            axes = np.array([axes])
            
        for idx, (_, row) in enumerate(top_cases.iterrows()):
            orig_path = row['Original_Path']
            res_path = os.path.join(output_dir, f"res_{row['Image Name']}")
            
            axes[idx, 0].imshow(Image.open(orig_path))
            axes[idx, 0].set_title(f"Original Image: {row['Image Name']}", fontsize=14)
            axes[idx, 0].axis('off')
            
            if os.path.exists(res_path):
                axes[idx, 1].imshow(Image.open(res_path))
                axes[idx, 1].set_title(f"Processed: Area {row['Area (m²)']:.4f} m²\nType: {row['Behavior Type']}", fontsize=12)
            axes[idx, 1].axis('off')
            
        plt.tight_layout()
        grid_path = os.path.join(output_dir, "Top_Cases_Comparison_Grid.png")
        plt.savefig(grid_path, dpi=300, bbox_inches='tight')
        plt.close()

    def batch_process_directory(self):
        input_dir, output_dir = self.cfg["batch_input_dir"], self.cfg["batch_output_dir"]
        os.makedirs(output_dir, exist_ok=True)
            
        image_files = glob.glob(os.path.join(input_dir, "*.[jp][pn]g")) + glob.glob(os.path.join(input_dir, "*.jpeg"))
        if not image_files: return print(f"[警告] {input_dir} 未找到图片。请放入驾驶员视角的图片。")
            
        print(f"\n[🚀] 开始批量处理 {len(image_files)} 张图像...")
        
        records = [] 
        
        for img_path in tqdm(image_files, desc="Processing Images"):
            filename = os.path.basename(img_path)
            out_img_path = os.path.join(output_dir, f"res_{filename}")
            
            try:
                image_pil = Image.open(img_path).convert("RGB")
                initial_desc = self.step1_macro_understanding(image_pil, verbose=False)
                bbox, conf, phrase = self.step2_open_vocabulary_localization(img_path, verbose=False)
                
                if bbox is not None:
                    mask, area, length = self.step3_micro_measurement(image_pil, bbox, verbose=False)
                    final_report = self.step4_final_report(image_pil, initial_desc, area, length, verbose=False)
                    
                    # [核心优化 3] 智能风险评级：拦截大面积异常
                    risk_level = "Medium"
                    if conf > 0.5 or area > 0.15:
                        risk_level = "High"
                        
                    if phrase == "Unknown Behavior" and area > 0.1:
                        risk_level = "High (Manual Check Required)"
                        phrase = "Suspected Distraction/Occlusion"
                    
                    self.visualize_and_save(image_pil, bbox, mask, area, length, phrase, out_img_path)
                    
                    # 将本次结果存入台账字典
                    records.append({
                        "Image Name": filename,
                        "Behavior Type": phrase,
                        "Risk Level": risk_level,
                        "Confidence": float(conf),
                        "Area (m²)": round(area, 4),
                        "Max Length (m)": round(length, 4),
                        "AI Report": final_report,
                        "Original_Path": img_path
                    })
            except Exception as e:
                print(f"    [!] 跳过 {filename}: {e}")
                continue
                
        # === 数据落盘与可视化生成 ===
        if records:
            df = pd.DataFrame(records)
            excel_path = os.path.join(output_dir, "Driver_Monitoring_Report.xlsx")
            
            # 导出为标准的 Excel 格式，并删除用于辅助绘图的 Original_Path
            df_to_export = df.drop(columns=['Original_Path'])
            
            try:
                # 使用 openpyxl 引擎确保完美的 .xlsx 输出
                df_to_export.to_excel(excel_path, index=False, engine='openpyxl')
            except ImportError:
                print("\n[提示] 未找到 openpyxl，尝试使用默认引擎导出 Excel。建议运行: pip install openpyxl")
                df_to_export.to_excel(excel_path, index=False)
                
            print(f"\n[📈] 全局数据台账已保存至: {excel_path}")
            
            # 触发科研可视化
            self.generate_research_dashboard(df, output_dir)
            self.generate_top_cases_grid(df, output_dir)
            print(f"[🎉] 顶级可视化图表已生成，请前往 {output_dir} 查阅！")

if __name__ == "__main__":
    system = DangerousDrivingPerceptionSystem(CONFIG)
    system.batch_process_directory()