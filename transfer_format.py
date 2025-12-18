import json

def convert_jsonl_to_json_list(input_file, output_file):
    data_list = []
    
    # 1. 讀取 JSONL (一行一行讀)
    print(f"正在讀取 {input_file} ...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): # 忽略空行
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"跳過錯誤行: {e}")

    # 2. 寫入標準 JSON List (整個存成一個 List)
    print(f"正在轉換並寫入 {output_file} ...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
    
    print("完成！")

# --- 設定路徑 ---
# 把這裡改成你的實際檔案路徑
input_path = "/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/data/selfplay_Llama321B_SPRecSet_4096_1/valid.json"      # 原始的 JSONL 檔案
output_path = "/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/data/selfplay_Llama321B_SPRecSet_4096_1/valid_fixed.json" # 轉換後的檔案

if __name__ == "__main__":
    convert_jsonl_to_json_list(input_path, output_path)