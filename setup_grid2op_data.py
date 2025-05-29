import grid2op
import numpy as np
import os # 引入 os 模組

print(f"Using Grid2Op version: {grid2op.__version__}")

env_name = "rte_case14_realistic"
print(f"Loading base environment: {env_name}")
try:
    env = grid2op.make(env_name)
except Exception as e:
    print(f"Error loading base environment: {e}")
    print("Ensure the dataset is downloaded, e.g., run 'grid2op install rte_case14_realistic'")
    exit()

# *** 提前從環境獲取實際可用的 chronic 名稱 ***
all_available_chronics = [os.path.basename(p) for p in env.chronics_handler.subpaths]
print(f"Total chronics available in '{env_name}': {len(all_available_chronics)}")
if not all_available_chronics:
    print(f"Error: No chronics found in the base environment at {env.chronics_handler.path}")
    exit()

val_split_path = "grid2op_env/train_val_test_split/val_chronics.npy"
test_split_path = "grid2op_env/train_val_test_split/test_chronics.npy" # 修正了原始作者可能的路徑筆誤

print(f"Loading validation split IDs from: {val_split_path}")
print(f"Loading test split IDs from: {test_split_path}")

try:
    val_chron_ids_original = np.load(val_split_path)
    test_chron_ids_original = np.load(test_split_path)
except FileNotFoundError:
    print(f"Error: Cannot find .npy split files. Make sure you are in the project root directory.")
    print(f"Current directory: {os.getcwd()}")
    exit()

print(f"Original intended validation chronic IDs count: {len(val_chron_ids_original)}")
print(f"Original intended test chronic IDs count: {len(test_chron_ids_original)}")

# *** 將 NumPy ID 轉換為預期的 chronic 名稱 (例如 "000", "001") ***
# 假設 ID 對應文件夾名稱，不足三位補零
try:
    val_chron_names_intended = [str(int(i)).zfill(3) for i in val_chron_ids_original]
    test_chron_names_intended = [str(int(i)).zfill(3) for i in test_chron_ids_original]
except ValueError:
    print("Error: Could not convert IDs from .npy files to zero-padded strings. Check file content.")
    exit()

# *** 過濾，確保名稱存在於基礎環境中 ***
val_chron_names_filtered = [name for name in val_chron_names_intended if name in all_available_chronics]
test_chron_names_filtered = [name for name in test_chron_names_intended if name in all_available_chronics]

print(f"Number of valid intended validation chronic names found: {len(val_chron_names_filtered)}")
print(f"Number of valid intended test chronic names found: {len(test_chron_names_filtered)}")

if not val_chron_names_filtered:
     print("Error: No valid chronic names found for the validation set within the base environment.")
     print("Available base chronics:", all_available_chronics)
     print("Original intended validation IDs mapped to names:", val_chron_names_intended)
     exit()

if not test_chron_names_filtered:
     print("Error: No valid chronic names found for the test set within the base environment.")
     print("Available base chronics:", all_available_chronics)
     print("Original intended test IDs mapped to names:", test_chron_names_intended)
     exit()

try:
    print("\nPerforming direct Train / Validation / Test split...")
    # 使用 train_val_split 進行三向分割
    # 傳入 test_scen_id 和 val_scen_id
    # add_for_train, add_for_val, add_for_test 控制生成的環境名稱後綴
    nm_env_train, nm_env_val, nm_env_test = env.train_val_split(
        test_scen_id=test_chron_names_filtered, # 使用過濾後的 test chronic 名稱列表
        val_scen_id=val_chron_names_filtered,   # 使用過濾後的 val chronic 名稱列表
        add_for_train="train",                  # 後綴為 _train
        add_for_val="val",                      # 後綴為 _val
        add_for_test="test"                     # 後綴為 _test
    )
    print(f"Environment names created: train='{nm_env_train}', val='{nm_env_val}', test='{nm_env_test}'")

    # Step 2: Load final environments to verify
    print("\nLoading final split environments...")
    # 這裡我們假設 nm_env_train, nm_env_val, nm_env_test 返回的就是 env_name + 後綴
    # 如果不是，則需要根據實際返回的名稱加載
    # 但通常 grid2op 的行為是這樣的
    env_train = grid2op.make(nm_env_train) # 或者 grid2op.make(env_name + "_train")
    env_val = grid2op.make(nm_env_val)     # 或者 grid2op.make(env_name + "_val")
    env_test = grid2op.make(nm_env_test)   # 或者 grid2op.make(env_name + "_test")

    print("\nGrid2Op train/val/test environments created successfully!")
    print(f"Training set chronics: {len(env_train.chronics_handler.subpaths)}")
    print(f"Validation set chronics: {len(env_val.chronics_handler.subpaths)}")
    print(f"Test set chronics: {len(env_test.chronics_handler.subpaths)}")

    # 驗證數量是否符合預期
    # 訓練集數量 = 總數 - 驗證集數量 - 測試集數量 (假設沒有交集)
    # 注意：如果驗證集和測試集 chronic 名稱有重疊，實際分割行為可能不同。
    # 但在此處，我們是從 `.npy` 文件加載，通常它們是互斥的。
    expected_train_count = len(all_available_chronics) - len(val_chron_names_filtered) - len(test_chron_names_filtered)
    # 考慮到 val_chron_names_filtered 和 test_chron_names_filtered 可能有重疊的情況（儘管不常見於標準分割）
    # 更準確的計算是：
    combined_val_test = set(val_chron_names_filtered) | set(test_chron_names_filtered)
    expected_train_count_robust = len(all_available_chronics) - len(combined_val_test)


    print(f"Expected training set chronics (approx, assuming no overlap in specified val/test): {expected_train_count}")
    print(f"Expected training set chronics (robust, accounting for potential overlap): {expected_train_count_robust}")
    print(f"Expected validation set chronics: {len(val_chron_names_filtered)}")
    print(f"Expected test set chronics: {len(test_chron_names_filtered)}")

    # 檢查是否有 chronic 被錯誤地分配
    train_actual_names = {os.path.basename(p) for p in env_train.chronics_handler.subpaths}
    val_actual_names = {os.path.basename(p) for p in env_val.chronics_handler.subpaths}
    test_actual_names = {os.path.basename(p) for p in env_test.chronics_handler.subpaths}

    # 檢查 val 和 test 集合是否與指定的完全一致
    if set(val_chron_names_filtered) != val_actual_names:
        print(f"Warning: Mismatch in validation set chronics!")
        print(f"  Expected in val: {set(val_chron_names_filtered)}")
        print(f"  Actual in val:   {val_actual_names}")
        print(f"  Missing from val: {set(val_chron_names_filtered) - val_actual_names}")
        print(f"  Extra in val:    {val_actual_names - set(val_chron_names_filtered)}")


    if set(test_chron_names_filtered) != test_actual_names:
        print(f"Warning: Mismatch in test set chronics!")
        print(f"  Expected in test: {set(test_chron_names_filtered)}")
        print(f"  Actual in test:   {test_actual_names}")
        print(f"  Missing from test: {set(test_chron_names_filtered) - test_actual_names}")
        print(f"  Extra in test:    {test_actual_names - set(test_chron_names_filtered)}")


    # 檢查是否有重疊
    if val_actual_names & test_actual_names:
        print(f"Warning: Overlap between final validation and test sets: {val_actual_names & test_actual_names}")
    if train_actual_names & val_actual_names:
        print(f"Warning: Overlap between final train and validation sets: {train_actual_names & val_actual_names}")
    if train_actual_names & test_actual_names:
        print(f"Warning: Overlap between final train and test sets: {train_actual_names & test_actual_names}")


except Exception as e:
    print(f"\nAn error occurred during the split process: {e}")
    import traceback
    traceback.print_exc()