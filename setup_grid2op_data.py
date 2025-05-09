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
test_split_path = "grid2op_env/train_val_test_split/test_chronics.npy"

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
val_chron_names_for_val_split = [name for name in val_chron_names_intended if name in all_available_chronics]
test_chron_names_for_test_split = [name for name in test_chron_names_intended if name in all_available_chronics] # <--- 新增對 test 名稱的過濾

print(f"Number of valid intended validation chronic names found: {len(val_chron_names_for_val_split)}")
print(f"Number of valid intended test chronic names found: {len(test_chron_names_for_test_split)}") # <--- 打印過濾後的 test 數量

if len(test_chron_names_for_test_split) == 0: # <--- 檢查 test 列表是否為空
     print("Error: No valid chronic names found for the test set within the base environment.")
     print("Available base chronics:", all_available_chronics)
     print("Original intended test IDs mapped to names:", test_chron_names_intended)
     exit()

try:
    # Step 1: Split into TrainVal and Test
    print("\nPerforming first split (TrainVal / Test) using Grid2Op v1.6.4 method...")
    nm_env_trainval, nm_env_test = env.train_val_split(
        val_scen_id=test_chron_names_for_test_split, # *** 使用過濾後的 test chronic 名稱列表 ***
        add_for_val="test",
        add_for_train="trainval"
    )
    print(f"Intermediate environment names: trainval='{nm_env_trainval}', test='{nm_env_test}'")

    # Step 2: Split TrainVal into Train and Val
    print("\nLoading intermediate trainval environment...")
    env_trainval = grid2op.make(nm_env_trainval)

    available_trainval_chronics = [os.path.basename(p) for p in env_trainval.chronics_handler.subpaths]
    print(f"Number of chronics available in '{nm_env_trainval}': {len(available_trainval_chronics)}")

    # *** 從 intended 驗證集名稱中，再次過濾，確保它們還在 trainval 環境中 ***
    val_chron_names_final = [name for name in val_chron_names_for_val_split if name in available_trainval_chronics] # 使用 val_chron_names_for_val_split

    print(f"Number of chronics selected for final validation set: {len(val_chron_names_final)}")
    if len(val_chron_names_final) == 0:
        print("Error: No valid chronic names found for the validation set within the trainval environment.")
        print("Available chronics in trainval:", available_trainval_chronics)
        print("Original intended validation IDs mapped to names:", val_chron_names_intended)
        exit()

    print("Performing second split (Train / Val) using Grid2Op v1.6.4 method...")
    nm_env_train, nm_env_val = env_trainval.train_val_split(
         val_scen_id=val_chron_names_final,
         add_for_val="val",
         add_for_train="train",
         remove_from_name="_trainval$"
    )
    print(f"Final environment names: train='{nm_env_train}', val='{nm_env_val}', test='{nm_env_test}'")

    # Step 3: Load final environments to verify
    print("\nLoading final split environments...")
    env_train = grid2op.make(env_name + "_train")
    env_val = grid2op.make(env_name + "_val")
    env_test = grid2op.make(env_name + "_test") # <--- 這裡上次出錯

    print("\nGrid2Op train/val/test environments created successfully!")
    print(f"Training set chronics: {len(env_train.chronics_handler.subpaths)}")
    print(f"Validation set chronics: {len(env_val.chronics_handler.subpaths)}")
    print(f"Test set chronics: {len(env_test.chronics_handler.subpaths)}")

except Exception as e:
    print(f"\nAn error occurred during the split process: {e}")
    import traceback
    traceback.print_exc()
