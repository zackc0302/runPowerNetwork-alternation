# experiments/custom_ppo_trainer.py

from ray.rllib.algorithms.ppo import PPOConfig

class CustomPPOTrainer(PPOTrainer):
    def training_step(self):
        # 調用父類別的 training_step
        train_results = PPOTrainer.training_step(self)

        # train_results = super(CustomPPOTrainer(), self).training_step

        # 從 callback 抓出我們的自訂 flag
        sync_callback = None
        for cb in self.callbacks.callbacks:
            if hasattr(cb, "substation_update_allowed"):
                sync_callback = cb
                break

        if sync_callback is not None and not sync_callback.substation_update_allowed:
            # 這裡直接刪掉中層的學習結果
            if "choose_substation_agent" in train_results.get("info", {}).get("learner", {}):
                print("[CustomPPOTrainer] 跳過中層更新（choose_substation_agent）！")
                train_results["info"]["learner"].pop("choose_substation_agent")

        return train_results
