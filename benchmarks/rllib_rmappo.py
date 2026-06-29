import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from netforge_rl.bridges.rllib_bridge import NetForgeRLlibEnv

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class NetForgeMetricsCallback(DefaultCallbacks):
    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        for agent_id in episode.get_agents():
            agent_info = episode.last_info_for(agent_id)
            if not isinstance(agent_info, dict):
                continue
            for k, v in agent_info.items():
                if isinstance(v, (int, float)):
                    episode.custom_metrics[f'{agent_id}_{k}'] = v

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        pass


def env_creator(config):
    return NetForgeRLlibEnv(config)


if __name__ == '__main__':
    ray.init(ignore_reinit_error=True)

    register_env('netforge_marl', env_creator)

    config = (
        PPOConfig()
        .environment('netforge_marl')
        .framework('torch')
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .multi_agent(
            policies={
                'red_rmappo': (None, None, None, {'model': {'use_lstm': True}}),
                'blue_rmappo': (None, None, None, {'model': {'use_lstm': True}}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: (
                'red_rmappo' if 'red' in agent_id else 'blue_rmappo'
            ),
        )
        .training(
            train_batch_size=1024,
        )
        .callbacks(NetForgeMetricsCallback)
    )

    algo = config.build()

    print('[Training Red (RMAPPO) vs Blue (RMAPPO) simultaneously...')
    for i in range(100):
        result = algo.train()
        print(f'\n--- Iteration {i} ---')
        print(f'Mean Reward: {result.get("episode_reward_mean", 0.0)}')
        metrics = result.get('custom_metrics', {})
        if metrics:
            print('Custom Security Metrics:')
            for k, v in list(metrics.items())[:10]:
                print(f'  {k}: {v:.2f}')

    print('\n[+] Training completed successfully!')
