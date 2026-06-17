"""Reference LoRA + PPO recipe; lazy-imports trl/transformers/peft."""

import argparse
from pathlib import Path
import torch
import yaml
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.semantic.finetune.adapter import LMPolicyAdapter

def main(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    tok = AutoTokenizer.from_pretrained(cfg['model']['name'])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lora = LoraConfig(
        r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['alpha'],
        lora_dropout=cfg['lora']['dropout'],
        target_modules=cfg['lora']['target_modules'],
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg['model']['name'],
        load_in_4bit=cfg['model']['load_in_4bit'],
        peft_config=lora,
    )

    ppo_cfg = PPOConfig(
        learning_rate=cfg['ppo']['learning_rate'],
        mini_batch_size=cfg['ppo']['mini_batch_size'],
        batch_size=cfg['ppo']['batch_size'],
        ppo_epochs=cfg['ppo']['ppo_epochs'],
        init_kl_coef=cfg['ppo']['init_kl_coef'],
        target_kl=cfg['ppo']['target_kl'],
        seed=cfg['run']['seed'],
    )
    trainer = PPOTrainer(ppo_cfg, model, tokenizer=tok)

    env = NetForgeRLEnv(
        {
            'scenario_type': cfg['env']['scenario'],
            'max_ticks': cfg['env']['max_steps_per_episode'],
        }
    )
    adapter = LMPolicyAdapter(
        env,
        controlled_agent=cfg['env']['controlled_agent'],
        seed=cfg['run']['seed'],
        invalid_penalty=cfg['env']['invalid_penalty'],
        max_hosts_in_prompt=cfg['env']['max_hosts_in_prompt'],
    )

    gen_kwargs = cfg['generation']
    out_dir = Path(cfg['run']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)


    for step in range(cfg['run']['total_steps']):
        queries = adapter.queries()
        query_ids = tok(queries, return_tensors='pt', padding=True).input_ids.to(
            trainer.accelerator.device
        )
        response_ids = trainer.generate(list(query_ids), **gen_kwargs)
        responses = tok.batch_decode(response_ids, skip_special_tokens=True)
        batch = adapter.step(responses)

        trainer.step(
            list(query_ids),
            response_ids,
            [torch.tensor(r) for r in batch.rewards],
        )

        if (step + 1) % cfg['run']['save_every'] == 0:
            trainer.save_pretrained(str(out_dir / f'step-{step + 1}'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    main(args.config)
