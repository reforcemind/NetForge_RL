# Scenarios

Pick a family and a topology. No simulator patch.

Families: `ransomware`, `apt_espionage`, `cloud_hybrid`, `iot_grid`, `ot_stuxnet`.

Packs: `hospital_ransomware`, `cloud_iam`, `enterprise_apt`, `finance`,
`iot_fleet`, `ot_plant`, `zero_trust`.

```bash
netforge run hospital_ransomware
netforge run examples/my_scenario.yaml
```

Reward knobs (`RewardDesignWrapper`): `sla_only`, `security_only`,
`fp_penalized`, `constrained_sla`, `safety_critical`, `sparse`.

YAML lives in `netforge_rl/scenarios/packs/`.
