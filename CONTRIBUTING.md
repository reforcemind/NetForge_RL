# Contributing to NetForge

We welcome contributions to NetForge! This guide explains how to propose changes, submit pull requests, and contribute new scenario packs or capability probes.

## Scope of Contributions

NetForge is an environment and benchmark platform for cyber defense reinforcement learning. Contributions should focus on:

- **Scenario Packs**: Realistic YAML topologies and attack configurations under `netforge_rl/scenarios/packs/`.
- **Defensive Wrappers & Probes**: New experimental wrappers, reward formulations, and diagnostic probes.
- **Documentation & Examples**: Tutorials, API documentation, and benchmark walkthroughs.
- **Bug Fixes & Performance**: Parity improvements, simulator optimizations, and bug fixes.

*Note: RL training algorithms (PPO, MAPPO, QMIX) and proprietary exploit code belong in external repositories evaluating on NetForge.*

## Development Workflow

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/<your-username>/NetForge_RL.git
   cd NetForge_RL
   pip install -e ".[dev,docs]"
   ```

2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/my-new-probe
   ```

3. **Implement Changes and Add Tests**:
   - Write clear, concise code adhering to PEP 8 standards.
   - Include unit tests in `tests/` covering new features or bug fixes.

4. **Verify Locally**:
   Run the verification suite before committing:
   ```bash
   # Windows PowerShell
   ./scripts/verify_local.ps1

   # Linux / macOS / Git Bash
   ./scripts/verify_all.sh
   ```

5. **Submit a Pull Request**:
   - Push your branch to GitHub and open a pull request against `main`.
   - Provide a concise description of the motivation and changes made.
