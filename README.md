# Cybersec OpenEnv: Long-Horizon Multi-Agent Cyber Defense

A clean, production-shaped OpenEnv environment for training LLM defenders against scripted attackers walking real MITRE ATT&CK kill chains.

▶️ **Play the Environment:** [huggingface.co/spaces/Lonelyguyse1/cybersec](https://huggingface.co/spaces/Lonelyguyse1/cybersec)

---

## 1. The Problem
**Large Language Models struggle with long-horizon planning and delayed observability.** 
Cybersecurity defense is not a point-in-time classification task. Real attackers don't push one button—they steal a credential at tick *t*, sit on it, escalate at *t + d₁*, pivot at *t + d₂*, and exfiltrate at *t + d₃*. The defender operates with an honest information disadvantage: detection is lagged, alerts are noisy, and actions like isolating a server have a real business cost. We built an environment to teach LLMs how to balance proactive threat containment against operational disruption.

## 2. Tasks & Grading
To evaluate the agent's performance programmatically, the environment scores the defender across three primary tasks:
* **Detection Task:** The agent must accurately identify and confirm compromised targets using the `INVESTIGATE` action. Scored by the number of true positive confirmations.
* **Containment Task:** The agent must successfully block an attacker's in-progress lateral movement or exfiltration using `ISOLATE_ASSET`, `REVOKE_IDENTITY`, or `BLOCK_EGRESS`. Scored by the number of attack stages prevented.
* **Survival Task:** The ultimate goal is to prevent the attacker from completing the exfiltration stage. Scored via a massive terminal penalty if exfiltration occurs, or a survival bonus if the network is preserved.

## 3. The Environment
There are two agents in the room: a scripted attacker with a chosen personality (stealthy, aggressive, opportunistic) driving ground truth, and the controllable LLM defender. The defender reads partial observations (lagged alerts, investigation forensics) and chooses one of six actions per tick (e.g., `MONITOR`, `INVESTIGATE`, `ISOLATE_ASSET`). 

The reward function forces the LLM to learn surgical defense:
* **+ Reward:** Confirming a real compromise, preemptively blocking an attack path.
* **- Penalty:** False-positive isolations, invalid formatting, and a scaling "disruption penalty" for keeping critical business assets offline.

*(Read the full environment spec in `cybersec/README.md`)*

## 3. The Results
We trained Qwen2.5-1.5B-Instruct using an iterative, on-policy GRPO loop. 

* **Before Training:** The baseline models either acted randomly (destroying the network) or followed a static heuristic.
* **After Training:** The LLM learned to wait for telemetry, investigate alerts, and surgically isolate only the compromised assets, successfully preventing exfiltration across unseen topologies.

![Before/After Reward Curves](_artifacts/before_after_curves.png)
*Caption: The trained LLM consistently outperforms random and heuristic baselines, maintaining a positive cumulative reward without triggering mass-disruption penalties.*

![Training Diagnostics](_artifacts/training_diagnostics.png)
*Caption: GRPO loss, KL divergence, and per-component reward curves during the iterative on-policy training phase.*

## 4. Why It Matters
A model that can successfully navigate this environment has learned fundamental concepts of **risk-aware operational strategy**. If an LLM can learn to quarantine a compromised Kubernetes cluster while leaving the payments database online despite noisy alerts, it proves that open-source models can be fine-tuned to act as autonomous, SRE-grade SOC analysts.

---

## Install and train

```bash
pip install -e ./cybersec[dev]
pytest -q
```

**Dependencies** (from repo root; match your CUDA):

```bash
pip install -e "./cybersec[grpo]"
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
```

**Run the iterative on-policy training script**

```bash
python scripts/train_cybersec_grpo.py --output-dir ./_artifacts
```

`--fast` uses tiny budgets for a quick run. Outputs: `qwen_cybersec_lora/`, `training_log.json`, `run_manifest.json`, per-outer checkpoints under `grpo_checkpoints_outer*`.

**Notebook (Colab / Jupyter)** — same algorithm, baselines, plots, eval: [notebooks/cybersec_grpo.ipynb](notebooks/cybersec_grpo.ipynb).

---

## Server

```bash
cybersec-server
```

Docker: `docker build -t cybersec-env:latest -f cybersec/server/Dockerfile cybersec` then `docker run --rm -p 8000:8000 cybersec-env:latest`.

---

## License

MIT — see `cybersec/pyproject.toml`.
