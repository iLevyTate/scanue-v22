# **SCANUE-V22**

[![DOI](https://zenodo.org/badge/893601857.svg)](https://doi.org/10.5281/zenodo.14510406)

![SCANUE-V22Logo](https://github.com/user-attachments/assets/35f53dfa-5b63-4f5a-8fc2-643ddad8ab28)

## **Overview**
SCANUE v22 is a brain-inspired, **multi-agent** CLI that orchestrates specialized “PFC region” agents using **LangGraph**. It focuses on decomposing a task (DLPFC) and then conditionally invoking only the necessary specialist agents (VMPFC/OFC/ACC) before final integration (MPFC).

## **Name Change Notification**
This repository was previously referred to as SCANJS, a deprecated project by another developer. To reflect the enhancements introduced—such as human-in-the-loop (HITL) mechanisms and customized fine-tuned models—the project has been rebranded as SCANUE-V22.

For clarity:
- Instances of "SCANJS" in older documentation or code refer to pre-rebranding materials
- The current version reflects multiple iterations leading to this enhanced release

## **Cognitive Agents**
- **DLPFC Agent:** Task delegation and executive control
- **VMPFC Agent:** Emotional regulation and risk assessment
- **OFC Agent:** Reward processing and outcome evaluation
- **ACC Agent:** Conflict detection and error monitoring
- **MPFC Agent:** Value-based decision-making

![SCANUE-V22Info](https://github.com/user-attachments/assets/d26044f7-ac85-44ea-b358-90b373bcf452)

## **Technical Requirements**
- **Python:** 3.8+
- **An LLM provider**: OpenAI, Ollama (local), or HuggingFace (endpoint/TGI)
- **Environment variables (only if needed by your provider)**:
  - OpenAI: `OPENAI_API_KEY`
  - HuggingFace: `HUGGINGFACEHUB_API_TOKEN`
  - Legacy fallback model names (optional): `DLPFC_MODEL`, `VMPFC_MODEL`, `OFC_MODEL`, `ACC_MODEL`, `MPFC_MODEL`

## **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/iLevyTate/scanue-v22.git
   cd scanue-v22
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Set up environment variables** in a `.env` file (recommended)

4. **Run the application:**
   ```bash
   python main.py
   ```

## **Configuration**
The primary configuration is `config/agents.yaml`. Each agent can use a different provider/model:

- **Ollama (local)**: set `provider: "ollama"` and (optionally) `base_url` (default is `http://localhost:11434`)
- **OpenAI**: set `provider: "openai"` and either set `OPENAI_API_KEY` or put `api_key:` in the YAML
- **HuggingFace**: set `provider: "huggingface"` and either set `HUGGINGFACEHUB_API_TOKEN` or put `api_key:` in the YAML

See `docs/local_models.md` for examples and recommendations.

## **Workflow**
1. User inputs a task or problem
2. **DLPFC Agent:** Breaks down the task and delegates which specialist agents are needed
3. Specialized agents run (only if delegated):
   - **VMPFC:** Emotional regulation
   - **OFC:** Reward processing
   - **ACC:** Conflict detection
4. **MPFC:** Integrates all prior insights into the final response
5. (Optional) User provides feedback (persisted to `feedback_history.json`)

## **Testing**
Run the test suite:
```bash
pytest tests/
```

## **Architecture**

Key modules:

- `main.py`: CLI entrypoint, feedback persistence, session logging
- `workflow.py`: LangGraph workflow graph (stages + dynamic delegation)
- `agents/`: agent implementations and provider/model factory
- `config/agents.yaml`: per-agent model/provider configuration
- `feedback_history.json`: persistent Human-in-the-Loop (HITL) feedback
- `logs/`: per-run session logs

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Contributing**
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## **Acknowledgments**
- Thanks to all contributors who have helped shape SCANUE-V22
- Special thanks to the cognitive science community for their research and insights
