# **SCANUE-V22**

[![DOI](https://zenodo.org/badge/893601857.svg)](https://doi.org/10.5281/zenodo.14510406)

![SCANUE-V22Logo](https://github.com/user-attachments/assets/35f53dfa-5b63-4f5a-8fc2-643ddad8ab28)

## **Overview**
SCANUE aims to develop AI-based extensions of the PFC by creating AI agents that simulate various PFC functions to assist in real-time cognitive tasks. Built using modern AI technologies, this project represents a significant step forward in cognitive augmentation and decision science.

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
- **OpenAI API Key**
- **Required Environment Variables:**
  ```plaintext
  OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
  DLPFC_MODEL=DLPFC_MODEL_ID_HERE
  ACC_MODEL=ACC_MODEL_ID_HERE
  OFC_MODEL=OFC_MODEL_ID_HERE
  VMPFC_MODEL=VMPFC_MODEL_ID_HERE
  MPFC_MODEL=MPFC_MODEL_ID_HERE
  ```

## **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/SCANUE-V22.git
   cd SCANUE-V22
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** in `.env` file

4. **Run the application:**
   ```bash
   python main.py
   ```

## **Workflow**
1. User inputs a task or problem
2. **DLPFC Agent:** Breaks down the task and delegates subtasks
3. Specialized agents process their aspects:
   - **VMPFC:** Emotional regulation
   - **OFC:** Reward processing
   - **ACC:** Conflict detection
   - **MPFC:** Value assessment
4. Results are integrated and presented to the user
5. User provides feedback for continuous improvement

## **Testing**
Run the test suite:
```bash
pytest tests/
```

## **Architecture**

### **Data Flow Diagram**
```mermaid
%%{init: {'flowchart': {'defaultRenderer': 'elk'}, 'themeVariables': { 'primaryColor': '#e0ffe0', 'primaryBorderColor': '#32cd32', 'primaryTextColor': '#006400' }}}%%
flowchart TB
    subgraph UserInteraction
        A[User Input] --> B[Task Description]
        Y[System Response] --> Z[User Feedback]
    end

    subgraph CoreProcessing
        C[DLPFC Agent] -->|Task Delegation| D[Task Breakdown]
        D -->|Parallel Processing| E[Specialized Agents]
    end

    subgraph SpecializedProcessing
        E -->|Emotional| F[VMPFC Agent]
        E -->|Reward| G[OFC Agent]
        E -->|Conflict| H[ACC Agent]
        E -->|Value| I[MPFC Agent]
    end

    subgraph ResultIntegration
        F & G & H & I -->|Results| J[Integration]
        J -->|Final Output| K[Response Generation]
    end

    B -->|Input| C
    K -->|Output| Y
    Z -->|Feedback Loop| C
```

### **User Interaction Flow**
```mermaid
%%{init: {'flowchart': {'defaultRenderer': 'elk'}, 'themeVariables': { 'primaryColor': '#fffacd', 'primaryBorderColor': '#ffd700', 'primaryTextColor': '#b8860b' }}}%%
sequenceDiagram
    participant U as User
    participant D as DLPFC Agent
    participant S as Specialized Agents
    participant I as Integration

    U->>D: Submit Task
    activate D
    D->>D: Parse Task
    D->>S: Delegate Subtasks
    activate S
    S->>S: Process Subtasks
    S->>I: Send Results
    deactivate S
    activate I
    I->>I: Integrate Results
    I->>U: Present Response
    deactivate I
    U->>D: Provide Feedback
    D->>D: Update State
    deactivate D

    Note over U,I: Feedback Loop Continues
```

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
