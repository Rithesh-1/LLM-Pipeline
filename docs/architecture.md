
```mermaid
graph TD
    subgraph User Interaction
        A[User] --> C[Pipelines];
    end

    subgraph Core Logic
        C -- Manages --> D[LLM Pipeline System Package];
        D -- Contains --> E[Domain];
        D -- Contains --> F[Application];
        D -- Contains --> G[Model];
        D -- Contains --> H[Infrastructure];
    end

    subgraph External Services
        H -- Interacts with --> J[MongoDB];
        H -- Interacts with --> K[Qdrant];
        H -- Interacts with --> L[Hugging Face];
        H -- Interacts with --> M[MLflow];
        H -- Interacts with --> N[ClearML];
    end

    subgraph CI/CD
        O[GitHub Repository] -- Triggers --> P[GitHub Actions];
    end

    F -- Uses --> E;
    G -- Uses --> E;
    H -- Supports --> F;
    H -- Supports --> G;
```

**Explanation:**

*   **User Interaction:** The user interacts with the system by running the defined pipelines.
*   **Core Logic:** The `llm_pipeline_system` package contains the core logic of the application, which is organized according to Domain-Driven Design principles:
    *   `Domain`: Defines the core business entities and data structures.
    *   `Application`: Contains the business logic, including data crawling and the RAG implementation.
    *   `Model`: Handles LLM training and inference.
    *   `Infrastructure`: Manages integrations with external services.
*   **External Services:** The system relies on several external services for its functionality:
    *   `MongoDB`: As the NoSQL database.
    *   `Qdrant`: As the vector database for the RAG system.
    *   `Hugging Face`: As the model registry.
    *   `MLflow`: For experiment tracking.
    *   `ClearML`: For experiment automation and orchestration.
*   **CI/CD:** The project uses GitHub Actions for continuous integration and deployment.
