from steps import feature_engineering as fe_steps


def feature_engineering(author_full_names: list[str], wait_for: str | list[str] | None = None) -> None:
    raw_documents = fe_steps.query_data_warehouse(author_full_names)

    cleaned_documents = fe_steps.clean_documents(raw_documents)
    fe_steps.load_to_vector_db(cleaned_documents)

    embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)
    fe_steps.load_to_vector_db(embedded_documents)
