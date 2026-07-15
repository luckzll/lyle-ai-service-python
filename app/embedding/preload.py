from app.embedding.client import warm_up_embedding_model


def main() -> None:
    status = warm_up_embedding_model()
    print(f"Embedding model preloaded: {status['provider']} / {status['model']}")


if __name__ == "__main__":
    main()
