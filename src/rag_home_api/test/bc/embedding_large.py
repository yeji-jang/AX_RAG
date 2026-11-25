"""교안 OpenAIEmbeddings + PGVector로 large 임베딩을 저장한다."""
import argparse
from lesson_pipeline import save_documents

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=['v0', 'v1'], default='v0')
    args = parser.parse_args()
    save_documents(args.version, 'large')
