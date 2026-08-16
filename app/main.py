from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

from app.chains.rag_chain import RAGChain

def main ():
    question = " ".join(sys.argv[1:]).strip() or "What does the company handbook say about the project?"
    chain = RAGChain()

    handbook_path = Path("data/raw/company_handbook.txt")
    if handbook_path.exists():
        chain.index_source(
            source=str(handbook_path),
            metadata={"source": "company_handbook"},
        )

    response = chain.ask(question)

    print(response.answer)

if __name__ == "__main__":
    main()