from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350.000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever properly.

"""


# # Example 1 :  (Character Text Splitter)
# Only splits according to the separator arg

# splitter1= CharacterTextSplitter(
#     separator="\n", # ["\n\n", "\n", ".", " ", ""]
#     chunk_size=100,
#     chunk_overlap=0
# )

# chunks1 = splitter1.split_text(tesla_text)

# for i, chunk in enumerate(chunks1, 1):
#     print(f"Chunk {i}: ({len(chunk)} chars)")
#     print(f"{chunk}\n")

# # Example 2 : (Recursive Character Text Splitter)

splitter2 = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " ", ""],
    chunk_size=100,
    chunk_overlap=0
)

chunks2 = splitter2.split_text(tesla_text)

for i, chunk in enumerate(chunks2, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f"{chunk}\n")