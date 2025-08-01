# query_rewriter.py
from InstructorEmbedding import INSTRUCTOR

# Load the Instructor model once (you can switch to 'instructor-xl' if preferred)
instructor_model = INSTRUCTOR('hkunlp/instructor-base')

def generate_semantic_query(prompt: str, theme: str, main: str, sub: str) -> list:
    """
    Generate a semantically rich embedding for a user prompt using thematic context.

    Args:
        prompt (str): The raw user question.
        theme (str): The selected theme.
        main (str): The main category under the theme.
        sub (str): The subcategory under the main category.

    Returns:
        List[float]: The embedding vector for ChromaDB search.
    """
    task_instruction = "Represent the user query for retrieving Jewish Torah sources:"
    combined_input = (
        f"Theme: {theme}\n"
        f"Main Category: {main}\n"
        f"Subcategory: {sub}\n"
        f"User Question: {prompt}"
    )
    return instructor_model.encode([[task_instruction, combined_input]])[0]
