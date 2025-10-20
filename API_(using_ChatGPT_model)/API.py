import openai

openai.api_key = "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"   # your key

# Define your fixed system prompt here
SYSTEM_PROMPT = """You are an intelligent agent that can give 5 most reliable online materials that are related to the given content. You can also give link and summarize the content of each website and you can re-check whether each website is still able to be found or not.
"""

# Initialize the conversation
messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# Read question text from file
with open("D:/PYlearning/text_recognition_project/questions.txt", "r", encoding="utf-8") as file:
    user_question = file.read().strip()

if user_question:
    messages.append({"role": "user", "content": user_question})

    # Send request to the model
    chat = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )

    # Extract reply
    reply = chat.choices[0].message.content

    # Write answer to file
    with open("D:/PYlearning/text_recognition_project/answer.txt", "w", encoding="utf-8") as file:
        file.write("The answer: ")
        file.write(reply)

    print("Response saved to answer.txt")
else:
    print("questions.txt is empty — nothing to process.")
