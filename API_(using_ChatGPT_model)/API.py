import openai
openai.api_key = "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"
messages = [ {"role": "system", "content": 
              "You are a intelligent assistant."} ]
while True:
    # Getting data from the questions.txt file
    content = "nothing"
    with open("D:/PYlearning/text_recognition_project/questions.txt", 'r') as file:
        content = file.read()
        # print("The content in the file is:", content)
    message = content

    # Processing using API
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-4", messages=messages
        )
    reply = chat.choices[0].message.content
    
    # Output the answer to the answer.txt file
    with open("D:/PYlearning/text_recognition_project/answer.txt", "w") as file:
        file.write("The answer: ")
        file.write(reply)

    messages.append({"role": "assistant", "content": reply})

    break